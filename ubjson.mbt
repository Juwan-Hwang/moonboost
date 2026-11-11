// Universal Binary JSON, the format XGBoost writes for `.ubj` model files.
//
// XGBoost uses big-endian scalars and the optimized container form
// (`[$d#l<count>` followed by raw float32 values) for its large arrays. The
// reader below turns a document into MoonBit's `Json`, so the same loader
// serves both model formats.

///|
priv struct UbjReader {
  data : BytesView
  mut pos : Int32
}

///|
fn UbjReader::byte(self : UbjReader) -> Byte raise ModelError {
  if self.pos >= self.data.length() {
    raise InvalidModel("UBJSON ends early")
  }
  let b = self.data[self.pos]
  self.pos += 1
  b
}

///|
fn UbjReader::be(self : UbjReader, n : Int32) -> UInt64 raise ModelError {
  if self.pos + n > self.data.length() {
    raise InvalidModel("UBJSON ends early")
  }
  let mut v = 0UL
  for i in 0..<n {
    v = (v << 8) | self.data[self.pos + i].to_uint64()
  }
  self.pos += n
  v
}

///|
/// A scalar of type `marker`, as a double when it is numeric.
fn UbjReader::number(
  self : UbjReader,
  marker : Byte,
) -> Double raise ModelError {
  match marker {
    b'i' => (self.be(1).to_int() << 24 >> 24).to_double()
    b'U' => self.be(1).to_double()
    b'I' => (self.be(2).to_int() << 16 >> 16).to_double()
    b'l' => self.be(4).to_int().to_double()
    b'L' => self.be(8).reinterpret_as_int64().to_double()
    b'd' => Float::reinterpret_from_uint(self.be(4).to_uint()).to_double()
    b'D' => self.be(8).reinterpret_as_double()
    _ => raise InvalidModel("UBJSON: \{marker.to_int()} is not a number type")
  }
}

///|
fn UbjReader::length(self : UbjReader) -> Int32 raise ModelError {
  let n = self.number(self.byte())
  if n < 0.0 || n > self.data.length().to_double() * 8.0 + 8.0 {
    raise InvalidModel("UBJSON: bad length \{n}")
  }
  n.to_int()
}

///|
fn UbjReader::text(self : UbjReader) -> String raise ModelError {
  let n = self.length()
  if self.pos + n > self.data.length() {
    raise InvalidModel("UBJSON ends early")
  }
  let s = @utf8.decode(self.data[self.pos:self.pos + n]) catch {
    _ => raise InvalidModel("UBJSON: string is not UTF-8")
  }
  self.pos += n
  s
}

///|
/// The optional `$type` and `#count` that follow `[` or `{`.
fn UbjReader::header(self : UbjReader) -> (Byte, Int32) raise ModelError {
  let mut typ = b'\x00'
  let mut count = -1
  if self.pos < self.data.length() && self.data[self.pos] == b'$' {
    self.pos += 1
    typ = self.byte()
  }
  if self.pos < self.data.length() && self.data[self.pos] == b'#' {
    self.pos += 1
    count = self.length()
  } else if typ != b'\x00' {
    raise InvalidModel("UBJSON: typed container without a count")
  }
  (typ, count)
}

///|
fn UbjReader::value(
  self : UbjReader,
  marker : Byte,
  depth : Int32,
) -> Json raise ModelError {
  if depth > 64 {
    raise InvalidModel("UBJSON nests too deep")
  }
  match marker {
    b'Z' => Json::null()
    b'T' => Json::boolean(true)
    b'F' => Json::boolean(false)
    b'i' | b'U' | b'I' | b'l' | b'L' | b'd' | b'D' =>
      Json::number(self.number(marker))
    b'C' => Json::string(self.byte().to_int().unsafe_to_char().to_string())
    b'S' => Json::string(self.text())
    b'H' => {
      let s = self.text()
      let d = @string.parse_double(s) catch {
        _ => raise InvalidModel("UBJSON: bad high-precision number")
      }
      Json::number(d)
    }
    b'[' => {
      let (typ, count) = self.header()
      let out : Array[Json] = []
      if count >= 0 {
        for _ in 0..<count {
          let m = if typ != b'\x00' { typ } else { self.next_marker() }
          out.push(self.value(m, depth + 1))
        }
      } else {
        while true {
          let m = self.next_marker()
          if m == b']' {
            break
          }
          out.push(self.value(m, depth + 1))
        }
      }
      Json::array(out)
    }
    b'{' => {
      let (typ, count) = self.header()
      let out : Map[String, Json] = Map([])
      if count >= 0 {
        for _ in 0..<count {
          let k = self.text()
          let m = if typ != b'\x00' { typ } else { self.next_marker() }
          out[k] = self.value(m, depth + 1)
        }
      } else {
        while true {
          if self.pos < self.data.length() && self.data[self.pos] == b'}' {
            self.pos += 1
            break
          }
          let k = self.text()
          out[k] = self.value(self.next_marker(), depth + 1)
        }
      }
      Json::object(out)
    }
    _ => raise InvalidModel("UBJSON: unknown marker \{marker.to_int()}")
  }
}

///|
/// The next value marker, skipping no-ops.
fn UbjReader::next_marker(self : UbjReader) -> Byte raise ModelError {
  let mut m = self.byte()
  while m == b'N' {
    m = self.byte()
  }
  m
}

///|
fn ubjson_parse(data : BytesView) -> Json raise ModelError {
  let r = { data, pos: 0, }
  let v = r.value(r.next_marker(), 0)
  if r.pos != data.length() {
    raise InvalidModel("UBJSON: trailing bytes")
  }
  v
}

///|
/// Loads a model saved by XGBoost as UBJSON (`booster.save_model("m.ubj")`).
pub fn Model::from_xgboost_ubjson(data : BytesView) -> Model raise ModelError {
  model_from_src(Src::from_json(ubjson_parse(data)))
}
