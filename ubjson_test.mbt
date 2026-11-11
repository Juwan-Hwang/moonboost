///|
fn hex_bytes(s : String) -> Bytes {
  let out : Array[Byte] = []
  let digit = (c : Int32) => {
    if c >= 'a'.to_int() {
      c - 'a'.to_int() + 10
    } else {
      c - '0'.to_int()
    }
  }
  let mut hi = -1
  for i in 0..<s.length() {
    let c = s.code_unit_at(i).to_int()
    if c == '\n'.to_int() {
      continue
    }
    if hi < 0 {
      hi = digit(c)
    } else {
      out.push((hi * 16 + digit(c)).to_byte())
      hi = -1
    }
  }
  Bytes::from_array(out)
}

///|
test "UBJSON models predict exactly like their JSON originals" {
  for pair in ubj_fixtures {
    let (hex, f) = pair
    let ubj = @moonboost.Model::from_xgboost_ubjson(hex_bytes(hex))
    let json = @moonboost.Model::from_xgboost_json(f.model)
    let x = to_floats(f.x)
    let rows = x.length() / f.num_feature
    for r in 0..<rows {
      let row = x[r * f.num_feature:(r + 1) * f.num_feature]
      let a = ubj.predict(row)
      let b = json.predict(row)
      for k in 0..<a.length() {
        if a[k].reinterpret_as_uint() != b[k].reinterpret_as_uint() {
          fail("\{f.name}: row \{r} output \{k}: \{a[k]} != \{b[k]}")
        }
      }
      let m = ubj.predict_margin(row)
      for g in 0..<f.num_group {
        if m[g].reinterpret_as_uint() != f.margin[r * f.num_group + g] {
          fail("\{f.name}: row \{r} margin differs from XGBoost")
        }
      }
    }
  }
}

///|
test "malformed UBJSON is rejected" {
  for bad in [b"", b"{", b"[$d#U\x02\x00", b"{U\x01a", b"Q", b"{}}"] {
    let ok = try {
      ignore(@moonboost.Model::from_xgboost_ubjson(bad))
      true
    } catch {
      _ => false
    }
    assert_false(ok)
  }
}
