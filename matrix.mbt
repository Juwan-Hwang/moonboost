///|
/// Row-major dense feature matrix. Missing values are NaN.
pub struct DenseMatrix {
  data : FixedArray[Float]
  rows : Int32
  cols : Int32
}

///|
pub fn DenseMatrix::new(
  data : Array[Float],
  cols~ : Int32,
) -> DenseMatrix raise ModelError {
  if cols <= 0 || data.length() % cols != 0 {
    raise InvalidModel(
      "data length \{data.length()} is not a multiple of \{cols} columns",
    )
  }
  { data: FixedArray::from_array(data), rows: data.length() / cols, cols, }
}

///|
/// Like `DenseMatrix::new`, but takes ownership of `data` without copying it.
pub fn DenseMatrix::from_fixed(
  data : FixedArray[Float],
  cols~ : Int32,
) -> DenseMatrix raise ModelError {
  if cols <= 0 || data.length() % cols != 0 {
    raise InvalidModel(
      "data length \{data.length()} is not a multiple of \{cols} columns",
    )
  }
  { data, rows: data.length() / cols, cols, }
}

///|
/// View of row `i`.
pub fn DenseMatrix::row(self : DenseMatrix, i : Int32) -> ArrayView[Float] {
  self.data[i * self.cols:(i + 1) * self.cols]
}

///|
fn nan() -> Float {
  let inf = Float::from_double(1.0e300)
  inf - inf
}

///|
fn parse_cell(cell : StringView, line : Int32) -> Float raise ModelError {
  let s = cell.trim(chars=" \t\r")
  match s {
    "" | "NA" | "NaN" | "nan" | "?" => nan()
    _ =>
      Float::from_double(
        @string.parse_double(s) catch {
          _ => raise InvalidModel("line \{line}: not a number: \{s}")
        },
      )
  }
}

///|
/// Parse numeric CSV. Empty cells, `NA`, `NaN` and `?` become missing values.
/// When `header` is true the first non-empty line is skipped.
pub fn DenseMatrix::from_csv(
  text : StringView,
  header~ : Bool = false,
  sep~ : StringView = ",",
) -> DenseMatrix raise ModelError {
  if sep.length() != 1 {
    return csv_by_split(text, header, sep)
  }
  csv_scan(text, header, sep.unsafe_get(0).to_int())
}

///|
/// Parse numeric CSV from its UTF-8 bytes (a file read with
/// `read_file_to_bytes`), without decoding it to a string first; the same
/// rules and values as `from_csv`. `sep` is one ASCII character.
pub fn DenseMatrix::from_csv_bytes(
  bytes : BytesView,
  header~ : Bool = false,
  sep~ : Char = ',',
) -> DenseMatrix raise ModelError {
  csv_scan(bytes, header, sep.to_int())
}

///|
/// Text a CSV is read from: code units of a string, or bytes of UTF-8.
priv trait CsvText {
  fn csv_len(Self) -> Int32
  fn csv_at(Self, Int32) -> Int32
  fn csv_slice(Self, Int32, Int32) -> String
}

///|
impl CsvText for StringView with fn csv_len(self) {
  self.length()
}

///|
impl CsvText for StringView with fn csv_at(self, i) {
  self.unsafe_get(i).to_int()
}

///|
impl CsvText for StringView with fn csv_slice(self, a, b) {
  self.clamped_view(start=a, end=b).to_owned()
}

///|
impl CsvText for BytesView with fn csv_len(self) {
  self.length()
}

///|
impl CsvText for BytesView with fn csv_at(self, i) {
  self.unsafe_get(i).to_int()
}

///|
impl CsvText for BytesView with fn csv_slice(self, a, b) {
  let buf = StringBuilder()
  for i in a..<b {
    buf.write_char(Int32::unsafe_to_char(self.unsafe_get(i).to_int()))
  }
  buf.to_string()
}

///|
fn[T : CsvText] csv_scan(
  text : T,
  header : Bool,
  comma : Int32,
) -> DenseMatrix raise ModelError {
  let n = text.csv_len()
  let data : Array[Float] = []
  let mut cols = -1
  let mut skip = header
  let mut line_no = 0
  let mut pos = 0
  while pos < n {
    line_no = line_no + 1
    let mut end = pos
    while end < n && text.csv_at(end) != 10 {
      end = end + 1
    }
    let next = end + 1
    let mut blank = true
    for k in pos..<end {
      let c = text.csv_at(k)
      if c != 32 && c != 9 && c != 13 {
        blank = false
        break
      }
    }
    if blank {
      pos = next
      continue
    }
    if skip {
      skip = false
      pos = next
      continue
    }
    let before = data.length()
    let mut a = pos
    while true {
      let mut b = a
      while b < end && text.csv_at(b) != comma {
        b = b + 1
      }
      data.push(csv_cell(text, a, b, line_no))
      if b >= end {
        break
      }
      a = b + 1
    }
    let got = data.length() - before
    if cols == -1 {
      cols = got
    } else if got != cols {
      raise InvalidModel(
        "line \{line_no}: expected \{cols} columns, got \{got}",
      )
    }
    pos = next
  }
  if cols == -1 {
    raise InvalidModel("CSV has no data rows")
  }
  { data: FixedArray::from_array(data), rows: data.length() / cols, cols, }
}

///|
/// Exact powers of ten, 1e0 to 1e22.
let pow10 : FixedArray[Double] = FixedArray::makei(23, i => {
  let mut x = 1.0
  for _ in 0..<i {
    x = x * 10.0
  }
  x
})

///|
/// Cell `text[a:b]`. Plain decimals of at most 15 significant digits and a
/// small exponent are computed directly (one multiplication or division of
/// exact doubles, so correctly rounded); anything else goes to
/// `parse_double`, so the value is the same either way.
fn[T : CsvText] csv_cell(
  text : T,
  a : Int32,
  b : Int32,
  line : Int32,
) -> Float raise ModelError {
  let mut i = a
  let mut j = b
  while i < j && is_csv_space(text.csv_at(i)) {
    i = i + 1
  }
  while j > i && is_csv_space(text.csv_at(j - 1)) {
    j = j - 1
  }
  if i == j {
    return nan()
  }
  let start = i
  let neg = text.csv_at(i) == 45
  if neg {
    i = i + 1
  }
  let mut mant = 0L
  let mut digits = 0
  let mut scale = 0
  let mut any = false
  while i < j {
    let c = text.csv_at(i) - 48
    if c < 0 || c > 9 {
      break
    }
    any = true
    if digits > 0 || c != 0 {
      digits = digits + 1
    }
    mant = mant * 10L + c.to_int64()
    i = i + 1
  }
  if i < j && text.csv_at(i) == 46 {
    i = i + 1
    while i < j {
      let c = text.csv_at(i) - 48
      if c < 0 || c > 9 {
        break
      }
      any = true
      if digits > 0 || c != 0 {
        digits = digits + 1
      }
      mant = mant * 10L + c.to_int64()
      scale = scale - 1
      i = i + 1
    }
  }
  if any && i < j {
    let e = text.csv_at(i)
    if e == 101 || e == 69 {
      let mut k = i + 1
      let mut eneg = false
      if k < j {
        let c = text.csv_at(k)
        if c == 45 || c == 43 {
          eneg = c == 45
          k = k + 1
        }
      }
      let mut ev = 0
      let mut edig = 0
      while k < j && edig < 4 {
        let c = text.csv_at(k) - 48
        if c < 0 || c > 9 {
          break
        }
        ev = ev * 10 + c
        edig = edig + 1
        k = k + 1
      }
      if edig > 0 {
        scale = scale + (if eneg { -ev } else { ev })
        i = k
      }
    }
  }
  if any && i == j && digits <= 15 && scale >= -22 && scale <= 22 {
    let m = mant.to_double()
    let v = if scale >= 0 {
      m * pow10.unsafe_get(scale)
    } else {
      m / pow10.unsafe_get(-scale)
    }
    return Float::from_double(if neg { -v } else { v })
  }
  parse_cell(text.csv_slice(start, j), line)
}

///|
fn is_csv_space(c : Int32) -> Bool {
  c == 32 || c == 9 || c == 13
}

///|
fn csv_by_split(
  text : StringView,
  header : Bool,
  sep : StringView,
) -> DenseMatrix raise ModelError {
  let data : Array[Float] = []
  let mut cols = -1
  let mut skip = header
  let mut line_no = 0
  for line in text.split("\n") {
    line_no = line_no + 1
    if line.trim(chars=" \t\r").length() == 0 {
      continue
    }
    if skip {
      skip = false
      continue
    }
    let before = data.length()
    for cell in line.split(sep) {
      data.push(parse_cell(cell, line_no))
    }
    let n = data.length() - before
    if cols == -1 {
      cols = n
    } else if n != cols {
      raise InvalidModel("line \{line_no}: expected \{cols} columns, got \{n}")
    }
  }
  if cols == -1 {
    raise InvalidModel("CSV has no data rows")
  }
  { data: FixedArray::from_array(data), rows: data.length() / cols, cols, }
}

///|
/// Rows processed together by `predict_batch`: each tree is walked for the
/// whole block before moving to the next, so it stays in cache.
const BLOCK : Int32 = 64

///|
/// Predict every row of `m`. The result is row-major with
/// `self.output_width()` values per row. Gives exactly the same values as
/// calling `predict` on each row.
pub fn Model::predict_batch(self : Model, m : DenseMatrix) -> Array[Float] {
  let c = self.compiled
  let groups = self.num_group
  let out : Array[Float] = Array(capacity=m.rows * self.output_width())
  // rows per block: the lockstep form may ask for more to stream a large
  // forest fewer times
  let block = match self.fast {
    Some(fast) => fast.block
    None => BLOCK
  }
  let keys = FixedArray::makei(block, _ => FixedArray::make(m.cols, 0))
  let margin = FixedArray::make(block * groups, (0.0 : Float))
  let row_out : Array[Float] = Array(capacity=groups)
  let ranks = match self.fast {
    Some(fast) => Buf::new(block * 2 * fast.num_feature)
    None => Buf::new(0)
  }
  let idx = Buf::new(2 * block)
  let vals = match self.fast {
    Some(fast) if fast.walks_values() => FBuf::new(block * 2 * fast.num_feature)
    _ => FBuf::new(0)
  }
  for start = 0; start < m.rows; start = start + block {
    let n = if m.rows - start < block { m.rows - start } else { block }
    for r in 0..<n {
      for g in 0..<groups {
        margin[r * groups + g] = self.base_margin[g]
      }
    }
    if self.linear.length() != 0 {
      for r in 0..<n {
        for g in 0..<groups {
          margin[r * groups + g] = self.linear_margin(m.row(start + r), g)
        }
      }
    } else if self.fast is Some(fast) && fast.walks_values() && groups == 1 {
      if fast.vheap.length() > 0 {
        fast.predict_vheap(m.data, m.cols, start, n, ranks, vals, margin)
      } else {
        fast.values_block(m.data, m.cols, start, n, vals)
        fast.accumulate_values(vals, n, margin)
      }
    } else if self.fast is Some(fast) {
      fast.ranks_block(m.data, m.cols, start, n, ranks)
      fast.accumulate(self.tree_group, groups, ranks, n, idx, margin)
    } else {
      for r in 0..<n {
        c.encode(m.row(start + r), keys[r])
      }
      for t in 0..<self.tree_group.length() {
        let g = self.tree_group[t]
        for r in 0..<n {
          let k = r * groups + g
          margin[k] = margin[k] + c.leaf_value[c.leaf(t, keys[r])]
        }
      }
    }
    for r in 0..<n {
      // `transform` may shrink the array (softmax keeps only the class)
      row_out.clear()
      for g in 0..<groups {
        row_out.push(margin[r * groups + g])
      }
      self.objective.transform(row_out)
      for i in 0..<self.output_width() {
        out.push(row_out[i])
      }
    }
  }
  out
}

///|
/// Number of values `predict` returns per row.
pub fn Model::output_width(self : Model) -> Int32 {
  match self.objective {
    Softmax(_) => 1
    _ => self.num_group
  }
}
