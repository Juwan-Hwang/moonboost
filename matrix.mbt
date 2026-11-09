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
    margin.fill(self.base_margin)
    if self.fast is Some(fast) && fast.walks_values() && groups == 1 {
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
