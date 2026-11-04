///|
/// Row-major dense feature matrix. Missing values are NaN.
pub struct DenseMatrix {
  data : Array[Float]
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
  { data, rows: data.length() / cols, cols, }
}

///|
/// Predict every row of `m`. The result is row-major with
/// `self.output_width()` values per row.
pub fn Model::predict_batch(self : Model, m : DenseMatrix) -> Array[Float] {
  let out : Array[Float] = Array(capacity=m.rows * self.output_width())
  for i in 0..<m.rows {
    for v in self.predict(m.row(i)) {
      out.push(v)
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
