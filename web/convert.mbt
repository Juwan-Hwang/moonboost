///|
fn to_float_matrix(xs : FixedArray[Double]) -> FixedArray[Float] {
  FixedArray::makei(xs.length(), i => Float::from_double(xs[i]))
}
