///|
/// Rows as a `Float32Array`: one native conversion that rounds each value to
/// float32 the way XGBoost does, and no copy when the caller already passes one.
extern "js" fn to_float_matrix(xs : FixedArray[Double]) -> FixedArray[Float] =
  #|(a) => a instanceof Float32Array ? a : new Float32Array(a)
