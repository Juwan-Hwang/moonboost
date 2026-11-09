///|
/// Rows as a `Float32Array`, each value rounded to float32 the way XGBoost
/// does, and no copy when the caller already passes one. A plain loop beats
/// the `Float32Array` constructor on a JS array by about 2x in V8.
extern "js" fn to_float_matrix(xs : FixedArray[Double]) -> FixedArray[Float] =
  #|(a) => {
  #|  if (a instanceof Float32Array) return a;
  #|  const n = a.length, o = new Float32Array(n);
  #|  for (let i = 0; i < n; i++) o[i] = a[i];
  #|  return o;
  #|}
