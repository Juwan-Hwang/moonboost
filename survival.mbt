// survival:cox: Cox proportional hazards on right-censored times.

///|
/// Rows in increasing order of |label| (the time), ties in row order.
fn cox_order(label : ArrayView[Float]) -> FixedArray[Int32] {
  let idx = Array::makei(label.length(), i => i)
  idx.sort_by((a, b) => {
    let c = label[a].abs().compare(label[b].abs())
    if c != 0 {
      c
    } else {
      a.compare(b)
    }
  })
  FixedArray::makei(idx.length(), i => idx[i])
}

///|
/// Gradient and hessian of the Cox partial likelihood for every row, at
/// 2r and 2r + 1, as XGBoost computes them: a positive label is an event
/// at that time, a negative one a censoring at its absolute value, and
/// tied times share their risk set (Breslow).
fn cox_gradients(
  label : ArrayView[Float],
  margin : FixedArray[Float],
  order : FixedArray[Int32],
) -> FixedArray[Double] {
  let n = label.length()
  let out = FixedArray::make(n * 2, 0.0)
  // exp of each margin, in time order
  let es = FixedArray::makei(n, i => train_exp(margin[order[i]].to_double()))
  let mut exp_sum = 0.0
  for i in 0..<n {
    exp_sum += es[i]
  }
  let mut r_k = 0.0
  let mut s_k = 0.0
  let mut last_exp = 0.0
  let mut last_t = 0.0
  let mut acc = 0.0
  for i in 0..<n {
    let r = order[i]
    let e = es[i]
    let y = label[r].to_double()
    let t = y.abs()
    // the risk set shrinks only once the time moves forward
    acc += last_exp
    if last_t < t {
      exp_sum -= acc
      acc = 0.0
    }
    if y > 0.0 {
      r_k += 1.0 / exp_sum
      s_k += 1.0 / (exp_sum * exp_sum)
    }
    let event = if y > 0.0 { 1.0 } else { 0.0 }
    out[r * 2] = e * r_k - event
    out[r * 2 + 1] = e * r_k - e * e * s_k
    last_t = t
    last_exp = e
  }
  out
}
