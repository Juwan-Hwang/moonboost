///|
/// `step` when the threshold key `t` is `<= key`, else 0. Native compilers
/// turn the comparison into a conditional move.
fn advance(t : Int32, key : Int32, step : Int32) -> Int32 {
  if t <= key {
    step
  } else {
    0
  }
}

///|
/// Whether batches walk each tree eight rows at a time with node ids in
/// locals (see `Lockstep::accumulate_rows8`).
const ROWS8 : Bool = true

///|
/// `step` when cut i is `<= v`: a float comparison times step, which
/// native compilers keep free of branches (the order keys are not needed).
fn cut_step(
  fv : FixedArray[Float],
  fc : FixedArray[Int32],
  i : Int32,
  v : Float,
  x : Int32,
  step : Int32,
) -> Int32 {
  ignore(fc)
  ignore(x)
  (fv.unsafe_get(i) <= v).to_int() * step
}
