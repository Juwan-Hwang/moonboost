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
