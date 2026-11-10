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
