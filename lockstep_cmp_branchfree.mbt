///|
/// `step` when the threshold key `t` is `<= key`, else 0, without a branch:
/// JS and wasm engines compile an `if` here to a jump, and a rank search is
/// a coin flip per halving.
fn advance(t : Int32, key : Int32, step : Int32) -> Int32 {
  let d = key - t
  // all ones when `key < t`, computed without overflow
  let lt = (d ^ ((key ^ t) & (d ^ key))) >> 31
  step & (lt ^ -1)
}

///|
/// Whether batches walk each tree eight rows at a time with node ids in
/// locals (see `Lockstep::accumulate_rows8`).
const ROWS8 : Bool = false

///|
/// `step` when cut i is `<= v`, compared on the order keys without a
/// branch.
fn cut_step(
  fv : FixedArray[Float],
  fc : FixedArray[Int32],
  i : Int32,
  v : Float,
  x : Int32,
  step : Int32,
) -> Int32 {
  ignore(fv)
  ignore(v)
  advance(fc.unsafe_get(i), x, step)
}
