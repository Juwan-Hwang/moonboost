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
