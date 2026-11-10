///|
/// `exp` for LightGBM outputs on web targets, see `soft_exp`.
fn lgb_exp(x : Double) -> Double {
  soft_exp(x)
}

///|
/// Walk LightGBM ranks through packed 64-bit nodes.
const LGB_PACKED : Bool = false
