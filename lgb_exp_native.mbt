///|
/// The C library's `exp`, which LightGBM itself calls. On glibc this is the
/// function `soft_exp` emulates.
extern "C" fn c_exp(x : Double) -> Double = "exp"

///|
/// `exp` for LightGBM outputs on native builds. A NaN takes the portable
/// path, which returns it unchanged.
fn lgb_exp(x : Double) -> Double {
  if x.is_nan() {
    soft_exp(x)
  } else {
    c_exp(x)
  }
}
