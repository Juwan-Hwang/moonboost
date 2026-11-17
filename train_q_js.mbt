///|
/// Quantised gradient pairs for JS, where Int64 is a BigInt: two
/// integer-valued Doubles per row, g at 2r and h at 2r + 1. The sums stay
/// below 2^53, so they are exact and equal to the packed sums of the other
/// backends.
type QArr = F64

///|
/// Histogram slots for JS: the g sum of slot s at 2s, its h sum at 2s + 1.
type Hist = F64

///|
/// A `Float64Array`: zero-filled on allocation and read without boxing,
/// unlike a JS array built element by element.
#external
priv type F64

///|
extern "js" fn F64::new(n : Int32) -> F64 = "(n) => new Float64Array(n)"

///|
extern "js" fn F64::length(self : F64) -> Int32 = "(a) => a.length"

///|
extern "js" fn F64::unsafe_get(self : F64, i : Int32) -> Double = "(a, i) => a[i]"

///|
extern "js" fn F64::unsafe_set(self : F64, i : Int32, v : Double) -> Unit = "(a, i, v) => { a[i] = v; }"

///|
/// Two-byte bins as a `Uint16Array`: half the memory of a JS array.
#external
priv type U16s

///|
extern "js" fn u16s_make(n : Int32) -> U16s = "(n) => new Uint16Array(n)"

///|
extern "js" fn U16s::unsafe_get(self : U16s, i : Int32) -> UInt16 = "(a, i) => a[i]"

///|
extern "js" fn U16s::unsafe_set(self : U16s, i : Int32, v : UInt16) -> Unit = "(a, i, v) => { a[i] = v; }"

///|
/// Row numbers and positions as an `Int32Array`.
#external
priv type I32s

///|
extern "js" fn i32s_make(n : Int32) -> I32s = "(n) => new Int32Array(n)"

///|
extern "js" fn I32s::length(self : I32s) -> Int32 = "(a) => a.length"

///|
extern "js" fn I32s::unsafe_get(self : I32s, i : Int32) -> Int32 = "(a, i) => a[i]"

///|
extern "js" fn I32s::unsafe_set(self : I32s, i : Int32, v : Int32) -> Unit = "(a, i, v) => { a[i] = v; }"

///|
/// Exact gradient pairs as a `Float32Array`.
#external
priv type F32s

///|
extern "js" fn f32s_make(n : Int32) -> F32s = "(n) => new Float32Array(n)"

///|
extern "js" fn F32s::length(self : F32s) -> Int32 = "(a) => a.length"

///|
extern "js" fn F32s::unsafe_get(self : F32s, i : Int32) -> Float = "(a, i) => a[i]"

///|
extern "js" fn F32s::unsafe_set(self : F32s, i : Int32, v : Float) -> Unit = "(a, i, v) => { a[i] = v; }"

///|
/// Bins packed into 64-bit words would be BigInts in JS: keep two-byte bins.
let narrow_bins : Bool = false

///|
fn qarr_make(n : Int32) -> QArr {
  F64::new(n * 2)
}

///|
fn q_copy(dst : QArr, p : Int32, src : QArr, r : Int32) -> Unit {
  dst.unsafe_set(p * 2, src.unsafe_get(r * 2))
  dst.unsafe_set(p * 2 + 1, src.unsafe_get(r * 2 + 1))
}

///|
fn hist_len(hist : Hist) -> Int32 {
  hist.length() / 2
}

///|
fn hist_g(hist : Hist, slot : Int32) -> Double {
  hist.unsafe_get(slot * 2)
}

///|
fn hist_h(hist : Hist, slot : Int32) -> Double {
  hist.unsafe_get(slot * 2 + 1)
}

///|
fn hist_sub(hist : Hist, small : Hist) -> Unit {
  for i in 0..<hist.length() {
    hist.unsafe_set(i, hist.unsafe_get(i) - small.unsafe_get(i))
  }
}

///|
/// Quantises a gradient pair as `pack_at` does on the other backends:
/// g * gs and h * hs rounded half away from zero.
fn Grower::pack_at(
  self : Grower,
  qa : QArr,
  r : Int32,
  g : Double,
  h : Double,
) -> Unit {
  let x = g * self.gs
  let half = if x < 0.0 { -0.5 } else { 0.5 }
  qa.unsafe_set(r * 2, (x + half).trunc())
  qa.unsafe_set(r * 2 + 1, (h * self.hs + 0.5).trunc())
}

///|
/// 2^(k - 128) for k in 0..=256, built by exact doublings and halvings.
let pow2_table : FixedArray[Double] = {
  let t = FixedArray::make(257, 1.0)
  for k in 129..<257 {
    t[k] = t[k - 1] * 2.0
  }
  for k = 127; k >= 0; k = k - 1 {
    t[k] = t[k + 1] * 0.5
  }
  t
}

///|
/// The same exp as the other backends, with 2^k from a table instead of
/// Int64 bit operations.
fn train_exp(x : Double) -> Double {
  let x = if x > 80.0 { 80.0 } else if x < -80.0 { -80.0 } else { x }
  let kd = x * 1.4426950408889634 + 6755399441055744.0
  let kf = kd - 6755399441055744.0
  let r = x - kf * 0.6931471803691238 - kf * 1.9082149292705877e-10
  let r2 = r * r
  let r4 = r2 * r2
  let p = 1.0 +
    r +
    r2 * (0.5 + r * (1.0 / 6.0)) +
    r4 *
    (1.0 / 24.0 + r * (1.0 / 120.0) + r2 * (1.0 / 720.0 + r * (1.0 / 5040.0)))
  p * pow2_table.unsafe_get(kf.to_int() + 128)
}

///|
/// Gradient and hessian sums per bin for rows[lo:hi], from two-byte bins.
fn Grower::histogram(
  self : Grower,
  lo : Int32,
  hi : Int32,
  level : Int32,
) -> Hist {
  guard !self.narrow else { abort("one-byte bins are not used in JS") }
  // the one-byte slot bases are only read by the other backends
  ignore(self.wbase)
  let rows = self.rows
  let hist = F64::new(self.nbins * 2)
  let cols = self.data.cols
  let bins = if level > 0 { self.cbins16[level - 1] } else { self.data.bins }
  let qa = if level > 0 { self.cq[level - 1] } else { self.gq }
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let g = qa.unsafe_get(r * 2)
    let h = qa.unsafe_get(r * 2 + 1)
    let base = r * cols
    for f in 0..<cols {
      let s = bins.unsafe_get(base + f).to_int() * 2
      hist.unsafe_set(s, hist.unsafe_get(s) + g)
      hist.unsafe_set(s + 1, hist.unsafe_get(s + 1) + h)
    }
  }
  hist
}
