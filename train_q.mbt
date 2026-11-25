///|
/// Quantised gradient pairs, one per row: (g << 32) + h. Sums over all rows
/// keep h inside 32 bits, so the packed sums never carry between halves.
type QArr = FixedArray[Int64]

///|
/// Histogram slots of packed quantised sums.
type Hist = FixedArray[Int64]

///|
/// Two-byte bins.
type U16s = FixedArray[UInt16]

///|
fn u16s_make(n : Int32) -> U16s {
  FixedArray::make(n, 0)
}

///|
/// Row numbers and positions.
type I32s = FixedArray[Int32]

///|
fn i32s_make(n : Int32) -> I32s {
  FixedArray::make(n, 0)
}

///|
/// Exact gradient pairs, g at 2r and h at 2r + 1.
type F32s = FixedArray[Float]

///|
fn f32s_make(n : Int32) -> F32s {
  FixedArray::make(n, 0.0)
}

///|
/// Whether bins may be packed eight to a 64-bit word.
let narrow_bins : Bool = true

///|
/// One-byte bins, eight to a 64-bit word.
type Words = FixedArray[UInt64]

///|
let word_bins : Int32 = 8

///|
fn words_make(n : Int32) -> Words {
  FixedArray::make(n, 0UL)
}

///|
/// The byte of a word at bit `shift`.
fn word_byte(x : UInt64, shift : Int32) -> Int32 {
  ((x >> shift) & 255UL).to_int()
}

///|
/// A word with byte `b` set at bit `shift` (which was zero).
fn word_put(x : UInt64, b : Byte, shift : Int32) -> UInt64 {
  x | (b.to_int().to_uint64() << shift)
}

///|
/// Whether copy levels can carry each row's quantised pair in the free
/// high half of its second bin word (9 to 12 features): per-row values
/// then fit 16 bits each (|g| and h are at most 2^31 / n, n > 65536), so
/// the pair packs into 32 bits and rides with the bins.
fn qpad_ok(cols : Int32, words : Int32, n : Int32) -> Bool {
  words == 2 && cols % 8 >= 1 && cols % 8 <= 4 && n > 65536
}

///|
/// Word w (its high half zero) with the pair (g << 32) + h packed into
/// that half as (g << 16) | h.
fn q_pad(w : UInt64, q : Int64) -> UInt64 {
  let g = (q >> 32).to_int()
  let h = (q & 0xFFFFL).to_int()
  w | (((g << 16) | h).reinterpret_as_uint().to_uint64() << 32)
}

///|
/// The pair (g << 32) + h packed in the high half of w by q_pad.
fn q_unpad(w : UInt64) -> Int64 {
  let x = (w >> 32).to_int()
  ((x >> 16).to_int64() << 32) + (x & 0xFFFF).to_int64()
}

///|
fn qarr_make(n : Int32) -> QArr {
  FixedArray::make(n, 0L)
}

///|
fn q_copy(dst : QArr, p : Int32, src : QArr, r : Int32) -> Unit {
  dst.unsafe_set(p, src.unsafe_get(r))
}

///|
fn hist_len(hist : Hist) -> Int32 {
  hist.length()
}

///|
/// The quantised gradient sum of a slot, as an integer-valued Double.
fn hist_g(hist : Hist, slot : Int32) -> Double {
  let q = hist.unsafe_get(slot)
  ((q - (q & 0xFFFFFFFFL)) >> 32).to_double()
}

///|
/// The quantised hessian sum of a slot.
fn hist_h(hist : Hist, slot : Int32) -> Double {
  (hist.unsafe_get(slot) & 0xFFFFFFFFL).to_double()
}

///|
/// hist -= small, slot by slot.
fn hist_sub(hist : Hist, small : Hist) -> Unit {
  for i in 0..<hist.length() {
    hist[i] = hist[i] - small[i]
  }
}

///|
/// Truncation toward zero for a value known to be in Int64 range: the
/// quantised gradients are bounded by the scales, so the NaN and range
/// checks of `Double::to_int64` are not needed.
fn trunc64(x : Double) -> Int64 = "%f64_to_i64"

///|
/// Quantises a gradient pair, rounding half away from zero (h is never
/// negative), and packs it as (g << 32) + h.
fn Grower::pack_at(
  self : Grower,
  qa : QArr,
  r : Int32,
  g : Double,
  h : Double,
) -> Unit {
  let x = g * self.gs
  // 0.5 with the sign of x, from its bits: the sign of a gradient is
  // unpredictable, so a branch on it would often be mispredicted
  let half = ((x.reinterpret_as_int64() & (1L << 63)) | 0x3FE0000000000000L).reinterpret_as_double()
  let gi = trunc64(x + half)
  qa.unsafe_set(r, (gi << 32) + trunc64(h * self.hs + 0.5))
}

///|
/// exp to about 1e-9 relative error, ample for gradients that are rounded
/// to float32: x = k ln2 + r with |r| <= ln2 / 2, a degree-7 polynomial for
/// e^r, and 2^k built from its exponent bits. x is clamped to +-80, where
/// the losses are saturated.
fn train_exp(x : Double) -> Double {
  let x = if x > 80.0 { 80.0 } else if x < -80.0 { -80.0 } else { x }
  // k = round(x / ln2) by the 1.5 * 2^52 trick, without a branch or a
  // checked conversion
  let kd = x * 1.4426950408889634 + 6755399441055744.0
  let ki = kd.reinterpret_as_int64() - 0x4338000000000000L
  let kf = kd - 6755399441055744.0
  let r = x - kf * 0.6931471803691238 - kf * 1.9082149292705877e-10
  // Estrin's scheme: a shorter dependency chain than Horner's
  let r2 = r * r
  let r4 = r2 * r2
  let p = 1.0 +
    r +
    r2 * (0.5 + r * (1.0 / 6.0)) +
    r4 *
    (1.0 / 24.0 + r * (1.0 / 120.0) + r2 * (1.0 / 720.0 + r * (1.0 / 5040.0)))
  p * ((ki + 1023L) << 52).reinterpret_as_double()
}

///|
/// Histogram buffers by slot (a depth, or a loss-guided slot + 1), reused
/// across nodes and trees: a slot's buffer is free whenever a new
/// histogram is built for it.
let hist_pool : Array[Hist] = []

///|
/// The zeroed buffer of slot `slot`, of `len` slots.
fn pooled_hist(slot : Int32, len : Int32) -> Hist {
  while hist_pool.length() <= slot {
    hist_pool.push(FixedArray::make(len, 0L))
  }
  let h = hist_pool[slot]
  if h.length() != len {
    let fresh = FixedArray::make(len, 0L)
    hist_pool[slot] = fresh
    return fresh
  }
  h.fill(0L)
  h
}

///|
/// Gradient and hessian sums per bin for rows[lo:hi], into the buffer of
/// slot `depth`.
fn Grower::histogram(
  self : Grower,
  lo : Int32,
  hi : Int32,
  level : Int32,
  depth : Int32,
) -> Hist {
  let rows = self.rows
  if self.narrow {
    let hist = pooled_hist(depth, self.nbins + 8)
    let (cb, cq) = if level > 0 {
      (self.cbins[level - 1], self.cq[level - 1])
    } else {
      (self.bins8, self.gq)
    }
    let wb = self.wbase
    let words = self.words
    if words == 1 {
      // the slot bases in locals: hist stores could alias wbase for the C
      // compiler, which would reload it for every byte
      let b0 = wb[0]
      let b1 = wb[1]
      let b2 = wb[2]
      let b3 = wb[3]
      let b4 = wb[4]
      let b5 = wb[5]
      let b6 = wb[6]
      let b7 = wb[7]
      for i in lo..<hi {
        let r = rows.unsafe_get(i)
        let q = cq.unsafe_get(r)
        let x = cb.unsafe_get(r)
        let s0 = (x & 255UL).to_int() + b0
        let s1 = ((x >> 8) & 255UL).to_int() + b1
        let s2 = ((x >> 16) & 255UL).to_int() + b2
        let s3 = ((x >> 24) & 255UL).to_int() + b3
        let s4 = ((x >> 32) & 255UL).to_int() + b4
        let s5 = ((x >> 40) & 255UL).to_int() + b5
        let s6 = ((x >> 48) & 255UL).to_int() + b6
        let s7 = (x >> 56).to_int() + b7
        hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
        hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
        hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
        hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
        hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
        hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
        hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
        hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
      }
      return hist
    }
    let tail = self.data.cols - 8
    if self.qpad && level > 0 {
      // the pair is in the second word's high half (tail 1 to 4)
      match tail {
        1 => histq_two_1(hist, rows, cb, wb, lo, hi)
        2 => histq_two_2(hist, rows, cb, wb, lo, hi)
        3 => histq_two_3(hist, rows, cb, wb, lo, hi)
        _ => histq_two_4(hist, rows, cb, wb, lo, hi)
      }
      return hist
    }
    if words == 2 && tail < 8 {
      // nine to fifteen features: the padding bytes of the second word
      // are skipped, as their slots would chain every row's updates
      match tail {
        1 => hist_two_1(hist, rows, cq, cb, wb, lo, hi)
        2 => hist_two_2(hist, rows, cq, cb, wb, lo, hi)
        3 => hist_two_3(hist, rows, cq, cb, wb, lo, hi)
        4 => hist_two_4(hist, rows, cq, cb, wb, lo, hi)
        5 => hist_two_5(hist, rows, cq, cb, wb, lo, hi)
        6 => hist_two_6(hist, rows, cq, cb, wb, lo, hi)
        _ => hist_two_7(hist, rows, cq, cb, wb, lo, hi)
      }
      return hist
    }
    if words == 2 {
      // sixteen features: both words unrolled, bases in locals
      let b0 = wb[0]
      let b1 = wb[1]
      let b2 = wb[2]
      let b3 = wb[3]
      let b4 = wb[4]
      let b5 = wb[5]
      let b6 = wb[6]
      let b7 = wb[7]
      let b8 = wb[8]
      let b9 = wb[9]
      let b10 = wb[10]
      let b11 = wb[11]
      let b12 = wb[12]
      let b13 = wb[13]
      let b14 = wb[14]
      let b15 = wb[15]
      for i in lo..<hi {
        let r = rows.unsafe_get(i)
        let q = cq.unsafe_get(r)
        let x = cb.unsafe_get(r * 2)
        let y = cb.unsafe_get(r * 2 + 1)
        let s0 = (x & 255UL).to_int() + b0
        let s1 = ((x >> 8) & 255UL).to_int() + b1
        let s2 = ((x >> 16) & 255UL).to_int() + b2
        let s3 = ((x >> 24) & 255UL).to_int() + b3
        let s4 = ((x >> 32) & 255UL).to_int() + b4
        let s5 = ((x >> 40) & 255UL).to_int() + b5
        let s6 = ((x >> 48) & 255UL).to_int() + b6
        let s7 = (x >> 56).to_int() + b7
        let s8 = (y & 255UL).to_int() + b8
        let s9 = ((y >> 8) & 255UL).to_int() + b9
        let s10 = ((y >> 16) & 255UL).to_int() + b10
        let s11 = ((y >> 24) & 255UL).to_int() + b11
        let s12 = ((y >> 32) & 255UL).to_int() + b12
        let s13 = ((y >> 40) & 255UL).to_int() + b13
        let s14 = ((y >> 48) & 255UL).to_int() + b14
        let s15 = (y >> 56).to_int() + b15
        hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
        hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
        hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
        hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
        hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
        hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
        hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
        hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
        hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
        hist.unsafe_set(s9, hist.unsafe_get(s9) + q)
        hist.unsafe_set(s10, hist.unsafe_get(s10) + q)
        hist.unsafe_set(s11, hist.unsafe_get(s11) + q)
        hist.unsafe_set(s12, hist.unsafe_get(s12) + q)
        hist.unsafe_set(s13, hist.unsafe_get(s13) + q)
        hist.unsafe_set(s14, hist.unsafe_get(s14) + q)
        hist.unsafe_set(s15, hist.unsafe_get(s15) + q)
      }
      return hist
    }
    // whole words, then the real bytes of a partial last word
    let full = self.data.cols / word_bins
    let rest = self.data.cols % word_bins
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let q = cq.unsafe_get(r)
      let base = r * words
      if rest > 0 {
        let x = cb.unsafe_get(base + full)
        let o = full * 8
        for j in 0..<rest {
          let s = ((x >> (j * 8)) & 255UL).to_int() + wb.unsafe_get(o + j)
          hist.unsafe_set(s, hist.unsafe_get(s) + q)
        }
      }
      for w in 0..<full {
        let x = cb.unsafe_get(base + w)
        let o = w * 8
        let s0 = (x & 255UL).to_int() + wb.unsafe_get(o)
        let s1 = ((x >> 8) & 255UL).to_int() + wb.unsafe_get(o + 1)
        let s2 = ((x >> 16) & 255UL).to_int() + wb.unsafe_get(o + 2)
        let s3 = ((x >> 24) & 255UL).to_int() + wb.unsafe_get(o + 3)
        hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
        hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
        hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
        hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
        let s4 = ((x >> 32) & 255UL).to_int() + wb.unsafe_get(o + 4)
        let s5 = ((x >> 40) & 255UL).to_int() + wb.unsafe_get(o + 5)
        let s6 = ((x >> 48) & 255UL).to_int() + wb.unsafe_get(o + 6)
        let s7 = (x >> 56).to_int() + wb.unsafe_get(o + 7)
        hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
        hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
        hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
        hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
      }
    }
    return hist
  }
  let hist = pooled_hist(depth, self.nbins)
  let cols = self.data.cols
  let bins = if level > 0 { self.cbins16[level - 1] } else { self.data.bins }
  let gq = if level > 0 { self.cq[level - 1] } else { self.gq }
  // indices are in range by construction: rows hold row numbers and bins
  // hold histogram slots
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let q = gq.unsafe_get(r)
    let base = r * cols
    let mut f = 0
    while f + 4 <= cols {
      let s0 = bins.unsafe_get(base + f).to_int()
      let s1 = bins.unsafe_get(base + f + 1).to_int()
      let s2 = bins.unsafe_get(base + f + 2).to_int()
      let s3 = bins.unsafe_get(base + f + 3).to_int()
      hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
      hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
      hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
      hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
      f += 4
    }
    while f < cols {
      let slot = bins.unsafe_get(base + f).to_int()
      hist.unsafe_set(slot, hist.unsafe_get(slot) + q)
      f += 1
    }
  }
  hist
}

///|
/// histogram for 8 + 1 features (two words, 7 padding bytes skipped).
fn hist_two_1(
  hist : Hist,
  rows : I32s,
  cq : QArr,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let q = cq.unsafe_get(r)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
  }
}

///|
/// histogram for 8 + 2 features (two words, 6 padding bytes skipped).
fn hist_two_2(
  hist : Hist,
  rows : I32s,
  cq : QArr,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  let b9 = wb[9]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let q = cq.unsafe_get(r)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    let s9 = ((y >> 8) & 255UL).to_int() + b9
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
    hist.unsafe_set(s9, hist.unsafe_get(s9) + q)
  }
}

///|
/// histogram for 8 + 3 features (two words, 5 padding bytes skipped).
fn hist_two_3(
  hist : Hist,
  rows : I32s,
  cq : QArr,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  let b9 = wb[9]
  let b10 = wb[10]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let q = cq.unsafe_get(r)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    let s9 = ((y >> 8) & 255UL).to_int() + b9
    let s10 = ((y >> 16) & 255UL).to_int() + b10
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
    hist.unsafe_set(s9, hist.unsafe_get(s9) + q)
    hist.unsafe_set(s10, hist.unsafe_get(s10) + q)
  }
}

///|
/// histogram for 8 + 4 features (two words, 4 padding bytes skipped).
fn hist_two_4(
  hist : Hist,
  rows : I32s,
  cq : QArr,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  let b9 = wb[9]
  let b10 = wb[10]
  let b11 = wb[11]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let q = cq.unsafe_get(r)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    let s9 = ((y >> 8) & 255UL).to_int() + b9
    let s10 = ((y >> 16) & 255UL).to_int() + b10
    let s11 = ((y >> 24) & 255UL).to_int() + b11
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
    hist.unsafe_set(s9, hist.unsafe_get(s9) + q)
    hist.unsafe_set(s10, hist.unsafe_get(s10) + q)
    hist.unsafe_set(s11, hist.unsafe_get(s11) + q)
  }
}

///|
/// histogram for 8 + 5 features (two words, 3 padding bytes skipped).
fn hist_two_5(
  hist : Hist,
  rows : I32s,
  cq : QArr,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  let b9 = wb[9]
  let b10 = wb[10]
  let b11 = wb[11]
  let b12 = wb[12]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let q = cq.unsafe_get(r)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    let s9 = ((y >> 8) & 255UL).to_int() + b9
    let s10 = ((y >> 16) & 255UL).to_int() + b10
    let s11 = ((y >> 24) & 255UL).to_int() + b11
    let s12 = ((y >> 32) & 255UL).to_int() + b12
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
    hist.unsafe_set(s9, hist.unsafe_get(s9) + q)
    hist.unsafe_set(s10, hist.unsafe_get(s10) + q)
    hist.unsafe_set(s11, hist.unsafe_get(s11) + q)
    hist.unsafe_set(s12, hist.unsafe_get(s12) + q)
  }
}

///|
/// histogram for 8 + 6 features (two words, 2 padding bytes skipped).
fn hist_two_6(
  hist : Hist,
  rows : I32s,
  cq : QArr,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  let b9 = wb[9]
  let b10 = wb[10]
  let b11 = wb[11]
  let b12 = wb[12]
  let b13 = wb[13]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let q = cq.unsafe_get(r)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    let s9 = ((y >> 8) & 255UL).to_int() + b9
    let s10 = ((y >> 16) & 255UL).to_int() + b10
    let s11 = ((y >> 24) & 255UL).to_int() + b11
    let s12 = ((y >> 32) & 255UL).to_int() + b12
    let s13 = ((y >> 40) & 255UL).to_int() + b13
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
    hist.unsafe_set(s9, hist.unsafe_get(s9) + q)
    hist.unsafe_set(s10, hist.unsafe_get(s10) + q)
    hist.unsafe_set(s11, hist.unsafe_get(s11) + q)
    hist.unsafe_set(s12, hist.unsafe_get(s12) + q)
    hist.unsafe_set(s13, hist.unsafe_get(s13) + q)
  }
}

///|
/// histogram for 8 + 7 features (two words, 1 padding bytes skipped).
fn hist_two_7(
  hist : Hist,
  rows : I32s,
  cq : QArr,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  let b9 = wb[9]
  let b10 = wb[10]
  let b11 = wb[11]
  let b12 = wb[12]
  let b13 = wb[13]
  let b14 = wb[14]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let q = cq.unsafe_get(r)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    let s9 = ((y >> 8) & 255UL).to_int() + b9
    let s10 = ((y >> 16) & 255UL).to_int() + b10
    let s11 = ((y >> 24) & 255UL).to_int() + b11
    let s12 = ((y >> 32) & 255UL).to_int() + b12
    let s13 = ((y >> 40) & 255UL).to_int() + b13
    let s14 = ((y >> 48) & 255UL).to_int() + b14
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
    hist.unsafe_set(s9, hist.unsafe_get(s9) + q)
    hist.unsafe_set(s10, hist.unsafe_get(s10) + q)
    hist.unsafe_set(s11, hist.unsafe_get(s11) + q)
    hist.unsafe_set(s12, hist.unsafe_get(s12) + q)
    hist.unsafe_set(s13, hist.unsafe_get(s13) + q)
    hist.unsafe_set(s14, hist.unsafe_get(s14) + q)
  }
}

///|
/// histogram, pairs packed in the second word, for 8 + 1 features (two words, 7 padding bytes skipped).
fn histq_two_1(
  hist : Hist,
  rows : I32s,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let q = q_unpad(y)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
  }
}

///|
/// histogram, pairs packed in the second word, for 8 + 2 features (two words, 6 padding bytes skipped).
fn histq_two_2(
  hist : Hist,
  rows : I32s,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  let b9 = wb[9]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let q = q_unpad(y)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    let s9 = ((y >> 8) & 255UL).to_int() + b9
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
    hist.unsafe_set(s9, hist.unsafe_get(s9) + q)
  }
}

///|
/// histogram, pairs packed in the second word, for 8 + 3 features (two words, 5 padding bytes skipped).
fn histq_two_3(
  hist : Hist,
  rows : I32s,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  let b9 = wb[9]
  let b10 = wb[10]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let q = q_unpad(y)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    let s9 = ((y >> 8) & 255UL).to_int() + b9
    let s10 = ((y >> 16) & 255UL).to_int() + b10
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
    hist.unsafe_set(s9, hist.unsafe_get(s9) + q)
    hist.unsafe_set(s10, hist.unsafe_get(s10) + q)
  }
}

///|
/// histogram, pairs packed in the second word, for 8 + 4 features (two words, 4 padding bytes skipped).
fn histq_two_4(
  hist : Hist,
  rows : I32s,
  cb : Words,
  wb : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let b0 = wb[0]
  let b1 = wb[1]
  let b2 = wb[2]
  let b3 = wb[3]
  let b4 = wb[4]
  let b5 = wb[5]
  let b6 = wb[6]
  let b7 = wb[7]
  let b8 = wb[8]
  let b9 = wb[9]
  let b10 = wb[10]
  let b11 = wb[11]
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let x = cb.unsafe_get(r * 2)
    let y = cb.unsafe_get(r * 2 + 1)
    let q = q_unpad(y)
    let s0 = (x & 255UL).to_int() + b0
    let s1 = ((x >> 8) & 255UL).to_int() + b1
    let s2 = ((x >> 16) & 255UL).to_int() + b2
    let s3 = ((x >> 24) & 255UL).to_int() + b3
    let s4 = ((x >> 32) & 255UL).to_int() + b4
    let s5 = ((x >> 40) & 255UL).to_int() + b5
    let s6 = ((x >> 48) & 255UL).to_int() + b6
    let s7 = (x >> 56).to_int() + b7
    let s8 = (y & 255UL).to_int() + b8
    let s9 = ((y >> 8) & 255UL).to_int() + b9
    let s10 = ((y >> 16) & 255UL).to_int() + b10
    let s11 = ((y >> 24) & 255UL).to_int() + b11
    hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
    hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
    hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
    hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
    hist.unsafe_set(s4, hist.unsafe_get(s4) + q)
    hist.unsafe_set(s5, hist.unsafe_get(s5) + q)
    hist.unsafe_set(s6, hist.unsafe_get(s6) + q)
    hist.unsafe_set(s7, hist.unsafe_get(s7) + q)
    hist.unsafe_set(s8, hist.unsafe_get(s8) + q)
    hist.unsafe_set(s9, hist.unsafe_get(s9) + q)
    hist.unsafe_set(s10, hist.unsafe_get(s10) + q)
    hist.unsafe_set(s11, hist.unsafe_get(s11) + q)
  }
}
