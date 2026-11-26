///|
/// train_exp in both lanes: the same operations in the same order, so each
/// lane is bit for bit train_exp of its input.
fn exp_v(x : V128) -> V128 {
  let hi_c = @simd.f64x2_splat(80.0)
  let lo_c = @simd.f64x2_splat(-80.0)
  let log2e = @simd.f64x2_splat(1.4426950408889634)
  let shifter = @simd.f64x2_splat(6755399441055744.0)
  let bias = @simd.i64x2_splat(0x4338000000000000UL - 1023UL)
  let ln2hi = @simd.f64x2_splat(0.6931471803691238)
  let ln2lo = @simd.f64x2_splat(1.9082149292705877e-10)
  let c2 = @simd.f64x2_splat(0.5)
  let c3 = @simd.f64x2_splat(1.0 / 6.0)
  let c4 = @simd.f64x2_splat(1.0 / 24.0)
  let c5 = @simd.f64x2_splat(1.0 / 120.0)
  let c6 = @simd.f64x2_splat(1.0 / 720.0)
  let c7 = @simd.f64x2_splat(1.0 / 5040.0)
  let one = @simd.f64x2_splat(1.0)
  let x = @simd.f64x2_pmax(@simd.f64x2_pmin(x, hi_c), lo_c)
  let kd = @simd.f64x2_add(@simd.f64x2_mul(x, log2e), shifter)
  // (ki + 1023) << 52 with ki = bits(kd) - 0x4338000000000000
  let scale = @simd.i64x2_shl(@simd.i64x2_sub(kd, bias), 52)
  let kf = @simd.f64x2_sub(kd, shifter)
  let t = @simd.f64x2_sub(
    @simd.f64x2_sub(x, @simd.f64x2_mul(kf, ln2hi)),
    @simd.f64x2_mul(kf, ln2lo),
  )
  let t2 = @simd.f64x2_mul(t, t)
  let t4 = @simd.f64x2_mul(t2, t2)
  let a = @simd.f64x2_add(
    @simd.f64x2_add(one, t),
    @simd.f64x2_mul(t2, @simd.f64x2_add(c2, @simd.f64x2_mul(t, c3))),
  )
  let b = @simd.f64x2_add(
    @simd.f64x2_add(c4, @simd.f64x2_mul(t, c5)),
    @simd.f64x2_mul(t2, @simd.f64x2_add(c6, @simd.f64x2_mul(t, c7))),
  )
  @simd.f64x2_mul(@simd.f64x2_add(a, @simd.f64x2_mul(t4, b)), scale)
}

///|
/// Softmax probabilities of every row and class into out, two rows at a
/// time; each lane follows the scalar order, so the values are the same.
fn softmax_probs(
  margin : FixedArray[Float],
  n : Int32,
  groups : Int32,
  out : FixedArray[Double],
) -> Unit {
  let mut r = 0
  while r + 1 < n {
    let a = r * groups
    let b = a + groups
    let mut mx = @simd.f64x2_const(
      margin.unsafe_get(a).to_double(),
      margin.unsafe_get(b).to_double(),
    )
    for k in 1..<groups {
      let v = @simd.f64x2_const(
        margin.unsafe_get(a + k).to_double(),
        margin.unsafe_get(b + k).to_double(),
      )
      mx = @simd.f64x2_pmax(mx, v)
    }
    let mut sum = @simd.f64x2_splat(0.0)
    for k in 0..<groups {
      let v = @simd.f64x2_const(
        margin.unsafe_get(a + k).to_double(),
        margin.unsafe_get(b + k).to_double(),
      )
      let e = exp_v(@simd.f64x2_sub(v, mx))
      sum = @simd.f64x2_add(sum, e)
      out.unsafe_set(a + k, @simd.f64x2_extract_lane(e, 0))
      out.unsafe_set(b + k, @simd.f64x2_extract_lane(e, 1))
    }
    let s0 = @simd.f64x2_extract_lane(sum, 0)
    let s1 = @simd.f64x2_extract_lane(sum, 1)
    for k in 0..<groups {
      out.unsafe_set(a + k, out.unsafe_get(a + k) / s0)
      out.unsafe_set(b + k, out.unsafe_get(b + k) / s1)
    }
    r += 2
  }
  if r < n {
    softmax_row(margin, r, groups, out)
  }
}

///|
/// `gradients` for binary:logistic, two rows at a time in the lanes of a
/// 128-bit vector: the same operations in the same order as train_exp and
/// the scalar loop, so every gradient is bit for bit the scalar one.
fn Grower::logistic_gradients(
  self : Grower,
  label : ArrayView[Float],
  margin : FixedArray[Float],
  groups : Int32,
  k : Int32,
  lo : Int32,
  hi : Int32,
  gsum : Double,
  hsum : Double,
) -> (Double, Double) {
  let gh = self.gh
  let gq = self.gq
  let rows = self.rows
  let mut gsum = gsum
  let mut hsum = hsum
  let one = @simd.f64x2_splat(1.0)
  let tiny = @simd.f64x2_splat(1.0e-16)
  let mut r = lo
  while r + 1 < hi {
    let m0 = margin.unsafe_get(r * groups + k).to_double()
    let m1 = margin.unsafe_get((r + 1) * groups + k).to_double()
    let x = @simd.f64x2_const(-m0, -m1)
    let e = exp_v(x)
    let pr = @simd.f64x2_div(one, @simd.f64x2_add(one, e))
    let h0 = @simd.f64x2_mul(pr, @simd.f64x2_sub(one, pr))
    let y = @simd.f64x2_const(label[r].to_double(), label[r + 1].to_double())
    // rounded to float32, the hessian floored at 1e-16
    let gf = @simd.f32x4_demote_f64x2_zero(@simd.f64x2_sub(pr, y))
    let hf = @simd.f32x4_demote_f64x2_zero(@simd.f64x2_pmax(h0, tiny))
    let g = @simd.f64x2_promote_low_f32x4(gf)
    let h = @simd.f64x2_promote_low_f32x4(hf)
    gh.unsafe_set(r * 2, @simd.f32x4_extract_lane(gf, 0))
    gh.unsafe_set(r * 2 + 1, @simd.f32x4_extract_lane(hf, 0))
    gh.unsafe_set(r * 2 + 2, @simd.f32x4_extract_lane(gf, 1))
    gh.unsafe_set(r * 2 + 3, @simd.f32x4_extract_lane(hf, 1))
    let g0 = @simd.f64x2_extract_lane(g, 0)
    let g1 = @simd.f64x2_extract_lane(g, 1)
    let h0 = @simd.f64x2_extract_lane(h, 0)
    let h1 = @simd.f64x2_extract_lane(h, 1)
    self.pack_at(gq, r, g0, h0)
    self.pack_at(gq, r + 1, g1, h1)
    rows.unsafe_set(r, r)
    rows.unsafe_set(r + 1, r + 1)
    gsum += g0
    gsum += g1
    hsum += h0
    hsum += h1
    r += 2
  }
  if r < hi {
    let y = label[r].to_double()
    let m = margin.unsafe_get(r * groups + k).to_double()
    let pr = 1.0 / (1.0 + train_exp(-m))
    let h0 = pr * (1.0 - pr)
    let g = Float::from_double(pr - y).to_double()
    let h = Float::from_double(if h0 < 1.0e-16 { 1.0e-16 } else { h0 }).to_double()
    gh.unsafe_set(r * 2, Float::from_double(g))
    gh.unsafe_set(r * 2 + 1, Float::from_double(h))
    self.pack_at(gq, r, g, h)
    rows.unsafe_set(r, r)
    gsum += g
    hsum += h
  }
  (gsum, hsum)
}
