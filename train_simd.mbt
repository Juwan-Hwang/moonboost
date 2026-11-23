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
  let hi_c = @v128.f64x2_splat(80.0)
  let lo_c = @v128.f64x2_splat(-80.0)
  let log2e = @v128.f64x2_splat(1.4426950408889634)
  let shifter = @v128.f64x2_splat(6755399441055744.0)
  let bias = @v128.i64x2_splat(0x4338000000000000UL - 1023UL)
  let ln2hi = @v128.f64x2_splat(0.6931471803691238)
  let ln2lo = @v128.f64x2_splat(1.9082149292705877e-10)
  let one = @v128.f64x2_splat(1.0)
  let c2 = @v128.f64x2_splat(0.5)
  let c3 = @v128.f64x2_splat(1.0 / 6.0)
  let c4 = @v128.f64x2_splat(1.0 / 24.0)
  let c5 = @v128.f64x2_splat(1.0 / 120.0)
  let c6 = @v128.f64x2_splat(1.0 / 720.0)
  let c7 = @v128.f64x2_splat(1.0 / 5040.0)
  let tiny = @v128.f64x2_splat(1.0e-16)
  let mut r = lo
  while r + 1 < hi {
    let m0 = margin.unsafe_get(r * groups + k).to_double()
    let m1 = margin.unsafe_get((r + 1) * groups + k).to_double()
    let x = @v128.f64x2_const(-m0, -m1)
    let x = @v128.f64x2_pmax(@v128.f64x2_pmin(x, hi_c), lo_c)
    let kd = @v128.f64x2_add(@v128.f64x2_mul(x, log2e), shifter)
    // (ki + 1023) << 52 with ki = bits(kd) - 0x4338000000000000
    let scale = @v128.i64x2_shl(@v128.i64x2_sub(kd, bias), 52)
    let kf = @v128.f64x2_sub(kd, shifter)
    let t = @v128.f64x2_sub(
      @v128.f64x2_sub(x, @v128.f64x2_mul(kf, ln2hi)),
      @v128.f64x2_mul(kf, ln2lo),
    )
    let t2 = @v128.f64x2_mul(t, t)
    let t4 = @v128.f64x2_mul(t2, t2)
    let a = @v128.f64x2_add(
      @v128.f64x2_add(one, t),
      @v128.f64x2_mul(t2, @v128.f64x2_add(c2, @v128.f64x2_mul(t, c3))),
    )
    let b = @v128.f64x2_add(
      @v128.f64x2_add(c4, @v128.f64x2_mul(t, c5)),
      @v128.f64x2_mul(t2, @v128.f64x2_add(c6, @v128.f64x2_mul(t, c7))),
    )
    let p = @v128.f64x2_add(a, @v128.f64x2_mul(t4, b))
    let e = @v128.f64x2_mul(p, scale)
    let pr = @v128.f64x2_div(one, @v128.f64x2_add(one, e))
    let h0 = @v128.f64x2_mul(pr, @v128.f64x2_sub(one, pr))
    let y = @v128.f64x2_const(label[r].to_double(), label[r + 1].to_double())
    // rounded to float32, the hessian floored at 1e-16
    let gf = @v128.f32x4_demote_f64x2_zero(@v128.f64x2_sub(pr, y))
    let hf = @v128.f32x4_demote_f64x2_zero(@v128.f64x2_pmax(h0, tiny))
    let g = @v128.f64x2_promote_low_f32x4(gf)
    let h = @v128.f64x2_promote_low_f32x4(hf)
    gh.unsafe_set(r * 2, @v128.f32x4_extract_lane(gf, 0))
    gh.unsafe_set(r * 2 + 1, @v128.f32x4_extract_lane(hf, 0))
    gh.unsafe_set(r * 2 + 2, @v128.f32x4_extract_lane(gf, 1))
    gh.unsafe_set(r * 2 + 3, @v128.f32x4_extract_lane(hf, 1))
    let g0 = @v128.f64x2_extract_lane(g, 0)
    let g1 = @v128.f64x2_extract_lane(g, 1)
    let h0 = @v128.f64x2_extract_lane(h, 0)
    let h1 = @v128.f64x2_extract_lane(h, 1)
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
