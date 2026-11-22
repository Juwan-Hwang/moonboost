///|
test "aft_erf matches erf" {
  let cases = [
    (0.1, 0.1124629160182849),
    (0.5, 0.5204998778130465),
    (1.0, 0.8427007929497149),
    (2.0, 0.9953222650189527),
    (3.0, 0.9999779095030014),
    (-1.5, -0.9661051464753108),
  ]
  for c in cases {
    let (x, want) = c
    assert_true((aft_erf(x) - want).abs() < 1.0e-9, msg="erf(\{x})")
  }
}

///|
test "aft_gradients are the derivatives of the AFT loss" {
  // finite differences of -log likelihood for each distribution and
  // censoring
  let nll = (d : Int32, lo : Double, hi : Double, m : Double) => {
    if lo == hi {
      let z = (@math.ln(lo) - m) / 0.8
      -@math.ln(aft_pdf(d, z) / (0.8 * lo))
    } else {
      let cu = if hi == aft_inf {
        1.0
      } else {
        aft_cdf(d, (@math.ln(hi) - m) / 0.8)
      }
      let cl = if lo <= 0.0 {
        0.0
      } else {
        aft_cdf(d, (@math.ln(lo) - m) / 0.8)
      }
      -@math.ln(cu - cl)
    }
  }
  let rows = [(2.0, 2.0), (1.5, aft_inf), (0.0, 3.0), (1.0, 4.0)]
  for d in 0..<3 {
    for row in rows {
      let (lo, hi) = row
      let a = aft_rows([Float::from_double(lo)][:], [Float::from_double(hi)])
      for m in [-0.3, 0.4, 1.2] {
        let g = aft_gradients(a, [Float::from_double(m)], d, 0.8)
        let mm = Float::from_double(m).to_double()
        let e = 1.0e-5
        let num_g = (nll(d, lo, hi, mm + e) - nll(d, lo, hi, mm - e)) /
          (2.0 * e)
        let num_h = (
            nll(d, lo, hi, mm + e) -
            2.0 * nll(d, lo, hi, mm) +
            nll(d, lo, hi, mm - e)
          ) /
          (e * e)
        assert_true(
          (g[0] - num_g).abs() < 1.0e-5,
          msg="grad d=\{d} \{lo} \{hi} m=\{m}: \{g[0]} vs \{num_g}",
        )
        assert_true(
          (g[1] - num_h).abs() < 1.0e-3 * (1.0 + num_h.abs()),
          msg="hess d=\{d} \{lo} \{hi} m=\{m}: \{g[1]} vs \{num_h}",
        )
      }
    }
  }
}
