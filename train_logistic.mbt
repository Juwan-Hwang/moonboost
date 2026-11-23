///|
/// `gradients` for binary:logistic.
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
  for r in lo..<hi {
    let y = label[r].to_double()
    let m = margin[r * groups + k].to_double()
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

///|
/// Softmax probabilities of every row and class into out.
fn softmax_probs(
  margin : FixedArray[Float],
  n : Int32,
  groups : Int32,
  out : FixedArray[Double],
) -> Unit {
  for r in 0..<n {
    softmax_row(margin, r, groups, out)
  }
}
