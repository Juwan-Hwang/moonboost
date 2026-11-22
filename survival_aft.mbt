// survival:aft, XGBoost's accelerated failure time loss: log T = margin +
// sigma Z, Z normal, logistic or extreme-value, on interval-censored labels
// [lower, upper] (lower == upper an observed time, upper = +inf right
// censored, lower = 0 left censored). Gradients, hessians, their clipping
// and their limits where the likelihood underflows follow XGBoost's
// survival_util.h.

///|
let aft_inf : Double = 0x7FF0000000000000L.reinterpret_as_double()

///|
let aft_min_grad = -15.0

///|
let aft_max_grad = 15.0

///|
let aft_min_hess = 1.0e-16

///|
let aft_max_hess = 15.0

///|
let aft_eps = 1.0e-12

///|
/// The distribution of `aft_loss_distribution`: 0 normal, 1 logistic, 2
/// extreme.
fn aft_distribution(name : String) -> Int32 raise ModelError {
  match name {
    "normal" => 0
    "logistic" => 1
    "extreme" => 2
    _ =>
      raise InvalidModel(
        "aft_loss_distribution must be normal, logistic or extreme",
      )
  }
}

///|
/// e^x over the whole double range, the same on every backend: train_exp
/// of x / 16 raised to the 16th power, +inf above 709.8 and 0 far below.
fn aft_exp(x : Double) -> Double {
  if x > 709.782712893384 {
    return aft_inf
  }
  if x < -745.2 {
    return 0.0
  }
  let s = train_exp(x * 0.0625)
  let s2 = s * s
  let s4 = s2 * s2
  let s8 = s4 * s4
  s8 * s8
}

///|
fn aft_is_inf(x : Double) -> Bool {
  x == aft_inf || x == -aft_inf
}

///|
/// erf, fdlibm's rational approximations, with aft_exp for its tail.
fn aft_erf(x : Double) -> Double {
  if x != x {
    return x
  }
  let ax = x.abs()
  let sign = x < 0.0
  if ax < 0.84375 {
    if ax < 3.725290298461914e-9 {
      return 0.125 * (8.0 * x + 1.02703333676410069053 * x)
    }
    let z = x * x
    let r = 1.28379167095512558561e-01 +
      z *
      (
        -3.25042107247001499370e-01 +
        z *
        (
          -2.84817495755985104766e-02 +
          z * (-5.77027029648944159157e-03 + z * -2.37630166566501626084e-05)
        )
      )
    let s = 1.0 +
      z *
      (
        3.97917223959155352819e-01 +
        z *
        (
          6.50222499887672944485e-02 +
          z *
          (
            5.08130628187576562776e-03 +
            z * (1.32494738004321644526e-04 + z * -3.96022827877536812320e-06)
          )
        )
      )
    return x + x * (r / s)
  }
  let y = if ax >= 6.0 {
    1.0 - 2.2250738585072014e-308
  } else if ax < 1.25 {
    let s = ax - 1.0
    let p = -2.36211856075265944077e-03 +
      s *
      (
        4.14856118683748331666e-01 +
        s *
        (
          -3.72207876035701323847e-01 +
          s *
          (
            3.18346619901161753674e-01 +
            s *
            (
              -1.10894694282396677476e-01 +
              s * (3.54783043256182359371e-02 + s * -2.16637559486879084300e-03)
            )
          )
        )
      )
    let q = 1.0 +
      s *
      (
        1.06420880400844228286e-01 +
        s *
        (
          5.40397917702171048937e-01 +
          s *
          (
            7.18286544141962662868e-02 +
            s *
            (
              1.26171219808761642112e-01 +
              s * (1.36370839120290507362e-02 + s * 1.19844998467991074170e-02)
            )
          )
        )
      )
    // 1 - erfc(x) with erfc = 1 - erx - P/Q
    8.45062911510467529297e-01 + p / q
  } else {
    let s = 1.0 / (ax * ax)
    let (r, ss) = if ax < 2.857142857142857 {
      (
        -9.86494403484714822705e-03 +
        s *
        (
          -6.93858572707181764372e-01 +
          s *
          (
            -1.05586262253232909814e+01 +
            s *
            (
              -6.23753324503260060396e+01 +
              s *
              (
                -1.62396669462573470355e+02 +
                s *
                (
                  -1.84605092906711035994e+02 +
                  s *
                  (
                    -8.12874355063065934246e+01 +
                    s * -9.81432934416914548592e+00
                  )
                )
              )
            )
          )
        ),
        1.0 +
        s *
        (
          1.96512716674392571292e+01 +
          s *
          (
            1.37657754143519042600e+02 +
            s *
            (
              4.34565877475229228821e+02 +
              s *
              (
                6.45387271733267880336e+02 +
                s *
                (
                  4.29008140027567833386e+02 +
                  s *
                  (
                    1.08635005541779435134e+02 +
                    s *
                    (
                      6.57024977031928170135e+00 +
                      s * -6.04244152148580987438e-02
                    )
                  )
                )
              )
            )
          )
        ),
      )
    } else {
      (
        -9.86494292470009928597e-03 +
        s *
        (
          -7.99283237680523006574e-01 +
          s *
          (
            -1.77579549177547519889e+01 +
            s *
            (
              -1.60636384855821916062e+02 +
              s *
              (
                -6.37566443368389627722e+02 +
                s *
                (-1.02509513161107724954e+03 + s * -4.83519191608651397019e+02)
              )
            )
          )
        ),
        1.0 +
        s *
        (
          3.03380607434824582924e+01 +
          s *
          (
            3.25792512996573918826e+02 +
            s *
            (
              1.53672958608443695994e+03 +
              s *
              (
                3.19985821950859553908e+03 +
                s *
                (
                  2.55305040643316442583e+03 +
                  s *
                  (4.74528541206955367215e+02 + s * -2.24409524465858183362e+01)
                )
              )
            )
          )
        ),
      )
    }
    // z: x with the low 32 bits of its mantissa cleared, so z * z is exact
    let z = (ax.reinterpret_as_int64() & -4294967296L).reinterpret_as_double()
    let erfc = aft_exp(-z * z - 0.5625) *
      aft_exp((z - ax) * (z + ax) + r / ss) /
      ax
    1.0 - erfc
  }
  if sign {
    -y
  } else {
    y
  }
}

///|
let inv_sqrt_2pi = 0.3989422804014327

///|
fn aft_pdf(d : Int32, z : Double) -> Double {
  match d {
    0 => aft_exp(-z * z / 2.0) * inv_sqrt_2pi
    1 => {
      let w = aft_exp(z)
      if aft_is_inf(w) || aft_is_inf(w * w) {
        0.0
      } else {
        w / ((1.0 + w) * (1.0 + w))
      }
    }
    _ => {
      let w = aft_exp(z)
      if aft_is_inf(w) {
        0.0
      } else {
        w * aft_exp(-w)
      }
    }
  }
}

///|
fn aft_cdf(d : Int32, z : Double) -> Double {
  match d {
    0 => 0.5 * (1.0 + aft_erf(z / 1.4142135623730951))
    1 => {
      let w = aft_exp(z)
      if aft_is_inf(w) {
        1.0
      } else {
        w / (1.0 + w)
      }
    }
    _ => 1.0 - aft_exp(-aft_exp(z))
  }
}

///|
fn aft_grad_pdf(d : Int32, z : Double) -> Double {
  match d {
    0 => -z * aft_pdf(0, z)
    1 => {
      let w = aft_exp(z)
      if aft_is_inf(w) {
        0.0
      } else {
        aft_pdf(1, z) * (1.0 - w) / (1.0 + w)
      }
    }
    _ => {
      let w = aft_exp(z)
      if aft_is_inf(w) {
        0.0
      } else {
        (1.0 - w) * aft_pdf(2, z)
      }
    }
  }
}

///|
fn aft_hess_pdf(d : Int32, z : Double) -> Double {
  match d {
    0 => (z * z - 1.0) * aft_pdf(0, z)
    1 => {
      let w = aft_exp(z)
      if aft_is_inf(w) || aft_is_inf(w * w) {
        0.0
      } else {
        aft_pdf(1, z) * (w * w - 4.0 * w + 1.0) / ((1.0 + w) * (1.0 + w))
      }
    }
    _ => {
      let w = aft_exp(z)
      if aft_is_inf(w) || aft_is_inf(w * w) {
        0.0
      } else {
        (w * w - 3.0 * w + 1.0) * aft_pdf(2, z)
      }
    }
  }
}

///|
/// XGBoost's gradient where the likelihood underflows, by censoring (0
/// uncensored, 1 right, 2 left, 3 interval) and the sign of z.
fn aft_limit_grad(d : Int32, c : Int32, sign : Bool, sigma : Double) -> Double {
  let hi = match d {
    0 => aft_max_grad
    _ => 1.0 / sigma
  }
  let lo = match d {
    1 => -1.0 / sigma
    _ => aft_min_grad
  }
  match c {
    1 => if sign { lo } else { 0.0 }
    2 => if sign { 0.0 } else { hi }
    _ => if sign { lo } else { hi }
  }
}

///|
fn aft_limit_hess(d : Int32, c : Int32, sign : Bool, sigma : Double) -> Double {
  match d {
    0 =>
      match c {
        1 => if sign { 1.0 / (sigma * sigma) } else { aft_min_hess }
        2 => if sign { aft_min_hess } else { 1.0 / (sigma * sigma) }
        _ => 1.0 / (sigma * sigma)
      }
    1 => aft_min_hess
    _ =>
      match c {
        2 => aft_min_hess
        _ => if sign { aft_max_hess } else { aft_min_hess }
      }
  }
}

///|
fn aft_clip(x : Double, lo : Double, hi : Double) -> Double {
  if x < lo {
    lo
  } else if x > hi {
    hi
  } else {
    x
  }
}

///|
/// Per row: the log of the lower and upper bound (unused for a censored
/// side) and the censoring, 0 observed, 1 right, 2 left, 3 interval, 4
/// both bounds open.
priv struct AftRows {
  log_lower : FixedArray[Double]
  log_upper : FixedArray[Double]
  censor : FixedArray[Int32]
}

///|
fn aft_rows(
  lower : ArrayView[Float],
  upper : Array[Float],
) -> AftRows raise ModelError {
  let n = lower.length()
  if upper.length() != n {
    raise InvalidModel(
      "survival:aft needs label_upper_bound for every row (\{upper.length()} for \{n} rows)",
    )
  }
  let ll = FixedArray::make(n, 0.0)
  let lu = FixedArray::make(n, 0.0)
  let cs = FixedArray::make(n, 0)
  for r in 0..<n {
    let lo = lower[r].to_double()
    let hi = upper[r].to_double()
    if !(lo >= 0.0) || !(hi >= lo) || hi <= 0.0 {
      raise InvalidModel(
        "survival:aft bounds must satisfy 0 <= lower <= upper, upper > 0",
      )
    }
    if lo == hi {
      ll[r] = @math.ln(lo)
      lu[r] = ll[r]
      cs[r] = 0
    } else {
      let right = hi == aft_inf
      let left = lo <= 0.0
      if !left {
        ll[r] = @math.ln(lo)
      }
      if !right {
        lu[r] = @math.ln(hi)
      }
      cs[r] = if right && left {
        4
      } else if right {
        1
      } else if left {
        2
      } else {
        3
      }
    }
  }
  { log_lower: ll, log_upper: lu, censor: cs, }
}

///|
/// Gradient and hessian of the AFT negative log likelihood, interleaved.
fn aft_gradients(
  rows : AftRows,
  margin : FixedArray[Float],
  d : Int32,
  sigma : Double,
) -> FixedArray[Double] {
  let n = rows.censor.length()
  let out = FixedArray::make(n * 2, 0.0)
  for r in 0..<n {
    let pred = margin[r].to_double()
    let c = rows.censor[r]
    let mut grad = 0.0
    let mut hess = 0.0
    if c == 0 {
      let z = (rows.log_lower[r] - pred) / sigma
      let sign = z > 0.0
      let pdf = aft_pdf(d, z)
      let gp = aft_grad_pdf(d, z)
      let hp = aft_hess_pdf(d, z)
      let den = sigma * pdf
      grad = gp / den
      if den < aft_eps && (grad != grad || aft_is_inf(grad)) {
        grad = aft_limit_grad(d, 0, sign, sigma)
      }
      let hden = sigma * sigma * pdf * pdf
      hess = -(pdf * hp - gp * gp) / hden
      if hden < aft_eps && (hess != hess || aft_is_inf(hess)) {
        hess = aft_limit_hess(d, 0, sign, sigma)
      }
    } else {
      let mut zu = 0.0
      let mut zl = 0.0
      // right-censored (and unbounded) rows have no upper side, left-censored
      // ones no lower side
      let (pdf_u, cdf_u, gpdf_u) = if c == 1 || c == 4 {
        (0.0, 1.0, 0.0)
      } else {
        zu = (rows.log_upper[r] - pred) / sigma
        (aft_pdf(d, zu), aft_cdf(d, zu), aft_grad_pdf(d, zu))
      }
      let (pdf_l, cdf_l, gpdf_l) = if c == 2 || c == 4 {
        (0.0, 0.0, 0.0)
      } else {
        zl = (rows.log_lower[r] - pred) / sigma
        (aft_pdf(d, zl), aft_cdf(d, zl), aft_grad_pdf(d, zl))
      }
      let sign = zu > 0.0 || zl > 0.0
      let cdf_diff = cdf_u - cdf_l
      let pdf_diff = pdf_u - pdf_l
      let den = sigma * cdf_diff
      grad = pdf_diff / den
      if den < aft_eps && (grad != grad || aft_is_inf(grad)) {
        grad = aft_limit_grad(d, if c == 4 { 2 } else { c }, sign, sigma)
      }
      let hden = den * den
      hess = -(cdf_diff * (gpdf_u - gpdf_l) - pdf_diff * pdf_diff) / hden
      if hden < aft_eps && (hess != hess || aft_is_inf(hess)) {
        hess = aft_limit_hess(d, if c == 4 { 2 } else { c }, sign, sigma)
      }
    }
    out[r * 2] = aft_clip(grad, aft_min_grad, aft_max_grad)
    out[r * 2 + 1] = aft_clip(hess, aft_min_hess, aft_max_hess)
  }
  out
}

///|
/// The constant log time of highest likelihood: the root of the summed
/// gradient (increasing in the margin, the loss being convex), by Newton
/// steps kept inside a bracket that bisection narrows when they leave it.
fn aft_intercept(rows : AftRows, d : Int32, sigma : Double) -> Double {
  let n = rows.censor.length()
  let mut lo = 1.0e300
  let mut hi = -1.0e300
  let mut sum = 0.0
  let mut cnt = 0
  for r in 0..<n {
    let c = rows.censor[r]
    if c != 2 && c != 4 {
      let v = rows.log_lower[r]
      sum += v
      cnt += 1
      lo = if v < lo { v } else { lo }
      hi = if v > hi { v } else { hi }
    }
    if c == 2 || c == 3 {
      let v = rows.log_upper[r]
      sum += v
      cnt += 1
      lo = if v < lo { v } else { lo }
      hi = if v > hi { v } else { hi }
    }
  }
  if cnt == 0 {
    return 0.0
  }
  // the root lies within the bounds widened by a few scales
  lo -= 10.0 * sigma
  hi += 10.0 * sigma
  let mut m = sum / cnt.to_double()
  let margin = FixedArray::make(n, Float::from_double(m))
  for _ in 0..<40 {
    margin.fill(Float::from_double(m))
    let gh = aft_gradients(rows, margin, d, sigma)
    let mut g = 0.0
    let mut h = 0.0
    for r in 0..<n {
      g += gh[r * 2]
      h += gh[r * 2 + 1]
    }
    if g > 0.0 {
      hi = m
    } else if g < 0.0 {
      lo = m
    } else {
      break
    }
    let next = m - g / h
    let next = if next > lo && next < hi { next } else { 0.5 * (lo + hi) }
    if (next - m).abs() < 1.0e-7 || hi - lo < 1.0e-7 {
      m = next
      break
    }
    m = next
  }
  m
}
