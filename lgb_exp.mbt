///|
/// `exp` for LightGBM outputs, bit for bit as glibc computes it on x86-64.
///
/// LightGBM's transforms call the C library's `exp`, so its probabilities are
/// only reproducible with the same `exp`. This is the algorithm glibc 2.28+
/// uses (from Arm's optimized-routines, MIT OR Apache-2.0 WITH
/// LLVM-exception): a 128-entry table of `2^(k/128)` and a degree-5
/// polynomial, with the fused multiply-adds of the FMA build that x86-64
/// CPUs with AVX2 and FMA select. A fused multiply-add is emulated exactly
/// (Boldo and Melquiond's algorithm with rounding to odd), so the result is
/// the same on every target.
fn soft_exp(x : Double) -> Double {
  let abstop0 = (x.reinterpret_as_uint64() >> 52).to_int() & 0x7ff
  let mut abstop = abstop0
  if abstop - 0x3c9 >= 0x408 - 0x3c9 || abstop < 0x3c9 {
    if abstop < 0x3c9 {
      // |x| < 2^-54
      return 1.0 + x
    }
    if abstop >= 0x409 {
      if x.reinterpret_as_uint64() == 0xfff0000000000000UL {
        return 0.0
      }
      if abstop >= 0x7ff {
        return 1.0 + x
      }
      return if x < 0.0 {
        0.0
      } else {
        0x7FF0000000000000UL.reinterpret_as_double()
      }
    }
    // large |x|: handled by `exp_special` below
    abstop = 0
  }
  let kd0 = fma(x, exp_invln2n, exp_shift)
  let ki = kd0.reinterpret_as_uint64()
  let kd = kd0 - exp_shift
  let r = fma(kd, exp_negln2hin, x)
  let r = fma(kd, exp_negln2lon, r)
  let idx = (2UL * (ki & 127UL)).to_int()
  let top = ki << 45
  let tail = exp_tab[idx].reinterpret_as_double()
  let sbits = exp_tab[idx + 1] + top
  let r2 = r * r
  let a = fma(r, exp_c3, exp_c2)
  let b = fma(r, exp_c5, exp_c4)
  let tmp = fma(r2 * r2, b, fma(a, r2, tail + r))
  if abstop == 0 {
    return exp_special(tmp, sbits, ki)
  }
  let scale = sbits.reinterpret_as_double()
  fma(scale, tmp, scale)
}

///|
/// `scale * (1 + tmp)` when the scale's exponent over- or underflows.
fn exp_special(tmp : Double, sbits : UInt64, ki : UInt64) -> Double {
  if (ki & 0x80000000UL) == 0UL {
    let scale = (sbits - (1009UL << 52)).reinterpret_as_double()
    return fma(scale, tmp, scale) * 0x1.0p1009
  }
  let scale = (sbits + (1022UL << 52)).reinterpret_as_double()
  let st = scale * tmp
  let mut y = scale + st
  if y < 1.0 {
    // round y before scaling it into the subnormal range
    let hi = 1.0 + y
    let lo = scale - y + st
    let lo = 1.0 - hi + y + lo
    y = lo + hi - 1.0
    if y == 0.0 {
      y = 0.0
    }
  }
  0x1.0p-1022 * y
}

///|
/// `a * b + c` with a single rounding. Exact product by Veltkamp splitting,
/// then `c + product` rounded once by summing the low parts with rounding to
/// odd (Boldo and Melquiond, 2008). Valid while the product neither
/// overflows nor loses bits to underflow, which holds for every use in
/// `lgb_exp`.
fn fma(a : Double, b : Double, c : Double) -> Double {
  let ph = a * b
  let pl = exact_mul_err(a, b, ph)
  let th = c + ph
  let tl = two_sum_err(c, ph, th)
  let v = tl + pl
  let ve = two_sum_err(tl, pl, v)
  let v = if ve != 0.0 && (v.reinterpret_as_uint64() & 1UL) == 0UL {
    let u = v.reinterpret_as_uint64()
    (if (ve > 0.0) == (v > 0.0) { u + 1UL } else { u - 1UL }).reinterpret_as_double()
  } else {
    v
  }
  th + v
}

///|
/// The rounding error of `p = a * b`: `a * b - p` exactly.
fn exact_mul_err(a : Double, b : Double, p : Double) -> Double {
  let split = 134217729.0 // 2^27 + 1
  let t = split * a
  let ah = t - (t - a)
  let al = a - ah
  let t = split * b
  let bh = t - (t - b)
  let bl = b - bh
  ah * bh - p + ah * bl + al * bh + al * bl
}

///|
/// The rounding error of `s = a + b`: `a + b - s` exactly.
fn two_sum_err(a : Double, b : Double, s : Double) -> Double {
  let bb = s - a
  a - (s - bb) + (b - bb)
}

///|
let exp_invln2n : Double = 0x1.71547652b82fep7

///|
let exp_negln2hin : Double = -0x1.62e42fefa0000p-8

///|
let exp_negln2lon : Double = -0x1.cf79abc9e3b3ap-47

///|
let exp_shift : Double = 0x1.8p52

///|
let exp_c2 : Double = 0x1.ffffffffffdbdp-2

///|
let exp_c3 : Double = 0x1.555555555543cp-3

///|
let exp_c4 : Double = 0x1.55555cf172b91p-5

///|
let exp_c5 : Double = 0x1.1111167a4d017p-7

///|
/// `2^(k/128)` as (tail, scale bits with the exponent of k removed).
let exp_tab : FixedArray[UInt64] = [
  0x0UL, 0x3ff0000000000000UL, 0x3c9b3b4f1a88bf6eUL, 0x3feff63da9fb3335UL, 0xbc7160139cd8dc5dUL,
  0x3fefec9a3e778061UL, 0xbc905e7a108766d1UL, 0x3fefe315e86e7f85UL, 0x3c8cd2523567f613UL,
  0x3fefd9b0d3158574UL, 0xbc8bce8023f98efaUL, 0x3fefd06b29ddf6deUL, 0x3c60f74e61e6c861UL,
  0x3fefc74518759bc8UL, 0x3c90a3e45b33d399UL, 0x3fefbe3ecac6f383UL, 0x3c979aa65d837b6dUL,
  0x3fefb5586cf9890fUL, 0x3c8eb51a92fdeffcUL, 0x3fefac922b7247f7UL, 0x3c3ebe3d702f9cd1UL,
  0x3fefa3ec32d3d1a2UL, 0xbc6a033489906e0bUL, 0x3fef9b66affed31bUL, 0xbc9556522a2fbd0eUL,
  0x3fef9301d0125b51UL, 0xbc5080ef8c4eea55UL, 0x3fef8abdc06c31ccUL, 0xbc91c923b9d5f416UL,
  0x3fef829aaea92de0UL, 0x3c80d3e3e95c55afUL, 0x3fef7a98c8a58e51UL, 0xbc801b15eaa59348UL,
  0x3fef72b83c7d517bUL, 0xbc8f1ff055de323dUL, 0x3fef6af9388c8deaUL, 0x3c8b898c3f1353bfUL,
  0x3fef635beb6fcb75UL, 0xbc96d99c7611eb26UL, 0x3fef5be084045cd4UL, 0x3c9aecf73e3a2f60UL,
  0x3fef54873168b9aaUL, 0xbc8fe782cb86389dUL, 0x3fef4d5022fcd91dUL, 0x3c8a6f4144a6c38dUL,
  0x3fef463b88628cd6UL, 0x3c807a05b0e4047dUL, 0x3fef3f49917ddc96UL, 0x3c968efde3a8a894UL,
  0x3fef387a6e756238UL, 0x3c875e18f274487dUL, 0x3fef31ce4fb2a63fUL, 0x3c80472b981fe7f2UL,
  0x3fef2b4565e27cddUL, 0xbc96b87b3f71085eUL, 0x3fef24dfe1f56381UL, 0x3c82f7e16d09ab31UL,
  0x3fef1e9df51fdee1UL, 0xbc3d219b1a6fbffaUL, 0x3fef187fd0dad990UL, 0x3c8b3782720c0ab4UL,
  0x3fef1285a6e4030bUL, 0x3c6e149289cecb8fUL, 0x3fef0cafa93e2f56UL, 0x3c834d754db0abb6UL,
  0x3fef06fe0a31b715UL, 0x3c864201e2ac744cUL, 0x3fef0170fc4cd831UL, 0x3c8fdd395dd3f84aUL,
  0x3feefc08b26416ffUL, 0xbc86a3803b8e5b04UL, 0x3feef6c55f929ff1UL, 0xbc924aedcc4b5068UL,
  0x3feef1a7373aa9cbUL, 0xbc9907f81b512d8eUL, 0x3feeecae6d05d866UL, 0xbc71d1e83e9436d2UL,
  0x3feee7db34e59ff7UL, 0xbc991919b3ce1b15UL, 0x3feee32dc313a8e5UL, 0x3c859f48a72a4c6dUL,
  0x3feedea64c123422UL, 0xbc9312607a28698aUL, 0x3feeda4504ac801cUL, 0xbc58a78f4817895bUL,
  0x3feed60a21f72e2aUL, 0xbc7c2c9b67499a1bUL, 0x3feed1f5d950a897UL, 0x3c4363ed60c2ac11UL,
  0x3feece086061892dUL, 0x3c9666093b0664efUL, 0x3feeca41ed1d0057UL, 0x3c6ecce1daa10379UL,
  0x3feec6a2b5c13cd0UL, 0x3c93ff8e3f0f1230UL, 0x3feec32af0d7d3deUL, 0x3c7690cebb7aafb0UL,
  0x3feebfdad5362a27UL, 0x3c931dbdeb54e077UL, 0x3feebcb299fddd0dUL, 0xbc8f94340071a38eUL,
  0x3feeb9b2769d2ca7UL, 0xbc87deccdc93a349UL, 0x3feeb6daa2cf6642UL, 0xbc78dec6bd0f385fUL,
  0x3feeb42b569d4f82UL, 0xbc861246ec7b5cf6UL, 0x3feeb1a4ca5d920fUL, 0x3c93350518fdd78eUL,
  0x3feeaf4736b527daUL, 0x3c7b98b72f8a9b05UL, 0x3feead12d497c7fdUL, 0x3c9063e1e21c5409UL,
  0x3feeab07dd485429UL, 0x3c34c7855019c6eaUL, 0x3feea9268a5946b7UL, 0x3c9432e62b64c035UL,
  0x3feea76f15ad2148UL, 0xbc8ce44a6199769fUL, 0x3feea5e1b976dc09UL, 0xbc8c33c53bef4da8UL,
  0x3feea47eb03a5585UL, 0xbc845378892be9aeUL, 0x3feea34634ccc320UL, 0xbc93cedd78565858UL,
  0x3feea23882552225UL, 0x3c5710aa807e1964UL, 0x3feea155d44ca973UL, 0xbc93b3efbf5e2228UL,
  0x3feea09e667f3bcdUL, 0xbc6a12ad8734b982UL, 0x3feea012750bdabfUL, 0xbc6367efb86da9eeUL,
  0x3fee9fb23c651a2fUL, 0xbc80dc3d54e08851UL, 0x3fee9f7df9519484UL, 0xbc781f647e5a3ecfUL,
  0x3fee9f75e8ec5f74UL, 0xbc86ee4ac08b7db0UL, 0x3fee9f9a48a58174UL, 0xbc8619321e55e68aUL,
  0x3fee9feb564267c9UL, 0x3c909ccb5e09d4d3UL, 0x3feea0694fde5d3fUL, 0xbc7b32dcb94da51dUL,
  0x3feea11473eb0187UL, 0x3c94ecfd5467c06bUL, 0x3feea1ed0130c132UL, 0x3c65ebe1abd66c55UL,
  0x3feea2f336cf4e62UL, 0xbc88a1c52fb3cf42UL, 0x3feea427543e1a12UL, 0xbc9369b6f13b3734UL,
  0x3feea589994cce13UL, 0xbc805e843a19ff1eUL, 0x3feea71a4623c7adUL, 0xbc94d450d872576eUL,
  0x3feea8d99b4492edUL, 0x3c90ad675b0e8a00UL, 0x3feeaac7d98a6699UL, 0x3c8db72fc1f0eab4UL,
  0x3feeace5422aa0dbUL, 0xbc65b6609cc5e7ffUL, 0x3feeaf3216b5448cUL, 0x3c7bf68359f35f44UL,
  0x3feeb1ae99157736UL, 0xbc93091fa71e3d83UL, 0x3feeb45b0b91ffc6UL, 0xbc5da9b88b6c1e29UL,
  0x3feeb737b0cdc5e5UL, 0xbc6c23f97c90b959UL, 0x3feeba44cbc8520fUL, 0xbc92434322f4f9aaUL,
  0x3feebd829fde4e50UL, 0xbc85ca6cd7668e4bUL, 0x3feec0f170ca07baUL, 0x3c71affc2b91ce27UL,
  0x3feec49182a3f090UL, 0x3c6dd235e10a73bbUL, 0x3feec86319e32323UL, 0xbc87c50422622263UL,
  0x3feecc667b5de565UL, 0x3c8b1c86e3e231d5UL, 0x3feed09bec4a2d33UL, 0xbc91bbd1d3bcbb15UL,
  0x3feed503b23e255dUL, 0x3c90cc319cee31d2UL, 0x3feed99e1330b358UL, 0x3c8469846e735ab3UL,
  0x3feede6b5579fdbfUL, 0xbc82dfcd978e9db4UL, 0x3feee36bbfd3f37aUL, 0x3c8c1a7792cb3387UL,
  0x3feee89f995ad3adUL, 0xbc907b8f4ad1d9faUL, 0x3feeee07298db666UL, 0xbc55c3d956dcaebaUL,
  0x3feef3a2b84f15fbUL, 0xbc90a40e3da6f640UL, 0x3feef9728de5593aUL, 0xbc68d6f438ad9334UL,
  0x3feeff76f2fb5e47UL, 0xbc91eee26b588a35UL, 0x3fef05b030a1064aUL, 0x3c74ffd70a5fddcdUL,
  0x3fef0c1e904bc1d2UL, 0xbc91bdfbfa9298acUL, 0x3fef12c25bd71e09UL, 0x3c736eae30af0cb3UL,
  0x3fef199bdd85529cUL, 0x3c8ee3325c9ffd94UL, 0x3fef20ab5fffd07aUL, 0x3c84e08fd10959acUL,
  0x3fef27f12e57d14bUL, 0x3c63cdaf384e1a67UL, 0x3fef2f6d9406e7b5UL, 0x3c676b2c6c921968UL,
  0x3fef3720dcef9069UL, 0xbc808a1883ccb5d2UL, 0x3fef3f0b555dc3faUL, 0xbc8fad5d3ffffa6fUL,
  0x3fef472d4a07897cUL, 0xbc900dae3875a949UL, 0x3fef4f87080d89f2UL, 0x3c74a385a63d07a7UL,
  0x3fef5818dcfba487UL, 0xbc82919e2040220fUL, 0x3fef60e316c98398UL, 0x3c8e5a50d5c192acUL,
  0x3fef69e603db3285UL, 0x3c843a59ac016b4bUL, 0x3fef7321f301b460UL, 0xbc82d52107b43e1fUL,
  0x3fef7c97337b9b5fUL, 0xbc892ab93b470dc9UL, 0x3fef864614f5a129UL, 0x3c74b604603a88d3UL,
  0x3fef902ee78b3ff6UL, 0x3c83c5ec519d7271UL, 0x3fef9a51fbc74c83UL, 0xbc8ff7128fd391f0UL,
  0x3fefa4afa2a490daUL, 0xbc8dae98e223747dUL, 0x3fefaf482d8e67f1UL, 0x3c8ec3bc41aa2008UL,
  0x3fefba1bee615a27UL, 0x3c842b94c3a9eb32UL, 0x3fefc52b376bba97UL, 0x3c8a64a931d185eeUL,
  0x3fefd0765b6e4540UL, 0xbc8e37bae43be3edUL, 0x3fefdbfdad9cbe14UL, 0x3c77893b4d91cd9dUL,
  0x3fefe7c1819e90d8UL, 0x3c5305c14160cc89UL, 0x3feff3c22b8f71f1UL,
]
