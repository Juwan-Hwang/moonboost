// Number scanning for LightGBM model text.
//
// LightGBM writes doubles with up to 17 significant digits. This scans one
// token straight into a decimal mantissa and exponent, then converts with
// Clinger's exact fast path or the Eisel-Lemire algorithm, and hands anything
// it cannot certify to `@string.parse_double`, so every result is the
// correctly rounded double.
//
// The 128-bit powers of ten below are the slice 10^-64..10^40 of the table
// in moonbitlang/core (internal/strconv, generated from Go's
// internal/strconv/pow10tab.go), Apache-2.0.

///|
const LGB_POW10_MIN : Int32 = -64

///|
const LGB_POW10_MAX : Int32 = 40

///|
let lgb_pow10 : FixedArray[UInt64] = [
  0xa87fea27a539e9a5UL, 0x3f2398d747b36224UL, 0xd29fe4b18e88640eUL, 0x8eec7f0d19a03aadUL,
  0x83a3eeeef9153e89UL, 0x1953cf68300424acUL, 0xa48ceaaab75a8e2bUL, 0x5fa8c3423c052dd7UL,
  0xcdb02555653131b6UL, 0x3792f412cb06794dUL, 0x808e17555f3ebf11UL, 0xe2bbd88bbee40bd0UL,
  0xa0b19d2ab70e6ed6UL, 0x5b6aceaeae9d0ec4UL, 0xc8de047564d20a8bUL, 0xf245825a5a445275UL,
  0xfb158592be068d2eUL, 0xeed6e2f0f0d56712UL, 0x9ced737bb6c4183dUL, 0x55464dd69685606bUL,
  0xc428d05aa4751e4cUL, 0xaa97e14c3c26b886UL, 0xf53304714d9265dfUL, 0xd53dd99f4b3066a8UL,
  0x993fe2c6d07b7fabUL, 0xe546a8038efe4029UL, 0xbf8fdb78849a5f96UL, 0xde98520472bdd033UL,
  0xef73d256a5c0f77cUL, 0x963e66858f6d4440UL, 0x95a8637627989aadUL, 0xdde7001379a44aa8UL,
  0xbb127c53b17ec159UL, 0x5560c018580d5d52UL, 0xe9d71b689dde71afUL, 0xaab8f01e6e10b4a6UL,
  0x9226712162ab070dUL, 0xcab3961304ca70e8UL, 0xb6b00d69bb55c8d1UL, 0x3d607b97c5fd0d22UL,
  0xe45c10c42a2b3b05UL, 0x8cb89a7db77c506aUL, 0x8eb98a7a9a5b04e3UL, 0x77f3608e92adb242UL,
  0xb267ed1940f1c61cUL, 0x55f038b237591ed3UL, 0xdf01e85f912e37a3UL, 0x6b6c46dec52f6688UL,
  0x8b61313bbabce2c6UL, 0x2323ac4b3b3da015UL, 0xae397d8aa96c1b77UL, 0xabec975e0a0d081aUL,
  0xd9c7dced53c72255UL, 0x96e7bd358c904a21UL, 0x881cea14545c7575UL, 0x7e50d64177da2e54UL,
  0xaa242499697392d2UL, 0xdde50bd1d5d0b9e9UL, 0xd4ad2dbfc3d07787UL, 0x955e4ec64b44e864UL,
  0x84ec3c97da624ab4UL, 0xbd5af13bef0b113eUL, 0xa6274bbdd0fadd61UL, 0xecb1ad8aeacdd58eUL,
  0xcfb11ead453994baUL, 0x67de18eda5814af2UL, 0x81ceb32c4b43fcf4UL, 0x80eacf948770ced7UL,
  0xa2425ff75e14fc31UL, 0xa1258379a94d028dUL, 0xcad2f7f5359a3b3eUL, 0x096ee45813a04330UL,
  0xfd87b5f28300ca0dUL, 0x8bca9d6e188853fcUL, 0x9e74d1b791e07e48UL, 0x775ea264cf55347dUL,
  0xc612062576589ddaUL, 0x95364afe032a819dUL, 0xf79687aed3eec551UL, 0x3a83ddbd83f52204UL,
  0x9abe14cd44753b52UL, 0xc4926a9672793542UL, 0xc16d9a0095928a27UL, 0x75b7053c0f178293UL,
  0xf1c90080baf72cb1UL, 0x5324c68b12dd6338UL, 0x971da05074da7beeUL, 0xd3f6fc16ebca5e03UL,
  0xbce5086492111aeaUL, 0x88f4bb1ca6bcf584UL, 0xec1e4a7db69561a5UL, 0x2b31e9e3d06c32e5UL,
  0x9392ee8e921d5d07UL, 0x3aff322e62439fcfUL, 0xb877aa3236a4b449UL, 0x09befeb9fad487c2UL,
  0xe69594bec44de15bUL, 0x4c2ebe687989a9b3UL, 0x901d7cf73ab0acd9UL, 0x0f9d37014bf60a10UL,
  0xb424dc35095cd80fUL, 0x538484c19ef38c94UL, 0xe12e13424bb40e13UL, 0x2865a5f206b06fb9UL,
  0x8cbccc096f5088cbUL, 0xf93f87b7442e45d3UL, 0xafebff0bcb24aafeUL, 0xf78f69a51539d748UL,
  0xdbe6fecebdedd5beUL, 0xb573440e5a884d1bUL, 0x89705f4136b4a597UL, 0x31680a88f8953030UL,
  0xabcc77118461cefcUL, 0xfdc20d2b36ba7c3dUL, 0xd6bf94d5e57a42bcUL, 0x3d32907604691b4cUL,
  0x8637bd05af6c69b5UL, 0xa63f9a49c2c1b10fUL, 0xa7c5ac471b478423UL, 0x0fcf80dc33721d53UL,
  0xd1b71758e219652bUL, 0xd3c36113404ea4a8UL, 0x83126e978d4fdf3bUL, 0x645a1cac083126e9UL,
  0xa3d70a3d70a3d70aUL, 0x3d70a3d70a3d70a3UL, 0xccccccccccccccccUL, 0xccccccccccccccccUL,
  0x8000000000000000UL, 0x0000000000000000UL, 0xa000000000000000UL, 0x0000000000000000UL,
  0xc800000000000000UL, 0x0000000000000000UL, 0xfa00000000000000UL, 0x0000000000000000UL,
  0x9c40000000000000UL, 0x0000000000000000UL, 0xc350000000000000UL, 0x0000000000000000UL,
  0xf424000000000000UL, 0x0000000000000000UL, 0x9896800000000000UL, 0x0000000000000000UL,
  0xbebc200000000000UL, 0x0000000000000000UL, 0xee6b280000000000UL, 0x0000000000000000UL,
  0x9502f90000000000UL, 0x0000000000000000UL, 0xba43b74000000000UL, 0x0000000000000000UL,
  0xe8d4a51000000000UL, 0x0000000000000000UL, 0x9184e72a00000000UL, 0x0000000000000000UL,
  0xb5e620f480000000UL, 0x0000000000000000UL, 0xe35fa931a0000000UL, 0x0000000000000000UL,
  0x8e1bc9bf04000000UL, 0x0000000000000000UL, 0xb1a2bc2ec5000000UL, 0x0000000000000000UL,
  0xde0b6b3a76400000UL, 0x0000000000000000UL, 0x8ac7230489e80000UL, 0x0000000000000000UL,
  0xad78ebc5ac620000UL, 0x0000000000000000UL, 0xd8d726b7177a8000UL, 0x0000000000000000UL,
  0x878678326eac9000UL, 0x0000000000000000UL, 0xa968163f0a57b400UL, 0x0000000000000000UL,
  0xd3c21bcecceda100UL, 0x0000000000000000UL, 0x84595161401484a0UL, 0x0000000000000000UL,
  0xa56fa5b99019a5c8UL, 0x0000000000000000UL, 0xcecb8f27f4200f3aUL, 0x0000000000000000UL,
  0x813f3978f8940984UL, 0x4000000000000000UL, 0xa18f07d736b90be5UL, 0x5000000000000000UL,
  0xc9f2c9cd04674edeUL, 0xa400000000000000UL, 0xfc6f7c4045812296UL, 0x4d00000000000000UL,
  0x9dc5ada82b70b59dUL, 0xf020000000000000UL, 0xc5371912364ce305UL, 0x6c28000000000000UL,
  0xf684df56c3e01bc6UL, 0xc732000000000000UL, 0x9a130b963a6c115cUL, 0x3c7f400000000000UL,
  0xc097ce7bc90715b3UL, 0x4b9f100000000000UL, 0xf0bdc21abb48db20UL, 0x1e86d40000000000UL,
  0x96769950b50d88f4UL, 0x1314448000000000UL, 0xbc143fa4e250eb31UL, 0x17d955a000000000UL,
  0xeb194f8e1ae525fdUL, 0x5dcfab0800000000UL,
]

///|
let lgb_exact_pow10 : FixedArray[Double] = [
  1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11,
  1.0e12, 1.0e13, 1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19, 1.0e20, 1.0e21,
  1.0e22,
]

///|
fn lgb_umul128_hi(a : UInt64, b : UInt64) -> UInt64 {
  let a_lo = a & 0xffffffffUL
  let a_hi = a >> 32
  let b_lo = b & 0xffffffffUL
  let b_hi = b >> 32
  let x = a_lo * b_lo
  let y = a_hi * b_lo + (x >> 32)
  let z = a_lo * b_hi + (y & 0xffffffffUL)
  a_hi * b_hi + (y >> 32) + (z >> 32)
}

///|
/// Eisel-Lemire for `mantissa * 10^exponent`; NaN when it cannot certify the
/// rounding (the caller then falls back to the exact parser).
fn lgb_eisel_lemire(
  mantissa : UInt64,
  exponent : Int32,
  negative : Bool,
) -> Double {
  let nan = 0x7FF8000000000000UL.reinterpret_as_double()
  if exponent < LGB_POW10_MIN || exponent > LGB_POW10_MAX {
    return nan
  }
  let ix = (exponent - LGB_POW10_MIN) * 2
  let pow_hi = lgb_pow10[ix]
  let pow_lo = lgb_pow10[ix + 1]
  let pow_exp2 = 1 + ((exponent * 108853) >> 15)
  let lz = mantissa.clz()
  let normalized = mantissa << lz
  let mut result_exp2 = pow_exp2 + 63 + 1023 - lz
  let mut product_hi = lgb_umul128_hi(normalized, pow_hi)
  let mut product_lo = normalized * pow_hi
  if (product_hi & 0x1ffUL) == 0x1ffUL && product_lo + normalized < normalized {
    let wider_hi = lgb_umul128_hi(normalized, pow_lo)
    let wider_lo = normalized * pow_lo
    let mut merged_hi = product_hi
    let merged_lo = product_lo + wider_hi
    if merged_lo < product_lo {
      merged_hi += 1UL
    }
    if (merged_hi & 0x1ffUL) == 0x1ffUL &&
      merged_lo + 1UL == 0UL &&
      wider_lo + normalized < normalized {
      return nan
    }
    product_hi = merged_hi
    product_lo = merged_lo
  }
  let top_bit = (product_hi >> 63).to_int()
  let mut result_mantissa = product_hi >> (top_bit + 9)
  result_exp2 -= 1 - top_bit
  if product_lo == 0UL &&
    (product_hi & 0x1ffUL) == 0UL &&
    (result_mantissa & 3UL) == 1UL {
    return nan
  }
  result_mantissa += result_mantissa & 1UL
  result_mantissa = result_mantissa >> 1
  if result_mantissa >> 53 > 0UL {
    result_mantissa = result_mantissa >> 1
    result_exp2 += 1
  }
  if result_exp2 <= 0 || result_exp2 >= 0x7ff {
    return nan
  }
  let mut bits = (result_exp2.to_uint64() << 52) |
    (result_mantissa & 0x000fffffffffffffUL)
  if negative {
    bits = bits | 0x8000000000000000UL
  }
  bits.reinterpret_as_double()
}

///|
/// Parses one token `s[start:end]` as a double, correctly rounded.
fn lgb_parse_double(s : StringView, start : Int32, end : Int32) -> Double raise {
  let mut i = start
  let negative = i < end && s.unsafe_get(i) == '-'
  if negative || (i < end && s.unsafe_get(i) == '+') {
    i += 1
  }
  let mut mantissa = 0UL
  let mut sig = 0 // significant digits taken into `mantissa`
  let mut exp = 0
  let mut any = false
  while i < end {
    let c = s.unsafe_get(i).to_int() - 48
    if c < 0 || c > 9 {
      break
    }
    any = true
    if sig > 0 || c != 0 {
      mantissa = mantissa * 10UL + c.to_uint64()
      sig += 1
    }
    i += 1
  }
  if i < end && s.unsafe_get(i) == '.' {
    i += 1
    while i < end {
      let c = s.unsafe_get(i).to_int() - 48
      if c < 0 || c > 9 {
        break
      }
      any = true
      if sig > 0 || c != 0 {
        mantissa = mantissa * 10UL + c.to_uint64()
        sig += 1
      }
      exp -= 1
      i += 1
    }
  }
  if i < end && (s.unsafe_get(i) == 'e' || s.unsafe_get(i) == 'E') && any {
    i += 1
    let eneg = i < end && s.unsafe_get(i) == '-'
    if eneg || (i < end && s.unsafe_get(i) == '+') {
      i += 1
    }
    let mut e = 0
    let mut edigits = 0
    while i < end && edigits < 6 {
      let c = s.unsafe_get(i).to_int() - 48
      if c < 0 || c > 9 {
        break
      }
      e = e * 10 + c
      edigits += 1
      i += 1
    }
    if edigits == 0 {
      i = -1 // malformed: let the general parser report it
    }
    exp += if eneg { -e } else { e }
  }
  if i == end && any && sig <= 19 {
    if mantissa == 0UL {
      return if negative { -0.0 } else { 0.0 }
    }
    if mantissa <= 0x20000000000000UL && exp >= -22 && exp <= 22 {
      let m = mantissa.to_double()
      let v = if exp >= 0 {
        m * lgb_exact_pow10[exp]
      } else {
        m / lgb_exact_pow10[-exp]
      }
      return if negative { -v } else { v }
    }
    let v = lgb_eisel_lemire(mantissa, exp, negative)
    if !v.is_nan() {
      return v
    }
  }
  @string.parse_double(s.unsafe_view(start~, end~))
}
