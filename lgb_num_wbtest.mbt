///|
test "lgb_parse_double matches the general parser" {
  let cases = [
    "0", "-0", "1", "-1.5", "0.12345678901234568", "1.2345678901234567e-05", "-3.0000000000000004",
    "1e22", "1e23", "9007199254740993", "123456789012345678901", "4.9406564584124654e-324",
    "2.2250738585072014e-308", "1.7976931348623157e+308", "0.1", "0.30000000000000004",
    "1E-7", "+2.5", "5e-1", "0.000000000000000000001", "123456.78901234567", "-0.0099999999999999985",
    "17179869184.000004",
  ]
  for c in cases {
    let a = lgb_parse_double(c, 0, c.length())
    let b = @string.parse_double(c)
    assert_eq(a.reinterpret_as_uint64(), b.reinterpret_as_uint64())
  }
  // a pseudo-random sweep over 17-digit values in LightGBM's range
  let mut x = 0x9E3779B97F4A7C15UL
  for _ in 0..<20000 {
    x = x ^ (x << 13)
    x = x ^ (x >> 7)
    x = x ^ (x << 17)
    let m = (x % 100000000000000000UL).to_string()
    let e = (x >> 40).to_int() % 60 - 40
    let c = "\{if x % 2UL == 0UL { "-" } else { "" }}0.\{m}e\{e}"
    let a = lgb_parse_double(c, 0, c.length())
    let b = @string.parse_double(c)
    assert_eq(a.reinterpret_as_uint64(), b.reinterpret_as_uint64())
  }
  let bad = ["", "-", ".", "1e", "abc", "1.2.3"]
  for c in bad {
    let ok = try {
      ignore(lgb_parse_double(c, 0, c.length()))
      true
    } catch {
      _ => false
    }
    assert_false(ok)
  }
}
