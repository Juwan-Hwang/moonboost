///|
test "csv_cell gives the bits parse_double gives" {
  let cases = [
    "0", "-0", "1", "-1", "0.1", "0.3", "123.456", "-7.25e-3", "1e22", "1e23", "9.999999e-23",
    "3.4028235e38", "1.17549435e-38", "123456789012345", "1234567890123456789", "0.000000000000000000001",
    " 42 ", "5e+2", "2E-1", "+3", "007", "0.0001234", "65504", "1.5e-45", "8388609",
    "16777217",
  ]
  let mut seed = 12345UL
  for _ in 0..<3000 {
    seed = seed * 6364136223846793005UL + 1442695040888963407UL
    let m = (seed >> 20).reinterpret_as_int64() % 1000000000L
    let e = (seed >> 50).to_int() % 40 - 20
    let neg = seed >> 63 == 1UL
    let digits = m.to_string()
    let k = (seed >> 8).to_int() & 7
    let body = if k < digits.length() {
      digits.clamped_view(end=k).to_owned() +
      "." +
      digits.clamped_view(start=k).to_owned()
    } else {
      digits
    }
    let s = (if neg { "-" } else { "" }) +
      body +
      (if (seed & 3UL) == 0UL { "e\{e}" } else { "" })
    cases.push(s)
  }
  for s in cases {
    let a = csv_cell(s[:], 0, s.length(), 1)
    let b = parse_cell(s, 1)
    if a.reinterpret_as_uint() != b.reinterpret_as_uint() {
      fail("\{s}: \{a} vs \{b}")
    }
  }
  for s in cases {
    let a = csv_cell(@utf8.encode(s)[:], 0, s.length(), 1)
    let b = parse_cell(s, 1)
    if a.reinterpret_as_uint() != b.reinterpret_as_uint() {
      fail("bytes \{s}: \{a} vs \{b}")
    }
  }
  let t = "x;y\n1;2.5\r\n\n-3;NA\n"
  let mb = DenseMatrix::from_csv_bytes(@utf8.encode(t)[:], header=true, sep=';')
  let ms = DenseMatrix::from_csv(t, header=true, sep=";")
  assert_eq(
    Array::makei(mb.data.length(), i => mb.data[i].reinterpret_as_uint()),
    Array::makei(ms.data.length(), i => ms.data[i].reinterpret_as_uint()),
  )
  let m = DenseMatrix::from_csv("a,b\n1, 2.5\r\n\n-3,NA\n", header=true)
  assert_eq(m.rows, 2)
  assert_eq(m.data[1], 2.5)
  assert_true(m.data[3].is_nan())
}
