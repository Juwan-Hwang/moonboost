///|
test "train_exp accuracy" {
  let mut worst = 0.0
  let mut x = -80.0
  while x <= 80.0 {
    let e = @math.exp(x)
    let d = ((train_exp(x) - e) / e).abs()
    if d > worst {
      worst = d
    }
    x += 0.0137
  }
  assert_true(worst < 1.0e-8)
}

///|
test "sorted_column matches a comparison sort" {
  let vals : Array[Float] = [
    3.5, -2.0, 0.0, -0.0, 1.0e30, -1.0e-30, 7.25, -2.0, 3.5, 1.0, -1000.0, 0.5,
  ]
  let inf = Float::from_double(1.0 / 0.0)
  vals.push(inf)
  vals.push(-inf)
  vals.push(Float::from_double(0.0 / 0.0))
  let m = DenseMatrix::new(vals, cols=1)
  let got = sorted_column(m, 0)
  let want = vals.filter(v => !v.is_nan())
  want.sort()
  assert_eq(got.length(), want.length())
  for i in 0..<want.length() {
    assert_true(got[i] == want[i])
  }
}
