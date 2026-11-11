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
