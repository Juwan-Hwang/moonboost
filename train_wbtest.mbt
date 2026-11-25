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

///|
test "one-byte and two-byte copied bins train the same model" {
  // a few thousand rows, some features with missing values, deep enough
  // for subtrees to be copied; feature counts that fill one, two and
  // three row words fully and partly
  let rows = 3000
  for cols in [3, 8, 10, 15, 16, 20] {
    let data = Array::make(rows * cols, (0.0 : Float))
    let label = Array::make(rows, (0.0 : Float))
    let mut seed = 12345
    for r in 0..<rows {
      for f in 0..<cols {
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        let v = (seed % 1000).to_double() / 10.0
        data[r * cols + f] = if f == 2 && seed % 7 == 0 {
          Float::from_double(0.0 / 0.0)
        } else {
          Float::from_double(v)
        }
      }
      let x = data[r * cols].to_double() +
        data[r * cols + 1].to_double() +
        data[r * cols + cols - 1].to_double() * 0.5
      label[r] = if x > 125.0 { 1.0 } else { 0.0 }
    }
    let m = DenseMatrix::new(data, cols~)
    let p = {
      ..TrainParams::default(),
      objective: "binary:logistic",
      max_depth: 6,
      num_round: 3,
      max_bin: 64,
    }
    assert_eq(train_json(m, label, p, true), train_json(m, label, p, false))
    let lg = { ..p, grow_policy: "lossguide", max_depth: 0, max_leaves: 40, }
    assert_eq(train_json(m, label, lg, true), train_json(m, label, lg, false))
  }
}

///|
test "pairs packed in the bin words train the same model" {
  // more than 65536 rows, 9 and 12 features and loss-guided trees deep
  // enough for several copy levels: the copies carry the quantised pairs
  // in their bin words, against the two-byte path without them
  let rows = 70000
  for cols in [9, 12] {
    let data = Array::make(rows * cols, (0.0 : Float))
    let label = Array::make(rows, (0.0 : Float))
    let mut seed = 777
    for r in 0..<rows {
      for f in 0..<cols {
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        let v = (seed % 1000).to_double() / 10.0
        data[r * cols + f] = if f == 2 && seed % 7 == 0 {
          Float::from_double(0.0 / 0.0)
        } else {
          Float::from_double(v)
        }
      }
      seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
      let x = data[r * cols].to_double() +
        data[r * cols + 1].to_double() * data[r * cols + 3].to_double() / 50.0 +
        data[r * cols + cols - 1].to_double() * 0.5 +
        (seed % 100).to_double()
      label[r] = if x > 175.0 { 1.0 } else { 0.0 }
    }
    let m = DenseMatrix::new(data, cols~)
    let p = {
      ..TrainParams::default(),
      objective: "binary:logistic",
      grow_policy: "lossguide",
      max_depth: 0,
      max_leaves: 255,
      num_round: 2,
      max_bin: 64,
    }
    assert_eq(train_json(m, label, p, true), train_json(m, label, p, false))
  }
}

///|
test "quantile by selection matches quantile of the sorted values" {
  let mut s = 7
  for n in [1, 2, 3, 5, 17, 100, 1000, 4097] {
    for dup in [false, true] {
      let xs = FixedArray::makei(n, _ => {
        s = (s * 1103515245 + 12345) & 0x7fffffff
        let v = s % 100000
        Float::from_int(if dup { v % 7 } else { v })
      })
      let sorted = FixedArray::makei(n, i => xs[i])
      sorted.sort()
      for alpha in [0.01, 0.1, 0.25, 0.5, 0.77, 0.9, 0.99] {
        let nd = n.to_double()
        let want = if alpha <= 1.0 / (nd + 1.0) {
          sorted[0].to_double()
        } else if alpha >= nd / (nd + 1.0) {
          sorted[n - 1].to_double()
        } else {
          let x = alpha * (nd + 1.0)
          let k = x.floor() - 1.0
          let d = x - 1.0 - k
          let v0 = sorted[k.to_int()].to_double()
          v0 + d * (sorted[k.to_int() + 1].to_double() - v0)
        }
        let work = FixedArray::makei(n + 2, i => {
          if i == 0 || i == n + 1 {
            Float::from_int(-1)
          } else {
            xs[i - 1]
          }
        })
        assert_eq(quantile_range(work, 1, n + 1, alpha), want)
        assert_eq(work[0], Float::from_int(-1))
        assert_eq(work[n + 1], Float::from_int(-1))
      }
    }
  }
}
