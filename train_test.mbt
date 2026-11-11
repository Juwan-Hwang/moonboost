///|
/// A small deterministic data set: y = 3 * x0 - 2 * x1 (+ noise in x2).
fn toy(n : Int32) -> (@moonboost.DenseMatrix, Array[Float]) raise {
  let xs : Array[Float] = []
  let ys : Array[Float] = []
  let mut s = 12345
  let next = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    (s % 2000).to_double() / 1000.0 - 1.0
  }
  for i in 0..<n {
    let a = next()
    let b = next()
    let c = next()
    xs.push(Float::from_double(a))
    // every seventh row misses its second feature
    xs.push(
      if i % 7 == 0 {
        @float.not_a_number
      } else {
        Float::from_double(b)
      },
    )
    xs.push(Float::from_double(c))
    ys.push(
      Float::from_double(3.0 * a - 2.0 * (if i % 7 == 0 { 0.5 } else { b })),
    )
  }
  (@moonboost.DenseMatrix::new(xs, cols=3), ys)
}

///|
test "training reduces the error and the model reloads unchanged" {
  let (x, y) = toy(2000)
  let params = { ..@moonboost.TrainParams::default(), num_round: 50, }
  let json = @moonboost.train_xgboost_json(x, y, params~)
  let model = @moonboost.Model::from_xgboost_json(json)
  let mut se = 0.0
  let mut total = 0.0
  let mut mean = 0.0
  for v in y {
    mean += v.to_double()
  }
  mean = mean / y.length().to_double()
  let pred = model.predict_batch(x)
  for i in 0..<y.length() {
    let d = pred[i].to_double() - y[i].to_double()
    se += d * d
    total += (y[i].to_double() - mean) * (y[i].to_double() - mean)
  }
  // explains more than 99% of the variance
  assert_true(se < 0.01 * total)
  // training is deterministic
  assert_eq(@moonboost.train_xgboost_json(x, y, params~), json)
}

///|
test "binary and multiclass training" {
  let (x, y) = toy(1500)
  let yb = y.map(v => if v > 0.0 { Float::from_double(1.0) } else { 0.0 })
  let mb = @moonboost.Model::train(x, yb, params={
    ..@moonboost.TrainParams::default(),
    objective: "binary:logistic",
    num_round: 30,
  })
  let pb = mb.predict_batch(x)
  let mut right = 0
  for i in 0..<yb.length() {
    if (pb[i] > 0.5) == (yb[i] > 0.5) {
      right += 1
    }
  }
  assert_true(right > yb.length() * 95 / 100)
  let yc = y.map(v => {
    if v < -1.0 {
      Float::from_double(0.0)
    } else if v < 1.0 {
      1.0
    } else {
      2.0
    }
  })
  let mc = @moonboost.Model::train(x, yc, params={
    ..@moonboost.TrainParams::default(),
    objective: "multi:softmax",
    num_class: 3,
    num_round: 30,
  })
  let pc = mc.predict_batch(x)
  let mut ok = 0
  for i in 0..<yc.length() {
    if pc[i] == yc[i] {
      ok += 1
    }
  }
  assert_true(ok > yc.length() * 90 / 100)
}

///|
test "bad training input is rejected" {
  let (x, y) = toy(10)
  let bad = [
    () => @moonboost.train_xgboost_json(x, y[:5]),
    () => {
      @moonboost.train_xgboost_json(x, y, params={
        ..@moonboost.TrainParams::default(),
        objective: "binary:logistic",
      })
    },
    () => {
      @moonboost.train_xgboost_json(x, y, params={
        ..@moonboost.TrainParams::default(),
        objective: "rank:pairwise",
      })
    },
  ]
  for f in bad {
    let ok = try {
      ignore(f())
      true
    } catch {
      _ => false
    }
    assert_false(ok)
  }
}
