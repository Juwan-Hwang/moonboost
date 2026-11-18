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

///|
/// Category codes 0 .. 39 in column 0 (missing every 11th row) and noise in
/// column 1; the label is 1 for categories 3, 17 and 29.
fn cat_toy(n : Int32) -> (@moonboost.DenseMatrix, Array[Float]) raise {
  let xs : Array[Float] = []
  let ys : Array[Float] = []
  let mut s = 777
  for i in 0..<n {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    let c = s % 40
    xs.push(if i % 11 == 0 { @float.not_a_number } else { Float::from_int(c) })
    xs.push(Float::from_int(s % 1000) / 1000.0)
    ys.push(
      if i % 11 != 0 && (c == 3 || c == 17 || c == 29) {
        1.0
      } else {
        0.0
      },
    )
  }
  (@moonboost.DenseMatrix::new(xs, cols=2), ys)
}

///|
test "categorical splits learn a set of categories" {
  let (x, y) = cat_toy(4000)
  for onehot in [4, 64] {
    let params = {
      ..@moonboost.TrainParams::default(),
      objective: "binary:logistic",
      num_round: 20,
      max_depth: 3,
      categorical: [0],
      max_cat_to_onehot: onehot,
    }
    let json = @moonboost.train_xgboost_json(x, y, params~)
    assert_true(json.contains("\"split_type\":[1"))
    let model = @moonboost.Model::from_xgboost_json(json)
    let p = model.predict_batch(x)
    for r in 0..<x.rows {
      assert_eq(p[r] > 0.5, y[r] > 0.5)
    }
  }
}

///|
test "loss-guided growth keeps to max_leaves" {
  let (x, y) = toy(3000)
  let params = {
    ..@moonboost.TrainParams::default(),
    num_round: 5,
    max_depth: 0,
    grow_policy: "lossguide",
    max_leaves: 7,
  }
  let json = @moonboost.train_xgboost_json(x, y, params~)
  let model = @moonboost.Model::from_xgboost_json(json)
  // a tree with 7 leaves has 13 nodes
  assert_true(json.contains("\"num_nodes\":\"13\""))
  assert_eq(model.predict_batch(x).length(), 3000)
}

///|
test "categorical columns must hold category codes" {
  let x = @moonboost.DenseMatrix::new([0.5, 1.0], cols=1)
  let params = { ..@moonboost.TrainParams::default(), categorical: [0], }
  let ok = @moonboost.train_xgboost_json(x, [0.0, 1.0], params~) catch {
    _ => "error"
  }
  assert_eq(ok, "error")
}

///|
test "row and feature sampling: seeded, and every tree still loads" {
  let (x, y) = toy(3000)
  let base = {
    ..@moonboost.TrainParams::default(),
    num_round: 8,
    max_depth: 4,
  }
  let plain = @moonboost.train_xgboost_json(x, y, params=base)
  let sampled = {
    ..base,
    subsample: 0.7,
    colsample_bytree: 0.8,
    colsample_bynode: 0.5,
    seed: 3,
  }
  let a = @moonboost.train_xgboost_json(x, y, params=sampled)
  let b = @moonboost.train_xgboost_json(x, y, params=sampled)
  let c = @moonboost.train_xgboost_json(x, y, params={ ..sampled, seed: 4, })
  assert_eq(a, b)
  assert_true(a != plain)
  assert_true(a != c)
  let model = @moonboost.Model::from_xgboost_json(a)
  assert_eq(model.predict_batch(x).length(), 3000)
}
