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

///|
test "min_data_in_leaf bounds the rows of every child" {
  // squared error: every row's hessian is 1, so the estimate is exact
  let (x, y) = toy(3000)
  let params = {
    ..@moonboost.TrainParams::default(),
    num_round: 4,
    max_depth: 4,
    min_data_in_leaf: 1400.0,
  }
  let json = @moonboost.train_xgboost_json(x, y, params~)
  // only a near-even root split leaves both children 1400 rows
  for n in ["5", "7", "9", "11", "13", "15"] {
    assert_false(json.contains("\"num_nodes\":\"\{n}\""))
  }
  assert_true(json.contains("\"num_nodes\":\"3\""))
}

///|
test "gradient_based sampling: seeded, unbiased enough to fit, method checked" {
  let (x, y) = toy(3000)
  let base = {
    ..@moonboost.TrainParams::default(),
    num_round: 20,
    max_depth: 4,
    subsample: 0.5,
  }
  let mvs = { ..base, sampling_method: "gradient_based", seed: 1, }
  let a = @moonboost.train_xgboost_json(x, y, params=mvs)
  let b = @moonboost.train_xgboost_json(x, y, params=mvs)
  let u = @moonboost.train_xgboost_json(x, y, params={
    ..mvs,
    sampling_method: "uniform",
  })
  assert_eq(a, b)
  assert_true(a != u)
  let mse = fn(json : String) -> Double raise {
    let p = @moonboost.Model::from_xgboost_json(json).predict_batch(x)
    let mut s = 0.0
    for i in 0..<y.length() {
      let d = p[i].to_double() - y[i].to_double()
      s += d * d
    }
    s / y.length().to_double()
  }
  let full = mse(
    @moonboost.train_xgboost_json(x, y, params={ ..base, subsample: 1.0, }),
  )
  assert_true(mse(a) < full * 1.5)
  let bad = @moonboost.train_xgboost_json(x, y, params={
    ..mvs,
    sampling_method: "goss",
  }) catch {
    _ => "error"
  }
  assert_eq(bad, "error")
}

///|
test "the other regression and binary objectives fit their targets" {
  let (x, y) = toy(3000)
  let objectives = [
    "count:poisson", "reg:gamma", "reg:tweedie", "reg:pseudohubererror", "reg:squaredlogerror",
    "reg:logistic", "binary:logitraw", "reg:absoluteerror",
  ]
  for o in objectives {
    // labels in each objective's domain, from y in about [-5, 5]
    let t = y.map(v => {
      let v = v.to_double()
      let e = @math.exp(v / 3.0)
      Float::from_double(
        match o {
          "count:poisson" | "reg:tweedie" => (e * 2.0).round()
          "reg:gamma" | "reg:squaredlogerror" => e
          "reg:logistic" => 1.0 / (1.0 + @math.exp(-v))
          "binary:logitraw" => if v > 0.0 { 1.0 } else { 0.0 }
          _ => v
        },
      )
    })
    let params = {
      ..@moonboost.TrainParams::default(),
      objective: o,
      num_round: 40,
      max_depth: 4,
      // a slope near the residuals' scale: with slope 1, residuals of 5
      // have hessians near 0 and Newton steps overshoot (as in XGBoost)
      huber_slope: 5.0,
    }
    let err = fn(rounds : Int32) -> Double raise {
      let json = @moonboost.train_xgboost_json(x, t, params={
        ..params,
        num_round: rounds,
      })
      let p = @moonboost.Model::from_xgboost_json(json).predict_batch(x)
      let mut s = 0.0
      for i in 0..<t.length() {
        let q = if o == "binary:logitraw" {
          1.0 / (1.0 + @math.exp(-p[i].to_double()))
        } else {
          p[i].to_double()
        }
        let d = q - t[i].to_double()
        s += d * d
      }
      s / t.length().to_double()
    }
    let fitted = err(40)
    let constant = err(0)
    assert_true(fitted < constant * 0.5, msg="\{o}: \{fitted} vs \{constant}")
  }
}

///|
test "reg:quantileerror predicts the asked quantile" {
  let (x, y) = toy(3000)
  for alpha in [0.1, 0.5, 0.9] {
    let json = @moonboost.train_xgboost_json(x, y, params={
      ..@moonboost.TrainParams::default(),
      objective: "reg:quantileerror",
      quantile_alpha: alpha,
      num_round: 60,
      max_depth: 4,
    })
    let p = @moonboost.Model::from_xgboost_json(json).predict_batch(x)
    let mut below = 0
    for i in 0..<y.length() {
      if y[i].to_double() <= p[i].to_double() {
        below += 1
      }
    }
    let share = below.to_double() / y.length().to_double()
    assert_true((share - alpha).abs() < 0.08, msg="\{alpha}: \{share}")
  }
  let bad = @moonboost.train_xgboost_json(x, y, params={
    ..@moonboost.TrainParams::default(),
    objective: "reg:quantileerror",
    quantile_alpha: 1.0,
  }) catch {
    _ => "error"
  }
  assert_eq(bad, "error")
}

///|
test "labels and settings outside an objective's domain are refused" {
  let (x, y) = toy(100)
  let cases = [
    ("count:poisson", -1.0, 1.5, 1.0),
    ("reg:gamma", 0.0, 1.5, 1.0),
    ("reg:squaredlogerror", -1.0, 1.5, 1.0),
    ("reg:logistic", 2.0, 1.5, 1.0),
    ("reg:tweedie", 1.0, 2.0, 1.0),
    ("reg:pseudohubererror", 1.0, 1.5, 0.0),
  ]
  for c in cases {
    let (o, label, power, slope) = c
    let t = y.map(_ => Float::from_double(label))
    let params = {
      ..@moonboost.TrainParams::default(),
      objective: o,
      num_round: 2,
      tweedie_variance_power: power,
      huber_slope: slope,
    }
    let r = @moonboost.train_xgboost_json(x, t, params~) catch { _ => "error" }
    assert_eq(r, "error")
  }
}

///|
/// Mean NDCG@10 over queries of `size` consecutive rows.
fn mean_ndcg(pred : Array[Float], rel : Array[Float], size : Int32) -> Double {
  let mut total = 0.0
  let nq = rel.length() / size
  for q in 0..<nq {
    let idx = Array::makei(size, i => q * size + i)
    let dcg_of = (order : Array[Int32]) => {
      let mut d = 0.0
      for i in 0..<10 {
        let g = @math.pow(2.0, rel[order[i]].to_double()) - 1.0
        d += g / (@math.ln(2.0 + i.to_double()) / @math.ln(2.0))
      }
      d
    }
    let by_pred = idx.copy()
    by_pred.sort_by((a, b) => pred[b].compare(pred[a]))
    let ideal = idx.copy()
    ideal.sort_by((a, b) => rel[b].compare(rel[a]))
    let best = dcg_of(ideal)
    total += if best > 0.0 { dcg_of(by_pred) / best } else { 1.0 }
  }
  total / nq.to_double()
}

///|
test "rank:ndcg and rank:pairwise order the documents of each query" {
  let (x, y) = toy(2000)
  // relevance 0 .. 3 from the target, in queries of 20 rows
  let rel = y.map(v => {
    Float::from_int(
      if v < -1.0 {
        0
      } else if v < 0.5 {
        1
      } else if v < 2.0 {
        2
      } else {
        3
      },
    )
  })
  let group = Array::make(100, 20)
  let flat = mean_ndcg(Array::make(2000, Float::from_double(0.0)), rel, 20)
  for o in ["rank:ndcg", "rank:pairwise"] {
    let json = @moonboost.train_xgboost_json(x, rel, params={
      ..@moonboost.TrainParams::default(),
      objective: o,
      group,
      num_round: 30,
      eta: 0.3,
      max_depth: 4,
    })
    let m = @moonboost.Model::from_xgboost_json(json)
    let score = mean_ndcg(m.predict_batch(x), rel, 20)
    assert_true(
      score > 0.9 && score > flat + 0.2,
      msg="\{o}: \{score} vs \{flat}",
    )
  }
  // the query sizes must cover the rows, and labels be small integers
  let bad = @moonboost.train_xgboost_json(x, rel, params={
    ..@moonboost.TrainParams::default(),
    objective: "rank:ndcg",
    group: [1000],
  }) catch {
    _ => "error"
  }
  assert_eq(bad, "error")
  let bad = @moonboost.train_xgboost_json(x, y, params={
    ..@moonboost.TrainParams::default(),
    objective: "rank:ndcg",
    group,
  }) catch {
    _ => "error"
  }
  assert_eq(bad, "error")
}

///|
test "TrainParams::from_json reads XGBoost names and aliases" {
  let p = @moonboost.TrainParams::from_json(
    (
      #|{"objective": "reg:quantileerror", "learning_rate": "0.05",
      #| "max_depth": 8, "n_estimators": 50, "reg_lambda": 2,
      #| "categorical": "0, 3", "quantile_alpha": 0.9, "group": [2, 3],
      #| "grow_policy": "lossguide", "max_leaves": 31}
    ),
  )
  assert_eq(p.objective, "reg:quantileerror")
  assert_eq(p.eta, 0.05)
  assert_eq(p.max_depth, 8)
  assert_eq(p.num_round, 50)
  assert_eq(p.lambda, 2.0)
  assert_eq(p.categorical, [0, 3])
  assert_eq(p.quantile_alpha, 0.9)
  assert_eq(p.group, [2, 3])
  assert_eq(p.grow_policy, "lossguide")
  assert_eq(p.max_leaves, 31)
  assert_eq(p.min_child_weight, 1.0)
  let d = @moonboost.TrainParams::from_json("{}")
  assert_eq(d.num_round, @moonboost.TrainParams::default().num_round)
  assert_true(
    try {
      ignore(@moonboost.TrainParams::from_json("{\"nope\": 1}"))
      false
    } catch {
      _ => true
    },
  )
  assert_true(
    try {
      ignore(@moonboost.TrainParams::from_json("{\"max_depth\": 2.5}"))
      false
    } catch {
      _ => true
    },
  )
  assert_true(
    try {
      ignore(@moonboost.TrainParams::from_json("[1]"))
      false
    } catch {
      _ => true
    },
  )
}

///|
test "TrainParams::from_json keeps unnamed settings of base" {
  let base = {
    ..@moonboost.TrainParams::default(),
    group: [2, 3],
    num_round: 7,
  }
  let p = @moonboost.TrainParams::from_json(
    "{\"objective\": \"rank:ndcg\", \"eta\": 0.2}",
    base~,
  )
  assert_eq(p.group, [2, 3])
  assert_eq(p.num_round, 7)
  assert_eq(p.eta, 0.2)
  assert_eq(p.objective, "rank:ndcg")
}

///|
test "binary:hinge trains a classifier that predicts 0 or 1" {
  let rows = 200
  let data = Array::makei(rows * 2, i => {
    let r = i / 2
    if i % 2 == 0 {
      Float::from_int(r % 20)
    } else {
      Float::from_int(r % 7)
    }
  })
  let label : Array[Float] = Array::makei(rows, r => {
    if r % 20 >= 10 {
      1.0
    } else {
      0.0
    }
  })
  let m = @moonboost.DenseMatrix::new(data, cols=2)
  let json = @moonboost.train_xgboost_json(m, label[:], params={
    ..@moonboost.TrainParams::default(),
    objective: "binary:hinge",
    num_round: 5,
  })
  let model = @moonboost.Model::from_xgboost_json(json)
  let mut right = 0
  for r in 0..<rows {
    let p = model.predict(m.row(r))[0]
    assert_true(p == 0.0 || p == 1.0)
    if p == label[r] {
      right += 1
    }
  }
  assert_eq(right, rows)
}
