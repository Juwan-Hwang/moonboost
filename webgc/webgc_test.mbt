///|
test "trains the same model as train_xgboost_json" {
  let rows = 200
  let cols = 3
  let data = FixedArray::makei(rows * cols, i => {
    let r = i / cols
    let c = i % cols
    if (r * 7 + c) % 13 == 0 {
      Float::from_double(@double.not_a_number)
    } else {
      Float::from_int(r * (c + 3) % 17)
    }
  })
  let labels : Array[Float] = Array::makei(rows, r => {
    if r * 5 % 9 < 4 {
      1.0
    } else {
      0.0
    }
  })
  @webgc.train_alloc(rows, cols)
  for i in 0..<(rows * cols) {
    @webgc.train_set(i, data[i].to_double())
  }
  for r in 0..<rows {
    @webgc.train_label(r, labels[r].to_double())
  }
  let n = @webgc.train_run(1, -1, 5, 3, 0.3)
  let text = String::from_array(
    Array::makei(n, i => @webgc.train_char(i).unsafe_to_char()),
  )
  let expected = @moonboost.train_xgboost_json(
    @moonboost.DenseMatrix::from_fixed(data, cols~),
    labels,
    params={
      ..@moonboost.TrainParams::default(),
      objective: "binary:logistic",
      num_round: 5,
      max_depth: 3,
      eta: 0.3,
    },
  )
  assert_eq(text, expected)
  // a bad objective setting reports through the error text
  @webgc.train_alloc(2, 1)
  @webgc.train_label(0, 3.0)
  assert_eq(@webgc.train_run(1, -1, 1, 1, 0.3), -1)
  assert_true(@webgc.train_error_char(-1) > 0)
}

///|
test "trains with parameters given as JSON" {
  let rows = 120
  let cols = 2
  let data = FixedArray::makei(rows * cols, i => Float::from_int(i * 7 % 23))
  let labels : Array[Float] = Array::makei(rows, r => Float::from_int(r % 3))
  @webgc.train_alloc(rows, cols)
  for i in 0..<(rows * cols) {
    @webgc.train_set(i, data[i].to_double())
  }
  for r in 0..<rows {
    @webgc.train_label(r, labels[r].to_double())
  }
  let json = "{\"objective\": \"rank:ndcg\", \"group\": [40, 50, 30], \"learning_rate\": 0.2, \"n_estimators\": 4, \"grow_policy\": \"lossguide\", \"max_leaves\": 7}"
  @webgc.train_params_alloc(json.length())
  for i in 0..<json.length() {
    @webgc.train_params_char(i, json.code_unit_at(i).to_int())
  }
  let n = @webgc.train_run_params()
  assert_true(n > 0)
  let text = String::from_array(
    Array::makei(n, i => @webgc.train_char(i).unsafe_to_char()),
  )
  let expected = @moonboost.train_xgboost_json(
    @moonboost.DenseMatrix::from_fixed(data, cols~),
    labels,
    params=@moonboost.TrainParams::from_json(json),
  )
  assert_eq(text, expected)
  let bad = "{\"no_such\": 1}"
  @webgc.train_params_alloc(bad.length())
  for i in 0..<bad.length() {
    @webgc.train_params_char(i, bad.code_unit_at(i).to_int())
  }
  assert_eq(@webgc.train_run_params(), -1)
}
