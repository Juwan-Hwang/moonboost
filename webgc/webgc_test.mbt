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
