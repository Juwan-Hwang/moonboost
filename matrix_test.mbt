///|
test "csv with header and missing cells" {
  let m = @moonboost.DenseMatrix::from_csv(
    (
      #|a,b,c
      #|1,2,3
      #|4,,NA
      #|
    ),
    header=true,
  )
  inspect(m.rows, content="2")
  inspect(m.cols, content="3")
  assert_eq(m.row(0).to_owned(), [1.0, 2.0, 3.0])
  assert_true(m.row(1)[1].is_nan())
  assert_true(m.row(1)[2].is_nan())
}

///|
test "ragged csv is rejected" {
  let r = Ok(@moonboost.DenseMatrix::from_csv("1,2\n3\n")) catch { e => Err(e) }
  inspect(r is Err(InvalidModel(_)), content="true")
}

///|
test "batch prediction matches row prediction" {
  let model = @moonboost.Model::from_xgboost_json(fixture_multi_softprob.model)
  let x = fixture_multi_softprob.x.map(Float::reinterpret_from_uint)
  let m = @moonboost.DenseMatrix::new(
    x,
    cols=fixture_multi_softprob.num_feature,
  )
  let out = model.predict_batch(m)
  inspect(model.output_width(), content="3")
  assert_eq(out.length(), m.rows * 3)
  assert_eq(out[3:6].to_owned(), model.predict(m.row(1)))
}
