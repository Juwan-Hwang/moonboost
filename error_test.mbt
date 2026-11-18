///|
test "ModelError prints its kind and message" {
  let e : @moonboost.ModelError = InvalidModel("missing learner")
  inspect(e, content="InvalidModel: missing learner")
  let u : @moonboost.ModelError = Unsupported("objective rank:ndcg")
  inspect("\{u}", content="Unsupported: objective rank:ndcg")
}

///|
test "a load error can be caught and printed" {
  let msg = try @moonboost.Model::from_xgboost_json("{}") catch {
    e => "\{e}"
  } noraise {
    _ => "loaded"
  }
  assert_true(
    msg.has_prefix("InvalidModel: ") || msg.has_prefix("Unsupported: "),
  )
}
