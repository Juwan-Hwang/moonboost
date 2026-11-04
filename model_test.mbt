///|
fn leaf(v : Float) -> @moonboost.Tree raise @moonboost.ModelError {
  @moonboost.Tree::new(left=[-1], right=[-1], feature=[0], value=[v], default_left=[
    false,
  ])
}

///|
test "margins accumulate per group on top of base score" {
  let m = @moonboost.Model::new(
    trees=[leaf(1.0), leaf(2.0), leaf(0.25)],
    tree_group=[0, 1, 0],
    num_group=2,
    num_feature=1,
    base_score=0.5,
    objective=Identity("reg:squarederror"),
  )
  assert_eq(m.predict_margin([0.0]), [1.75, 2.5])
  assert_eq(m.predict_leaf([0.0]), [0, 0, 0])
}

///|
test "logistic base score is applied in margin space" {
  let m = @moonboost.Model::new(
    trees=[leaf(0.0)],
    tree_group=[0],
    num_group=1,
    num_feature=1,
    base_score=0.5,
    objective=Logistic("binary:logistic"),
  )
  assert_eq(m.predict_one([]), 0.5)
}

///|
test "groups out of range are rejected" {
  let r = Ok(
    @moonboost.Model::new(
      trees=[leaf(1.0)],
      tree_group=[3],
      num_group=2,
      num_feature=1,
      base_score=0.0,
      objective=Identity("reg:squarederror"),
    ),
  ) catch {
    e => Err(e)
  }
  inspect(r is Err(InvalidModel(_)), content="true")
}
