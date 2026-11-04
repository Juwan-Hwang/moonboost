///|
fn stump() -> @moonboost.Tree raise {
  // x0 < 0.5 ? -1.0 : 2.0, missing goes right
  @moonboost.Tree::new(
    left=[1, -1, -1],
    right=[2, -1, -1],
    feature=[0, 0, 0],
    value=[0.5, -1.0, 2.0],
    default_left=[false, false, false],
  )
}

///|
test "stump routes by threshold" {
  let t = stump()
  inspect(t.predict([0.0]), content="-1")
  inspect(t.predict([0.5]), content="2")
  inspect(t.predict([0.9]), content="2")
}

///|
test "missing values follow default direction" {
  let t = stump()
  let nan = @float.infinity - @float.infinity
  inspect(t.predict([nan]), content="2")
  inspect(t.predict([]), content="2")
}

///|
test "invalid children are rejected" {
  let r = Ok(
    @moonboost.Tree::new(
      left=[0, -1],
      right=[1, -1],
      feature=[0, 0],
      value=[0.0, 1.0],
      default_left=[false, false],
    ),
  ) catch {
    e => Err(e)
  }
  inspect(r is Err(_), content="true")
}

///|
test "max_feature ignores leaves" {
  let t = @moonboost.Tree::new(
    left=[1, -1, 3, -1, -1],
    right=[2, -1, 4, -1, -1],
    feature=[3, 99, 7, 99, 99],
    value=[0.0, 1.0, 0.0, 2.0, 3.0],
    default_left=[true, false, true, false, false],
  )
  inspect(t.max_feature(), content="7")
  inspect(t.num_nodes(), content="5")
}

///|
test "categorical split sends listed codes right" {
  // node 0 splits on categories {1, 3} of feature 0; missing goes left
  let t = @moonboost.Tree::new(
    left=[1, -1, -1],
    right=[2, -1, -1],
    feature=[0, 0, 0],
    value=[0.0, -1.0, 1.0],
    default_left=[true, false, false],
    categories=[Some([3, 1]), None, None],
  )
  inspect(t.predict([1.0]), content="1")
  inspect(t.predict([3.0]), content="1")
  inspect(t.predict([3.7]), content="1") // truncated to code 3
  inspect(t.predict([2.0]), content="-1")
  inspect(t.predict([-1.0]), content="-1") // invalid codes go left
  inspect(t.predict([1.0e10]), content="-1")
  inspect(t.predict([]), content="-1") // missing follows default_left
}
