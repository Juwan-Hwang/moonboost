///|
test "objective names round-trip" {
  for
    name in [
      "reg:squarederror", "binary:logistic", "binary:logitraw", "count:poisson",
      "reg:gamma",
    ] {
    assert_eq(@moonboost.Objective::from_name(name).name(), name)
  }
  assert_eq(
    @moonboost.Objective::from_name("multi:softprob", num_class=3),
    Softprob(3),
  )
}

///|
test "unknown objective is unsupported" {
  let r = Ok(@moonboost.Objective::from_name("foo:bar")) catch { e => Err(e) }
  inspect(r is Err(Unsupported(_)), content="true")
}

///|
test "logistic base margin is the logit" {
  let o = @moonboost.Objective::from_name("binary:logistic")
  inspect(o.base_margin(0.5), content="0")
  let m : Array[Float] = [0.0, 2.0]
  o.transform(m)
  inspect(m[0], content="0.5")
}

///|
test "softprob sums to one and softmax picks argmax" {
  let m : Array[Float] = [1.0, 2.0, 3.0]
  @moonboost.Objective::Softprob(3).transform(m)
  let s = m[0] + m[1] + m[2]
  assert_true((s - 1.0).abs() < 1.0e-6)
  let k : Array[Float] = [0.1, 5.0, -2.0]
  @moonboost.Objective::Softmax(3).transform(k)
  assert_eq(k, [1.0])
}
