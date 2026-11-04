///|
/// Small deterministic generator so the test needs no dependencies.
fn lcg(state : Ref[UInt32]) -> UInt32 {
  state.val = state.val * 1664525U + 1013904223U
  state.val
}

///|
test "float_key preserves float32 order" {
  let specials : Array[Float] = [
    0.0, -0.0, 1.0, -1.0, 1.0e-45, -1.0e-45, 3.4028235e38, -3.4028235e38, @float.infinity,
    @float.neg_infinity,
  ]
  let s = Ref(12345U)
  let xs = specials.copy()
  for _ in 0..<2000 {
    // random bit patterns cover subnormals and every exponent
    let x = Float::reinterpret_from_uint(lcg(s))
    if !x.is_nan() {
      xs.push(x)
    }
  }
  for a in xs {
    for b in specials {
      assert_eq(a < b, @moonboost.float_key(a) < @moonboost.float_key(b))
      assert_eq(b < a, @moonboost.float_key(b) < @moonboost.float_key(a))
    }
  }
  for i in 1..<xs.length() {
    let a = xs[i - 1]
    let b = xs[i]
    assert_eq(a < b, @moonboost.float_key(a) < @moonboost.float_key(b))
  }
}

///|
test "category_key truncates like XGBoost" {
  inspect(@moonboost.category_key(3.0), content="3")
  inspect(@moonboost.category_key(3.9), content="3")
  inspect(@moonboost.category_key(-0.5), content="-1")
  inspect(@moonboost.category_key(3.0e9), content="-1")
}
