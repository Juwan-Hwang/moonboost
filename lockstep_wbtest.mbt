///|
/// Margin through the verified kernel, ignoring the lockstep engine.
fn kernel_margin(m : Model, row : ArrayView[Float]) -> Array[Float] {
  let out = Array::make(m.num_group, m.base_margin)
  let keys = m.compiled.keys(row)
  for t in 0..<m.tree_group.length() {
    let g = m.tree_group[t]
    out[g] = out[g] + m.compiled.leaf_value[m.compiled.leaf(t, keys)]
  }
  out
}

///|
/// Tiny deterministic generator so the test needs no extra package.
fn next_rand(state : Ref[UInt64]) -> Int32 {
  state.val = state.val * 6364136223846793005UL + 1442695040888963407UL
  (state.val >> 33).to_int()
}

///|
test "lockstep agrees with the verified kernel on boundary values" {
  let nan = Float::from_double(0.0) / 0.0
  let inf = Float::from_double(1.0) / 0.0
  let thresholds : Array[Float] = [-1.5, 0.0, 0.25, 3.0e38, -0.0]
  let seed = Ref(42UL)
  // a few random trees of depth up to 5 over 4 features
  let trees : Array[Tree] = []
  for _ in 0..<20 {
    let left : Array[Int32] = []
    let right : Array[Int32] = []
    let feature : Array[Int32] = []
    let value : Array[Float] = []
    let default_left : Array[Bool] = []
    fn add(d : Int32) -> Int32 {
      let id = left.length()
      left.push(-1)
      right.push(-1)
      feature.push(0)
      value.push(0.0)
      default_left.push(false)
      if d < 5 && next_rand(seed) % 4 != 0 {
        feature[id] = next_rand(seed) % 4
        value[id] = thresholds[next_rand(seed) % thresholds.length()]
        default_left[id] = next_rand(seed) % 2 == 0
        let l = add(d + 1)
        let r = add(d + 1)
        left[id] = l
        right[id] = r
      } else {
        value[id] = Float::from_int(next_rand(seed) % 1000) / 7.0
      }
      id
    }
    let _ = add(0)
    trees.push(Tree::new(left~, right~, feature~, value~, default_left~))
  }
  let m = Model::new(
    trees~,
    tree_group=Array::make(20, 0),
    num_group=1,
    num_feature=4,
    base_score=0.5,
    objective=Identity("reg:squarederror"),
  )
  inspect(m.fast is Some(_), content="true")
  let pool : Array[Float] = [
    nan,
    inf,
    -inf,
    -1.5,
    -1.4999999,
    -1.5000001,
    0.0,
    -0.0,
    1.0e-45,
    -1.0e-45,
    0.25,
    0.24999999,
    3.0e38,
    3.4028235e38,
  ]
  for _ in 0..<2000 {
    let row : Array[Float] = Array::makei(4, _ => {
      pool[next_rand(seed) % pool.length()]
    })
    let a = m.predict_margin(row)
    let b = kernel_margin(m, row)
    assert_eq(a[0].reinterpret_as_int(), b[0].reinterpret_as_int())
  }
  // short rows: trailing features are missing
  let a = m.predict_margin([0.25])
  let b = kernel_margin(m, [0.25])
  assert_eq(a[0].reinterpret_as_int(), b[0].reinterpret_as_int())
}

///|
test "rank tables count thresholds exactly" {
  let inf = Float::from_double(1.0) / 0.0
  let nan = Float::from_double(0.0) / 0.0
  let per_feature : Array[Array[Float]] = [
    [-inf, -1.0, -0.0, 1.0, inf],
    [1.0e-30, 1.0e-29, 1.0e30],
    [5.0],
    [1.0, Float::reinterpret_from_int((1.0 : Float).reinterpret_as_int() + 1)],
    [],
    // crowded near zero with one far value: falls back to binary search
    Array::makei(40, i => Float::from_int(i) * 1.0e-6) + [1.0e20],
    [Float::reinterpret_from_int(1), 0.0, -Float::reinterpret_from_int(1)],
  ]
  let nf = per_feature.length()
  let trees : Array[Tree] = []
  for f, ts in per_feature {
    for t in ts {
      trees.push(
        Tree::new(
          left=[1, -1, -1],
          right=[2, -1, -1],
          feature=[f, 0, 0],
          value=[t, 0.0, 1.0],
          default_left=[true, false, false],
        ),
      )
    }
  }
  let ls = Lockstep::build(trees[:], nf).unwrap()
  // probe values: every threshold, its float neighbours, and specials
  let probes : Array[Float] = [nan, inf, -inf, 0.0, -0.0, 2.0, -2.0, 1.0e10]
  for ts in per_feature {
    for t in ts {
      let b = t.reinterpret_as_int()
      probes.push(t)
      probes.push(Float::reinterpret_from_int(b + 1))
      probes.push(Float::reinterpret_from_int(b - 1))
    }
  }
  let data : Array[Float] = []
  for p in probes {
    for _ in 0..<nf {
      data.push(p)
    }
  }
  let n = probes.length()
  let out = Buf::new(n * 2 * nf)
  ls.ranks_block(data, nf, 0, n, out)
  for r, p in probes {
    for f, ts in per_feature {
      let lo = out.get(r * 2 * nf + 2 * f)
      let hi = out.get(r * 2 * nf + 2 * f + 1)
      if p.is_nan() {
        assert_eq(lo, 0)
        assert_eq(hi, RANK_HIGH)
      } else {
        let mut want = 0
        let keys : Array[Int32] = ts.map(float_key)
        keys.sort()
        keys.dedup()
        for k in keys {
          if k <= float_key(p) {
            want += 1
          }
        }
        assert_eq(lo, want)
        assert_eq(hi, want)
      }
    }
  }
}
