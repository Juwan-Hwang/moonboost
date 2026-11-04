///|
/// Integer key of a non-NaN float32 whose order matches float order:
/// `x < y` if and only if `float_key(x) < float_key(y)`. Both zeros map to 0.
/// Keys of finite and infinite floats are all greater than `@kernel.MISSING`.
pub fn float_key(x : Float) -> Int32 {
  let b = x.reinterpret_as_int()
  if b >= 0 {
    b
  } else {
    -(b & 0x7FFFFFFF)
  }
}

///|
/// Key of a categorical feature value, following XGBoost: the value is
/// truncated to an integer code, and negative or too-large codes become -1,
/// which never matches a category and so always goes left.
pub fn category_key(x : Float) -> Int32 {
  if x < 0.0 || x >= 2147483648.0 {
    -1
  } else {
    x.to_int()
  }
}

///|
/// Every tree of a model packed into one verified `@kernel.Forest`, plus the
/// leaf weights indexed by global node id.
struct Compiled {
  forest : @kernel.Forest
  offsets : FixedArray[Int32]
  leaf_value : FixedArray[Float]
  // categorical[j] is true when feature j is used by a categorical split
  categorical : FixedArray[Bool]
}

///|
fn compile(
  trees : ArrayView[Tree],
  num_feature : Int32,
) -> Compiled raise ModelError {
  let total = trees.fold(init=0, (acc, t) => acc + t.num_nodes())
  let nodes = FixedArray::make(4 * total, 0)
  let leaf_value = FixedArray::make(total, (0.0 : Float))
  let offsets = FixedArray::make(trees.length(), 0)
  let categorical = FixedArray::make(num_feature, false)
  let cat_ranges = []
  let cats = []
  let mut o = 0
  for ti, t in trees {
    offsets[ti] = o
    for i in 0..<t.num_nodes() {
      let g = o + i
      leaf_value[g] = t.value[i]
      if t.left[i] == -1 {
        nodes[4 * g] = -1
        nodes[4 * g + 1] = -1
        continue
      }
      let f = t.feature[i]
      if f >= num_feature || f >= 0x20000000 {
        raise InvalidModel("node \{i} splits on feature \{f} out of range")
      }
      nodes[4 * g] = t.left[i] + o
      nodes[4 * g + 1] = t.right[i] + o
      let dl = if t.default_left[i] { 1 } else { 0 }
      match t.categories[i] {
        None => {
          nodes[4 * g + 2] = f * 4 + dl
          nodes[4 * g + 3] = float_key(t.value[i])
        }
        Some(set) => {
          categorical[f] = true
          nodes[4 * g + 2] = f * 4 + 2 + dl
          nodes[4 * g + 3] = cat_ranges.length() / 2
          cat_ranges.push(cats.length())
          cats.push_iter(set.iter())
          cat_ranges.push(cats.length())
        }
      }
    }
    o += t.num_nodes()
  }
  let roots = FixedArray::makei(trees.length(), i => offsets[i])
  let forest = @kernel.Forest::new(
    nodes,
    roots,
    FixedArray::from_array(cat_ranges),
    FixedArray::from_array(cats),
  )
  match forest {
    Some(forest) => { forest, offsets, leaf_value, categorical, }
    None => raise InvalidModel("trees do not form a valid forest")
  }
}

///|
/// Fill `out[j]` with the key of `row[j]`.
fn Compiled::encode(
  self : Compiled,
  row : ArrayView[Float],
  out : FixedArray[Int32],
) -> Unit {
  for j in 0..<row.length() {
    let x = row[j]
    out[j] = if x.is_nan() {
      @kernel.MISSING
    } else if j < self.categorical.length() && self.categorical[j] {
      category_key(x)
    } else {
      float_key(x)
    }
  }
}

///|
fn Compiled::keys(self : Compiled, row : ArrayView[Float]) -> FixedArray[Int32] {
  let out = FixedArray::make(row.length(), 0)
  self.encode(row, out)
  out
}

///|
/// Global leaf node reached by `keys` in tree `t`.
fn Compiled::leaf(
  self : Compiled,
  t : Int32,
  keys : FixedArray[Int32],
) -> Int32 {
  self.forest.leaf(t, keys)
}

///|
fn empty_compiled() -> Compiled {
  {
    forest: @kernel.Forest::new([-1, -1, 0, 0], [0], [], []).unwrap(),
    offsets: [0],
    leaf_value: [0.0],
    categorical: [],
  }
}
