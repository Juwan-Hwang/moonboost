///|
/// Key of a missing feature value. Every real key is larger.
pub const MISSING : Int32 = -2147483648

///|
/// The tree-walking core of moonboost, kept small enough to be verified by
/// `moon prove`.
///
/// A `Forest` holds every tree of a model in one flat array. Feature values
/// and numerical thresholds are passed as integer keys whose order matches
/// float32 order (see `@moonboost.float_key`), so the kernel needs no
/// floating-point arithmetic. Proven properties:
///
/// - `Forest::new` only returns forests satisfying `forest_inv`;
/// - for such a forest, `Forest::leaf` terminates, never indexes out of
///   bounds, and returns a leaf node.
struct Forest {
  n : Int32
  nodes : FixedArray[Int32]
  roots : FixedArray[Int32]
  cat_ranges : FixedArray[Int32]
  cats : FixedArray[Int32]
}

///|
fn check_node(
  f : Forest,
  i : Int32,
) -> Bool where {
  proof_require: 0 <= f.n && f.nodes.length() == 4 * f.n,
  proof_require: 0 <= i && i < f.n,
  proof_ensure: result => result → node_ok(f, i),
} {
  let n = f.n
  let l = f.nodes[4 * i]
  let r = f.nodes[4 * i + 1]
  let info = f.nodes[4 * i + 2]
  let k = f.nodes[4 * i + 3]
  if l == -1 {
    true
  } else if i < l && l < n && i < r && r < n && 0 <= info {
    if info % 4 < 2 {
      true
    } else {
      0 <= k &&
      2 * k + 1 < f.cat_ranges.length() &&
      0 <= f.cat_ranges[2 * k] &&
      f.cat_ranges[2 * k] <= f.cat_ranges[2 * k + 1] &&
      f.cat_ranges[2 * k + 1] <= f.cats.length()
    }
  } else {
    false
  }
}

///|
fn check_nodes(
  f : Forest,
) -> Bool where {
  proof_require: 0 <= f.n && f.nodes.length() == 4 * f.n,
  proof_ensure: result => {
    result → (∀ i : Int32, 0 <= i && i < n_nodes(f) → node_ok(f, i))
  },
} {
  let n = f.n
  for i = 0; i < n; {
    if !check_node(f, i) {
      break false
    }
    continue i + 1
  } nobreak {
    true
  } where {
    proof_invariant: 0 <= i && i <= n,
    proof_invariant: ∀ k : Int32, 0 <= k && k < i → node_ok(f, k),
  }
}

///|
fn check_roots(
  f : Forest,
) -> Bool where {
  proof_require: 0 <= f.n && f.nodes.length() == 4 * f.n,
  proof_ensure: result => {
    result → (∀ t : Int32, 0 <= t && t < f.roots.length() → root_ok(f, t))
  },
} {
  let n = f.n
  for t = 0; t < f.roots.length(); {
    if f.roots[t] < 0 || f.roots[t] >= n {
      break false
    }
    continue t + 1
  } nobreak {
    true
  } where {
    proof_invariant: 0 <= t && t <= f.roots.length(),
    proof_invariant: ∀ k : Int32, 0 <= k && k < t → root_ok(f, k),
  }
}

///|
fn validate(
  f : Forest,
) -> Bool where {
  proof_ensure: result => result → forest_inv(f),
} {
  if f.n < 0 || f.nodes.length() != 4 * f.n {
    false
  } else {
    check_nodes(f) && check_roots(f)
  }
}

///|
/// Build a forest, or return `None` if the arrays do not describe a valid
/// one. The arrays are owned by the forest afterwards and must not be
/// modified.
///
/// - `nodes`: 4 ints per node, `[left, right, info, arg]`. A leaf has
///   `left == -1`. Otherwise `info = feature * 4 + 2 * categorical +
///   default_left`, and `arg` is the threshold key of a numerical split or,
///   for a categorical split, an index `k` such that the codes going right
///   are `cats[cat_ranges[2k] : cat_ranges[2k+1]]` in ascending order.
/// - `roots`: root node of each tree.
pub fn Forest::new(
  nodes : FixedArray[Int32],
  roots : FixedArray[Int32],
  cat_ranges : FixedArray[Int32],
  cats : FixedArray[Int32],
) -> Forest? {
  let f = { n: nodes.length() / 4, nodes, roots, cat_ranges, cats, }
  if validate(f) {
    Some(f)
  } else {
    None
  }
}

///|
/// Number of trees.
pub fn Forest::num_trees(self : Forest) -> Int32 {
  self.roots.length()
}

///|
/// Number of nodes across all trees.
pub fn Forest::num_nodes(self : Forest) -> Int32 {
  self.n
}

///|
fn contains(
  cats : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
  x : Int32,
) -> Bool where {
  proof_require: 0 <= lo && lo <= hi && hi <= cats.length(),
} {
  for i = lo, j = hi; i < j; {
    let mid = i + (j - i) / 2
    let c = cats[mid]
    if c == x {
      break true
    } else if c < x {
      continue mid + 1, j
    } else {
      continue i, mid
    }
  } nobreak {
    false
  } where {
    proof_invariant: lo <= i && i <= j && j <= hi,
  }
}

///|
fn go_left(
  f : Forest,
  i : Int32,
  v : Int32,
) -> Bool where {
  proof_require: forest_inv(f),
  proof_require: 0 <= i && i < n_nodes(f),
  proof_require: left_of(f, i) != -1,
} {
  proof_assert node_ok(f, i)
  let info = f.nodes[4 * i + 2]
  let arg = f.nodes[4 * i + 3]
  if v == MISSING {
    info % 2 == 1
  } else if info % 4 < 2 {
    v < arg
  } else {
    v < 0 ||
    !contains(f.cats, f.cat_ranges[2 * arg], f.cat_ranges[2 * arg + 1], v)
  }
}

///|
fn walk(
  f : Forest,
  row : FixedArray[Int32],
  i : Int32,
) -> Int32 where {
  proof_require: forest_inv(f),
  proof_require: 0 <= i && i < n_nodes(f),
  proof_ensure: result => is_leaf(f, result),
  proof_decrease: remaining(f, i),
} {
  proof_assert node_ok(f, i)
  let l = f.nodes[4 * i]
  if l == -1 {
    i
  } else {
    proof_assert i < l && l < f.n
    proof_assert i < f.nodes[4 * i + 1] && f.nodes[4 * i + 1] < f.n
    let r = f.nodes[4 * i + 1]
    let info = f.nodes[4 * i + 2]
    proof_assert 0 <= info
    let feat = info / 4
    proof_assert 0 <= feat
    let v = if feat < row.length() { row[feat] } else { MISSING }
    let next = if go_left(f, i, v) { l } else { r }
    proof_assert i < next && next < f.n
    walk(f, row, next)
  }
}

///|
/// Leaf node reached by `row` in tree `t`, or -1 if `t` is out of range.
/// `row[j]` is the key of feature `j`; features past the end are missing.
pub fn Forest::leaf(
  self : Forest,
  t : Int32,
  row : FixedArray[Int32],
) -> Int32 where {
  proof_require: forest_inv(self),
  proof_ensure: result => result == -1 || is_leaf(self, result),
} {
  if t < 0 || t >= self.roots.length() {
    -1
  } else {
    walk(self, row, self.roots[t])
  }
}
