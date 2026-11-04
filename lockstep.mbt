///|
/// Rank of a missing value for splits whose default direction is right. Every
/// split threshold rank is smaller, so the comparison always goes right.
const RANK_HIGH : Int32 = 0xFFFE

///|
/// Threshold of a leaf. No rank reaches it, so a leaf always steps to its
/// own "left" child, which is itself.
const LEAF_THR : Int32 = 0xFFFF

///|
/// Branch-free, rank-encoded form of a forest of numerical trees.
///
/// Three changes make every step of a tree walk the same few instructions:
///
/// - **Ranks instead of floats.** For each feature, the distinct split
///   thresholds used anywhere in the model are sorted once. A row value is
///   replaced by its rank, the number of thresholds `<=` it, and a split on
///   the `k`-th threshold becomes `rank >= k + 1`. This is exact: `x < t_k`
///   holds exactly when fewer than `k + 1` thresholds are `<= x`.
/// - **Missing values folded into the feature index.** Each feature gets two
///   rank slots: slot `2f` holds 0 for a missing value and slot `2f + 1`
///   holds `RANK_HIGH`. A split that sends missing values left reads slot
///   `2f`, one that sends them right reads `2f + 1`. No test for missing.
/// - **Self-looping leaves.** Children are stored side by side, so the next
///   node is `left + go_right`. A leaf is its own left child with a threshold
///   no rank reaches, so a row that arrives early just stays put. Every row
///   then takes exactly `depth` steps through a tree, which lets a block of
///   rows walk in lockstep with independent, branch-free updates.
///
/// Leaves accumulate in tree order in float32, exactly as before.
priv struct Lockstep {
  // per node: left child and (rank slot << 16) | threshold; see `Nodes`
  nodes : Nodes
  leaf_value : FixedArray[Float]
  roots : FixedArray[Int32]
  depth : FixedArray[Int32]
  // sorted distinct threshold keys of each feature, padded with `Int32::max`
  // to a power of two, so a rank is a fixed number of branch-free halvings
  search : FixedArray[Int32]
  search_start : FixedArray[Int32]
  search_half : FixedArray[Int32]
  num_feature : Int32
}

///|
/// Build the lockstep form, or `None` when the model has a categorical split
/// or a feature with too many distinct thresholds for 16-bit ranks.
fn Lockstep::build(trees : ArrayView[Tree], num_feature : Int32) -> Lockstep? {
  // distinct threshold keys per feature
  let per_feature : Array[Array[Int32]] = Array::makei(num_feature, _ => [])
  for t in trees {
    for i in 0..<t.num_nodes() {
      if t.left[i] == -1 {
        continue
      }
      if t.categories[i] is Some(_) {
        return None
      }
      per_feature[t.feature[i]].push(float_key(t.value[i]))
    }
  }
  let cut_start = FixedArray::make(num_feature + 1, 0)
  let cuts : Array[Int32] = []
  for f in 0..<num_feature {
    let keys = per_feature[f]
    keys.sort()
    keys.dedup()
    if keys.length() >= RANK_HIGH {
      return None
    }
    cuts.push_iter(keys.iter())
    cut_start[f + 1] = cuts.length()
  }
  let cuts = FixedArray::from_array(cuts)
  let search : Array[Int32] = []
  let search_start = FixedArray::make(num_feature, 0)
  let search_half = FixedArray::make(num_feature, 0)
  for f in 0..<num_feature {
    let c = cut_start[f + 1] - cut_start[f]
    let mut p = 1
    while p < c + 1 {
      p *= 2
    }
    search_start[f] = search.length()
    search_half[f] = p / 2
    for i in 0..<p {
      search.push(if i < c { cuts[cut_start[f] + i] } else { 0x7FFFFFFF })
    }
  }
  let search = FixedArray::from_array(search)
  let total = trees.fold(init=0, (acc, t) => acc + t.num_nodes())
  let nodes = Nodes::make(total)
  let leaf_value = FixedArray::make(total, (0.0 : Float))
  let roots = FixedArray::make(trees.length(), 0)
  let depth = FixedArray::make(trees.length(), 0)
  let mut base = 0
  let queue : Array[(Int32, Int32, Int32)] = [] // (old id, new id, depth)
  for ti, t in trees {
    roots[ti] = base
    // breadth-first renumbering, so the two children of a node are adjacent
    queue.clear()
    queue.push((0, base, 0))
    let mut next = base + 1
    let mut head = 0
    let mut d_max = 0
    while head < queue.length() {
      let (old, id, d) = queue[head]
      head += 1
      if d > d_max {
        d_max = d
      }
      if t.left[old] == -1 {
        nodes.set(id, id, LEAF_THR)
        leaf_value[id] = t.value[old]
        continue
      }
      let f = t.feature[old]
      let slot = 2 * f + (if t.default_left[old] { 0 } else { 1 })
      let k = lower_bound(
        cuts,
        cut_start[f],
        cut_start[f + 1],
        float_key(t.value[old]),
      )
      nodes.set(id, next, (slot << 16) | (k - cut_start[f] + 1))
      queue.push((t.left[old], next, d + 1))
      queue.push((t.right[old], next + 1, d + 1))
      next += 2
    }
    depth[ti] = d_max
    base = next
  }
  if 2 * num_feature >= 0x8000 {
    return None
  }
  Some({
    nodes,
    leaf_value,
    roots,
    depth,
    search,
    search_start,
    search_half,
    num_feature,
  })
}

///|
/// First index in `a[lo:hi]` whose value is `>= x`.
fn lower_bound(
  a : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
  x : Int32,
) -> Int32 {
  for i = lo, j = hi; i < j; {
    let mid = i + (j - i) / 2
    if a[mid] < x {
      continue mid + 1, j
    } else {
      continue i, mid
    }
  } nobreak {
    i
  }
}

///|
/// Write the two rank slots of every feature of `row` to
/// `out[at : at + 2 * num_feature]`.
fn Lockstep::ranks(
  self : Lockstep,
  row : ArrayView[Float],
  out : FixedArray[Int32],
  at : Int32,
) -> Unit {
  for f in 0..<self.num_feature {
    let x = if f < row.length() {
      row[f]
    } else {
      Float::from_double(0.0) / 0.0
    }
    if x.is_nan() {
      out[at + 2 * f] = 0
      out[at + 2 * f + 1] = RANK_HIGH
    } else {
      // number of thresholds <= x
      let key = float_key(x)
      let s = self.search
      let lo = self.search_start[f]
      let mut b = lo
      let mut step = self.search_half[f]
      while step > 0 {
        // in bounds: b + step - 1 < lo + 2 * half - 1, the padded length
        b = b + (if s.unsafe_get(b + step - 1) <= key { step } else { 0 })
        step = step / 2
      }
      let r = b - lo
      out[at + 2 * f] = r
      out[at + 2 * f + 1] = r
    }
  }
}

///|
/// Add the leaves of every tree to `margin` for `n` rows whose rank slots are
/// stored consecutively in `ranks`. `margin` holds `groups` values per row.
fn Lockstep::accumulate(
  self : Lockstep,
  tree_group : Array[Int32],
  groups : Int32,
  ranks : FixedArray[Int32],
  n : Int32,
  idx : FixedArray[Int32],
  margin : FixedArray[Float],
) -> Unit {
  let width = 2 * self.num_feature
  let nodes = self.nodes
  for t in 0..<self.roots.length() {
    let root = self.roots[t]
    for r in 0..<n {
      idx[r] = root
    }
    for _ in 0..<self.depth[t] {
      for r in 0..<n {
        // In bounds by construction: `build` stores only node ids of this
        // forest and slots below `width`, and `ranks` holds `n` rows, so
        // `step` reads without bounds checks.
        let i = idx.unsafe_get(r)
        idx.unsafe_set(r, nodes.step(i, ranks, r * width))
      }
    }
    let g = tree_group[t]
    for r in 0..<n {
      let k = r * groups + g
      margin[k] = margin[k] + self.leaf_value[idx[r]]
    }
  }
}
