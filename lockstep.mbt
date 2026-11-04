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
  // per node: [left child, (rank slot << 16) | threshold]
  nodes : FixedArray[Int32]
  leaf_value : FixedArray[Float]
  roots : FixedArray[Int32]
  depth : FixedArray[Int32]
  // sorted distinct threshold keys, feature f at cuts[cut_start[f]:cut_start[f+1]]
  cuts : FixedArray[Int32]
  cut_start : FixedArray[Int32]
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
  let total = trees.fold(init=0, (acc, t) => acc + t.num_nodes())
  let nodes = FixedArray::make(2 * total, 0)
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
        nodes[2 * id] = id
        nodes[2 * id + 1] = LEAF_THR
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
      nodes[2 * id] = next
      nodes[2 * id + 1] = (slot << 16) | (k - cut_start[f] + 1)
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
  Some({ nodes, leaf_value, roots, depth, cuts, cut_start, num_feature, })
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
      let lo = self.cut_start[f]
      let r = for i = lo, j = self.cut_start[f + 1]; i < j; {
        let mid = i + (j - i) / 2
        if self.cuts[mid] <= key {
          continue mid + 1, j
        } else {
          continue i, mid
        }
      } nobreak {
        i - lo
      }
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
        let i = idx[r]
        let w = nodes[2 * i + 1]
        let go = if ranks[r * width + (w >> 16)] >= (w & 0xFFFF) {
          1
        } else {
          0
        }
        idx[r] = nodes[2 * i] + go
      }
    }
    let g = tree_group[t]
    for r in 0..<n {
      let k = r * groups + g
      margin[k] = margin[k] + self.leaf_value[idx[r]]
    }
  }
}
