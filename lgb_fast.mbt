///|
/// Rank of a value that every split sends right.
const LGB_HIGH : Int32 = 0xFFFE

///|
/// Threshold rank of a self-looping leaf: no rank exceeds it.
const LGB_LEAF : Int32 = 0xFFFF

///|
/// Rank slots per feature. LightGBM decides a numerical split by the node's
/// missing type and default direction as well as the threshold, so each
/// feature gets one rank per combination a split can ask for:
///
/// - 0: missing type NaN, default left: NaN reads 0
/// - 1: missing type NaN, default right: NaN reads `LGB_HIGH`
/// - 2: missing type None: NaN reads as 0.0
/// - 3: missing type Zero, default left: NaN and zero read 0
/// - 4: missing type Zero, default right: NaN and zero read `LGB_HIGH`
const LGB_SLOTS : Int32 = 5

///|
/// Branch-free form of a numerical LightGBM forest, the double-precision
/// counterpart of `Lockstep`.
///
/// A split on threshold `t` sends `v` left when `v <= t`. With a feature's
/// distinct thresholds sorted as `t_0 < t_1 < ...`, write `rank(v)` for the
/// number of thresholds below `v`; then `v <= t_k` exactly when
/// `rank(v) <= k`, so a split on `t_k` goes right when `rank > k`. A row's
/// ranks are computed once, by comparing doubles as LightGBM does, and every
/// tree step is then an integer compare. Children sit side by side
/// (`next = left + go_right`) and a leaf loops to itself, so each row takes
/// the same number of steps and several rows or trees walk at once. Leaves
/// are added in tree order in double precision, as LightGBM adds them.
priv struct LgbFast {
  left : FixedArray[Int32]
  slot : FixedArray[Int32]
  // each node as `(left << 32) | (slot << 16) | rank`, one load per step
  node : FixedArray[Int64]
  rank : FixedArray[Int32]
  // threshold of each split (+inf at leaves), for walking by value
  thr : FixedArray[Double]
  // the slots in use: feature and variant (see `LGB_SLOTS`) of each
  slot_feature : FixedArray[Int32]
  slot_variant : FixedArray[Int32]
  // walk by comparing values rather than ranks (short forests)
  by_value : Bool
  // Implicit-heap copy for walking by value: every tree padded to
  // `heap_depth` levels, split `h` (1-based) of tree `t` at `t * heap_size +
  // h` with children `2h` and `2h + 1`, so a step loads no child index; leaf
  // `h` at `t * heap_size + h - heap_size` of `heap_leaf`. A padded split
  // compares against +inf and so goes left. `heap_depth` is 0 when absent.
  heap_slot : FixedArray[Int32]
  heap_thr : FixedArray[Double]
  heap_leaf : FixedArray[Double]
  heap_depth : Int32
  heap_size : Int32
  // The same heap for float inputs, each split packed as `(t' bits << 32) |
  // slot`, where `t'` is the largest float not above the threshold `t`: for
  // a float `x`, `x <= t` exactly when `x <= t'`, so one load and a float
  // compare decide the step. Padded splits hold +inf. Empty when absent.
  fheap : FixedArray[Int64]
  leaf_value : FixedArray[Double]
  roots : FixedArray[Int32]
  depth : FixedArray[Int32]
  // sorted distinct thresholds of feature f at cuts[cut_start[f] ..
  // cut_start[f + 1]]
  cuts : FixedArray[Double]
  cut_start : FixedArray[Int32]
  // rank of 0.0 for each feature, for missing type None
  zero_rank : FixedArray[Int32]
  groups : Int32
}

///|
/// Number of rank or value slots per row.
fn LgbFast::width(self : LgbFast) -> Int32 {
  self.slot_feature.length()
}

///|
/// Build the fast form, or `None` for a model with a categorical split.
fn LgbFast::build(
  trees : Array[LightGBMTree],
  num_feature : Int32,
  groups : Int32,
) -> LgbFast? {
  let per_feature : Array[Array[Double]] = Array::makei(num_feature, _ => [])
  for t in trees {
    for i in 0..<t.left.length() {
      if t.decision_type[i] % 2 == 1 {
        return None
      }
      per_feature[t.split_feature[i]].push(t.threshold[i])
    }
  }
  let cut_start = FixedArray::make(num_feature + 1, 0)
  let cuts : Array[Double] = []
  for f in 0..<num_feature {
    let c = per_feature[f]
    c.sort()
    c.dedup()
    cuts.push_iter(c.iter())
    cut_start[f + 1] = cuts.length()
    // ranks and threshold ranks must stay below `LGB_HIGH` (16 bits)
    if c.length() >= LGB_HIGH {
      return None
    }
  }
  let cuts = FixedArray::from_array(cuts)
  let zero_rank = FixedArray::make(num_feature, 0)
  for f in 0..<num_feature {
    zero_rank[f] = lgb_rank(cuts, cut_start[f], cut_start[f + 1], 0.0)
  }
  // Lay each tree out breadth-first: a split's two children at consecutive
  // ids, a leaf looping to itself.
  let left : Array[Int32] = []
  let slot : Array[Int32] = []
  let rank : Array[Int32] = []
  let thr : Array[Double] = []
  // slot of each (feature, variant), -1 until used
  let slot_id = FixedArray::make(num_feature * LGB_SLOTS, -1)
  let slot_feature : Array[Int32] = []
  let slot_variant : Array[Int32] = []
  // a finite threshold beyond float range rules out walking by value
  let mut huge = false
  let leaf_value : Array[Double] = []
  let roots = FixedArray::make(trees.length(), 0)
  let depth = FixedArray::make(trees.length(), 0)
  for ti, t in trees {
    let base = left.length()
    roots[ti] = base
    // queue of (lightgbm node, depth); a node >= 0 is a split, ~i is leaf i
    let queue : Array[(Int32, Int32)] = [
      (if t.left.length() == 0 { -1 } else { 0 }, 0),
    ]
    let mut head = 0
    let mut maxd = 0
    while head < queue.length() {
      let (node, d) = queue[head]
      let id = base + head
      head += 1
      if d > maxd {
        maxd = d
      }
      if node < 0 {
        left.push(id)
        slot.push(0)
        rank.push(LGB_LEAF)
        thr.push(0x7FF0000000000000UL.reinterpret_as_double())
        leaf_value.push(t.leaf_value[-node - 1])
        continue
      }
      let f = t.split_feature[node]
      let dt = t.decision_type[node]
      let missing = (dt >> 2) & 3
      let default_left = (dt & 2) != 0
      let s = match missing {
        2 => if default_left { 0 } else { 1 }
        1 => if default_left { 3 } else { 4 }
        _ => 2
      }
      left.push(base + queue.length())
      let key = f * LGB_SLOTS + s
      let id = if slot_id[key] >= 0 {
        slot_id[key]
      } else {
        let i = slot_feature.length()
        slot_id[key] = i
        slot_feature.push(f)
        slot_variant.push(s)
        i
      }
      slot.push(id)
      // `x <= +inf` holds for every non-missing x; so does `x <= max`
      // once a +inf input reads as max (see `lgb_slot_value`)
      let tv = t.threshold[node]
      if tv.is_inf() && tv > 0.0 {
        thr.push(lgb_max)
      } else {
        if !tv.is_inf() && tv.abs() > lgb_fmax {
          huge = true
        }
        thr.push(tv)
      }
      rank.push(
        lgb_index(cuts, cut_start[f], cut_start[f + 1], t.threshold[node]),
      )
      leaf_value.push(0.0)
      queue.push((t.left[node], d + 1))
      queue.push((t.right[node], d + 1))
    }
    depth[ti] = maxd
  }
  let mut hd = 0
  for d in depth {
    if d > hd {
      hd = d
    }
  }
  // Walk by value over the implicit heap when every tree fits in 10 levels
  // and the heap stays cache-sized (at most 2^18 nodes):
  // no ranks to compute, one load and a compare per step.
  let by_value = !huge && hd >= 1 && hd <= 10 && trees.length() << hd <= 1 << 18
  let inf = 0x7FF0000000000000UL.reinterpret_as_double()
  let (heap_slot, heap_thr, heap_leaf, heap_depth, heap_size) = if by_value {
    let size = 1 << hd
    let hs = FixedArray::make(trees.length() * size, 0)
    let ht = FixedArray::make(trees.length() * size, inf)
    let hl = FixedArray::make(trees.length() * size, 0.0)
    for ti in 0..<trees.length() {
      let base = ti * size
      // (node id in `left`/`slot`, heap position)
      let stack = [(roots[ti], 1)]
      while stack.pop() is Some((id, h)) {
        if rank[id] == LGB_LEAF {
          // a leaf: always left down to the last level
          let mut p = h
          while p < size {
            p = 2 * p
          }
          hl[base + p - size] = leaf_value[id]
        } else {
          hs[base + h] = slot[id]
          ht[base + h] = thr[id]
          stack.push((left[id], 2 * h))
          stack.push((left[id] + 1, 2 * h + 1))
        }
      }
    }
    (hs, ht, hl, hd, size)
  } else {
    (
      FixedArray::make(0, 0),
      FixedArray::make(0, 0.0),
      FixedArray::make(0, 0.0),
      0,
      0,
    )
  }
  let fheap = FixedArray::makei(heap_slot.length(), i => {
    let c = f32_down(heap_thr[i])
    (c.reinterpret_as_int().to_int64() << 32) | heap_slot[i].to_int64()
  })
  if slot_feature.length() > 0xFFFF {
    return None
  }
  let node = FixedArray::makei(left.length(), i => {
    (left[i].to_int64() << 32) | (slot[i].to_int64() << 16) | rank[i].to_int64()
  })
  Some({
    node,
    fheap,
    heap_slot,
    heap_thr,
    heap_leaf,
    heap_depth,
    heap_size,
    left: FixedArray::from_array(left),
    slot: FixedArray::from_array(slot),
    rank: FixedArray::from_array(rank),
    thr: FixedArray::from_array(thr),
    slot_feature: FixedArray::from_array(slot_feature),
    slot_variant: FixedArray::from_array(slot_variant),
    by_value,
    leaf_value: FixedArray::from_array(leaf_value),
    roots,
    depth,
    cuts,
    cut_start,
    zero_rank,
    groups,
  })
}

///|
/// Number of `cuts[lo .. hi]` strictly below `v` (`v` not NaN).
fn lgb_rank(
  cuts : FixedArray[Double],
  lo : Int32,
  hi : Int32,
  v : Double,
) -> Int32 {
  let mut a = lo
  let mut b = hi
  while a < b {
    let m = (a + b) / 2
    if cuts[m] < v {
      a = m + 1
    } else {
      b = m
    }
  }
  a - lo
}

///|
/// Index of threshold `t` among `cuts[lo .. hi]`.
fn lgb_index(
  cuts : FixedArray[Double],
  lo : Int32,
  hi : Int32,
  t : Double,
) -> Int32 {
  lgb_rank(cuts, lo, hi, t)
}

///|
/// The value a split of `variant` (see `LGB_SLOTS`) compares for input `v`,
/// with a forced direction written as +-inf. Values within
/// `kZeroThreshold` of zero read as 0.
#inline
fn lgb_slot_value(variant : Int32, v : Double) -> Double {
  let inf = 0x7FF0000000000000UL.reinterpret_as_double()
  if v.is_nan() {
    match variant {
      0 | 3 => -inf
      2 => 0.0
      _ => inf
    }
  } else if v >= -lgb_zero && v <= lgb_zero {
    match variant {
      3 => -inf
      4 => inf
      _ => 0.0
    }
  } else if v > lgb_max {
    // +inf: past every finite threshold, but not past a +inf one
    lgb_max
  } else {
    v
  }
}

///|
/// Largest finite double; a +inf threshold or input reads as this when
/// walking by value, which leaves +inf free to force a step right.
let lgb_max : Double = 0x7FEFFFFFFFFFFFFFUL.reinterpret_as_double()

///|
/// Largest finite float.
let lgb_fmax : Double = Float::reinterpret_from_int(0x7F7FFFFF).to_double()

///|
/// Rank of a slot value: forced directions map to the extremes.
#inline
fn LgbFast::slot_rank(self : LgbFast, f : Int32, w : Double) -> Int32 {
  if w.is_inf() {
    if w > 0.0 {
      LGB_HIGH
    } else {
      0
    }
  } else if w == 0.0 {
    self.zero_rank.unsafe_get(f)
  } else {
    lgb_rank(
      self.cuts,
      self.cut_start.unsafe_get(f),
      self.cut_start.unsafe_get(f + 1),
      w,
    )
  }
}

///|
/// Write the slots of row `row` of a row-major float matrix with `cols`
/// columns to `out[at ..]`: ranks, or values when the forest walks by value.
fn LgbFast::slots_of_floats(
  self : LgbFast,
  data : FixedArray[Float],
  row : Int32,
  cols : Int32,
  ranks : FixedArray[Int32],
  values : FixedArray[Double],
  at : Int32,
) -> Unit {
  let base = row * cols
  let sf = self.slot_feature
  let sv = self.slot_variant
  if self.by_value {
    for i in 0..<sf.length() {
      let f = sf.unsafe_get(i)
      let v = if f < cols { data[base + f].to_double() } else { 0.0 }
      values.unsafe_set(at + i, lgb_slot_value(sv.unsafe_get(i), v))
    }
  } else {
    for i in 0..<sf.length() {
      let f = sf.unsafe_get(i)
      let v = if f < cols { data[base + f].to_double() } else { 0.0 }
      ranks.unsafe_set(
        at + i,
        self.slot_rank(f, lgb_slot_value(sv.unsafe_get(i), v)),
      )
    }
  }
}

///|
/// `slots_of_floats` for one row of doubles.
fn LgbFast::slots_of_doubles(
  self : LgbFast,
  row : ArrayView[Double],
  ranks : FixedArray[Int32],
  values : FixedArray[Double],
) -> Unit {
  let n = row.length()
  for i in 0..<self.slot_feature.length() {
    let f = self.slot_feature.unsafe_get(i)
    let w = lgb_slot_value(
      self.slot_variant.unsafe_get(i),
      if f < n {
        row[f]
      } else {
        0.0
      },
    )
    if self.by_value {
      values.unsafe_set(i, w)
    } else {
      ranks.unsafe_set(i, self.slot_rank(f, w))
    }
  }
}

///|
/// Add every tree's leaf to `margin` (`groups` values per row) for `n` rows
/// whose slots sit consecutively in `ranks` or `values`.
fn LgbFast::accumulate(
  self : LgbFast,
  ranks : FixedArray[Int32],
  values : FixedArray[Double],
  n : Int32,
  margin : FixedArray[Double],
) -> Unit {
  if self.by_value {
    self.accumulate_values(values, n, margin)
  } else {
    self.accumulate_ranks(ranks, n, margin)
  }
}

///|
/// Add every tree's leaf to `margin` (`groups` values per row) for `n` rows
/// whose rank slots sit consecutively in `ranks`.
fn LgbFast::accumulate_ranks(
  self : LgbFast,
  ranks : FixedArray[Int32],
  n : Int32,
  margin : FixedArray[Double],
) -> Unit {
  let w = self.width()
  let left = self.left
  let slot = self.slot
  let rank = self.rank
  let node = self.node
  let lv = self.leaf_value
  let groups = self.groups
  let trees = self.roots.length()
  if n == 1 {
    self.single_ranks(ranks, margin)
    return
  }
  for t in 0..<trees {
    let root = self.roots.unsafe_get(t)
    let d = self.depth.unsafe_get(t)
    let g = t % groups
    let mut r = 0
    if LGB_PACKED {
      while r + 8 <= n {
        let b0 = r * w
        let b1 = b0 + w
        let b2 = b1 + w
        let b3 = b2 + w
        let b4 = b3 + w
        let b5 = b4 + w
        let b6 = b5 + w
        let b7 = b6 + w
        let mut i0 = root
        let mut i1 = root
        let mut i2 = root
        let mut i3 = root
        let mut i4 = root
        let mut i5 = root
        let mut i6 = root
        let mut i7 = root
        for _ in 0..<d {
          let n0 = node.unsafe_get(i0)
          let n1 = node.unsafe_get(i1)
          let n2 = node.unsafe_get(i2)
          let n3 = node.unsafe_get(i3)
          let n4 = node.unsafe_get(i4)
          let n5 = node.unsafe_get(i5)
          let n6 = node.unsafe_get(i6)
          let n7 = node.unsafe_get(i7)
          let w0 = n0.to_int()
          let w1 = n1.to_int()
          let w2 = n2.to_int()
          let w3 = n3.to_int()
          let w4 = n4.to_int()
          let w5 = n5.to_int()
          let w6 = n6.to_int()
          let w7 = n7.to_int()
          i0 = (n0 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b0 + (w0 >> 16)), w0 & 0xFFFF)
          i1 = (n1 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b1 + (w1 >> 16)), w1 & 0xFFFF)
          i2 = (n2 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b2 + (w2 >> 16)), w2 & 0xFFFF)
          i3 = (n3 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b3 + (w3 >> 16)), w3 & 0xFFFF)
          i4 = (n4 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b4 + (w4 >> 16)), w4 & 0xFFFF)
          i5 = (n5 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b5 + (w5 >> 16)), w5 & 0xFFFF)
          i6 = (n6 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b6 + (w6 >> 16)), w6 & 0xFFFF)
          i7 = (n7 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b7 + (w7 >> 16)), w7 & 0xFFFF)
        }
        let m = r * groups + g
        margin.unsafe_set(
          m + 0 * groups,
          margin.unsafe_get(m + 0 * groups) + lv.unsafe_get(i0),
        )
        margin.unsafe_set(
          m + 1 * groups,
          margin.unsafe_get(m + 1 * groups) + lv.unsafe_get(i1),
        )
        margin.unsafe_set(
          m + 2 * groups,
          margin.unsafe_get(m + 2 * groups) + lv.unsafe_get(i2),
        )
        margin.unsafe_set(
          m + 3 * groups,
          margin.unsafe_get(m + 3 * groups) + lv.unsafe_get(i3),
        )
        margin.unsafe_set(
          m + 4 * groups,
          margin.unsafe_get(m + 4 * groups) + lv.unsafe_get(i4),
        )
        margin.unsafe_set(
          m + 5 * groups,
          margin.unsafe_get(m + 5 * groups) + lv.unsafe_get(i5),
        )
        margin.unsafe_set(
          m + 6 * groups,
          margin.unsafe_get(m + 6 * groups) + lv.unsafe_get(i6),
        )
        margin.unsafe_set(
          m + 7 * groups,
          margin.unsafe_get(m + 7 * groups) + lv.unsafe_get(i7),
        )
        r += 8
      }
    }
    while r + 4 <= n {
      let b0 = r * w
      let b1 = b0 + w
      let b2 = b1 + w
      let b3 = b2 + w
      let mut i0 = root
      let mut i1 = root
      let mut i2 = root
      let mut i3 = root
      if LGB_PACKED {
        for _ in 0..<d {
          let n0 = node.unsafe_get(i0)
          let n1 = node.unsafe_get(i1)
          let n2 = node.unsafe_get(i2)
          let n3 = node.unsafe_get(i3)
          let w0 = n0.to_int()
          let w1 = n1.to_int()
          let w2 = n2.to_int()
          let w3 = n3.to_int()
          i0 = (n0 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b0 + (w0 >> 16)), w0 & 0xFFFF)
          i1 = (n1 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b1 + (w1 >> 16)), w1 & 0xFFFF)
          i2 = (n2 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b2 + (w2 >> 16)), w2 & 0xFFFF)
          i3 = (n3 >> 32).to_int() +
            lgb_gt(ranks.unsafe_get(b3 + (w3 >> 16)), w3 & 0xFFFF)
        }
      } else {
        for _ in 0..<d {
          let s0 = slot.unsafe_get(i0)
          let s1 = slot.unsafe_get(i1)
          let s2 = slot.unsafe_get(i2)
          let s3 = slot.unsafe_get(i3)
          let k0 = rank.unsafe_get(i0)
          let k1 = rank.unsafe_get(i1)
          let k2 = rank.unsafe_get(i2)
          let k3 = rank.unsafe_get(i3)
          i0 = left.unsafe_get(i0) + lgb_gt(ranks.unsafe_get(b0 + s0), k0)
          i1 = left.unsafe_get(i1) + lgb_gt(ranks.unsafe_get(b1 + s1), k1)
          i2 = left.unsafe_get(i2) + lgb_gt(ranks.unsafe_get(b2 + s2), k2)
          i3 = left.unsafe_get(i3) + lgb_gt(ranks.unsafe_get(b3 + s3), k3)
        }
      }
      let m = r * groups + g
      margin.unsafe_set(m, margin.unsafe_get(m) + lv.unsafe_get(i0))
      margin.unsafe_set(
        m + groups,
        margin.unsafe_get(m + groups) + lv.unsafe_get(i1),
      )
      let m2 = m + 2 * groups
      margin.unsafe_set(m2, margin.unsafe_get(m2) + lv.unsafe_get(i2))
      margin.unsafe_set(
        m2 + groups,
        margin.unsafe_get(m2 + groups) + lv.unsafe_get(i3),
      )
      r += 4
    }
    while r < n {
      let b = r * w
      let mut i = root
      for _ in 0..<d {
        i = left.unsafe_get(i) +
          lgb_gt(ranks.unsafe_get(b + slot.unsafe_get(i)), rank.unsafe_get(i))
      }
      let m = r * groups + g
      margin.unsafe_set(m, margin.unsafe_get(m) + lv.unsafe_get(i))
      r += 1
    }
  }
}

///|
/// One row: four trees walk at once, their leaves added in tree order.
fn LgbFast::single_ranks(
  self : LgbFast,
  ranks : FixedArray[Int32],
  margin : FixedArray[Double],
) -> Unit {
  let left = self.left
  let slot = self.slot
  let rank = self.rank
  let node = self.node
  let lv = self.leaf_value
  let groups = self.groups
  let trees = self.roots.length()
  let mut t = 0
  if LGB_PACKED {
    while t + 8 <= trees {
      let mut i0 = self.roots.unsafe_get(t + 0)
      let mut i1 = self.roots.unsafe_get(t + 1)
      let mut i2 = self.roots.unsafe_get(t + 2)
      let mut i3 = self.roots.unsafe_get(t + 3)
      let mut i4 = self.roots.unsafe_get(t + 4)
      let mut i5 = self.roots.unsafe_get(t + 5)
      let mut i6 = self.roots.unsafe_get(t + 6)
      let mut i7 = self.roots.unsafe_get(t + 7)
      let mut d = self.depth.unsafe_get(t)
      for j in 1..<8 {
        let dj = self.depth.unsafe_get(t + j)
        if dj > d {
          d = dj
        }
      }
      for _ in 0..<d {
        let n0 = node.unsafe_get(i0)
        let n1 = node.unsafe_get(i1)
        let n2 = node.unsafe_get(i2)
        let n3 = node.unsafe_get(i3)
        let n4 = node.unsafe_get(i4)
        let n5 = node.unsafe_get(i5)
        let n6 = node.unsafe_get(i6)
        let n7 = node.unsafe_get(i7)
        let w0 = n0.to_int()
        let w1 = n1.to_int()
        let w2 = n2.to_int()
        let w3 = n3.to_int()
        let w4 = n4.to_int()
        let w5 = n5.to_int()
        let w6 = n6.to_int()
        let w7 = n7.to_int()
        i0 = (n0 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w0 >> 16), w0 & 0xFFFF)
        i1 = (n1 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w1 >> 16), w1 & 0xFFFF)
        i2 = (n2 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w2 >> 16), w2 & 0xFFFF)
        i3 = (n3 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w3 >> 16), w3 & 0xFFFF)
        i4 = (n4 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w4 >> 16), w4 & 0xFFFF)
        i5 = (n5 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w5 >> 16), w5 & 0xFFFF)
        i6 = (n6 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w6 >> 16), w6 & 0xFFFF)
        i7 = (n7 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w7 >> 16), w7 & 0xFFFF)
      }
      let g = (t + 0) % groups
      margin[g] = margin[g] + lv.unsafe_get(i0)
      let g = (t + 1) % groups
      margin[g] = margin[g] + lv.unsafe_get(i1)
      let g = (t + 2) % groups
      margin[g] = margin[g] + lv.unsafe_get(i2)
      let g = (t + 3) % groups
      margin[g] = margin[g] + lv.unsafe_get(i3)
      let g = (t + 4) % groups
      margin[g] = margin[g] + lv.unsafe_get(i4)
      let g = (t + 5) % groups
      margin[g] = margin[g] + lv.unsafe_get(i5)
      let g = (t + 6) % groups
      margin[g] = margin[g] + lv.unsafe_get(i6)
      let g = (t + 7) % groups
      margin[g] = margin[g] + lv.unsafe_get(i7)
      t += 8
    }
  }
  while t + 4 <= trees {
    let mut i0 = self.roots.unsafe_get(t)
    let mut i1 = self.roots.unsafe_get(t + 1)
    let mut i2 = self.roots.unsafe_get(t + 2)
    let mut i3 = self.roots.unsafe_get(t + 3)
    let mut d = self.depth.unsafe_get(t)
    for j in 1..<4 {
      let dj = self.depth.unsafe_get(t + j)
      if dj > d {
        d = dj
      }
    }
    if LGB_PACKED {
      for _ in 0..<d {
        let n0 = node.unsafe_get(i0)
        let n1 = node.unsafe_get(i1)
        let n2 = node.unsafe_get(i2)
        let n3 = node.unsafe_get(i3)
        let w0 = n0.to_int()
        let w1 = n1.to_int()
        let w2 = n2.to_int()
        let w3 = n3.to_int()
        i0 = (n0 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w0 >> 16), w0 & 0xFFFF)
        i1 = (n1 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w1 >> 16), w1 & 0xFFFF)
        i2 = (n2 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w2 >> 16), w2 & 0xFFFF)
        i3 = (n3 >> 32).to_int() +
          lgb_gt(ranks.unsafe_get(w3 >> 16), w3 & 0xFFFF)
      }
    } else {
      for _ in 0..<d {
        let k0 = rank.unsafe_get(i0)
        let k1 = rank.unsafe_get(i1)
        let k2 = rank.unsafe_get(i2)
        let k3 = rank.unsafe_get(i3)
        i0 = left.unsafe_get(i0) +
          lgb_gt(ranks.unsafe_get(slot.unsafe_get(i0)), k0)
        i1 = left.unsafe_get(i1) +
          lgb_gt(ranks.unsafe_get(slot.unsafe_get(i1)), k1)
        i2 = left.unsafe_get(i2) +
          lgb_gt(ranks.unsafe_get(slot.unsafe_get(i2)), k2)
        i3 = left.unsafe_get(i3) +
          lgb_gt(ranks.unsafe_get(slot.unsafe_get(i3)), k3)
      }
    }
    let g = t % groups
    margin[g] = margin[g] + lv.unsafe_get(i0)
    let g = (t + 1) % groups
    margin[g] = margin[g] + lv.unsafe_get(i1)
    let g = (t + 2) % groups
    margin[g] = margin[g] + lv.unsafe_get(i2)
    let g = (t + 3) % groups
    margin[g] = margin[g] + lv.unsafe_get(i3)
    t += 4
  }
  while t < trees {
    let mut i = self.roots.unsafe_get(t)
    for _ in 0..<self.depth.unsafe_get(t) {
      i = left.unsafe_get(i) +
        lgb_gt(ranks.unsafe_get(slot.unsafe_get(i)), rank.unsafe_get(i))
    }
    let g = t % groups
    margin[g] = margin[g] + lv.unsafe_get(i)
    t += 1
  }
}

///|
/// Add every tree's leaf to `margin` (`groups` values per row) for `n` rows
/// whose rank slots sit consecutively in `ranks`.
fn LgbFast::accumulate_values(
  self : LgbFast,
  values : FixedArray[Double],
  n : Int32,
  margin : FixedArray[Double],
) -> Unit {
  let w = self.width()
  let left = self.left
  let slot = self.slot
  let thr = self.thr
  let lv = self.leaf_value
  let groups = self.groups
  let trees = self.roots.length()
  if n == 1 {
    self.single_values(values, margin)
    return
  }
  if self.heap_depth > 0 {
    self.accumulate_heap(values, n, margin)
    return
  }
  for t in 0..<trees {
    let root = self.roots.unsafe_get(t)
    let d = self.depth.unsafe_get(t)
    let g = t % groups
    let mut r = 0
    while r + 4 <= n {
      let b0 = r * w
      let b1 = b0 + w
      let b2 = b1 + w
      let b3 = b2 + w
      let mut i0 = root
      let mut i1 = root
      let mut i2 = root
      let mut i3 = root
      for _ in 0..<d {
        let s0 = slot.unsafe_get(i0)
        let s1 = slot.unsafe_get(i1)
        let s2 = slot.unsafe_get(i2)
        let s3 = slot.unsafe_get(i3)
        let k0 = thr.unsafe_get(i0)
        let k1 = thr.unsafe_get(i1)
        let k2 = thr.unsafe_get(i2)
        let k3 = thr.unsafe_get(i3)
        i0 = left.unsafe_get(i0) + lgb_gtd(values.unsafe_get(b0 + s0), k0)
        i1 = left.unsafe_get(i1) + lgb_gtd(values.unsafe_get(b1 + s1), k1)
        i2 = left.unsafe_get(i2) + lgb_gtd(values.unsafe_get(b2 + s2), k2)
        i3 = left.unsafe_get(i3) + lgb_gtd(values.unsafe_get(b3 + s3), k3)
      }
      let m = r * groups + g
      margin.unsafe_set(m, margin.unsafe_get(m) + lv.unsafe_get(i0))
      margin.unsafe_set(
        m + groups,
        margin.unsafe_get(m + groups) + lv.unsafe_get(i1),
      )
      let m2 = m + 2 * groups
      margin.unsafe_set(m2, margin.unsafe_get(m2) + lv.unsafe_get(i2))
      margin.unsafe_set(
        m2 + groups,
        margin.unsafe_get(m2 + groups) + lv.unsafe_get(i3),
      )
      r += 4
    }
    while r < n {
      let b = r * w
      let mut i = root
      for _ in 0..<d {
        i = left.unsafe_get(i) +
          lgb_gtd(values.unsafe_get(b + slot.unsafe_get(i)), thr.unsafe_get(i))
      }
      let m = r * groups + g
      margin.unsafe_set(m, margin.unsafe_get(m) + lv.unsafe_get(i))
      r += 1
    }
  }
}

///|
/// One row: four trees walk at once, their leaves added in tree order.
fn LgbFast::single_values(
  self : LgbFast,
  values : FixedArray[Double],
  margin : FixedArray[Double],
) -> Unit {
  let left = self.left
  let slot = self.slot
  let thr = self.thr
  let lv = self.leaf_value
  let groups = self.groups
  let trees = self.roots.length()
  let mut t = 0
  while t + 4 <= trees {
    let mut i0 = self.roots.unsafe_get(t)
    let mut i1 = self.roots.unsafe_get(t + 1)
    let mut i2 = self.roots.unsafe_get(t + 2)
    let mut i3 = self.roots.unsafe_get(t + 3)
    let mut d = self.depth.unsafe_get(t)
    for j in 1..<4 {
      let dj = self.depth.unsafe_get(t + j)
      if dj > d {
        d = dj
      }
    }
    for _ in 0..<d {
      let k0 = thr.unsafe_get(i0)
      let k1 = thr.unsafe_get(i1)
      let k2 = thr.unsafe_get(i2)
      let k3 = thr.unsafe_get(i3)
      i0 = left.unsafe_get(i0) +
        lgb_gtd(values.unsafe_get(slot.unsafe_get(i0)), k0)
      i1 = left.unsafe_get(i1) +
        lgb_gtd(values.unsafe_get(slot.unsafe_get(i1)), k1)
      i2 = left.unsafe_get(i2) +
        lgb_gtd(values.unsafe_get(slot.unsafe_get(i2)), k2)
      i3 = left.unsafe_get(i3) +
        lgb_gtd(values.unsafe_get(slot.unsafe_get(i3)), k3)
    }
    let g = t % groups
    margin[g] = margin[g] + lv.unsafe_get(i0)
    let g = (t + 1) % groups
    margin[g] = margin[g] + lv.unsafe_get(i1)
    let g = (t + 2) % groups
    margin[g] = margin[g] + lv.unsafe_get(i2)
    let g = (t + 3) % groups
    margin[g] = margin[g] + lv.unsafe_get(i3)
    t += 4
  }
  while t < trees {
    let mut i = self.roots.unsafe_get(t)
    for _ in 0..<self.depth.unsafe_get(t) {
      i = left.unsafe_get(i) +
        lgb_gtd(values.unsafe_get(slot.unsafe_get(i)), thr.unsafe_get(i))
    }
    let g = t % groups
    margin[g] = margin[g] + lv.unsafe_get(i)
    t += 1
  }
}

///|
#inline
fn lgb_gt(a : Int32, b : Int32) -> Int32 {
  if a > b {
    1
  } else {
    0
  }
}

///|
#inline
fn lgb_gtd(a : Double, b : Double) -> Int32 {
  if a > b {
    1
  } else {
    0
  }
}

///|
/// `accumulate_values` over the implicit heap, four rows at a time.
fn LgbFast::accumulate_heap(
  self : LgbFast,
  values : FixedArray[Double],
  n : Int32,
  margin : FixedArray[Double],
) -> Unit {
  let w = self.width()
  let hs = self.heap_slot
  let ht = self.heap_thr
  let hl = self.heap_leaf
  let d = self.heap_depth
  let size = self.heap_size
  let groups = self.groups
  let trees = self.roots.length()
  for t in 0..<trees {
    let base = t * size
    let lbase = base - size
    let g = t % groups
    let mut r = 0
    while r + 4 <= n {
      let b0 = r * w
      let b1 = b0 + w
      let b2 = b1 + w
      let b3 = b2 + w
      let mut h0 = 1
      let mut h1 = 1
      let mut h2 = 1
      let mut h3 = 1
      for _ in 0..<d {
        let p0 = base + h0
        let p1 = base + h1
        let p2 = base + h2
        let p3 = base + h3
        h0 = 2 * h0 +
          lgb_gtd(values.unsafe_get(b0 + hs.unsafe_get(p0)), ht.unsafe_get(p0))
        h1 = 2 * h1 +
          lgb_gtd(values.unsafe_get(b1 + hs.unsafe_get(p1)), ht.unsafe_get(p1))
        h2 = 2 * h2 +
          lgb_gtd(values.unsafe_get(b2 + hs.unsafe_get(p2)), ht.unsafe_get(p2))
        h3 = 2 * h3 +
          lgb_gtd(values.unsafe_get(b3 + hs.unsafe_get(p3)), ht.unsafe_get(p3))
      }
      let m = r * groups + g
      margin.unsafe_set(m, margin.unsafe_get(m) + hl.unsafe_get(lbase + h0))
      let m1 = m + groups
      margin.unsafe_set(m1, margin.unsafe_get(m1) + hl.unsafe_get(lbase + h1))
      let m2 = m1 + groups
      margin.unsafe_set(m2, margin.unsafe_get(m2) + hl.unsafe_get(lbase + h2))
      let m3 = m2 + groups
      margin.unsafe_set(m3, margin.unsafe_get(m3) + hl.unsafe_get(lbase + h3))
      r += 4
    }
    while r < n {
      let b = r * w
      let mut h = 1
      for _ in 0..<d {
        let p = base + h
        h = 2 * h +
          lgb_gtd(values.unsafe_get(b + hs.unsafe_get(p)), ht.unsafe_get(p))
      }
      let m = r * groups + g
      margin.unsafe_set(m, margin.unsafe_get(m) + hl.unsafe_get(lbase + h))
      r += 1
    }
  }
}

///|
/// The largest float not above `t`.
fn f32_down(t : Double) -> Float {
  let f = Float::from_double(t)
  if f.to_double() <= t {
    f
  } else if f > 0.0 {
    Float::reinterpret_from_int(f.reinterpret_as_int() - 1)
  } else if f == 0.0 {
    // t is a tiny negative number
    Float::reinterpret_from_int(0x80000001)
  } else {
    Float::reinterpret_from_int(f.reinterpret_as_int() + 1)
  }
}

///|
/// Slot values of rows `start .. start + n` of a row-major float matrix, as
/// floats, row `r` at `out[r * width]`.
fn LgbFast::float_slots(
  self : LgbFast,
  data : FixedArray[Float],
  cols : Int32,
  start : Int32,
  n : Int32,
  out : FixedArray[Float],
) -> Unit {
  let w = self.width()
  let sf = self.slot_feature
  let sv = self.slot_variant
  let mut fits = (start + n) * cols <= data.length()
  for i in 0..<w {
    if sf[i] >= cols {
      fits = false
    }
  }
  if fits {
    // Every slot reads a column: select the NaN and near-zero readings
    // from per-slot tables instead of branching on the variant.
    let nan = 0x7FF8000000000000UL.reinterpret_as_double()
    let nanv = FixedArray::makei(w, i => {
      Float::from_double(lgb_slot_value(sv[i], nan))
    })
    let zerov = FixedArray::makei(w, i => {
      Float::from_double(lgb_slot_value(sv[i], 0.0))
    })
    let zf = Float::from_double(lgb_zero)
    let inf = Float::reinterpret_from_int(0x7F800000)
    let fmax = Float::reinterpret_from_int(0x7F7FFFFF)
    for r in 0..<n {
      let base = (start + r) * cols
      let o = r * w
      for i in 0..<w {
        let x = data.unsafe_get(base + sf.unsafe_get(i))
        let y = if x == inf { fmax } else { x }
        let y = if x.abs() <= zf { zerov.unsafe_get(i) } else { y }
        let y = if x != x { nanv.unsafe_get(i) } else { y }
        out.unsafe_set(o + i, y)
      }
    }
    return
  }
  for r in 0..<n {
    let base = (start + r) * cols
    let o = r * w
    for i in 0..<w {
      let f = sf.unsafe_get(i)
      let v = if f < cols { data[base + f].to_double() } else { 0.0 }
      let w = lgb_slot_value(sv.unsafe_get(i), v)
      // max reads as the largest float, which every threshold's float
      // rounding below reaches only when the threshold is at least that
      out.unsafe_set(
        o + i,
        if w == lgb_max {
          Float::reinterpret_from_int(0x7F7FFFFF)
        } else {
          Float::from_double(w)
        },
      )
    }
  }
}

///|
/// Add every tree's leaf to `margin` (one group) for `n` rows of float slot
/// values, walking `fheap`: four rows and four trees at a time, each row's
/// margin in a local and its leaves added in tree order.
fn LgbFast::accumulate_fheap(
  self : LgbFast,
  vals : FixedArray[Float],
  n : Int32,
  margin : FixedArray[Double],
) -> Unit {
  let w = self.width()
  let fh = self.fheap
  let leaf = self.heap_leaf
  let d = self.heap_depth
  let size = self.heap_size
  let trees = self.roots.length()
  let mut r = 0
  while r + 3 < n {
    let r0 = (r + 0) * w
    let r1 = (r + 1) * w
    let r2 = (r + 2) * w
    let r3 = (r + 3) * w
    let mut m0 = margin.unsafe_get(r + 0)
    let mut m1 = margin.unsafe_get(r + 1)
    let mut m2 = margin.unsafe_get(r + 2)
    let mut m3 = margin.unsafe_get(r + 3)
    let mut t = 0
    while t + 3 < trees {
      let b0 = (t + 0) * size
      let b1 = (t + 1) * size
      let b2 = (t + 2) * size
      let b3 = (t + 3) * size
      let mut x0_0 = 1
      let mut x0_1 = 1
      let mut x0_2 = 1
      let mut x0_3 = 1
      let mut x1_0 = 1
      let mut x1_1 = 1
      let mut x1_2 = 1
      let mut x1_3 = 1
      let mut x2_0 = 1
      let mut x2_1 = 1
      let mut x2_2 = 1
      let mut x2_3 = 1
      let mut x3_0 = 1
      let mut x3_1 = 1
      let mut x3_2 = 1
      let mut x3_3 = 1
      for _ in 0..<d {
        x0_0 = {
          let nd = fh.unsafe_get(b0 + x0_0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_0 << 1) | lgb_gtf(vals.unsafe_get(r0 + nd.to_int()), c)
        }
        x0_1 = {
          let nd = fh.unsafe_get(b1 + x0_1)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_1 << 1) | lgb_gtf(vals.unsafe_get(r0 + nd.to_int()), c)
        }
        x0_2 = {
          let nd = fh.unsafe_get(b2 + x0_2)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_2 << 1) | lgb_gtf(vals.unsafe_get(r0 + nd.to_int()), c)
        }
        x0_3 = {
          let nd = fh.unsafe_get(b3 + x0_3)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_3 << 1) | lgb_gtf(vals.unsafe_get(r0 + nd.to_int()), c)
        }
        x1_0 = {
          let nd = fh.unsafe_get(b0 + x1_0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x1_0 << 1) | lgb_gtf(vals.unsafe_get(r1 + nd.to_int()), c)
        }
        x1_1 = {
          let nd = fh.unsafe_get(b1 + x1_1)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x1_1 << 1) | lgb_gtf(vals.unsafe_get(r1 + nd.to_int()), c)
        }
        x1_2 = {
          let nd = fh.unsafe_get(b2 + x1_2)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x1_2 << 1) | lgb_gtf(vals.unsafe_get(r1 + nd.to_int()), c)
        }
        x1_3 = {
          let nd = fh.unsafe_get(b3 + x1_3)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x1_3 << 1) | lgb_gtf(vals.unsafe_get(r1 + nd.to_int()), c)
        }
        x2_0 = {
          let nd = fh.unsafe_get(b0 + x2_0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x2_0 << 1) | lgb_gtf(vals.unsafe_get(r2 + nd.to_int()), c)
        }
        x2_1 = {
          let nd = fh.unsafe_get(b1 + x2_1)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x2_1 << 1) | lgb_gtf(vals.unsafe_get(r2 + nd.to_int()), c)
        }
        x2_2 = {
          let nd = fh.unsafe_get(b2 + x2_2)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x2_2 << 1) | lgb_gtf(vals.unsafe_get(r2 + nd.to_int()), c)
        }
        x2_3 = {
          let nd = fh.unsafe_get(b3 + x2_3)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x2_3 << 1) | lgb_gtf(vals.unsafe_get(r2 + nd.to_int()), c)
        }
        x3_0 = {
          let nd = fh.unsafe_get(b0 + x3_0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x3_0 << 1) | lgb_gtf(vals.unsafe_get(r3 + nd.to_int()), c)
        }
        x3_1 = {
          let nd = fh.unsafe_get(b1 + x3_1)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x3_1 << 1) | lgb_gtf(vals.unsafe_get(r3 + nd.to_int()), c)
        }
        x3_2 = {
          let nd = fh.unsafe_get(b2 + x3_2)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x3_2 << 1) | lgb_gtf(vals.unsafe_get(r3 + nd.to_int()), c)
        }
        x3_3 = {
          let nd = fh.unsafe_get(b3 + x3_3)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x3_3 << 1) | lgb_gtf(vals.unsafe_get(r3 + nd.to_int()), c)
        }
      }
      m0 = m0 + leaf.unsafe_get(b0 - size + x0_0)
      m0 = m0 + leaf.unsafe_get(b1 - size + x0_1)
      m0 = m0 + leaf.unsafe_get(b2 - size + x0_2)
      m0 = m0 + leaf.unsafe_get(b3 - size + x0_3)
      m1 = m1 + leaf.unsafe_get(b0 - size + x1_0)
      m1 = m1 + leaf.unsafe_get(b1 - size + x1_1)
      m1 = m1 + leaf.unsafe_get(b2 - size + x1_2)
      m1 = m1 + leaf.unsafe_get(b3 - size + x1_3)
      m2 = m2 + leaf.unsafe_get(b0 - size + x2_0)
      m2 = m2 + leaf.unsafe_get(b1 - size + x2_1)
      m2 = m2 + leaf.unsafe_get(b2 - size + x2_2)
      m2 = m2 + leaf.unsafe_get(b3 - size + x2_3)
      m3 = m3 + leaf.unsafe_get(b0 - size + x3_0)
      m3 = m3 + leaf.unsafe_get(b1 - size + x3_1)
      m3 = m3 + leaf.unsafe_get(b2 - size + x3_2)
      m3 = m3 + leaf.unsafe_get(b3 - size + x3_3)
      t += 4
    }
    while t < trees {
      let b = t * size
      let mut x0 = 1
      let mut x1 = 1
      let mut x2 = 1
      let mut x3 = 1
      for _ in 0..<d {
        x0 = {
          let nd = fh.unsafe_get(b + x0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0 << 1) | lgb_gtf(vals.unsafe_get(r0 + nd.to_int()), c)
        }
        x1 = {
          let nd = fh.unsafe_get(b + x1)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x1 << 1) | lgb_gtf(vals.unsafe_get(r1 + nd.to_int()), c)
        }
        x2 = {
          let nd = fh.unsafe_get(b + x2)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x2 << 1) | lgb_gtf(vals.unsafe_get(r2 + nd.to_int()), c)
        }
        x3 = {
          let nd = fh.unsafe_get(b + x3)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x3 << 1) | lgb_gtf(vals.unsafe_get(r3 + nd.to_int()), c)
        }
      }
      m0 = m0 + leaf.unsafe_get(b - size + x0)
      m1 = m1 + leaf.unsafe_get(b - size + x1)
      m2 = m2 + leaf.unsafe_get(b - size + x2)
      m3 = m3 + leaf.unsafe_get(b - size + x3)
      t += 1
    }
    margin.unsafe_set(r + 0, m0)
    margin.unsafe_set(r + 1, m1)
    margin.unsafe_set(r + 2, m2)
    margin.unsafe_set(r + 3, m3)
    r += 4
  }
  while r < n {
    let o = r * w
    let mut m = margin.unsafe_get(r)
    for t in 0..<trees {
      let b = t * size
      let mut x = 1
      for _ in 0..<d {
        let nd = fh.unsafe_get(b + x)
        let c = Float::reinterpret_from_int((nd >> 32).to_int())
        x = (x << 1) | lgb_gtf(vals.unsafe_get(o + nd.to_int()), c)
      }
      m = m + leaf.unsafe_get(b - size + x)
    }
    margin.unsafe_set(r, m)
    r += 1
  }
}

///|
#inline
fn lgb_gtf(a : Float, b : Float) -> Int32 {
  if a > b {
    1
  } else {
    0
  }
}
