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
///   the `k`-th threshold (counting from 0) goes right when `rank > k`. This
///   is exact: `x < t_k` holds exactly when at most `k` thresholds are
///   `<= x`. Storing `k` rather than `k + 1` saves a subtraction per step.
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
  // Bucket table over each feature's threshold values: bucket
  // `b = clamp(floor((x - lo) * scale), 0, top)` is a monotone function of
  // `x`, so every threshold in an earlier bucket is `<= x` and every one in a
  // later bucket is `> x`. `table[table_start + b]` counts the thresholds in
  // earlier buckets, and at most `scan` thresholds share a bucket, so a rank
  // is one lookup and `scan` compares. `scan` is -1 for a feature whose
  // thresholds crowd into few buckets; it keeps the binary search.
  bucket_lo : FixedArray[Double]
  bucket_scale : FixedArray[Double]
  bucket_top : FixedArray[Int32]
  bucket_scan : FixedArray[Int32]
  table_start : FixedArray[Int32]
  table : FixedArray[Int32]
  // threshold values of each feature followed by `scan` NaNs (never `<=`), at
  // `bcut_start`
  bcut : FixedArray[Double]
  bcut_start : FixedArray[Int32]
  num_feature : Int32
  // Implicit-heap copy of the forest (targets with `USE_HEAP`): every tree
  // padded to `heap_depth` levels, split node `h` of tree `t` at `t *
  // heap_size + h` with children `2h` and `2h + 1`, so a step reads no child
  // pointer; leaf `h` (`h >= heap_size`) at `t * heap_size + h - heap_size`.
  // `heap_depth` is 0 when the copy is not built.
  heap : Buf
  heap_leaf : FBuf
  heap_depth : Int32
  heap_size : Int32
  // rows per batch block (see `Model::predict_batch`)
  block : Int32
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
  let bucket_lo = FixedArray::make(num_feature, 0.0)
  let bucket_scale = FixedArray::make(num_feature, 0.0)
  let bucket_top = FixedArray::make(num_feature, 0)
  let bucket_scan = FixedArray::make(num_feature, 0)
  let table_start = FixedArray::make(num_feature, 0)
  let table : Array[Int32] = []
  let bcut_start = FixedArray::make(num_feature, 0)
  let bcut : Array[Double] = []
  for f in 0..<num_feature {
    let c = if USE_BUCKETS { cut_start[f + 1] - cut_start[f] } else { 0 }
    let cut_end = cut_start[f] + c
    let mut nb = 1
    while nb < 4 * c && nb < MAX_BUCKETS {
      nb *= 2
    }
    if c > 0 {
      let lo = key_value(cuts[cut_start[f]]).to_double()
      let hi = key_value(cuts[cut_start[f + 1] - 1]).to_double()
      bucket_lo[f] = lo
      bucket_scale[f] = if hi > lo {
        (nb - 1).to_double() / (hi - lo)
      } else {
        0.0
      }
    }
    bucket_top[f] = nb - 1
    let count = FixedArray::make(nb, 0)
    for i in cut_start[f]..<cut_end {
      let b = bucket_of(
        key_value(cuts[i]).to_double(),
        bucket_lo[f],
        bucket_scale[f],
        nb - 1,
      )
      count[b] += 1
    }
    let occ = count.fold(init=0, (a, n) => if n > a { n } else { a })
    bucket_scan[f] = if !USE_BUCKETS || occ > MAX_SCAN { -1 } else { occ }
    table_start[f] = table.length()
    let mut before = 0
    for b in 0..<nb {
      table.push(before)
      before += count[b]
    }
    bcut_start[f] = bcut.length()
    for i in cut_start[f]..<cut_end {
      bcut.push(key_value(cuts[i]).to_double())
    }
    for _ in 0..<occ {
      bcut.push(0.0 / 0.0)
    }
  }
  let table = FixedArray::from_array(table)
  let bcut = FixedArray::from_array(bcut)
  let total = trees.fold(init=0, (acc, t) => acc + t.num_nodes())
  let nodes = Nodes::make(total)
  let leaf_value = FixedArray::make(total, (0.0 : Float))
  let roots = FixedArray::make(trees.length(), 0)
  let depth = FixedArray::make(trees.length(), 0)
  let mut base = 0
  let queue : Array[(Int32, Int32, Int32)] = [] // (old id, new id, depth)
  let words : Array[Array[Int32]] = [] // per tree, the word of each old id
  for ti, t in trees {
    let word_of = Array::make(t.num_nodes(), 0)
    words.push(word_of)
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
      word_of[old] = (slot << 16) | (k - cut_start[f])
      nodes.set(id, next, word_of[old])
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
  let d_all = depth.fold(init=0, (a, d) => if d > a { d } else { a })
  let heap_size = if d_all < 20 { 1 << d_all } else { 0 }
  let use_heap = USE_HEAP &&
    d_all >= 2 &&
    heap_size > 0 &&
    trees.length().to_int64() * heap_size.to_int64() <=
    HEAP_FACTOR.to_int64() * total.to_int64()
  let (heap, heap_leaf, heap_depth) = if use_heap {
    let heap = Buf::new(trees.length() * heap_size)
    let heap_leaf = FBuf::new(trees.length() * heap_size)
    let stack : Array[(Int32, Int32, Int32)] = [] // (old id, heap id, depth)
    for ti, t in trees {
      let b = ti * heap_size
      stack.push((0, 1, 0))
      while stack.pop() is Some((old, h, d)) {
        if t.left[old] != -1 {
          heap.set(b + h, words[ti][old])
          stack.push((t.left[old], 2 * h, d + 1))
          stack.push((t.right[old], 2 * h + 1, d + 1))
        } else if d < d_all {
          // a shallow leaf: the leaf threshold always goes left
          heap.set(b + h, LEAF_THR)
          stack.push((old, 2 * h, d + 1))
        } else {
          heap_leaf.set(b + h - heap_size, t.value[old])
        }
      }
    }
    (heap, heap_leaf, d_all)
  } else {
    (Buf::new(0), FBuf::new(0), 0)
  }
  Some({
    heap,
    heap_leaf,
    heap_depth,
    heap_size,
    // a big forest is streamed once per block, so bigger blocks pay off:
    // heap forests over about 4 MB in V8, compact ones over 2 MB on wasm
    block: if (
        heap_depth > 0 &&
        trees.length().to_int64() * heap_size.to_int64() * 8L > 0x400000L
      ) ||
      (STREAM_BLOCK && heap_depth == 0 && total.to_int64() * 8L > 0x200000L) {
      256
    } else {
      BLOCK
    },
    // the heap copy replaces the compact one, so drop it
    nodes: if heap_depth > 0 {
      Nodes::make(0)
    } else {
      nodes
    },
    leaf_value: if heap_depth > 0 {
      []
    } else {
      leaf_value
    },
    roots,
    depth,
    search,
    search_start,
    search_half,
    bucket_lo,
    bucket_scale,
    bucket_top,
    bucket_scan,
    table_start,
    table,
    bcut,
    bcut_start,
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
/// `ranks` for rows `start .. start + n` of a row-major matrix with `cols`
/// columns, rank slots of row `r` at `out[r * 2 * num_feature]`. Works one
/// feature at a time, so each feature's table is looked up once per block.
fn Lockstep::ranks_block(
  self : Lockstep,
  data : Array[Float],
  cols : Int32,
  start : Int32,
  n : Int32,
  out : Buf,
) -> Unit {
  let width = 2 * self.num_feature
  for f in 0..<self.num_feature {
    let slot = 2 * f
    if f >= cols {
      for r in 0..<n {
        out.set(r * width + slot, 0)
        out.set(r * width + slot + 1, RANK_HIGH)
      }
      continue
    }
    let scan = self.bucket_scan.unsafe_get(f)
    if scan >= 0 {
      let lo = self.bucket_lo.unsafe_get(f)
      let scale = self.bucket_scale.unsafe_get(f)
      let top = self.bucket_top.unsafe_get(f)
      let ts = self.table_start.unsafe_get(f)
      let bs = self.bcut_start.unsafe_get(f)
      for r in 0..<n {
        let x = data[(start + r) * cols + f]
        let o = r * width + slot
        if x.is_nan() {
          out.set(o, 0)
          out.set(o + 1, RANK_HIGH)
        } else {
          let xd = x.to_double()
          let r0 = self.table.unsafe_get(ts + bucket_of(xd, lo, scale, top))
          let mut k = r0
          for j in 0..<scan {
            // a threshold equal to x counts: x >= t goes right
            if self.bcut.unsafe_get(bs + r0 + j) <= xd {
              k = k + 1
            }
          }
          out.set(o, k)
          out.set(o + 1, k)
        }
      }
    } else {
      let s = self.search
      let lo = self.search_start.unsafe_get(f)
      let half = self.search_half.unsafe_get(f)
      for r in 0..<n {
        let x = data[(start + r) * cols + f]
        let o = r * width + slot
        if x.is_nan() {
          out.set(o, 0)
          out.set(o + 1, RANK_HIGH)
        } else {
          let key = float_key(x)
          let mut b = lo
          let mut step = half
          while step > 0 {
            // in bounds: b + step - 1 < lo + 2 * half - 1, the padded length
            b = b + advance(s.unsafe_get(b + step - 1), key, step)
            step = step / 2
          }
          out.set(o, b - lo)
          out.set(o + 1, b - lo)
        }
      }
    }
  }
}

///|
/// Write the two rank slots of every feature of `row` to
/// `out[at : at + 2 * num_feature]`.
fn Lockstep::ranks(
  self : Lockstep,
  row : ArrayView[Float],
  out : Buf,
  at : Int32,
) -> Unit {
  for f in 0..<self.num_feature {
    let x = if f < row.length() {
      row[f]
    } else {
      Float::from_double(0.0) / 0.0
    }
    if x.is_nan() {
      out.set(at + 2 * f, 0)
      out.set(at + 2 * f + 1, RANK_HIGH)
    } else {
      // number of thresholds <= x
      let scan = self.bucket_scan.unsafe_get(f)
      if scan >= 0 {
        let xd = x.to_double()
        let b = bucket_of(
          xd,
          self.bucket_lo.unsafe_get(f),
          self.bucket_scale.unsafe_get(f),
          self.bucket_top.unsafe_get(f),
        )
        let r0 = self.table.unsafe_get(self.table_start.unsafe_get(f) + b)
        let at_cut = self.bcut_start.unsafe_get(f) + r0
        let mut r = r0
        for j in 0..<scan {
          // a threshold equal to x counts: x >= t goes right
          if self.bcut.unsafe_get(at_cut + j) <= xd {
            r = r + 1
          }
        }
        out.set(at + 2 * f, r)
        out.set(at + 2 * f + 1, r)
        continue
      }
      let key = float_key(x)
      let s = self.search
      let lo = self.search_start[f]
      let mut b = lo
      let mut step = self.search_half[f]
      while step > 0 {
        // in bounds: b + step - 1 < lo + 2 * half - 1, the padded length
        b = b + advance(s.unsafe_get(b + step - 1), key, step)
        step = step / 2
      }
      let r = b - lo
      out.set(at + 2 * f, r)
      out.set(at + 2 * f + 1, r)
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
  ranks : Buf,
  n : Int32,
  idx : Buf,
  margin : FixedArray[Float],
) -> Unit {
  if self.heap_depth > 0 {
    self.accumulate_heap(tree_group, groups, ranks, n, idx, margin)
    return
  }
  let width = 2 * self.num_feature
  let nodes = self.nodes
  let leaf_value = self.leaf_value
  let trees = self.roots.length()
  if n == 1 {
    nodes.single(self.roots, self.depth, tree_group, ranks, leaf_value, margin)
    return
  }
  for t = 0; t < trees; {
    let root = self.roots[t]
    let d = self.depth[t]
    let g = tree_group[t]
    if t + 1 < trees && d >= 2 && self.depth[t + 1] >= 2 {
      // Two trees walk side by side through the same rows: one loop, one
      // row offset and twice the independent loads per iteration. The
      // shallower tree's rows wait in their self-looping leaves.
      let root2 = self.roots[t + 1]
      let d2 = self.depth[t + 1]
      let g2 = tree_group[t + 1]
      let d = if d2 > d { d2 } else { d }
      nodes.walk_pair(
        root, root2, d, ranks, width, n, idx, leaf_value, margin, groups, g, g2,
      )
      continue t + 2
    }
    // The first level starts every row at the root, so it writes `idx`
    // directly instead of filling it; the last level feeds the leaf straight
    // into the margin instead of storing the node id.
    if d >= 2 {
      for r in 0..<n {
        idx.set(r, nodes.step(root, ranks, r * width))
      }
      for _ in 2..<d {
        for r in 0..<n {
          // In bounds by construction: `build` stores only node ids of this
          // forest and slots below `width`, and `ranks` holds `n` rows, so
          // `step` reads without bounds checks.
          let i = idx.get(r)
          idx.set(r, nodes.step(i, ranks, r * width))
        }
      }
    }
    for r in 0..<n {
      // in bounds: `margin` holds `n * groups` values and `idx` holds nodes
      let i = if d >= 2 { idx.get(r) } else { root }
      let leaf = if d >= 1 { nodes.step(i, ranks, r * width) } else { root }
      let k = r * groups + g
      margin.unsafe_set(k, margin.unsafe_get(k) + leaf_value.unsafe_get(leaf))
    }
    continue t + 1
  }
}

///|
/// `accumulate` over the implicit-heap copy, two trees per pass.
fn Lockstep::accumulate_heap(
  self : Lockstep,
  tree_group : Array[Int32],
  groups : Int32,
  ranks : Buf,
  n : Int32,
  idx : Buf,
  margin : FixedArray[Float],
) -> Unit {
  let width = 2 * self.num_feature
  let heap = self.heap
  let leaf = self.heap_leaf
  let d = self.heap_depth
  let size = self.heap_size
  let trees = self.roots.length()
  if n == 1 {
    // one row: walk trees to their leaves side by side, no `idx` traffic
    let rb = 0
    let mut t = 0
    while t + 15 < trees {
      let b0 = (t + 0) * size
      let mut x0 = 1
      let b1 = (t + 1) * size
      let mut x1 = 1
      let b2 = (t + 2) * size
      let mut x2 = 1
      let b3 = (t + 3) * size
      let mut x3 = 1
      let b4 = (t + 4) * size
      let mut x4 = 1
      let b5 = (t + 5) * size
      let mut x5 = 1
      let b6 = (t + 6) * size
      let mut x6 = 1
      let b7 = (t + 7) * size
      let mut x7 = 1
      let b8 = (t + 8) * size
      let mut x8 = 1
      let b9 = (t + 9) * size
      let mut x9 = 1
      let b10 = (t + 10) * size
      let mut x10 = 1
      let b11 = (t + 11) * size
      let mut x11 = 1
      let b12 = (t + 12) * size
      let mut x12 = 1
      let b13 = (t + 13) * size
      let mut x13 = 1
      let b14 = (t + 14) * size
      let mut x14 = 1
      let b15 = (t + 15) * size
      let mut x15 = 1
      for _ in 0..<d {
        x0 = {
          let w = heap.get(b0 + x0)
          (x0 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x1 = {
          let w = heap.get(b1 + x1)
          (x1 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x2 = {
          let w = heap.get(b2 + x2)
          (x2 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x3 = {
          let w = heap.get(b3 + x3)
          (x3 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x4 = {
          let w = heap.get(b4 + x4)
          (x4 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x5 = {
          let w = heap.get(b5 + x5)
          (x5 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x6 = {
          let w = heap.get(b6 + x6)
          (x6 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x7 = {
          let w = heap.get(b7 + x7)
          (x7 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x8 = {
          let w = heap.get(b8 + x8)
          (x8 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x9 = {
          let w = heap.get(b9 + x9)
          (x9 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x10 = {
          let w = heap.get(b10 + x10)
          (x10 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x11 = {
          let w = heap.get(b11 + x11)
          (x11 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x12 = {
          let w = heap.get(b12 + x12)
          (x12 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x13 = {
          let w = heap.get(b13 + x13)
          (x13 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x14 = {
          let w = heap.get(b14 + x14)
          (x14 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x15 = {
          let w = heap.get(b15 + x15)
          (x15 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
      }
      let g = tree_group[t + 0]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b0 + x0 - size))
      let g = tree_group[t + 1]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b1 + x1 - size))
      let g = tree_group[t + 2]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b2 + x2 - size))
      let g = tree_group[t + 3]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b3 + x3 - size))
      let g = tree_group[t + 4]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b4 + x4 - size))
      let g = tree_group[t + 5]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b5 + x5 - size))
      let g = tree_group[t + 6]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b6 + x6 - size))
      let g = tree_group[t + 7]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b7 + x7 - size))
      let g = tree_group[t + 8]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b8 + x8 - size))
      let g = tree_group[t + 9]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b9 + x9 - size))
      let g = tree_group[t + 10]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b10 + x10 - size))
      let g = tree_group[t + 11]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b11 + x11 - size))
      let g = tree_group[t + 12]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b12 + x12 - size))
      let g = tree_group[t + 13]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b13 + x13 - size))
      let g = tree_group[t + 14]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b14 + x14 - size))
      let g = tree_group[t + 15]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b15 + x15 - size))
      t += 16
    }
    while t + 3 < trees {
      let b0 = (t + 0) * size
      let mut x0 = 1
      let b1 = (t + 1) * size
      let mut x1 = 1
      let b2 = (t + 2) * size
      let mut x2 = 1
      let b3 = (t + 3) * size
      let mut x3 = 1
      for _ in 0..<d {
        x0 = {
          let w = heap.get(b0 + x0)
          (x0 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x1 = {
          let w = heap.get(b1 + x1)
          (x1 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x2 = {
          let w = heap.get(b2 + x2)
          (x2 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        x3 = {
          let w = heap.get(b3 + x3)
          (x3 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
      }
      let g = tree_group[t + 0]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b0 + x0 - size))
      let g = tree_group[t + 1]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b1 + x1 - size))
      let g = tree_group[t + 2]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b2 + x2 - size))
      let g = tree_group[t + 3]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b3 + x3 - size))
      t += 4
    }
    while t + 0 < trees {
      let b0 = (t + 0) * size
      let mut x0 = 1
      for _ in 0..<d {
        x0 = {
          let w = heap.get(b0 + x0)
          (x0 << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
      }
      let g = tree_group[t + 0]
      margin.unsafe_set(g, margin.unsafe_get(g) + leaf.get(b0 + x0 - size))
      t += 1
    }
    return
  }
  for t = 0; t < trees; {
    let ba = t * size
    let ga = tree_group[t]
    if t + 1 < trees {
      let bb = ba + size
      let gb = tree_group[t + 1]
      for r in 0..<n {
        let rb = r * width
        idx.set(
          r,
          {
            let i = 1
            let w = heap.get(ba + i)
            (i << 1) |
            (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
            31).reinterpret_as_int()
          },
        )
        idx.set(
          n + r,
          {
            let i = 1
            let w = heap.get(bb + i)
            (i << 1) |
            (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
            31).reinterpret_as_int()
          },
        )
      }
      for _ in 2..<d {
        for r in 0..<n {
          let rb = r * width
          idx.set(
            r,
            {
              let i = idx.get(r)
              let w = heap.get(ba + i)
              (i << 1) |
              (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
              31).reinterpret_as_int()
            },
          )
          idx.set(
            n + r,
            {
              let i = idx.get(n + r)
              let w = heap.get(bb + i)
              (i << 1) |
              (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
              31).reinterpret_as_int()
            },
          )
        }
      }
      for r in 0..<n {
        let rb = r * width
        let la = {
          let i = idx.get(r)
          let w = heap.get(ba + i)
          (i << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        let lb = {
          let i = idx.get(n + r)
          let w = heap.get(bb + i)
          (i << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        }
        let k = r * groups + ga
        margin.unsafe_set(k, margin.unsafe_get(k) + leaf.get(ba + la - size))
        let k = r * groups + gb
        margin.unsafe_set(k, margin.unsafe_get(k) + leaf.get(bb + lb - size))
      }
      continue t + 2
    }
    for r in 0..<n {
      let rb = r * width
      idx.set(
        r,
        {
          let i = 1
          let w = heap.get(ba + i)
          (i << 1) |
          (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
          31).reinterpret_as_int()
        },
      )
    }
    for _ in 2..<d {
      for r in 0..<n {
        let rb = r * width
        idx.set(
          r,
          {
            let i = idx.get(r)
            let w = heap.get(ba + i)
            (i << 1) |
            (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >>
            31).reinterpret_as_int()
          },
        )
      }
    }
    for r in 0..<n {
      let rb = r * width
      let la = {
        let i = idx.get(r)
        let w = heap.get(ba + i)
        (i << 1) |
        (((w & 0xFFFF) - ranks.get(rb + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      let k = r * groups + ga
      margin.unsafe_set(k, margin.unsafe_get(k) + leaf.get(ba + la - size))
    }
    continue t + 1
  }
}

///|
const MAX_BUCKETS : Int32 = 0x10000

///|
const MAX_SCAN : Int32 = 4

///|
/// Bucket of `x` (not NaN) in a feature's table; monotone in `x`.
fn bucket_of(x : Double, lo : Double, scale : Double, top : Int32) -> Int32 {
  let t = (x - lo) * scale
  if t > 0.0 {
    if t < top.to_double() {
      t.to_int()
    } else {
      top
    }
  } else {
    0
  }
}

///|
/// The threshold value whose `float_key` is `k`.
fn key_value(k : Int32) -> Float {
  Float::reinterpret_from_int(if k >= 0 { k } else { -k | (1 << 31) })
}
