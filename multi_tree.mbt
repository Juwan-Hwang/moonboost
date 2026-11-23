// Trees with vector leaves (XGBoost's `multi_strategy: "multi_output_tree"`):
// with num_target outputs, each round grows one tree whose splits serve all
// outputs and whose leaves hold one value per output. A split's gain is the
// sum over outputs of the usual gain, from histograms that keep a gradient
// and hessian sum per output in each bin.

///|
/// A vector-leaf tree as it grows, in XGBoost's node arrays; base_weight
/// holds k values per node.
priv struct MultiTree {
  left : Array[Int32]
  right : Array[Int32]
  parent : Array[Int32]
  split_index : Array[Int32]
  split_cond : Array[Float]
  default_left : Array[Bool]
  base_weight : Array[Float]
  loss_change : Array[Float]
  sum_hess : Array[Float]
}

///|
priv struct MultiSplit {
  gain : Double
  feature : Int32
  bin : Int32 // left takes bins 0 ..= bin, as in Split
  default_left : Bool
  // the left child's gradient and hessian sum per output, 2j and 2j + 1
  left_gh : FixedArray[Double]
}

///|
priv struct MultiGrower {
  p : TrainParams
  k : Int32 // outputs
  cols : Int32
  // histogram slot of feature f for row r at r * cols + f
  slots : FixedArray[UInt16]
  hist_start : FixedArray[Int32]
  cut_start : FixedArray[Int32]
  cuts : Array[Float]
  nbins : Int32
  // gradient and hessian of output j for row r at r * 2k + 2j (+ 1)
  gh : FixedArray[Double]
  // the same pair quantised and packed as (g << 32) + h at r * k + j, so
  // a histogram slot of one output is one integer add; per-output scales
  gq : FixedArray[Int64]
  igs : FixedArray[Double]
  ihs : FixedArray[Double]
  rows : FixedArray[Int32]
  scratch : FixedArray[Int32]
  // leaf reached by each row
  leaf_at : FixedArray[Int32]
}

///|
fn MultiGrower::histogram(
  self : MultiGrower,
  lo : Int32,
  hi : Int32,
) -> FixedArray[Int64] {
  let k = self.k
  let hist = FixedArray::make(self.nbins * k, 0L)
  let cols = self.cols
  let slots = self.slots
  let gq = self.gq
  let rows = self.rows
  // indices are in range: rows hold rows and slots are below nbins
  if k == 2 {
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let a = gq.unsafe_get(r * 2)
      let b = gq.unsafe_get(r * 2 + 1)
      let at = r * cols
      for f in 0..<cols {
        let base = slots.unsafe_get(at + f).to_int() * 2
        hist.unsafe_set(base, hist.unsafe_get(base) + a)
        hist.unsafe_set(base + 1, hist.unsafe_get(base + 1) + b)
      }
    }
  } else if k == 3 {
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let a = gq.unsafe_get(r * 3)
      let b = gq.unsafe_get(r * 3 + 1)
      let c = gq.unsafe_get(r * 3 + 2)
      let at = r * cols
      for f in 0..<cols {
        let base = slots.unsafe_get(at + f).to_int() * 3
        hist.unsafe_set(base, hist.unsafe_get(base) + a)
        hist.unsafe_set(base + 1, hist.unsafe_get(base + 1) + b)
        hist.unsafe_set(base + 2, hist.unsafe_get(base + 2) + c)
      }
    }
  } else {
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let src = r * k
      let at = r * cols
      for f in 0..<cols {
        let base = slots.unsafe_get(at + f).to_int() * k
        for j in 0..<k {
          hist.unsafe_set(
            base + j,
            hist.unsafe_get(base + j) + gq.unsafe_get(src + j),
          )
        }
      }
    }
  }
  hist
}

///|
/// Adds the gradient and hessian sums of output j in a packed slot,
/// unscaled, to a[2j] and a[2j + 1].
fn MultiGrower::unpack_add(
  self : MultiGrower,
  a : FixedArray[Double],
  q : Int64,
  j : Int32,
) -> Unit {
  let low = q & 0xFFFFFFFFL
  a[2 * j] = a[2 * j] + ((q - low) >> 32).to_double() * self.igs[j]
  a[2 * j + 1] = a[2 * j + 1] + low.to_double() * self.ihs[j]
}

///|
/// Sum over outputs of g^2 / (h + lambda), for the sums in a[2j], a[2j + 1].
fn MultiGrower::score(self : MultiGrower, a : FixedArray[Double]) -> Double {
  let mut s = 0.0
  for j in 0..<self.k {
    let g = a[2 * j]
    s += g * g / (a[2 * j + 1] + self.p.lambda)
  }
  s
}

///|
/// Whether every output's hessian sum reaches min_child_weight.
fn MultiGrower::heavy(self : MultiGrower, a : FixedArray[Double]) -> Bool {
  for j in 0..<self.k {
    if a[2 * j + 1] < self.p.min_child_weight {
      return false
    }
  }
  true
}

///|
fn MultiGrower::best_split(
  self : MultiGrower,
  hist : FixedArray[Int64],
  total : FixedArray[Double],
) -> MultiSplit? {
  let k = self.k
  let w = 2 * k
  let parent = self.score(total)
  let mut best : MultiSplit? = None
  let mut best_gain = self.p.gamma
  let acc = FixedArray::make(k, 0L)
  let missing = FixedArray::make(w, 0.0)
  let left = FixedArray::make(w, 0.0)
  let right = FixedArray::make(w, 0.0)
  for f in 0..<self.cols {
    // slot hist_start[f] holds the feature's missing values, then its bins
    let miss = self.hist_start[f] * k
    let start = self.hist_start[f] + 1
    let n = self.cut_start[f + 1] - self.cut_start[f] + 1
    if n < 2 {
      continue
    }
    missing.fill(0.0)
    let mut mq = 0L
    for j in 0..<k {
      self.unpack_add(missing, hist[miss + j], j)
      mq = mq | hist[miss + j]
    }
    acc.fill(0L)
    for b in 0..<(n - 1) {
      let at = (start + b) * k
      let mut any = 0L
      for j in 0..<k {
        acc[j] = acc[j] + hist[at + j]
        any = any | hist[at + j]
      }
      // an empty bin repeats the previous split
      if any == 0L {
        continue
      }
      left.fill(0.0)
      for j in 0..<k {
        self.unpack_add(left, acc[j], j)
      }
      // missing values to the right, then to the left
      for side in 0..<2 {
        if side == 1 {
          if mq == 0L {
            break
          }
          for j in 0..<w {
            left[j] = left[j] + missing[j]
          }
        }
        for j in 0..<w {
          right[j] = total[j] - left[j]
        }
        if self.heavy(left) && self.heavy(right) {
          let gain = self.score(left) + self.score(right) - parent
          if gain > best_gain {
            best_gain = gain
            best = Some({
              gain,
              feature: f,
              bin: b,
              default_left: side == 1,
              left_gh: FixedArray::makei(w, j => left[j]),
            })
          }
        }
      }
    }
  }
  best
}

///|
/// Adds a node with the leaf values -g / (h + lambda) * eta of its sums.
fn MultiGrower::add(
  self : MultiGrower,
  t : MultiTree,
  parent : Int32,
  sums : FixedArray[Double],
) -> Int32 {
  t.left.push(-1)
  t.right.push(-1)
  t.parent.push(parent)
  t.split_index.push(0)
  t.split_cond.push(0.0)
  t.default_left.push(false)
  let mut hs = 0.0
  for j in 0..<self.k {
    let g = sums[2 * j]
    let h = sums[2 * j + 1]
    hs += h
    t.base_weight.push(
      Float::from_double(-g / (h + self.p.lambda) * self.p.eta),
    )
  }
  t.loss_change.push(0.0)
  t.sum_hess.push(Float::from_double(hs))
  t.left.length() - 1
}

///|
/// Moves rows[lo:hi] going left to the front, keeping their order; returns
/// where the right part starts.
fn MultiGrower::partition(
  self : MultiGrower,
  lo : Int32,
  hi : Int32,
  s : MultiSplit,
) -> Int32 {
  let rows = self.rows
  let dst = self.scratch
  let slots = self.slots
  let cols = self.cols
  let f = s.feature
  let sub = self.hist_start[f]
  let limit = s.bin + 1
  let miss = if s.default_left { 0 } else { 1 << 20 }
  let mut a = lo
  let mut b = hi - 1
  // indices are in range: rows[lo:hi] are rows
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let bin = slots.unsafe_get(r * cols + f).to_int() - sub
    let key = bin + (((bin - 1) >> 31) & miss)
    let left = if key <= limit { 1 } else { 0 }
    dst.unsafe_set(a, r)
    dst.unsafe_set(b, r)
    a += left
    b -= 1 - left
  }
  // left rows in order, then right rows back in order
  for i in lo..<a {
    rows.unsafe_set(i, dst.unsafe_get(i))
  }
  for i in a..<hi {
    rows.unsafe_set(i, dst.unsafe_get(hi - 1 - (i - a)))
  }
  a
}

///|
fn MultiGrower::grow(
  self : MultiGrower,
  t : MultiTree,
  node : Int32,
  lo : Int32,
  hi : Int32,
  hist : FixedArray[Int64],
  sums : FixedArray[Double],
  depth : Int32,
) -> Unit {
  let split = if depth < self.p.max_depth && hi - lo >= 2 {
    self.best_split(hist, sums)
  } else {
    None
  }
  let k = self.k
  guard split is Some(s) else {
    // the leaf's values are set from exact sums once the tree is grown
    let rows = self.rows
    let leaf_at = self.leaf_at
    // indices are in range: rows[lo:hi] are rows
    for i in lo..<hi {
      leaf_at.unsafe_set(rows.unsafe_get(i), node)
    }
    return
  }
  let mid = self.partition(lo, hi, s)
  let w = 2 * k
  let lsum = s.left_gh
  let rsum = FixedArray::makei(w, j => sums[j] - lsum[j])
  let l = self.add(t, node, lsum)
  let r = self.add(t, node, rsum)
  t.left[node] = l
  t.right[node] = r
  t.split_index[node] = s.feature
  t.split_cond[node] = self.cuts[self.cut_start[s.feature] + s.bin]
  t.default_left[node] = s.default_left
  t.loss_change[node] = Float::from_double(s.gain)
  // the smaller child's histogram is built, the larger's is the parent's
  // minus it; children at the depth limit need none
  let (lh, rh) = if depth + 1 >= self.p.max_depth {
    (hist, hist)
  } else {
    let left_small = mid - lo <= hi - mid
    let small = if left_small {
      self.histogram(lo, mid)
    } else {
      self.histogram(mid, hi)
    }
    for i in 0..<hist.length() {
      hist[i] = hist[i] - small[i]
    }
    if left_small {
      (small, hist)
    } else {
      (hist, small)
    }
  }
  self.grow(t, l, lo, mid, lh, lsum, depth + 1)
  self.grow(t, r, mid, hi, rh, rsum, depth + 1)
}

///|
/// Grows `p.num_round` vector-leaf trees on the binned data, updating
/// margin (k values per row) and writing each tree's JSON to `out`.
fn train_multi_trees(
  binned : Binned,
  hist_start : FixedArray[Int32],
  n : Int32,
  label : ArrayView[Float],
  margin : FixedArray[Float],
  p : TrainParams,
  kind : Int32,
  out : StringBuilder,
) -> Unit {
  let k = p.num_target
  let cols = binned.cols
  let slots = FixedArray::make(n * cols, (0 : UInt16))
  for f in 0..<cols {
    for r in 0..<n {
      slots[r * cols + f] = (binned.by_col.unsafe_get(f * n + r).to_int() +
      hist_start[f]).to_uint16()
    }
  }
  let gr : MultiGrower = {
    p,
    k,
    cols,
    slots,
    hist_start,
    cut_start: binned.cut_start,
    cuts: binned.cuts,
    nbins: hist_start[cols],
    gh: FixedArray::make(n * 2 * k, 0.0),
    gq: FixedArray::make(n * k, 0L),
    igs: FixedArray::make(k, 1.0),
    ihs: FixedArray::make(k, 1.0),
    rows: FixedArray::make(n, 0),
    scratch: FixedArray::make(n, 0),
    leaf_at: FixedArray::make(n, 0),
  }
  let w = 2 * k
  for round in 0..<p.num_round {
    let sums = FixedArray::make(w, 0.0)
    let gh = gr.gh
    // indices are in range: margin, label and gh hold n rows of k outputs
    for j in 0..<k {
      let mut gs = 0.0
      let mut hs = 0.0
      for r in 0..<n {
        let m = margin.unsafe_get(r * k + j).to_double()
        let y = label.unsafe_get(r * k + j).to_double()
        let mut g = m - y
        let mut h = 1.0
        if kind == 1 {
          let pr = 1.0 / (1.0 + train_exp(-m))
          g = pr - y
          h = pr * (1.0 - pr)
          h = if h < 1.0e-16 { 1.0e-16 } else { h }
        } else if kind != 0 {
          // pseudo-Huber
          let d = p.huber_slope
          let sq = (1.0 + g / d * (g / d)).sqrt()
          h = d * d / ((d * d + g * g) * sq)
          g = g / sq
        }
        // stored in float32, as XGBoost keeps gradients
        let g = Float::from_double(g).to_double()
        let h = Float::from_double(h).to_double()
        gh.unsafe_set(r * w + 2 * j, g)
        gh.unsafe_set(r * w + 2 * j + 1, h)
        gs += g
        hs += h
      }
      sums[2 * j] = gs
      sums[2 * j + 1] = hs
    }
    for r in 0..<n {
      gr.rows[r] = r
    }
    let t : MultiTree = {
      left: [],
      right: [],
      parent: [],
      split_index: [],
      split_cond: [],
      default_left: [],
      base_weight: [],
      loss_change: [],
      sum_hess: [],
    }
    // quantise each output's pairs so the sums over all n rows stay inside
    // 31 bits for g and 32 bits for h
    let lg = (2147483647 / n - 1).to_double()
    for j in 0..<k {
      let mut gmax = 0.0
      let mut hmax = 0.0
      for r in 0..<n {
        let g = gh.unsafe_get(r * w + 2 * j).abs()
        let h = gh.unsafe_get(r * w + 2 * j + 1)
        if g > gmax {
          gmax = g
        }
        if h > hmax {
          hmax = h
        }
      }
      let gs = if gmax > 0.0 { lg / gmax } else { 1.0 }
      let hs = if hmax > 0.0 { lg * 2.0 / hmax } else { 1.0 }
      gr.igs[j] = 1.0 / gs
      gr.ihs[j] = 1.0 / hs
      for r in 0..<n {
        let x = gr.gh[r * w + 2 * j] * gs
        // 0.5 with the sign of x, from its bits: the sign is unpredictable
        let half = ((x.reinterpret_as_int64() & (1L << 63)) |
        0x3FE0000000000000L).reinterpret_as_double()
        let gi = (x + half).to_int64()
        let hi = (gr.gh[r * w + 2 * j + 1] * hs + 0.5).to_int64()
        gr.gq[r * k + j] = (gi << 32) + hi
      }
    }
    let hist = gr.histogram(0, n)
    // the quantised totals, from feature 0's slots, which hold every row
    let qsums = FixedArray::make(w, 0.0)
    for j in 0..<k {
      let mut q = 0L
      for slot in hist_start[0]..<hist_start[1] {
        q = q + hist[slot * k + j]
      }
      gr.unpack_add(qsums, q, j)
    }
    let root = gr.add(t, -1, sums)
    gr.grow(t, root, 0, n, hist, qsums, 0)
    // each leaf's values from its rows' exact gradient sums, in row order
    let nodes = t.left.length()
    let exact = FixedArray::make(nodes * w, 0.0)
    let leaf_at = gr.leaf_at
    // indices are in range: leaf_at holds nodes of t
    for r in 0..<n {
      let at = leaf_at.unsafe_get(r) * w
      for j in 0..<w {
        exact.unsafe_set(
          at + j,
          exact.unsafe_get(at + j) + gh.unsafe_get(r * w + j),
        )
      }
    }
    for v in 0..<nodes {
      if t.left[v] == -1 {
        for j in 0..<k {
          t.base_weight[v * k + j] = Float::from_double(
            -exact[v * w + 2 * j] /
            (exact[v * w + 2 * j + 1] + p.lambda) *
            p.eta,
          )
        }
      }
    }
    let bw = t.base_weight
    for r in 0..<n {
      let at = leaf_at.unsafe_get(r) * k
      for j in 0..<k {
        margin.unsafe_set(r * k + j, margin.unsafe_get(r * k + j) + bw[at + j])
      }
    }
    if round > 0 {
      out.write_char(',')
    }
    multi_tree_json(out, t, round, cols, k)
  }
}

///|
/// A vector-leaf tree in XGBoost's JSON layout: a leaf's right child is
/// its index among the leaves, whose values are in leaf_weights.
fn multi_tree_json(
  buf : StringBuilder,
  t : MultiTree,
  id : Int32,
  num_feature : Int32,
  k : Int32,
) -> Unit {
  let n = t.left.length()
  let right = []
  let leaf_weights = []
  for i in 0..<n {
    if t.left[i] == -1 {
      right.push(leaf_weights.length() / k)
      for j in 0..<k {
        leaf_weights.push(t.base_weight[i * k + j])
      }
    } else {
      right.push(t.right[i])
    }
  }
  buf.write_string("{\"base_weights\":")
  json_floats(buf, t.base_weight)
  buf.write_string(
    ",\"categories\":[],\"categories_nodes\":[],\"categories_segments\":[],\"categories_sizes\":[],\"default_left\":",
  )
  json_ints(buf, t.default_left.map(b => if b { 1 } else { 0 }))
  buf.write_string(",\"id\":\{id},\"leaf_weights\":")
  json_floats(buf, leaf_weights)
  buf.write_string(",\"left_children\":")
  json_ints(buf, t.left)
  buf.write_string(",\"loss_changes\":")
  json_floats(buf, t.loss_change)
  buf.write_string(",\"parents\":")
  json_ints(buf, t.parent)
  buf.write_string(",\"right_children\":")
  json_ints(buf, right)
  buf.write_string(",\"split_conditions\":")
  json_floats(buf, t.split_cond)
  buf.write_string(",\"split_indices\":")
  json_ints(buf, t.split_index)
  buf.write_string(",\"split_type\":")
  json_ints(buf, Array::make(n, 0))
  buf.write_string(",\"sum_hessian\":")
  json_floats(buf, t.sum_hess)
  buf.write_string(
    ",\"tree_param\":{\"num_deleted\":\"0\",\"num_feature\":\"\{num_feature}\",\"num_nodes\":\"\{n}\",\"size_leaf_vector\":\"\{k}\"}}",
  )
}
