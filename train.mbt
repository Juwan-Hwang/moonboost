// Gradient boosting training with histograms, writing XGBoost JSON models.
//
// Each feature is cut into at most `max_bin` bins at observed values; trees
// grow depth-wise on per-node gradient histograms (the smaller child is
// built, its sibling is the parent minus it), and every split learns where
// missing values go. Trained models are ordinary XGBoost JSON documents, so
// XGBoost itself loads them and predicts exactly what moonboost predicts.

///|
/// Training settings, named as in XGBoost.
pub(all) struct TrainParams {
  /// `reg:squarederror`, `binary:logistic`, `multi:softprob` or
  /// `multi:softmax`
  objective : String
  /// number of classes, for the `multi:` objectives
  num_class : Int32
  num_round : Int32
  max_depth : Int32
  /// learning rate
  eta : Double
  /// L2 regularisation on leaf weights
  lambda : Double
  /// minimum loss reduction to make a split (`min_split_loss`)
  gamma : Double
  min_child_weight : Double
  max_bin : Int32
}

///|
/// XGBoost's defaults: squared error, 10 rounds, depth 6, eta 0.3,
/// lambda 1, 256 bins.
pub fn TrainParams::default() -> TrainParams {
  {
    objective: "reg:squarederror",
    num_class: 0,
    num_round: 10,
    max_depth: 6,
    eta: 0.3,
    lambda: 1.0,
    gamma: 0.0,
    min_child_weight: 1.0,
    max_bin: 256,
  }
}

///|
/// Feature values cut into bins. `bins[r * cols + f]` is the histogram slot
/// of row `r` in feature `f`: `cut_start[f] + 2 * f` when missing, then one
/// per bin; `by_col[f * rows + r]` is one more than the bin (`0` missing); a value `v` falls in bin `k` when
/// exactly `k` of the feature's cut points are `<= v`.
priv struct Binned {
  cols : Int32
  bins : FixedArray[UInt16]
  // the same, column by column: by_col[f * rows + r]
  by_col : FixedArray[UInt16]
  // cut points of feature f are cuts[cut_start[f] .. cut_start[f + 1]]
  cuts : Array[Float]
  cut_start : FixedArray[Int32]
}

///|
fn bin_matrix(m : DenseMatrix, max_bin : Int32) -> Binned {
  let rows = m.rows
  let cols = m.cols
  let cuts : Array[Float] = []
  let cut_start = FixedArray::make(cols + 1, 0)
  let bins : FixedArray[UInt16] = FixedArray::make(rows * cols, 0)
  let by_col : FixedArray[UInt16] = FixedArray::make(rows * cols, 0)
  let vals : Array[Float] = []
  for f in 0..<cols {
    vals.clear()
    for r in 0..<rows {
      let v = m.data[r * cols + f]
      if !v.is_nan() {
        vals.push(v)
      }
    }
    vals.sort()
    let base = cuts.length()
    let n = vals.length()
    if n > 0 {
      // count distinct values
      let mut distinct = 1
      for i in 1..<n {
        if vals[i] != vals[i - 1] {
          distinct += 1
        }
      }
      if distinct <= max_bin {
        // every distinct value but the smallest opens a bin
        for i in 1..<n {
          if vals[i] != vals[i - 1] {
            cuts.push(vals[i])
          }
        }
      } else {
        // cut at evenly spaced ranks
        for b in 1..<max_bin {
          let c = vals[(b.to_int64() * n.to_int64() / max_bin.to_int64()).to_int()]
          if c > vals[0] &&
            (cuts.length() == base || cuts[cuts.length() - 1] < c) {
            cuts.push(c)
          }
        }
      }
    }
    cut_start[f + 1] = cuts.length()
    let lo = base
    let hi = cuts.length()
    for r in 0..<rows {
      let v = m.data[r * cols + f]
      if v.is_nan() {
        bins[r * cols + f] = (lo + 2 * f).to_uint16()
      } else {
        // number of cuts <= v
        let mut a = lo
        let mut b = hi
        while a < b {
          let mid = (a + b) / 2
          if cuts[mid] <= v {
            a = mid + 1
          } else {
            b = mid
          }
        }
        bins[r * cols + f] = (a + 2 * f + 1).to_uint16()
        by_col[f * rows + r] = (a - lo + 1).to_uint16()
      }
    }
  }
  { cols, bins, by_col, cuts, cut_start, }
}

///|
/// A tree as it grows, in XGBoost's node arrays.
priv struct GrowTree {
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
fn GrowTree::new() -> GrowTree {
  {
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
}

///|
fn GrowTree::add(
  self : GrowTree,
  parent : Int32,
  weight : Double,
  hess : Double,
) -> Int32 {
  self.left.push(-1)
  self.right.push(-1)
  self.parent.push(parent)
  self.split_index.push(0)
  self.split_cond.push(0.0)
  self.default_left.push(false)
  self.base_weight.push(Float::from_double(weight))
  self.loss_change.push(0.0)
  self.sum_hess.push(Float::from_double(hess))
  self.left.length() - 1
}

///|
priv struct Split {
  gain : Double
  feature : Int32
  bin : Int32 // left takes bins 0 ..= bin
  default_left : Bool
  gl : Double
  hl : Double
}

///|
priv struct Grower {
  data : Binned
  p : TrainParams
  // gradient and hessian of row r at 2r and 2r + 1
  gh : FixedArray[Float]
  // the same pair quantised to integers and packed as (g << 32) + h, so a
  // histogram slot is one integer add; gs and hs are the scales
  gq : FixedArray[Int64]
  mut gs : Double
  mut hs : Double
  mut igs : Double // 1 / gs
  // deeper subtrees work on their own contiguous copy of their rows' bins
  // and gradients (indexed by position, with the original row in `orig`),
  // so deep, scattered nodes stay cache local.
  // level 0 is the original arrays; a subtree is copied to the next level
  // when its depth reaches compact_at[level]
  compact_at : FixedArray[Int32]
  cbins : FixedArray[FixedArray[UInt16]]
  cq : FixedArray[FixedArray[Int64]]
  cgh : FixedArray[FixedArray[Float]]
  orig : FixedArray[FixedArray[Int32]]
  mut ihs : Double // 1 / hs
  rows : FixedArray[Int32]
  scratch : FixedArray[Int32]
  // leaf value reached by each row, filled as leaves are finalised
  leaf_of_row : FixedArray[Double]
  nbins : Int32 // total histogram slots: per feature, missing + (cuts + 1)
  hist_start : FixedArray[Int32]
}

///|
fn Grower::set_scales(
  self : Grower,
  n : Int32,
  gmax : Double,
  hmax : Double,
) -> Unit {
  // sums over all n rows stay inside 31 bits for g and 32 bits for h
  let lg = (2147483647 / n - 1).to_double()
  let lh = (2147483647 / n - 1).to_double() * 2.0
  self.gs = if gmax > 0.0 { lg / gmax } else { 1.0 }
  self.hs = if hmax > 0.0 { lh / hmax } else { 1.0 }
  self.igs = 1.0 / self.gs
  self.ihs = 1.0 / self.hs
}

///|
/// Quantises a gradient pair, rounding half away from zero (h is never
/// negative), and packs it as (g << 32) + h.
fn Grower::pack(self : Grower, g : Double, h : Double) -> Int64 {
  let x = g * self.gs
  let gi = (x + (if x < 0.0 { -0.5 } else { 0.5 })).to_int64()
  (gi << 32) + (h * self.hs + 0.5).to_int64()
}

///|
fn Grower::weight(self : Grower, g : Double, h : Double) -> Double {
  -g / (h + self.p.lambda)
}

///|
fn Grower::score(self : Grower, g : Double, h : Double) -> Double {
  g * g / (h + self.p.lambda)
}

///|
/// Gradient and hessian sums per bin for rows[lo:hi].
fn Grower::histogram(
  self : Grower,
  lo : Int32,
  hi : Int32,
  level : Int32,
) -> FixedArray[Int64] {
  let hist = FixedArray::make(self.nbins, 0L)
  let cols = self.data.cols
  let bins = if level > 0 { self.cbins[level - 1] } else { self.data.bins }
  let rows = self.rows
  let gq = if level > 0 { self.cq[level - 1] } else { self.gq }
  // indices are in range by construction: rows hold row numbers and bins
  // hold histogram slots
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let q = gq.unsafe_get(r)
    let base = r * cols
    let mut f = 0
    while f + 4 <= cols {
      let s0 = bins.unsafe_get(base + f).to_int()
      let s1 = bins.unsafe_get(base + f + 1).to_int()
      let s2 = bins.unsafe_get(base + f + 2).to_int()
      let s3 = bins.unsafe_get(base + f + 3).to_int()
      hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
      hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
      hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
      hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
      f += 4
    }
    while f < cols {
      let slot = bins.unsafe_get(base + f).to_int()
      hist.unsafe_set(slot, hist.unsafe_get(slot) + q)
      f += 1
    }
  }
  hist
}

///|
/// Copies the bins and gradients of rows[lo:hi] to positions lo..hi of the
/// next level's arrays, and makes rows[lo:hi] hold those positions.
fn Grower::compact(
  self : Grower,
  lo : Int32,
  hi : Int32,
  level : Int32,
) -> Unit {
  let cols = self.data.cols
  let (bins, q, gh) = if level > 0 {
    (self.cbins[level - 1], self.cq[level - 1], self.cgh[level - 1])
  } else {
    (self.data.bins, self.gq, self.gh)
  }
  let tb = self.cbins[level]
  let tq = self.cq[level]
  let tg = self.cgh[level]
  let to = self.orig[level]
  for i in lo..<hi {
    let r = self.rows[i]
    for f in 0..<cols {
      tb[i * cols + f] = bins[r * cols + f]
    }
    tq[i] = q[r]
    tg[i * 2] = gh[r * 2]
    tg[i * 2 + 1] = gh[r * 2 + 1]
    to[i] = if level > 0 { self.orig[level - 1][r] } else { r }
    self.rows[i] = i
  }
}

///|
/// Splits a packed histogram sum into its gradient and hessian.
fn Grower::unpack(self : Grower, q : Int64) -> (Double, Double) {
  let h = q & 0xFFFFFFFFL
  (((q - h) >> 32).to_double() * self.igs, h.to_double() * self.ihs)
}

///|
fn Grower::best_split(
  self : Grower,
  hist : FixedArray[Int64],
  g : Double,
  h : Double,
) -> Split? {
  let parent = self.score(g, h)
  let mcw = self.p.min_child_weight
  let mut best : Split? = None
  let mut best_gain = self.p.gamma
  for f in 0..<self.data.cols {
    // slot hist_start[f] holds the feature's missing values, then its bins
    let start = self.hist_start[f] + 1
    let n = self.data.cut_start[f + 1] - self.data.cut_start[f] + 1
    if n < 2 {
      continue
    }
    let (mg, mh) = self.unpack(hist[start - 1])
    let mut ql = 0L
    for b in 0..<(n - 1) {
      let q = hist[start + b]
      // an empty bin repeats the previous split, which cannot gain more
      if q == 0L {
        continue
      }
      ql = ql + q
      let (gl, hl) = self.unpack(ql)
      // missing values to the right
      let gr = g - gl
      let hr = h - hl
      if hl >= mcw && hr >= mcw {
        let gain = self.score(gl, hl) + self.score(gr, hr) - parent
        if gain > best_gain {
          best_gain = gain
          best = Some({ gain, feature: f, bin: b, default_left: false, gl, hl, })
        }
      }
      // missing values to the left
      if mh > 0.0 {
        let gl2 = gl + mg
        let hl2 = hl + mh
        let gr2 = g - gl2
        let hr2 = h - hl2
        if hl2 >= mcw && hr2 >= mcw {
          let gain = self.score(gl2, hl2) + self.score(gr2, hr2) - parent
          if gain > best_gain {
            best_gain = gain
            best = Some({
              gain,
              feature: f,
              bin: b,
              default_left: true,
              gl: gl2,
              hl: hl2,
            })
          }
        }
      }
    }
  }
  best
}

///|
/// Moves rows[lo:hi] going left to the front; returns where the right part
/// starts.
fn Grower::partition(
  self : Grower,
  lo : Int32,
  hi : Int32,
  s : Split,
  level : Int32,
) -> Int32 {
  let limit = s.bin + 1
  // the feature's bin of row r is bins[r * stride + column] - sub
  let (bins, stride, column, sub) = if level > 0 {
    (
      self.cbins[level - 1],
      self.data.cols,
      s.feature,
      self.hist_start[s.feature],
    )
  } else {
    (self.data.by_col, 1, s.feature * self.gh.length() / 2, 0)
  }
  // a missing value (bin 0) goes left as bin 0, or right as a bin past limit
  let miss = if s.default_left { 0 } else { 1 << 20 }
  let rows = self.rows
  let scratch = self.scratch
  let mut a = lo
  let mut b = 0
  // without branches: each row is written to both sides and only one
  // cursor advances; indices stay in range because a + b counts rows seen
  for i in lo..<hi {
    let r = rows.unsafe_get(i)
    let bin = bins.unsafe_get(r * stride + column).to_int() - sub
    let key = bin + (((bin - 1) >> 31) & miss)
    let left = if key <= limit { 1 } else { 0 }
    rows.unsafe_set(a, r)
    scratch.unsafe_set(b, r)
    a += left
    b += 1 - left
  }
  for i in 0..<b {
    rows.unsafe_set(a + i, scratch.unsafe_get(i))
  }
  a
}

///|
fn Grower::grow(
  self : Grower,
  tree : GrowTree,
  node : Int32,
  lo : Int32,
  hi : Int32,
  hist : FixedArray[Int64],
  g : Double,
  h : Double,
  depth : Int32,
  level : Int32,
) -> Unit {
  let level = if level < self.compact_at.length() &&
    depth >= self.compact_at[level] {
    self.compact(lo, hi, level)
    level + 1
  } else {
    level
  }
  let split = if depth < self.p.max_depth && hi - lo >= 2 {
    self.best_split(hist, g, h)
  } else {
    None
  }
  match split {
    None => {
      // leaf values use the exact gradient sums
      let mut eg = 0.0
      let mut eh = 0.0
      let gh = if level > 0 { self.cgh[level - 1] } else { self.gh }
      for i in lo..<hi {
        let r = self.rows[i]
        eg += gh[r * 2].to_double()
        eh += gh[r * 2 + 1].to_double()
      }
      let leaf = Float::from_double(self.weight(eg, eh) * self.p.eta)
      tree.split_cond[node] = leaf
      for i in lo..<hi {
        // compact positions map to rows through orig; otherwise rows[i]
        // is the row itself and orig is rows
        let r = self.rows[i]
        self.leaf_of_row[if level > 0 { self.orig[level - 1][r] } else { r }] = leaf.to_double()
      }
    }
    Some(s) => {
      let mid = self.partition(lo, hi, s, level)
      let gl = s.gl
      let hl = s.hl
      let gr = g - gl
      let hr = h - hl
      let l = tree.add(node, self.weight(gl, hl), hl)
      let r = tree.add(node, self.weight(gr, hr), hr)
      tree.left[node] = l
      tree.right[node] = r
      tree.split_index[node] = s.feature
      tree.split_cond[node] = self.data.cuts[self.data.cut_start[s.feature] +
        s.bin]
      tree.default_left[node] = s.default_left
      tree.loss_change[node] = Float::from_double(s.gain)
      // children at the depth limit become leaves and need no histogram;
      // otherwise build the smaller child's and derive the other's
      let (hl_hist, hr_hist) = if depth + 1 >= self.p.max_depth {
        (hist, hist)
      } else {
        let (small_lo, small_hi) = if mid - lo <= hi - mid {
          (lo, mid)
        } else {
          (mid, hi)
        }
        let small = self.histogram(small_lo, small_hi, level)
        // the parent's histogram becomes the larger child's
        for i in 0..<hist.length() {
          hist[i] = hist[i] - small[i]
        }
        if small_lo == lo {
          (small, hist)
        } else {
          (hist, small)
        }
      }
      self.grow(tree, l, lo, mid, hl_hist, gl, hl, depth + 1, level)
      self.grow(tree, r, mid, hi, hr_hist, gr, hr, depth + 1, level)
    }
  }
}

///|
/// A float32 as JSON text that XGBoost reads as a number rather than an
/// integer: it always has a fraction or an exponent.
fn json_float(buf : StringBuilder, x : Float) -> Unit {
  let t = x.to_string()
  buf.write_string(t)
  if !(t.contains(".") || t.contains("e") || t.contains("E")) {
    buf.write_string(".0")
  }
}

///|
fn json_floats(buf : StringBuilder, a : Array[Float]) -> Unit {
  buf.write_char('[')
  for i, x in a {
    if i > 0 {
      buf.write_char(',')
    }
    json_float(buf, x)
  }
  buf.write_char(']')
}

///|
fn json_ints(buf : StringBuilder, a : Array[Int32]) -> Unit {
  buf.write_char('[')
  for i, x in a {
    if i > 0 {
      buf.write_char(',')
    }
    buf.write_string(x.to_string())
  }
  buf.write_char(']')
}

///|
fn tree_json(
  buf : StringBuilder,
  t : GrowTree,
  id : Int32,
  num_feature : Int32,
) -> Unit {
  let n = t.left.length()
  buf.write_string("{\"base_weights\":")
  json_floats(buf, t.base_weight)
  buf.write_string(
    ",\"categories\":[],\"categories_nodes\":[],\"categories_segments\":[],\"categories_sizes\":[],\"default_left\":",
  )
  json_ints(buf, t.default_left.map(b => if b { 1 } else { 0 }))
  buf.write_string(",\"id\":\{id},\"left_children\":")
  json_ints(buf, t.left)
  buf.write_string(",\"loss_changes\":")
  json_floats(buf, t.loss_change)
  buf.write_string(",\"parents\":")
  json_ints(buf, t.parent.map(p => if p < 0 { 2147483647 } else { p }))
  buf.write_string(",\"right_children\":")
  json_ints(buf, t.right)
  buf.write_string(",\"split_conditions\":")
  json_floats(buf, t.split_cond)
  buf.write_string(",\"split_indices\":")
  json_ints(buf, t.split_index)
  buf.write_string(",\"split_type\":")
  json_ints(buf, Array::make(n, 0))
  buf.write_string(",\"sum_hessian\":")
  json_floats(buf, t.sum_hess)
  buf.write_string(
    ",\"tree_param\":{\"num_deleted\":\"0\",\"num_feature\":\"\{num_feature}\",\"num_nodes\":\"\{n}\",\"size_leaf_vector\":\"1\"}}",
  )
}

///|
/// A float written the way XGBoost writes its parameters.
fn xgb_float_text(x : Float) -> String {
  x.to_string()
}

///|
/// Trains a model and returns it as an XGBoost JSON document, which both
/// moonboost and XGBoost load. `label` has one entry per row: the target,
/// a 0/1 label, or a class index.
pub fn train_xgboost_json(
  data : DenseMatrix,
  label : ArrayView[Float],
  params~ : TrainParams = TrainParams::default(),
) -> String raise ModelError {
  let p = params
  let n = data.rows
  if label.length() != n {
    raise InvalidModel("\{label.length()} labels for \{n} rows")
  }
  if n == 0 {
    raise InvalidModel("no training rows")
  }
  if p.max_bin < 2 || p.max_bin > 65535 || p.max_depth < 1 || p.num_round < 0 {
    raise InvalidModel("bad training parameters")
  }
  let groups = match p.objective {
    "reg:squarederror" | "binary:logistic" => 1
    "multi:softprob" | "multi:softmax" => {
      if p.num_class < 2 {
        raise InvalidModel("`num_class` must be at least 2")
      }
      p.num_class
    }
    o => raise Unsupported("training objective \{o}")
  }
  for y in label {
    if y.is_nan() {
      raise InvalidModel("missing label")
    }
    if p.objective == "binary:logistic" && (y < 0.0 || y > 1.0) {
      raise InvalidModel("binary labels must be in [0, 1]")
    }
    if groups > 1 &&
      (
        y < 0.0 ||
        y.to_int() >= groups ||
        y.to_int().to_double() != y.to_double()
      ) {
      raise InvalidModel("class labels must be 0 .. num_class - 1")
    }
  }
  // intercept, in the space XGBoost stores it
  let mut mean = 0.0
  for y in label {
    mean += y.to_double()
  }
  mean = mean / n.to_double()
  let (base_text, base_margin) = match p.objective {
    "reg:squarederror" => {
      let b = Float::from_double(mean)
      ("[" + xgb_float_text(b) + "]", b.to_double())
    }
    "binary:logistic" => {
      let pr = Float::from_double(
        if mean < 1.0e-6 {
          1.0e-6
        } else if mean > 1.0 - 1.0e-6 {
          1.0 - 1.0e-6
        } else {
          mean
        },
      )
      let x = pr.to_double()
      ("[" + xgb_float_text(pr) + "]", @math.ln(x / (1.0 - x)))
    }
    _ => ("[" + Array::make(groups, "0").join(",") + "]", 0.0)
  }
  let kind = match p.objective {
    "reg:squarederror" => 0
    "binary:logistic" => 1
    _ => 2
  }
  // histogram slots are stored as UInt16
  if data.cols * (p.max_bin + 1) > 65535 {
    raise InvalidModel("too many features times bins for training")
  }
  let binned = bin_matrix(data, p.max_bin)
  let hist_start = FixedArray::make(data.cols + 1, 0)
  for f in 0..<data.cols {
    hist_start[f + 1] = hist_start[f] +
      (binned.cut_start[f + 1] - binned.cut_start[f] + 2)
  }
  let gr : Grower = {
    data: binned,
    p,
    gh: FixedArray::make(n * 2, (0.0 : Float)),
    gq: FixedArray::make(n, 0L),
    gs: 1.0,
    hs: 1.0,
    igs: 1.0,
    compact_at: [2],
    cbins: FixedArray::makei(1, fn(_) { FixedArray::make(n * data.cols, 0) }),
    cq: FixedArray::makei(1, fn(_) { FixedArray::make(n, 0L) }),
    cgh: FixedArray::makei(1, fn(_) { FixedArray::make(n * 2, (0.0 : Float)) }),
    orig: FixedArray::makei(1, fn(_) { FixedArray::make(n, 0) }),
    ihs: 1.0,
    rows: FixedArray::make(n, 0),
    scratch: FixedArray::make(n, 0),
    leaf_of_row: FixedArray::make(n, 0.0),
    nbins: hist_start[data.cols],
    hist_start,
  }
  // margins accumulate in float32, as XGBoost predicts
  let margin = FixedArray::make(n * groups, Float::from_double(base_margin))
  let trees = StringBuilder()
  let mut num_trees = 0
  let prob = FixedArray::make(groups, 0.0)
  for _ in 0..<p.num_round {
    // softmax probabilities are computed once per round for all classes
    let probs = if groups > 1 {
      let out = FixedArray::make(n * groups, 0.0)
      for r in 0..<n {
        let mut mx = margin[r * groups].to_double()
        for k in 1..<groups {
          let v = margin[r * groups + k].to_double()
          if v > mx {
            mx = v
          }
        }
        let mut sum = 0.0
        for k in 0..<groups {
          prob[k] = @math.exp(margin[r * groups + k].to_double() - mx)
          sum += prob[k]
        }
        for k in 0..<groups {
          out[r * groups + k] = prob[k] / sum
        }
      }
      out
    } else {
      FixedArray::make(0, 0.0)
    }
    for k in 0..<groups {
      let mut gsum = 0.0
      let mut hsum = 0.0
      // bounds on |g| and h: known for the logistic losses, measured for
      // squared error
      match kind {
        0 => {
          let mut gmax = 0.0
          for r in 0..<n {
            let d = (margin[r * groups + k].to_double() - label[r].to_double()).abs()
            if d > gmax {
              gmax = d
            }
          }
          gr.set_scales(
            n,
            Float::from_double(gmax).to_double() * 1.0000001,
            1.0,
          )
        }
        1 => gr.set_scales(n, 1.0, 0.25)
        _ => gr.set_scales(n, 1.0, 0.5)
      }
      let mut qsum = 0L
      for r in 0..<n {
        let y = label[r].to_double()
        let m = margin[r * groups + k].to_double()
        let (g, h) = match kind {
          0 => (m - y, 1.0)
          1 => {
            let pr = 1.0 / (1.0 + @math.exp(-m))
            let h = pr * (1.0 - pr)
            (pr - y, if h < 1.0e-16 { 1.0e-16 } else { h })
          }
          _ => {
            let pr = probs[r * groups + k]
            let t = if y.to_int() == k { 1.0 } else { 0.0 }
            let h = 2.0 * pr * (1.0 - pr)
            (pr - t, if h < 1.0e-16 { 1.0e-16 } else { h })
          }
        }
        let g = Float::from_double(g).to_double()
        let h = Float::from_double(h).to_double()
        gr.gh[r * 2] = Float::from_double(g)
        gr.gh[r * 2 + 1] = Float::from_double(h)
        let q = gr.pack(g, h)
        gr.gq[r] = q
        qsum = qsum + q
        gr.rows[r] = r
        gsum += g
        hsum += h
      }
      let (qg, qh) = gr.unpack(qsum)
      let tree = GrowTree::new()
      let root = tree.add(-1, gr.weight(gsum, hsum), hsum)
      gr.grow(tree, root, 0, n, gr.histogram(0, n, 0), qg, qh, 0, 0)
      for r in 0..<n {
        margin[r * groups + k] = margin[r * groups + k] +
          Float::from_double(gr.leaf_of_row[r])
      }
      if num_trees > 0 {
        trees.write_char(',')
      }
      tree_json(trees, tree, num_trees, data.cols)
      num_trees += 1
    }
  }
  let buf = StringBuilder()
  buf.write_string(
    "{\"learner\":{\"attributes\":{},\"feature_names\":[],\"feature_types\":[],\"gradient_booster\":{\"model\":{\"cats\":{\"enc\":[],\"feature_segments\":[],\"sorted_idx\":[]},",
  )
  buf.write_string(
    "\"gbtree_model_param\":{\"num_parallel_tree\":\"1\",\"num_trees\":\"\{num_trees}\"},\"iteration_indptr\":",
  )
  json_ints(buf, Array::makei(p.num_round + 1, i => i * groups))
  buf.write_string(",\"tree_info\":")
  json_ints(buf, Array::makei(num_trees, i => i % groups))
  buf.write_string(",\"trees\":[")
  buf.write_string(trees.to_string())
  buf.write_string("]},\"name\":\"gbtree\"},\"learner_model_param\":{")
  let num_class = if groups > 1 { groups } else { 0 }
  buf.write_string(
    "\"base_score\":\"\{base_text}\",\"boost_from_average\":\"1\",\"num_class\":\"\{num_class}\",\"num_feature\":\"\{data.cols}\",\"num_target\":\"1\"},",
  )
  if groups > 1 {
    buf.write_string(
      "\"objective\":{\"name\":\"\{p.objective}\",\"softmax_multiclass_param\":{\"num_class\":\"\{groups}\"}}",
    )
  } else {
    buf.write_string(
      "\"objective\":{\"name\":\"\{p.objective}\",\"reg_loss_param\":{\"scale_pos_weight\":\"1\"}}",
    )
  }
  buf.write_string("},\"version\":[3,4,1]}")
  buf.to_string()
}

///|
/// Trains a model (see `train_xgboost_json`).
pub fn Model::train(
  data : DenseMatrix,
  label : ArrayView[Float],
  params~ : TrainParams = TrainParams::default(),
) -> Model raise ModelError {
  Model::from_xgboost_json(train_xgboost_json(data, label, params~))
}
