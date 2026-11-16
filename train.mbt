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
/// Feature values cut into bins. `by_col[f * rows + r]` is one more than
/// the bin of row `r` in feature `f` (`0` missing); a value `v` falls in bin
/// `k` when exactly `k` of the feature's cut points are `<= v`.
priv struct Binned {
  cols : Int32
  // for two-byte training, the histogram slot of row r in feature f at
  // bins[r * cols + f]: `cut_start[f] + 2 * f` when missing, then one per
  // bin (filled by training only when needed)
  mut bins : FixedArray[UInt16]
  by_col : FixedArray[UInt16]
  // cut points of feature f are cuts[cut_start[f] .. cut_start[f + 1]]
  cuts : Array[Float]
  cut_start : FixedArray[Int32]
  // whether feature f has a missing value
  missing : FixedArray[Bool]
}

///|
/// Sorts the non-missing values of one column: a radix sort on keys that
/// order like the floats, skipping the byte passes where all keys agree.
fn sorted_column(m : DenseMatrix, f : Int32) -> FixedArray[Float] {
  let rows = m.rows
  let cols = m.cols
  let mut n = 0
  for r in 0..<rows {
    if !m.data[r * cols + f].is_nan() {
      n += 1
    }
  }
  let mut keys : FixedArray[Int32] = FixedArray::make(n, 0)
  let mut tmp : FixedArray[Int32] = FixedArray::make(n, 0)
  let mut k = 0
  for r in 0..<rows {
    let v = m.data[r * cols + f]
    if !v.is_nan() {
      let b = v.reinterpret_as_int()
      // flip negatives so signed integer order matches float order, then
      // move the sign so unsigned byte order does too
      keys[k] = b ^ ((b >> 31) & 0x7FFFFFFF) ^ 0x80000000
      k += 1
    }
  }
  let count = FixedArray::make(256, 0)
  for pass in 0..<4 {
    let shift = pass * 8
    count.fill(0)
    for i in 0..<n {
      let d = (keys.unsafe_get(i) >> shift) & 255
      count.unsafe_set(d, count.unsafe_get(d) + 1)
    }
    if n == 0 || count[(keys[0] >> shift) & 255] == n {
      continue
    }
    let mut sum = 0
    for d in 0..<256 {
      let c = count[d]
      count[d] = sum
      sum += c
    }
    for i in 0..<n {
      let key = keys.unsafe_get(i)
      let d = (key >> shift) & 255
      let at = count.unsafe_get(d)
      tmp.unsafe_set(at, key)
      count.unsafe_set(d, at + 1)
    }
    let t = keys
    keys = tmp
    tmp = t
  }
  FixedArray::makei(n, fn(i) {
    let key = keys[i] ^ 0x80000000
    Float::reinterpret_from_int(key ^ ((key >> 31) & 0x7FFFFFFF))
  })
}

///|
fn bin_matrix(m : DenseMatrix, max_bin : Int32) -> Binned {
  let rows = m.rows
  let cols = m.cols
  let cuts : Array[Float] = []
  let cut_start = FixedArray::make(cols + 1, 0)
  let by_col : FixedArray[UInt16] = FixedArray::make(rows * cols, 0)
  let missing = FixedArray::make(cols, false)
  for f in 0..<cols {
    let vals = sorted_column(m, f)
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
    let fc = FixedArray::makei(hi - lo, fn(i) { cuts[lo + i] })
    for r in 0..<rows {
      let v = m.data[r * cols + f]
      if v.is_nan() {
        missing[f] = true
      } else {
        // number of cuts <= v, by steps that add a 0/1 multiple rather
        // than branch
        let mut k = 0
        let mut len = hi - lo
        while len > 0 {
          let half = len / 2
          k += (fc.unsafe_get(k + half) <= v).to_int() * (len - half)
          len = half
        }
        by_col[f * rows + r] = (k + 1).to_uint16()
      }
    }
  }
  { cols, bins: FixedArray::make(0, 0), by_col, cuts, cut_start, missing, }
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
  // copied bins take one byte, slot - cbase[feature], when every feature's
  // slots in use span at most 256 (narrow); otherwise two, the slot itself
  narrow : Bool
  // when narrow, the original bins in the same one-byte form
  bins8 : FixedArray[Byte]
  // and column by column: by8[f * rows + r]
  by8 : FixedArray[Byte]
  cbins : FixedArray[FixedArray[Byte]]
  cbase : FixedArray[Int32]
  cbins16 : FixedArray[FixedArray[UInt16]]
  cq : FixedArray[FixedArray[Int64]]
  cgh : FixedArray[FixedArray[Float]]
  orig : FixedArray[FixedArray[Int32]]
  mut ihs : Double // 1 / hs
  // rows of the nodes at the current depth; partitioning writes the
  // children's rows to `scratch`, and the two swap while the children grow
  mut rows : FixedArray[Int32]
  mut scratch : FixedArray[Int32]
  // leaf node reached by each copied position (orig[0] maps positions to
  // rows); a leaf above the copy depth fills its own positions and orig[0]
  leaf_at : FixedArray[Int32]
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
  // 0.5 with the sign of x, from its bits: the sign of a gradient is
  // unpredictable, so a branch on it would often be mispredicted
  let half = ((x.reinterpret_as_int64() & (1L << 63)) | 0x3FE0000000000000L).reinterpret_as_double()
  let gi = (x + half).to_int64()
  (gi << 32) + (h * self.hs + 0.5).to_int64()
}

///|
/// exp to about 1e-9 relative error, ample for gradients that are rounded
/// to float32: x = k ln2 + r with |r| <= ln2 / 2, a degree-7 polynomial for
/// e^r, and 2^k built from its exponent bits. x is clamped to +-80, where
/// the losses are saturated.
fn train_exp(x : Double) -> Double {
  let x = if x > 80.0 { 80.0 } else if x < -80.0 { -80.0 } else { x }
  // k = round(x / ln2) by the 1.5 * 2^52 trick, without a branch or a
  // checked conversion
  let kd = x * 1.4426950408889634 + 6755399441055744.0
  let ki = kd.reinterpret_as_int64() - 0x4338000000000000L
  let kf = kd - 6755399441055744.0
  let r = x - kf * 0.6931471803691238 - kf * 1.9082149292705877e-10
  // Estrin's scheme: a shorter dependency chain than Horner's
  let r2 = r * r
  let r4 = r2 * r2
  let p = 1.0 +
    r +
    r2 * (0.5 + r * (1.0 / 6.0)) +
    r4 *
    (1.0 / 24.0 + r * (1.0 / 120.0) + r2 * (1.0 / 720.0 + r * (1.0 / 5040.0)))
  p * ((ki + 1023L) << 52).reinterpret_as_double()
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
  let rows = self.rows
  if self.narrow {
    let (cb, cq) = if level > 0 {
      (self.cbins[level - 1], self.cq[level - 1])
    } else {
      (self.bins8, self.gq)
    }
    let base_of = self.cbase
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let q = cq.unsafe_get(r)
      let base = r * cols
      let mut f = 0
      while f + 4 <= cols {
        let s0 = cb.unsafe_get(base + f).to_int() + base_of.unsafe_get(f)
        let s1 = cb.unsafe_get(base + f + 1).to_int() +
          base_of.unsafe_get(f + 1)
        let s2 = cb.unsafe_get(base + f + 2).to_int() +
          base_of.unsafe_get(f + 2)
        let s3 = cb.unsafe_get(base + f + 3).to_int() +
          base_of.unsafe_get(f + 3)
        hist.unsafe_set(s0, hist.unsafe_get(s0) + q)
        hist.unsafe_set(s1, hist.unsafe_get(s1) + q)
        hist.unsafe_set(s2, hist.unsafe_get(s2) + q)
        hist.unsafe_set(s3, hist.unsafe_get(s3) + q)
        f += 4
      }
      while f < cols {
        let slot = cb.unsafe_get(base + f).to_int() + base_of.unsafe_get(f)
        hist.unsafe_set(slot, hist.unsafe_get(slot) + q)
        f += 1
      }
    }
    return hist
  }
  let bins = if level > 0 { self.cbins16[level - 1] } else { self.data.bins }
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
  // subtrees are copied from the original arrays only (compact_at has one
  // depth), so the source is always level 0
  guard level == 0 else { abort("subtrees are copied once") }
  let cols = self.data.cols
  let rows = self.rows
  let q = self.gq
  let gh = self.gh
  let bins = self.data.bins
  let tq = self.cq[0]
  let tg = self.cgh[0]
  let to = self.orig[0]
  // indices are in range: rows[lo:hi] are rows, i < n
  if self.narrow {
    let tb = self.cbins[0]
    let b8 = self.bins8
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let src = r * cols
      let dst = i * cols
      for f in 0..<cols {
        tb.unsafe_set(dst + f, b8.unsafe_get(src + f))
      }
      tq.unsafe_set(i, q.unsafe_get(r))
      tg.unsafe_set(i * 2, gh.unsafe_get(r * 2))
      tg.unsafe_set(i * 2 + 1, gh.unsafe_get(r * 2 + 1))
      to.unsafe_set(i, r)
      rows.unsafe_set(i, i)
    }
  } else {
    let tw = self.cbins16[0]
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let src = r * cols
      let dst = i * cols
      for f in 0..<cols {
        tw.unsafe_set(dst + f, bins.unsafe_get(src + f))
      }
      tq.unsafe_set(i, q.unsafe_get(r))
      tg.unsafe_set(i * 2, gh.unsafe_get(r * 2))
      tg.unsafe_set(i * 2 + 1, gh.unsafe_get(r * 2 + 1))
      to.unsafe_set(i, r)
      rows.unsafe_set(i, i)
    }
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
  // a missing value (bin 0) goes left as bin 0, or right as a bin past limit
  let miss = if s.default_left { 0 } else { 1 << 20 }
  let rows = self.rows
  let dst = self.scratch
  let mut a = lo
  let mut b = hi - 1
  // without branches: left rows fill dst upwards from lo and right rows
  // downwards from hi - 1; each row is written at both cursors and only
  // one advances, so every slot ends with the row last meant for it
  if self.narrow {
    // the feature's bin of position r is cb[r * stride + column] + add
    let (cb, stride, column) = if level > 0 {
      (self.cbins[level - 1], self.data.cols, s.feature)
    } else {
      (self.by8, 1, s.feature * self.leaf_at.length())
    }
    let add = self.cbase[s.feature] - self.hist_start[s.feature]
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let bin = cb.unsafe_get(r * stride + column).to_int() + add
      let key = bin + (((bin - 1) >> 31) & miss)
      let left = if key <= limit { 1 } else { 0 }
      dst.unsafe_set(a, r)
      dst.unsafe_set(b, r)
      a += left
      b -= 1 - left
    }
  } else {
    // the feature's bin of row r is bins[r * stride + column] - sub
    let (bins, stride, column, sub) = if level > 0 {
      (
        self.cbins16[level - 1],
        self.data.cols,
        s.feature,
        self.hist_start[s.feature],
      )
    } else {
      (self.data.by_col, 1, s.feature * self.gh.length() / 2, 0)
    }
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let bin = bins.unsafe_get(r * stride + column).to_int() - sub
      let key = bin + (((bin - 1) >> 31) & miss)
      let left = if key <= limit { 1 } else { 0 }
      dst.unsafe_set(a, r)
      dst.unsafe_set(b, r)
      a += left
      b -= 1 - left
    }
  }
  a
}

///|
fn Grower::swap_rows(self : Grower) -> Unit {
  let t = self.rows
  self.rows = self.scratch
  self.scratch = t
}

///|
/// Partitions rows[lo:hi] (original rows) like `partition`, but writes each
/// row's bins and gradients to the copy level at its new position, left
/// rows upwards from lo and right rows downwards from hi - 1; the
/// children's rows in scratch are then those positions.
fn Grower::partition_copy(
  self : Grower,
  lo : Int32,
  hi : Int32,
  s : Split,
) -> Int32 {
  let limit = s.bin + 1
  let miss = if s.default_left { 0 } else { 1 << 20 }
  let cols = self.data.cols
  let n = self.leaf_at.length()
  let rows = self.rows
  let dst = self.scratch
  let q = self.gq
  let gh = self.gh
  let tq = self.cq[0]
  let tg = self.cgh[0]
  let to = self.orig[0]
  let mut a = lo
  let mut b = hi - 1
  // indices are in range: rows[lo:hi] are rows, and a + (hi - 1 - b)
  // counts rows seen
  if self.narrow {
    let by8 = self.by8
    let b8 = self.bins8
    let tb = self.cbins[0]
    let column = s.feature * n
    let add = self.cbase[s.feature] - self.hist_start[s.feature]
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let bin = by8.unsafe_get(r + column).to_int() + add
      let key = bin + (((bin - 1) >> 31) & miss)
      let left = if key <= limit { 1 } else { 0 }
      let p = b + (a - b) * left
      let src = r * cols
      let at = p * cols
      for f in 0..<cols {
        tb.unsafe_set(at + f, b8.unsafe_get(src + f))
      }
      tq.unsafe_set(p, q.unsafe_get(r))
      tg.unsafe_set(p * 2, gh.unsafe_get(r * 2))
      tg.unsafe_set(p * 2 + 1, gh.unsafe_get(r * 2 + 1))
      to.unsafe_set(p, r)
      a += left
      b -= 1 - left
    }
  } else {
    let by_col = self.data.by_col
    let bins = self.data.bins
    let tw = self.cbins16[0]
    let column = s.feature * n
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let bin = by_col.unsafe_get(r + column).to_int()
      let key = bin + (((bin - 1) >> 31) & miss)
      let left = if key <= limit { 1 } else { 0 }
      let p = b + (a - b) * left
      let src = r * cols
      let at = p * cols
      for f in 0..<cols {
        tw.unsafe_set(at + f, bins.unsafe_get(src + f))
      }
      tq.unsafe_set(p, q.unsafe_get(r))
      tg.unsafe_set(p * 2, gh.unsafe_get(r * 2))
      tg.unsafe_set(p * 2 + 1, gh.unsafe_get(r * 2 + 1))
      to.unsafe_set(p, r)
      a += left
      b -= 1 - left
    }
  }
  for i in lo..<hi {
    dst.unsafe_set(i, i)
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
    depth >= self.compact_at[level] &&
    depth < self.p.max_depth {
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
      // positions in rows[lo:hi] are those of one copy level (there is at
      // most one); a leaf above it is given the positions lo..hi, which no
      // copied subtree uses
      if level > 0 {
        for i in lo..<hi {
          self.leaf_at[self.rows[i]] = node
        }
      } else {
        let o = self.orig[0]
        for i in lo..<hi {
          o[i] = self.rows[i]
          self.leaf_at[i] = node
        }
      }
    }
    Some(s) => {
      // a node just above the copy depth partitions its rows straight
      // into the copy, reading each row once in row order
      let copy = level == 0 &&
        self.compact_at.length() > 0 &&
        depth + 1 == self.compact_at[0] &&
        depth + 1 < self.p.max_depth
      let mid = if copy {
        self.partition_copy(lo, hi, s)
      } else {
        self.partition(lo, hi, s, level)
      }
      let level = if copy { 1 } else { level }
      // the children's rows are in scratch; they become the current rows
      // until both children have grown
      self.swap_rows()
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
      self.swap_rows()
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
  train_json(data, label, params, true)
}

///|
/// `allow_narrow` false forces two-byte copied bins (for tests).
fn train_json(
  data : DenseMatrix,
  label : ArrayView[Float],
  params : TrainParams,
  allow_narrow : Bool,
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
  // per-feature base of the one-byte copied bins: a feature without
  // missing values never uses its missing slot
  let cbase = FixedArray::makei(data.cols, fn(f) {
    hist_start[f] + (if binned.missing[f] { 0 } else { 1 })
  })
  let mut narrow = allow_narrow
  for f in 0..<data.cols {
    if hist_start[f + 1] - 1 - cbase[f] > 255 {
      narrow = false
    }
  }
  let copies = if narrow { 0 } else { n * data.cols }
  let cols = data.cols
  let bins8 = FixedArray::make(n * cols - copies, b'\x00')
  let by8 = FixedArray::make(n * cols - copies, b'\x00')
  let by_col = binned.by_col
  if narrow {
    // one-byte bins: the bin, plus one for a feature with missing values
    // (whose missing bin is 0)
    for f in 0..<cols {
      let adj = if binned.missing[f] { 0 } else { 1 }
      for r in 0..<n {
        let b = (by_col.unsafe_get(f * n + r).to_int() - adj).to_byte()
        bins8.unsafe_set(r * cols + f, b)
        by8.unsafe_set(f * n + r, b)
      }
    }
  } else {
    let bins = FixedArray::make(n * cols, (0 : UInt16))
    for f in 0..<cols {
      for r in 0..<n {
        bins[r * cols + f] = (by_col[f * n + r].to_int() + hist_start[f]).to_uint16()
      }
    }
    binned.bins = bins
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
    narrow,
    bins8,
    by8,
    cbins: FixedArray::makei(1, fn(_) {
      FixedArray::make(n * data.cols - copies, (0 : Byte))
    }),
    cbase,
    cbins16: FixedArray::makei(1, fn(_) { FixedArray::make(copies, 0) }),
    cq: FixedArray::makei(1, fn(_) { FixedArray::make(n, 0L) }),
    cgh: FixedArray::makei(1, fn(_) { FixedArray::make(n * 2, (0.0 : Float)) }),
    orig: FixedArray::makei(1, fn(_) { FixedArray::make(n, 0) }),
    ihs: 1.0,
    rows: FixedArray::make(n, 0),
    scratch: FixedArray::make(n, 0),
    leaf_at: FixedArray::make(n, 0),
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
          prob[k] = train_exp(margin[r * groups + k].to_double() - mx)
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
            let pr = 1.0 / (1.0 + train_exp(-m))
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
      let o = gr.orig[0]
      for i in 0..<n {
        let at = o[i] * groups + k
        margin[at] = margin[at] + tree.split_cond[gr.leaf_at[i]]
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
