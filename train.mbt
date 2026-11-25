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
  /// outputs per row for `reg:squarederror`, `reg:pseudohubererror`,
  /// `reg:absoluteerror`, `binary:logistic`, `reg:logistic` and
  /// `binary:logitraw`: the label then holds num_target values per row,
  /// row by row (XGBoost's two-dimensional label), and each round grows
  /// one tree per output
  num_target : Int32
  /// with several outputs, `"one_output_per_tree"` (each round grows one
  /// tree per output) or `"multi_output_tree"` (one tree whose leaves hold
  /// a value per output; squared, pseudo-Huber and logistic losses,
  /// depth-wise growth on numerical features without sampling)
  multi_strategy : String
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
  /// `depthwise` grows every node to `max_depth`; `lossguide` splits the
  /// leaf with the largest gain first, up to `max_leaves` leaves
  grow_policy : String
  /// leaf limit for `lossguide` (0: none)
  max_leaves : Int32
  /// columns holding category codes (non-negative integers); a split on
  /// one sends a set of categories one way
  categorical : Array[Int32]
  /// features with at most this many categories in a node split one
  /// category against the rest
  max_cat_to_onehot : Int32
  /// most categories a partition split sends to the smaller side
  max_cat_threshold : Int32
  /// partition splits order categories by g / (h + cat_smooth), leave out
  /// categories with fewer than min_data_per_group rows (estimated from
  /// the hessian) and add cat_l2 to lambda (LightGBM's names)
  cat_smooth : Double
  cat_l2 : Double
  min_data_per_group : Double
  /// fraction of rows each tree learns from, drawn anew per tree (the
  /// others get zero gradients, as in XGBoost)
  subsample : Double
  /// `uniform` draws every row with chance `subsample`; `gradient_based`
  /// keeps rows with chance proportional to sqrt(g^2 + lambda h^2), capped
  /// at 1, and divides a kept row's g and h by its chance (minimal variance
  /// sampling), so `subsample` of the rows are kept on average
  sampling_method : String
  /// fraction of features each tree may split on, and of those, each node
  colsample_bytree : Double
  colsample_bynode : Double
  /// seed of the row and feature sampling
  seed : Int32
  /// fewest rows a child may have, counted as LightGBM does from its share
  /// of the node's hessian
  min_data_in_leaf : Double
  /// variance power of `reg:tweedie`, in (1, 2)
  tweedie_variance_power : Double
  /// slope (delta) of `reg:pseudohubererror`
  huber_slope : Double
  /// the quantile `reg:quantileerror` predicts, in (0, 1)
  quantile_alpha : Double
  /// several quantiles at once for `reg:quantileerror`, each in (0, 1):
  /// when not empty it replaces `quantile_alpha` and the model predicts
  /// one value per quantile (XGBoost's `quantile_alpha` list)
  quantile_alphas : Array[Double]
  /// for `rank:ndcg` and `rank:pairwise`: the sizes of the queries, whose
  /// rows are consecutive and in this order (XGBoost's `group`)
  group : Array[Int32]
  /// `rank:ndcg` pairs only documents of which one is ranked in the top
  /// this many of its query by the current scores (0: all pairs; 30, as
  /// LightGBM's lambdarank_truncation_level)
  rank_truncation : Int32
  /// for `survival:aft`: the upper bound of each row's time, `label` being
  /// the lower (equal for an observed time, +inf right censored; a lower
  /// bound of 0 is left censored)
  label_upper_bound : Array[Float]
  /// for `survival:aft`: `normal`, `logistic` or `extreme`
  aft_loss_distribution : String
  /// for `survival:aft`: the scale of the noise on log time
  aft_loss_distribution_scale : Double
}

///|
/// XGBoost's defaults: squared error, 10 rounds, depth 6, eta 0.3,
/// lambda 1, 256 bins, depth-wise growth, no categorical columns.
pub fn TrainParams::default() -> TrainParams {
  {
    objective: "reg:squarederror",
    num_class: 0,
    num_target: 1,
    multi_strategy: "one_output_per_tree",
    num_round: 10,
    max_depth: 6,
    eta: 0.3,
    lambda: 1.0,
    gamma: 0.0,
    min_child_weight: 1.0,
    max_bin: 256,
    grow_policy: "depthwise",
    max_leaves: 0,
    categorical: [],
    max_cat_to_onehot: 4,
    max_cat_threshold: 64,
    cat_smooth: 10.0,
    cat_l2: 10.0,
    min_data_per_group: 100.0,
    subsample: 1.0,
    sampling_method: "uniform",
    colsample_bytree: 1.0,
    colsample_bynode: 1.0,
    seed: 0,
    min_data_in_leaf: 0.0,
    tweedie_variance_power: 1.5,
    huber_slope: 1.0,
    quantile_alpha: 0.5,
    quantile_alphas: [],
    group: [],
    rank_truncation: 30,
    label_upper_bound: [],
    aft_loss_distribution: "normal",
    aft_loss_distribution_scale: 1.0,
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
  mut bins : U16s
  // released once training has packed one-byte bins from it
  mut by_col : U16s
  // cut points of feature f are cuts[cut_start[f] .. cut_start[f + 1]]
  cuts : Array[Float]
  cut_start : FixedArray[Int32]
  // whether feature f has a missing value
  missing : FixedArray[Bool]
  // whether feature f holds categories; each of its values is a bin
  is_cat : FixedArray[Bool]
  // smallest value of feature f (the value of bin 1 of a categorical
  // feature; bin k + 1 holds its cut k)
  lowest : FixedArray[Float]
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
  // the byte counts of all four passes in one read of the keys: the
  // passes only permute the keys, so the counts hold for each of them
  let count = FixedArray::make(1024, 0)
  let mut k = 0
  for r in 0..<rows {
    let v = m.data[r * cols + f]
    if !v.is_nan() {
      let b = v.reinterpret_as_int()
      // flip negatives so signed integer order matches float order, then
      // move the sign so unsigned byte order does too
      let key = b ^ ((b >> 31) & 0x7FFFFFFF) ^ 0x80000000
      keys[k] = key
      let d0 = key & 255
      let d1 = ((key >> 8) & 255) + 256
      let d2 = ((key >> 16) & 255) + 512
      let d3 = ((key >> 24) & 255) + 768
      count.unsafe_set(d0, count.unsafe_get(d0) + 1)
      count.unsafe_set(d1, count.unsafe_get(d1) + 1)
      count.unsafe_set(d2, count.unsafe_get(d2) + 1)
      count.unsafe_set(d3, count.unsafe_get(d3) + 1)
      k += 1
    }
  }
  for pass in 0..<4 {
    let shift = pass * 8
    let o = pass * 256
    if n == 0 || count[o + ((keys[0] >> shift) & 255)] == n {
      continue
    }
    let mut sum = 0
    for d in o..<(o + 256) {
      let c = count[d]
      count[d] = sum
      sum += c
    }
    for i in 0..<n {
      let key = keys.unsafe_get(i)
      let d = ((key >> shift) & 255) + o
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
/// Rows per call of the loops that pass over all rows. Each loop is its
/// own function, called block by block: WebAssembly engines optimise a
/// function between calls but not while one call runs, so a loop over all
/// rows in a function called once would run unoptimised throughout.
let block : Int32 = 8192

///|
/// Bins of feature f for rows lo..hi into by_col (feature-major, `rows`
/// rows), against the feature's sorted cuts fv and their order keys fc.
fn bin_block(
  data : FixedArray[Float],
  cols : Int32,
  f : Int32,
  fv : FixedArray[Float],
  fc : FixedArray[Int32],
  by_col : U16s,
  base : Int32,
  lo : Int32,
  hi : Int32,
) -> Unit {
  // the bin of a value is one more than the number of cuts <= it, found
  // by steps that add without a branch (`cut_step`); four
  // rows search side by side, as each search is one long dependency chain.
  // A missing value keeps bin 0.
  let mut r = lo
  while r + 4 <= hi {
    let v0 = data[r * cols + f]
    let x0 = order_key(v0)
    let v1 = data[(r + 1) * cols + f]
    let x1 = order_key(v1)
    let v2 = data[(r + 2) * cols + f]
    let x2 = order_key(v2)
    let v3 = data[(r + 3) * cols + f]
    let x3 = order_key(v3)
    let mut k0 = 0
    let mut k1 = 0
    let mut k2 = 0
    let mut k3 = 0
    let mut len = fc.length()
    while len > 0 {
      let half = len / 2
      let step = len - half
      k0 += cut_step(fv, fc, k0 + half, v0, x0, step)
      k1 += cut_step(fv, fc, k1 + half, v1, x1, step)
      k2 += cut_step(fv, fc, k2 + half, v2, x2, step)
      k3 += cut_step(fv, fc, k3 + half, v3, x3, step)
      len = half
    }
    let at = base + r
    by_col.unsafe_set(at, bin_code(v0, k0))
    by_col.unsafe_set(at + 1, bin_code(v1, k1))
    by_col.unsafe_set(at + 2, bin_code(v2, k2))
    by_col.unsafe_set(at + 3, bin_code(v3, k3))
    r += 4
  }
  while r < hi {
    let v = data[r * cols + f]
    let x = order_key(v)
    let mut k = 0
    let mut len = fc.length()
    while len > 0 {
      let half = len / 2
      k += cut_step(fv, fc, k + half, v, x, len - half)
      len = half
    }
    by_col.unsafe_set(base + r, bin_code(v, k))
    r += 1
  }
}

///|
/// One-byte bins of the features of word w for rows lo..hi: each bin less
/// its feature's `adj`, into the row words (each word written once) and
/// the feature-major bytes.
fn pack_block(
  by_col : U16s,
  bins8 : Words,
  by8 : FixedArray[Byte],
  words : Int32,
  n : Int32,
  cols : Int32,
  w : Int32,
  adjs : FixedArray[Int32],
  lo : Int32,
  hi : Int32,
) -> Unit {
  let f0 = w * word_bins
  let f1 = if f0 + word_bins < cols { f0 + word_bins } else { cols }
  for r in lo..<hi {
    let at = r * words + w
    // the word is still zero: it is built whole and stored once
    let mut x = bins8.unsafe_get(at)
    for f in f0..<f1 {
      let b = (by_col.unsafe_get(f * n + r).to_int() - adjs.unsafe_get(f)).to_byte()
      by8.unsafe_set(f * n + r, b)
      x = word_put(x, b, (f - f0) * 8)
    }
    bins8.unsafe_set(at, x)
  }
}

///|
/// Two-byte bins of feature f for rows lo..hi: histogram slots, row-major.
fn wide_block(
  by_col : U16s,
  bins : U16s,
  cols : Int32,
  n : Int32,
  f : Int32,
  start : Int32,
  lo : Int32,
  hi : Int32,
) -> Unit {
  for r in lo..<hi {
    bins.unsafe_set(
      r * cols + f,
      (by_col.unsafe_get(f * n + r).to_int() + start).to_uint16(),
    )
  }
}

///|
/// An integer that orders like the float (with -0 equal to +0); the order
/// of NaN is not used.
fn order_key(x : Float) -> Int32 {
  let b = (x + 0.0).reinterpret_as_int()
  b ^ ((b >> 31) & 0x7FFFFFFF)
}

///|
fn block_end(lo : Int32, n : Int32) -> Int32 {
  if n - lo > block {
    lo + block
  } else {
    n
  }
}

///|
/// The bin of a value with k cuts at or below it: k + 1, or 0 when missing.
fn bin_code(v : Float, k : Int32) -> UInt16 {
  (if v.is_nan() { 0 } else { k + 1 }).to_uint16()
}

///|
/// The up to `cap` values that occur most often in sorted `vals` (ties to
/// the smaller value), in increasing order.
fn frequent_values(vals : FixedArray[Float], cap : Int32) -> Array[Float] {
  frequent_cover(vals, cap).0
}

///|
/// frequent_values with the number of values they cover.
fn frequent_cover(
  vals : FixedArray[Float],
  cap : Int32,
) -> (Array[Float], Int32) {
  let distinct : Array[Float] = []
  let counts : Array[Int32] = []
  for i, v in vals {
    if i == 0 || v != vals[i - 1] {
      distinct.push(v)
      counts.push(1)
    } else {
      counts[counts.length() - 1] = counts[counts.length() - 1] + 1
    }
  }
  if distinct.length() <= cap {
    return (distinct, vals.length())
  }
  let idx = Array::makei(distinct.length(), i => i)
  idx.sort_by((a, b) => {
    if counts[a] != counts[b] {
      counts[b].compare(counts[a])
    } else {
      a.compare(b)
    }
  })
  let keep = idx[:cap].to_owned()
  let mut covered = 0
  for i in keep {
    covered += counts[i]
  }
  keep.sort()
  (keep.map(i => distinct[i]), covered)
}

///|
/// Columns copied out of the row-major matrix at once when binning.
const COL_STRIP : Int32 = 16

///|
/// Copies columns f .. f + w of the row-major data into bufs[0 .. w].
fn copy_strip(
  data : FixedArray[Float],
  cols : Int32,
  rows : Int32,
  f : Int32,
  w : Int32,
  bufs : FixedArray[FixedArray[Float]],
) -> Unit {
  if w == 16 {
    // the sixteen buffers in locals: the stores could alias the array of
    // buffers for the C compiler, which would reload it for every value
    let c0 = bufs[0]
    let c1 = bufs[1]
    let c2 = bufs[2]
    let c3 = bufs[3]
    let c4 = bufs[4]
    let c5 = bufs[5]
    let c6 = bufs[6]
    let c7 = bufs[7]
    let c8 = bufs[8]
    let c9 = bufs[9]
    let c10 = bufs[10]
    let c11 = bufs[11]
    let c12 = bufs[12]
    let c13 = bufs[13]
    let c14 = bufs[14]
    let c15 = bufs[15]
    for r in 0..<rows {
      let at = r * cols + f
      c0.unsafe_set(r, data.unsafe_get(at))
      c1.unsafe_set(r, data.unsafe_get(at + 1))
      c2.unsafe_set(r, data.unsafe_get(at + 2))
      c3.unsafe_set(r, data.unsafe_get(at + 3))
      c4.unsafe_set(r, data.unsafe_get(at + 4))
      c5.unsafe_set(r, data.unsafe_get(at + 5))
      c6.unsafe_set(r, data.unsafe_get(at + 6))
      c7.unsafe_set(r, data.unsafe_get(at + 7))
      c8.unsafe_set(r, data.unsafe_get(at + 8))
      c9.unsafe_set(r, data.unsafe_get(at + 9))
      c10.unsafe_set(r, data.unsafe_get(at + 10))
      c11.unsafe_set(r, data.unsafe_get(at + 11))
      c12.unsafe_set(r, data.unsafe_get(at + 12))
      c13.unsafe_set(r, data.unsafe_get(at + 13))
      c14.unsafe_set(r, data.unsafe_get(at + 14))
      c15.unsafe_set(r, data.unsafe_get(at + 15))
    }
    return
  }
  for r in 0..<rows {
    let at = r * cols + f
    for j in 0..<w {
      bufs.unsafe_get(j).unsafe_set(r, data.unsafe_get(at + j))
    }
  }
}

///|
fn bin_matrix(
  m : DenseMatrix,
  max_bin : Int32,
  is_cat : FixedArray[Bool],
  cat_cap : Int32,
  cat_room : Int32,
) -> Binned {
  let rows = m.rows
  let cols = m.cols
  let cuts : Array[Float] = []
  let cut_start = FixedArray::make(cols + 1, 0)
  let by_col = u16s_make(rows * cols)
  let missing = FixedArray::make(cols, false)
  let lowest : FixedArray[Float] = FixedArray::make(cols, 0.0)
  // the columns are copied out a strip at a time, so that the passes over
  // each column read contiguous memory rather than one float per row
  let strip = if cols < COL_STRIP { cols } else { COL_STRIP }
  let colbuf = FixedArray::makei(strip, _ => {
    FixedArray::make(rows, (0.0 : Float))
  })
  for f in 0..<cols {
    if f % strip == 0 {
      let w = if cols - f < strip { cols - f } else { strip }
      copy_strip(m.data, cols, rows, f, w, colbuf)
    }
    let col = colbuf[f % strip]
    let vals = sorted_column({ data: col, rows, cols: 1, }, 0)
    let base = cuts.length()
    let n = vals.length()
    if n > 0 {
      lowest[f] = vals[0]
    }
    if is_cat[f] {
      // a bin for each of the cat_cap most frequent categories, in
      // increasing order; the others share bin 0 with missing values.
      // When those leave more than 1% of the rows in bin 0, as many as
      // the slots allow (cat_room) instead: a long tail of categories
      // holds information that merging it with missing values loses
      let (kept, covered) = frequent_cover(vals, cat_cap)
      let kept = if cat_room > cat_cap &&
        covered.to_double() < 0.99 * n.to_double() {
        frequent_values(vals, cat_room)
      } else {
        kept
      }
      if kept.length() > 0 {
        lowest[f] = kept[0]
      }
      for i in 1..<kept.length() {
        cuts.push(kept[i])
      }
      cut_start[f + 1] = cuts.length()
      for r in 0..<rows {
        let v = col[r]
        let mut b = 0
        if !v.is_nan() {
          // binary search for v among kept
          let mut a = 0
          let mut z = kept.length()
          while a < z {
            let mid = (a + z) / 2
            if kept[mid] < v {
              a = mid + 1
            } else {
              z = mid
            }
          }
          if a < kept.length() && kept[a] == v {
            b = a + 1
          }
        }
        if b == 0 {
          missing[f] = true
        }
        by_col.unsafe_set(f * rows + r, b.to_uint16())
      }
      continue
    }
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
    let fv = FixedArray::makei(hi - lo, fn(i) { cuts[lo + i] })
    let fc = FixedArray::makei(hi - lo, fn(i) { order_key(fv[i]) })
    for lo = 0; lo < rows; lo = lo + block {
      bin_block(col, 1, 0, fv, fc, by_col, f * rows, lo, block_end(lo, rows))
    }
    for r in 0..<rows {
      if col[r].is_nan() {
        missing[f] = true
        break
      }
    }
  }
  {
    cols,
    bins: u16s_make(0),
    by_col,
    cuts,
    cut_start,
    missing,
    is_cat,
    lowest,
  }
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
  // categorical splits: the sorted category codes that go right
  cats : Array[Array[Int32]]
  // each split node's split, for routing rows the tree did not grow on
  splits : Array[Split?]
}

///|
fn GrowTree::new() -> GrowTree {
  {
    cats: [],
    splits: [],
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
  self.cats.push([])
  self.splits.push(None)
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
  // for a categorical split, whether each bin (1 .. categories; bin 0 is
  // missing) goes right; empty for a numerical split
  right : FixedArray[Bool]
}

///|
let no_cats : FixedArray[Bool] = []

///|
priv struct Grower {
  data : Binned
  p : TrainParams
  // gradient and hessian of row r at 2r and 2r + 1
  gh : F32s
  // the same pair quantised to integers and packed as (g << 32) + h, so a
  // histogram slot is one integer add; gs and hs are the scales
  gq : QArr
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
  // copy levels carry each row's quantised pair in its bin words (cq
  // unused): loss-guided, one-byte bins, see qpad_ok
  qpad : Bool
  // when narrow, the original bins in the same one-byte form, word_bins
  // to a word: feature f of row r is byte f % word_bins of
  // bins8[r * words + f / word_bins], so
  // a row's bins load and copy a word at a time
  bins8 : Words
  words : Int32
  // and column by column: by8[f * rows + r]
  by8 : FixedArray[Byte]
  cbins : FixedArray[Words]
  cbase : FixedArray[Int32]
  // cbase per byte of the row words; the padding bytes of a last, partial
  // word count into spare slots past nbins
  wbase : FixedArray[Int32]
  cbins16 : FixedArray[U16s]
  cq : FixedArray[QArr]
  cgh : FixedArray[F32s]
  orig : FixedArray[I32s]
  mut ihs : Double // 1 / hs
  // rows of the nodes at the current depth; partitioning writes the
  // children's rows to `scratch`, and the two swap while the children grow
  mut rows : I32s
  mut scratch : I32s
  // leaf node reached by each copied position (orig[0] maps positions to
  // rows); a leaf above the copy depth fills its own positions and orig[0]
  leaf_at : I32s
  nbins : Int32 // total histogram slots: per feature, missing + (cuts + 1)
  hist_start : FixedArray[Int32]
  // splitmix64 state for row and feature sampling
  mut rng : UInt64
  // features the current tree (tree_feats[0 ..< tree_n]) and node may use
  tree_feats : FixedArray[Int32]
  mut tree_n : Int32
  node_on : FixedArray[Bool]
}

///|
/// The next splitmix64 output.
fn Grower::next(self : Grower) -> UInt64 {
  self.rng = self.rng + 0x9E3779B97F4A7C15UL
  let mut z = self.rng
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL
  z ^ (z >> 31)
}

///|
/// Uniform in [0, 1).
fn Grower::uniform(self : Grower) -> Double {
  (self.next() >> 11).to_double() * 1.1102230246251565e-16
}

///|
/// Moves `k` of feats[0 ..< m], drawn uniformly, to the front.
fn Grower::draw(
  self : Grower,
  feats : FixedArray[Int32],
  m : Int32,
  k : Int32,
) -> Unit {
  for i in 0..<k {
    let j = i + (self.uniform() * (m - i).to_double()).to_int()
    let j = if j >= m { m - 1 } else { j }
    let x = feats[i]
    feats[i] = feats[j]
    feats[j] = x
  }
}

///|
/// How many of m features a fraction keeps: at least one.
fn sample_count(m : Int32, frac : Double) -> Int32 {
  let k = (frac * m.to_double()).to_int()
  if k < 1 {
    1
  } else if k > m {
    m
  } else {
    k
  }
}

///|
/// Picks the features of a new tree.
fn Grower::sample_tree(self : Grower) -> Unit {
  let m = self.data.cols
  for f in 0..<m {
    self.tree_feats[f] = f
  }
  if self.p.colsample_bytree < 1.0 {
    let k = sample_count(m, self.p.colsample_bytree)
    self.draw(self.tree_feats, m, k)
    self.tree_n = k
  } else {
    self.tree_n = m
  }
}

///|
/// Marks the features a node may split on in node_on.
fn Grower::sample_node(self : Grower) -> Unit {
  let on = self.node_on
  for f in 0..<on.length() {
    on[f] = false
  }
  let m = self.tree_n
  let k = if self.p.colsample_bynode < 1.0 {
    let k = sample_count(m, self.p.colsample_bynode)
    self.draw(self.tree_feats, m, k)
    k
  } else {
    m
  }
  for i in 0..<k {
    on[self.tree_feats[i]] = true
  }
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
/// Minimal variance sampling: row r is kept with chance
/// min(1, s_r / mu), s_r = sqrt(g^2 + lambda h^2) with lambda the squared
/// ratio of mean |g| to mean h, and mu chosen so the chances sum to
/// fraction * n. Kept rows' g and h are divided by their chance, the others
/// set to zero; the quantised pairs are rescaled to the new bounds. Kept
/// rows go to rows[0:m] and the others to out_rows; returns the sums of g
/// and h, and m.
fn Grower::mvs_sample(
  self : Grower,
  n : Int32,
  fraction : Double,
  out_rows : I32s,
) -> (Double, Double, Int32) {
  let gh = self.gh
  let mut ag = 0.0
  let mut sh = 0.0
  for r in 0..<n {
    ag += gh.unsafe_get(r * 2).to_double().abs()
    sh += gh.unsafe_get(r * 2 + 1).to_double()
  }
  let lam = if sh > 0.0 { ag / sh * (ag / sh) } else { 0.0 }
  let sc = FixedArray::make(n, 0.0)
  let mut smax = 0.0
  for r in 0..<n {
    let g = gh.unsafe_get(r * 2).to_double()
    let h = gh.unsafe_get(r * 2 + 1).to_double()
    let v = (g * g + lam * h * h).sqrt()
    sc.unsafe_set(r, v)
    if v > smax {
      smax = v
    }
  }
  let target = fraction * n.to_double()
  let mu = match mvs_threshold(sc, n, target) {
    Some(mu) => mu
    None => {
      // sum of min(1, s / mu) falls as mu grows; at hi it is at most target
      let mut lo = 0.0
      let mut hi = if target > 0.0 {
        smax * n.to_double() / target
      } else {
        0.0
      }
      for _ in 0..<40 {
        let mu = 0.5 * (lo + hi)
        let mut f = 0.0
        for r in 0..<n {
          let c = sc.unsafe_get(r) / mu
          f += if c < 1.0 { c } else { 1.0 }
        }
        if f > target {
          lo = mu
        } else {
          hi = mu
        }
      }
      hi
    }
  }
  let mut gsum = 0.0
  let mut hsum = 0.0
  let rows = self.rows
  let mut m = 0
  let mut u = 0
  for r in 0..<n {
    let s = sc.unsafe_get(r)
    let c = if mu > 0.0 { s / mu } else { 1.0 }
    if s > 0.0 && (c >= 1.0 || self.uniform() < c) {
      let w = if c < 1.0 { 1.0 / c } else { 1.0 }
      let g = Float::from_double(gh.unsafe_get(r * 2).to_double() * w)
      let h = Float::from_double(gh.unsafe_get(r * 2 + 1).to_double() * w)
      gh.unsafe_set(r * 2, g)
      gh.unsafe_set(r * 2 + 1, h)
      gsum += g.to_double()
      hsum += h.to_double()
      rows.unsafe_set(m, r)
      m += 1
    } else {
      gh.unsafe_set(r * 2, 0.0)
      gh.unsafe_set(r * 2 + 1, 0.0)
      out_rows.unsafe_set(u, r)
      u += 1
    }
  }
  self.requantize(n)
  (gsum, hsum, m)
}

///|
/// The mu at which the sum over rows of min(1, s_r / mu) equals target,
/// solved exactly in two passes: the scores are counted and summed in
/// buckets by their leading bits (monotone for non-negative doubles), which
/// locates the bucket mu falls in, and that bucket's scores are then sorted.
/// None when no positive mu fits (too few positive scores).
fn mvs_threshold(
  sc : FixedArray[Double],
  n : Int32,
  target : Double,
) -> Double? {
  let shift = 48
  let nb = 1 << (63 - shift)
  let cnt = FixedArray::make(nb, 0)
  let sum = FixedArray::make(nb, 0.0)
  let mut total = 0.0
  for r in 0..<n {
    let v = sc.unsafe_get(r)
    let b = (v.reinterpret_as_uint64() >> shift).to_int()
    cnt.unsafe_set(b, cnt.unsafe_get(b) + 1)
    sum.unsafe_set(b, sum.unsafe_get(b) + v)
    total += v
  }
  // f(edge of bucket b) = rows in buckets >= b + (sum below b) / edge;
  // f falls as b rises, so take the highest b with f(edge b) >= target
  let mut above = 0 // rows in buckets > b
  let mut above_sum = 0.0
  let mut b = nb - 1
  let mut found = -1
  while b >= 1 {
    let c = cnt[b]
    if c > 0 {
      let edge = (b.to_uint64() << shift).reinterpret_as_double()
      let below = total - above_sum - sum[b]
      if (above + c).to_double() + below / edge >= target {
        found = b
        break
      }
      above += c
      above_sum += sum[b]
    }
    b -= 1
  }
  guard found >= 1 else { None }
  let below = total - above_sum - sum[found]
  // the bucket's scores, largest first
  let vs = Array(capacity=cnt[found])
  for r in 0..<n {
    let v = sc.unsafe_get(r)
    if (v.reinterpret_as_uint64() >> shift).to_int() == found {
      vs.push(v)
    }
  }
  vs.sort()
  vs.rev_in_place()
  // with the j largest of the bucket kept for sure, mu = (sum of the rest
  // below it) / (target - rows kept for sure), valid when it lies between
  // the j-th and (j + 1)-th scores
  let mut rest = sum[found]
  for j in 0..<=vs.length() {
    if j > 0 {
      rest -= vs[j - 1]
    }
    let k = (above + j).to_double()
    if target - k <= 0.0 {
      break
    }
    let mu = (below + rest) / (target - k)
    let upper = if j > 0 { vs[j - 1] } else { 1.0e308 }
    let lower = if j < vs.length() { vs[j] } else { 0.0 }
    if mu > lower && mu <= upper && mu > 0.0 {
      return Some(mu)
    }
  }
  None
}

///|
/// Sets the quantisation scales from the largest |g| and h of the n rows
/// and quantises every row's pair again.
fn Grower::requantize(self : Grower, n : Int32) -> Unit {
  let gh = self.gh
  let mut gmax = 0.0
  let mut hmax = 0.0
  for r in 0..<n {
    let g = gh.unsafe_get(r * 2).to_double().abs()
    let h = gh.unsafe_get(r * 2 + 1).to_double()
    if g > gmax {
      gmax = g
    }
    if h > hmax {
      hmax = h
    }
  }
  self.set_scales(n, gmax * 1.0000001, hmax * 1.0000001)
  let gq = self.gq
  for r in 0..<n {
    self.pack_at(
      gq,
      r,
      gh.unsafe_get(r * 2).to_double(),
      gh.unsafe_get(r * 2 + 1).to_double(),
    )
  }
}

///|
/// exp(0.7) in float, XGBoost's default max_delta_step for count:poisson.
let poisson_hess_scale : Double = 2.0137526834646735

///|
fn log1p(x : Double) -> Double {
  @math.ln_1p(x)
}

///|
/// The least hessian a child of a node (hessian h, count rows) may have:
/// min_child_weight, or min_data_in_leaf rows at the node's mean hessian.
fn Grower::min_hessian(self : Grower, h : Double, count : Int32) -> Double {
  let m = self.p.min_child_weight
  if self.p.min_data_in_leaf > 0.0 && count > 0 {
    let d = self.p.min_data_in_leaf * h / count.to_double()
    if d > m {
      d
    } else {
      m
    }
  } else {
    m
  }
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
    let words = self.words
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let src = r * words
      let dst = i * words
      for w in 0..<words {
        tb.unsafe_set(dst + w, b8.unsafe_get(src + w))
      }
      q_copy(tq, i, q, r)
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
      q_copy(tq, i, q, r)
      tg.unsafe_set(i * 2, gh.unsafe_get(r * 2))
      tg.unsafe_set(i * 2 + 1, gh.unsafe_get(r * 2 + 1))
      to.unsafe_set(i, r)
      rows.unsafe_set(i, i)
    }
  }
}

///|
fn Grower::best_split(
  self : Grower,
  hist : Hist,
  g : Double,
  h : Double,
  count : Int32,
) -> Split? {
  let parent = self.score(g, h)
  let mcw = self.min_hessian(h, count)
  let mut best : Split? = None
  let mut best_gain = self.p.gamma
  let sampled = self.p.colsample_bytree < 1.0 || self.p.colsample_bynode < 1.0
  if sampled {
    self.sample_node()
  }
  for f in 0..<self.data.cols {
    if sampled && !self.node_on[f] {
      continue
    }
    // slot hist_start[f] holds the feature's missing values, then its bins
    let start = self.hist_start[f] + 1
    let n = self.data.cut_start[f + 1] - self.data.cut_start[f] + 1
    if n < 2 {
      continue
    }
    guard start + n - 1 <= hist_len(hist) else { abort("short histogram") }
    let mg = hist_g(hist, start - 1) * self.igs
    let mh = hist_h(hist, start - 1) * self.ihs
    if self.data.is_cat[f] {
      if self.cat_split(
          hist, f, start, n, g, h, count, mg, mh, parent, best_gain,
        )
        is Some(s) {
        best_gain = s.gain
        best = Some(s)
      }
      continue
    }
    // the quantised sums are integers below 2^53, exact in a Double
    let mut qg = 0.0
    let mut qh = 0.0
    for b in 0..<(n - 1) {
      let bg = hist_g(hist, start + b)
      let bh = hist_h(hist, start + b)
      // an empty bin repeats the previous split, which cannot gain more
      if bg == 0.0 && bh == 0.0 {
        continue
      }
      qg = qg + bg
      qh = qh + bh
      let gl = qg * self.igs
      let hl = qh * self.ihs
      // missing values to the right
      let gr = g - gl
      let hr = h - hl
      if hl >= mcw && hr >= mcw {
        let gain = self.score(gl, hl) + self.score(gr, hr) - parent
        if gain > best_gain {
          best_gain = gain
          best = Some({
            gain,
            feature: f,
            bin: b,
            default_left: false,
            gl,
            hl,
            right: no_cats,
          })
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
              right: no_cats,
            })
          }
        }
      }
    }
  }
  best
}

///|
/// The best categorical split of feature f (categories in hist slots
/// start ..< start + n) that gains more than best_gain: one category
/// against the rest when the node has at most max_cat_to_onehot
/// categories, else a set of categories taken from either end of their
/// order by g / (h + cat_smooth). The set goes right; categories not in
/// it, including ones unseen in training, go left.
fn Grower::cat_split(
  self : Grower,
  hist : Hist,
  f : Int32,
  start : Int32,
  n : Int32,
  g : Double,
  h : Double,
  count : Int32,
  mg : Double,
  mh : Double,
  parent : Double,
  best_gain : Double,
) -> Split? {
  let mcw = self.min_hessian(h, count)
  let ids : Array[Int32] = []
  let cg : Array[Double] = []
  let ch : Array[Double] = []
  for b in 0..<n {
    let bg = hist_g(hist, start + b) * self.igs
    let bh = hist_h(hist, start + b) * self.ihs
    if bh > 0.0 || bg != 0.0 {
      ids.push(b)
      cg.push(bg)
      ch.push(bh)
    }
  }
  let k = ids.length()
  if k < 2 {
    return None
  }
  let mut best_gain = best_gain
  // the best right set: ids at order[lo ..< hi]
  let mut best_lo = -1
  let mut best_hi = -1
  let mut best_dl = false
  let mut best_gl = 0.0
  let mut best_hl = 0.0
  let order : Array[Int32] = []
  let onehot = k <= self.p.max_cat_to_onehot
  let lam = if onehot { 0.0 } else { self.p.cat_l2 }
  // try right sums (gr, hr) with missing left and right
  let try_set = fn(gr : Double, hr : Double, lo : Int32, hi : Int32) {
    // missing values and the categories without a bin go left, as
    // categories absent from the right set do in the model
    for dl in 1..<2 {
      // dl 1: missing left (in the rest); dl 0: missing right
      let (gr2, hr2) = if dl == 1 { (gr, hr) } else { (gr + mg, hr + mh) }
      let gl = g - gr2
      let hl = h - hr2
      if hl >= mcw && hr2 >= mcw {
        let gain = gl * gl / (hl + self.p.lambda + lam) +
          gr2 * gr2 / (hr2 + self.p.lambda + lam) -
          parent
        if gain > best_gain {
          best_gain = gain
          best_lo = lo
          best_hi = hi
          best_dl = dl == 1
          best_gl = gl
          best_hl = hl
        }
      }
    }
  }
  if onehot {
    for i in 0..<k {
      order.push(i)
    }
    for i in 0..<k {
      try_set(cg[i], ch[i], i, i + 1)
    }
  } else {
    // categories with enough rows, by their mean gradient
    let per_row = h / count.to_double()
    for i in 0..<k {
      if ch[i] >= self.p.min_data_per_group * per_row {
        order.push(i)
      }
    }
    let m = order.length()
    if m < 2 {
      return None
    }
    order.sort_by_key(i => cg[i] / (ch[i] + self.p.cat_smooth))
    let limit = if m - 1 < self.p.max_cat_threshold {
      m - 1
    } else {
      self.p.max_cat_threshold
    }
    let mut gr = 0.0
    let mut hr = 0.0
    for j in 0..<limit {
      gr += cg[order[j]]
      hr += ch[order[j]]
      try_set(gr, hr, 0, j + 1)
    }
    gr = 0.0
    hr = 0.0
    for j in 0..<limit {
      gr += cg[order[m - 1 - j]]
      hr += ch[order[m - 1 - j]]
      try_set(gr, hr, m - 1 - j, m)
    }
  }
  if best_lo < 0 {
    return None
  }
  // bin b + 1 holds category b; bin 0 is missing
  let right = FixedArray::make(n + 1, false)
  for j in best_lo..<best_hi {
    right[ids[order[j]] + 1] = true
  }
  Some({
    gain: best_gain,
    feature: f,
    bin: 0,
    default_left: best_dl,
    gl: best_gl,
    hl: best_hl,
    right,
  })
}

///|
/// For one-byte bins: whether each byte of the split feature goes left
/// (1) or right (0).
fn Grower::left_table(self : Grower, s : Split) -> FixedArray[Int32] {
  let add = self.cbase[s.feature] - self.hist_start[s.feature]
  FixedArray::makei(256, fn(byte) { self.goes_left(s, byte + add) })
}

///|
/// Whether a row in bin `bin` of the split feature (0: missing) goes left:
/// 1 or 0.
fn Grower::goes_left(self : Grower, s : Split, bin : Int32) -> Int32 {
  ignore(self)
  if s.right.length() > 0 {
    if bin == 0 {
      if s.default_left {
        1
      } else {
        0
      }
    } else if bin < s.right.length() && s.right[bin] {
      0
    } else {
      1
    }
  } else {
    let miss = if s.default_left { 0 } else { 1 << 20 }
    let key = bin + (((bin - 1) >> 31) & miss)
    if key <= s.bin + 1 {
      1
    } else {
      0
    }
  }
}

///|
/// goes_left for every bin of the split feature: index 0 (missing) to the
/// feature's last bin.
fn Grower::bin_table(self : Grower, s : Split) -> FixedArray[Int32] {
  let f = s.feature
  FixedArray::makei(self.hist_start[f + 1] - self.hist_start[f], fn(bin) {
    self.goes_left(s, bin)
  })
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
    let go = self.left_table(s)
    if level > 0 {
      // the feature's byte of position r is in word r * words + column
      let cb = self.cbins[level - 1]
      let words = self.words
      let column = s.feature / word_bins
      let shift = s.feature % word_bins * 8
      for i in lo..<hi {
        let r = rows.unsafe_get(i)
        let byte = word_byte(cb.unsafe_get(r * words + column), shift)
        let left = go.unsafe_get(byte)
        dst.unsafe_set(a, r)
        dst.unsafe_set(b, r)
        a += left
        b -= 1 - left
      }
    } else {
      let cb = self.by8
      let column = s.feature * self.leaf_at.length()
      for i in lo..<hi {
        let r = rows.unsafe_get(i)
        let left = go.unsafe_get(cb.unsafe_get(r + column).to_int())
        dst.unsafe_set(a, r)
        dst.unsafe_set(b, r)
        a += left
        b -= 1 - left
      }
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
    if s.right.length() > 0 {
      let go = self.bin_table(s)
      for i in lo..<hi {
        let r = rows.unsafe_get(i)
        let left = go[bins.unsafe_get(r * stride + column).to_int() - sub]
        dst.unsafe_set(a, r)
        dst.unsafe_set(b, r)
        a += left
        b -= 1 - left
      }
      return a
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
/// The category codes a categorical split sends right, in increasing order.
fn Grower::right_codes(self : Grower, s : Split) -> Array[Int32] {
  let f = s.feature
  let out = []
  for bin in 1..<s.right.length() {
    if s.right[bin] {
      let v = if bin == 1 {
        self.data.lowest[f]
      } else {
        self.data.cuts[self.data.cut_start[f] + bin - 2]
      }
      out.push(v.to_int())
    }
  }
  out
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
    let words = self.words
    let column = s.feature * n
    let go = self.left_table(s)
    if self.qpad {
      // words == 2; the pair goes into the second word's high half
      for i in lo..<hi {
        let r = rows.unsafe_get(i)
        let left = go.unsafe_get(by8.unsafe_get(r + column).to_int())
        let p = b + (a - b) * left
        let src = r << 1
        let at = p << 1
        tb.unsafe_set(at, b8.unsafe_get(src))
        tb.unsafe_set(at + 1, q_pad(b8.unsafe_get(src + 1), q.unsafe_get(r)))
        tg.unsafe_set(p * 2, gh.unsafe_get(r * 2))
        tg.unsafe_set(p * 2 + 1, gh.unsafe_get(r * 2 + 1))
        to.unsafe_set(p, r)
        a += left
        b -= 1 - left
      }
      for i in lo..<hi {
        dst.unsafe_set(i, i)
      }
      return a
    }
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let left = go.unsafe_get(by8.unsafe_get(r + column).to_int())
      let p = b + (a - b) * left
      let src = r * words
      let at = p * words
      for w in 0..<words {
        tb.unsafe_set(at + w, b8.unsafe_get(src + w))
      }
      q_copy(tq, p, q, r)
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
    let cat = s.right.length() > 0
    let go = if cat { self.bin_table(s) } else { FixedArray::make(0, 0) }
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let bin = by_col.unsafe_get(r + column).to_int()
      let key = bin + (((bin - 1) >> 31) & miss)
      let left = if cat { go[bin] } else if key <= limit { 1 } else { 0 }
      let p = b + (a - b) * left
      let src = r * cols
      let at = p * cols
      for f in 0..<cols {
        tw.unsafe_set(at + f, bins.unsafe_get(src + f))
      }
      q_copy(tq, p, q, r)
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
/// Copy levels for loss-guided trees, which grow deep along their largest
/// nodes: below the first copy, a fresh one every recopy_every levels.
/// (Depth-wise trees gained nothing from more than one.)
let max_copies = 6

///|
let recopy_every = 3

///|
/// Partitions rows[lo:hi] (positions of copy level `level`, at least 1,
/// one-byte bins) like `partition`, copying each row's bins, quantised
/// gradients and level-1 position to level + 1 at its new position, so the
/// children's rows are dense again; the children's rows in scratch are
/// those positions. (Leaves read exact gradients at level 1 through orig.)
fn Grower::recopy(
  self : Grower,
  lo : Int32,
  hi : Int32,
  s : Split,
  level : Int32,
) -> Int32 {
  let rows = self.rows
  let dst = self.scratch
  let words = self.words
  let cb = self.cbins[level - 1]
  let q = self.cq[level - 1]
  let tb = self.cbins[level]
  let tq = self.cq[level]
  let to = self.orig[level]
  let column = s.feature / word_bins
  let shift = s.feature % word_bins * 8
  let go = self.left_table(s)
  let mut a = lo
  let mut b = hi - 1
  // orig[level] holds level-1 positions: level 1's own, or mapped down
  if self.qpad {
    // words == 2, the pairs ride in the bin words
    if level == 1 {
      for i in lo..<hi {
        let r = rows.unsafe_get(i)
        let src = r << 1
        let left = go.unsafe_get(word_byte(cb.unsafe_get(src + column), shift))
        let p = b + (a - b) * left
        let at = p << 1
        tb.unsafe_set(at, cb.unsafe_get(src))
        tb.unsafe_set(at + 1, cb.unsafe_get(src + 1))
        to.unsafe_set(p, r)
        a += left
        b -= 1 - left
      }
    } else {
      let fo = self.orig[level - 1]
      for i in lo..<hi {
        let r = rows.unsafe_get(i)
        let src = r << 1
        let left = go.unsafe_get(word_byte(cb.unsafe_get(src + column), shift))
        let p = b + (a - b) * left
        let at = p << 1
        tb.unsafe_set(at, cb.unsafe_get(src))
        tb.unsafe_set(at + 1, cb.unsafe_get(src + 1))
        to.unsafe_set(p, fo.unsafe_get(r))
        a += left
        b -= 1 - left
      }
    }
  } else if words == 2 {
    // the common case (up to 16 features), with fewer live values
    if level == 1 {
      for i in lo..<hi {
        let r = rows.unsafe_get(i)
        let src = r << 1
        let left = go.unsafe_get(word_byte(cb.unsafe_get(src + column), shift))
        let p = b + (a - b) * left
        let at = p << 1
        tb.unsafe_set(at, cb.unsafe_get(src))
        tb.unsafe_set(at + 1, cb.unsafe_get(src + 1))
        q_copy(tq, p, q, r)
        to.unsafe_set(p, r)
        a += left
        b -= 1 - left
      }
    } else {
      let fo = self.orig[level - 1]
      for i in lo..<hi {
        let r = rows.unsafe_get(i)
        let src = r << 1
        let left = go.unsafe_get(word_byte(cb.unsafe_get(src + column), shift))
        let p = b + (a - b) * left
        let at = p << 1
        tb.unsafe_set(at, cb.unsafe_get(src))
        tb.unsafe_set(at + 1, cb.unsafe_get(src + 1))
        q_copy(tq, p, q, r)
        to.unsafe_set(p, fo.unsafe_get(r))
        a += left
        b -= 1 - left
      }
    }
  } else if level == 1 {
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let src = r * words
      let left = go.unsafe_get(word_byte(cb.unsafe_get(src + column), shift))
      let p = b + (a - b) * left
      let at = p * words
      if words == 2 {
        tb.unsafe_set(at, cb.unsafe_get(src))
        tb.unsafe_set(at + 1, cb.unsafe_get(src + 1))
      } else {
        for w in 0..<words {
          tb.unsafe_set(at + w, cb.unsafe_get(src + w))
        }
      }
      q_copy(tq, p, q, r)
      to.unsafe_set(p, r)
      a += left
      b -= 1 - left
    }
  } else {
    let fo = self.orig[level - 1]
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      let src = r * words
      let left = go.unsafe_get(word_byte(cb.unsafe_get(src + column), shift))
      let p = b + (a - b) * left
      let at = p * words
      if words == 2 {
        tb.unsafe_set(at, cb.unsafe_get(src))
        tb.unsafe_set(at + 1, cb.unsafe_get(src + 1))
      } else {
        for w in 0..<words {
          tb.unsafe_set(at + w, cb.unsafe_get(src + w))
        }
      }
      q_copy(tq, p, q, r)
      to.unsafe_set(p, fo.unsafe_get(r))
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
/// Whether a node at `depth` and copy level `level` partitions into the
/// next level's copy.
fn Grower::copies(self : Grower, depth : Int32, level : Int32) -> Bool {
  level < self.cq.length() &&
  (self.p.max_depth <= 0 || depth + 1 < self.p.max_depth) &&
  (if level == 0 {
    depth + 1 == self.compact_at[0]
  } else {
    self.narrow && depth + 1 == self.compact_at[level]
  })
}

///|
/// Partitions rows[lo:hi] of a node at `depth` and level `level`; returns
/// the right part's start and the children's level.
fn Grower::split_rows(
  self : Grower,
  lo : Int32,
  hi : Int32,
  s : Split,
  depth : Int32,
  level : Int32,
) -> (Int32, Int32) {
  if !self.copies(depth, level) {
    (self.partition(lo, hi, s, level), level)
  } else if level == 0 {
    (self.partition_copy(lo, hi, s), 1)
  } else {
    (self.recopy(lo, hi, s, level), level + 1)
  }
}

///|
/// Makes node a leaf of rows[lo:hi]: its value from the rows' exact
/// gradient sums, and leaf_at (and orig[0] above the copy depth) for
/// adding it to the margins.
fn Grower::finish_leaf(
  self : Grower,
  tree : GrowTree,
  node : Int32,
  lo : Int32,
  hi : Int32,
  level : Int32,
) -> Unit {
  let mut eg = 0.0
  let mut eh = 0.0
  // gradients are copied to level 1 only; deeper positions read level
  // 1's through orig
  let gh = if level > 0 { self.cgh[0] } else { self.gh }
  let rows = self.rows
  let leaf_at = self.leaf_at
  // indices are in range: rows[lo:hi] are rows or copied positions
  // positions in rows[lo:hi] are those of one copy level (there is at
  // most one); a leaf above it is given the positions lo..hi, which no
  // copied subtree uses
  if level > 1 {
    // positions of deeper levels map to level 1's in orig[level - 1]
    let o = self.orig[level - 1]
    for i in lo..<hi {
      let r = o.unsafe_get(rows.unsafe_get(i))
      eg += gh.unsafe_get(r * 2).to_double()
      eh += gh.unsafe_get(r * 2 + 1).to_double()
      leaf_at.unsafe_set(r, node)
    }
  } else if level > 0 {
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      eg += gh.unsafe_get(r * 2).to_double()
      eh += gh.unsafe_get(r * 2 + 1).to_double()
      leaf_at.unsafe_set(r, node)
    }
  } else {
    let o = self.orig[0]
    for i in lo..<hi {
      let r = rows.unsafe_get(i)
      eg += gh.unsafe_get(r * 2).to_double()
      eh += gh.unsafe_get(r * 2 + 1).to_double()
      o.unsafe_set(i, r)
      leaf_at.unsafe_set(i, node)
    }
  }
  let leaf = Float::from_double(self.weight(eg, eh) * self.p.eta)
  tree.split_cond[node] = leaf
}

///|
/// Sets node's split in the tree and adds its two children; returns them.
fn Grower::add_split(
  self : Grower,
  tree : GrowTree,
  node : Int32,
  s : Split,
  g : Double,
  h : Double,
) -> (Int32, Int32) {
  let gr = g - s.gl
  let hr = h - s.hl
  let l = tree.add(node, self.weight(s.gl, s.hl), s.hl)
  let r = tree.add(node, self.weight(gr, hr), hr)
  tree.left[node] = l
  tree.right[node] = r
  tree.split_index[node] = s.feature
  if s.right.length() > 0 {
    tree.split_cond[node] = 0.0
    tree.cats[node] = self.right_codes(s)
  } else {
    tree.split_cond[node] = self.data.cuts[self.data.cut_start[s.feature] +
      s.bin]
  }
  tree.default_left[node] = s.default_left
  tree.loss_change[node] = Float::from_double(s.gain)
  tree.splits[node] = Some(s)
  (l, r)
}

///|
/// A leaf waiting to be split by loss-guided growth.
priv struct Pending {
  node : Int32
  lo : Int32
  hi : Int32
  hist : Hist
  slot : Int32 // which histogram buffer holds hist
  g : Double
  h : Double
  depth : Int32
  level : Int32
  split : Split?
  // whether the leaf's rows are in the scratch buffer (else in rows)
  in_scratch : Bool
}

///|
fn pending_gain(p : Pending) -> Double {
  match p.split {
    Some(s) => s.gain
    None => 0.0
  }
}

///|
fn heap_push(heap : Array[Pending], p : Pending) -> Unit {
  heap.push(p)
  let mut i = heap.length() - 1
  while i > 0 {
    let up = (i - 1) / 2
    if pending_gain(heap[up]) >= pending_gain(heap[i]) {
      break
    }
    let t = heap[up]
    heap[up] = heap[i]
    heap[i] = t
    i = up
  }
}

///|
/// Removes and returns the leaf with the largest gain.
fn heap_pop(heap : Array[Pending]) -> Pending {
  let top = heap[0]
  let last = heap.unsafe_pop()
  let n = heap.length()
  if n > 0 {
    heap[0] = last
    let mut i = 0
    while true {
      let l = 2 * i + 1
      let r = l + 1
      let mut m = i
      if l < n && pending_gain(heap[l]) > pending_gain(heap[m]) {
        m = l
      }
      if r < n && pending_gain(heap[r]) > pending_gain(heap[m]) {
        m = r
      }
      if m == i {
        break
      }
      let t = heap[m]
      heap[m] = heap[i]
      heap[i] = t
      i = m
    }
  }
  top
}

///|
/// Makes `b` the current rows when `scratch`, else `a` (the other is the
/// scratch buffer).
fn Grower::use_rows(self : Grower, scratch : Bool, a : I32s, b : I32s) -> Unit {
  if scratch {
    self.rows = b
    self.scratch = a
  } else {
    self.rows = a
    self.scratch = b
  }
}

///|
/// Loss-guided growth: repeatedly splits the leaf whose best split gains
/// the most, until max_leaves leaves (when set) or no split gains.
fn Grower::grow_lossguide(
  self : Grower,
  tree : GrowTree,
  n : Int32,
  hist : Hist,
  g : Double,
  h : Double,
) -> Unit {
  let max_depth = self.p.max_depth
  let can_split = fn(depth : Int32, lo : Int32, hi : Int32) {
    (max_depth <= 0 || depth < max_depth) && hi - lo >= 2
  }
  // a leaf's rows are in one of the two row buffers: partitioning reads
  // them from one and writes the children's to the other
  let a0 = self.rows
  let b0 = self.scratch
  // splittable leaves in a max-heap on gain; leaves that cannot split
  let heap : Array[Pending] = []
  let done : Array[Pending] = []
  let add = fn(p : Pending) {
    if p.split is Some(_) {
      heap_push(heap, p)
    } else {
      done.push(p)
    }
  }
  add({
    node: 0,
    lo: 0,
    hi: n,
    hist,
    slot: 0,
    g,
    h,
    depth: 0,
    level: 0,
    in_scratch: false,
    split: if can_split(0, 0, n) {
      self.best_split(hist, g, h, n)
    } else {
      None
    },
  })
  // histogram buffers not in use (JS reuses a buffer per slot)
  let free : Array[Int32] = []
  let mut slots = 1
  let mut leaves = 1
  while self.p.max_leaves <= 0 || leaves < self.p.max_leaves {
    if heap.length() == 0 {
      break
    }
    let p = heap_pop(heap)
    guard p.split is Some(s) else { abort("pending leaf without split") }
    let lo = p.lo
    let hi = p.hi
    let depth = p.depth
    let level = p.level
    self.use_rows(p.in_scratch, a0, b0)
    let (mid, level) = self.split_rows(lo, hi, s, depth, level)
    // the children's rows are in the other buffer, which becomes theirs
    let child_scratch = !p.in_scratch
    self.use_rows(child_scratch, a0, b0)
    let (l, r) = self.add_split(tree, p.node, s, p.g, p.h)
    leaves += 1
    let gr = p.g - s.gl
    let hr = p.h - s.hl
    let dl = depth + 1
    let split_l = can_split(dl, lo, mid)
    let split_r = can_split(dl, mid, hi)
    if !split_l && !split_r {
      free.push(p.slot)
      add({
        ..p,
        node: l,
        hi: mid,
        g: s.gl,
        h: s.hl,
        depth: dl,
        level,
        split: None,
        in_scratch: child_scratch,
      })
      add({
        ..p,
        node: r,
        lo: mid,
        g: gr,
        h: hr,
        depth: dl,
        level,
        split: None,
        in_scratch: child_scratch,
      })
      continue
    }
    let small_left = mid - lo <= hi - mid
    let (small_lo, small_hi) = if small_left { (lo, mid) } else { (mid, hi) }
    let slot = match free.pop() {
      Some(x) => x
      None => {
        slots += 1
        slots - 1
      }
    }
    let small = self.histogram(small_lo, small_hi, level, 1 + slot)
    // the parent's histogram becomes the larger child's
    hist_sub(p.hist, small)
    let (hl_hist, sl, hr_hist, sr) = if small_left {
      (small, slot, p.hist, p.slot)
    } else {
      (p.hist, p.slot, small, slot)
    }
    add({
      node: l,
      lo,
      hi: mid,
      hist: hl_hist,
      slot: sl,
      g: s.gl,
      h: s.hl,
      depth: dl,
      level,
      in_scratch: child_scratch,
      split: if split_l {
        self.best_split(hl_hist, s.gl, s.hl, mid - lo)
      } else {
        None
      },
    })
    add({
      node: r,
      lo: mid,
      hi,
      hist: hr_hist,
      slot: sr,
      g: gr,
      h: hr,
      depth: dl,
      level,
      in_scratch: child_scratch,
      split: if split_r {
        self.best_split(hr_hist, gr, hr, hi - mid)
      } else {
        None
      },
    })
  }
  for p in heap {
    self.use_rows(p.in_scratch, a0, b0)
    self.finish_leaf(tree, p.node, p.lo, p.hi, p.level)
  }
  for p in done {
    self.use_rows(p.in_scratch, a0, b0)
    self.finish_leaf(tree, p.node, p.lo, p.hi, p.level)
  }
  self.use_rows(false, a0, b0)
}

///|
fn Grower::grow(
  self : Grower,
  tree : GrowTree,
  node : Int32,
  lo : Int32,
  hi : Int32,
  hist : Hist,
  g : Double,
  h : Double,
  depth : Int32,
  level : Int32,
) -> Unit {
  let level = if level == 0 &&
    self.compact_at.length() > 0 &&
    depth >= self.compact_at[level] &&
    depth < self.p.max_depth {
    self.compact(lo, hi, level)
    level + 1
  } else {
    level
  }
  let split = if depth < self.p.max_depth && hi - lo >= 2 {
    self.best_split(hist, g, h, hi - lo)
  } else {
    None
  }
  match split {
    None => self.finish_leaf(tree, node, lo, hi, level)
    Some(s) => {
      // a node just above the copy depth partitions its rows straight
      // into the copy, reading each row once in row order
      let (mid, level) = self.split_rows(lo, hi, s, depth, level)
      // the children's rows are in scratch; they become the current rows
      // until both children have grown
      self.swap_rows()
      let gl = s.gl
      let hl = s.hl
      let gr = g - gl
      let hr = h - hl
      let (l, r) = self.add_split(tree, node, s, g, h)
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
        let small = self.histogram(small_lo, small_hi, level, depth + 1)
        // the parent's histogram becomes the larger child's
        hist_sub(hist, small)
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
fn json_float(buf : @buffer.Buffer, x : Float) -> Unit {
  let t = x.to_string()
  buf.write_string_utf8(t)
  if !(t.contains(".") || t.contains("e") || t.contains("E")) {
    buf.write_string_utf8(".0")
  }
}

///|
fn json_floats(buf : @buffer.Buffer, a : Array[Float]) -> Unit {
  buf.write_char_utf8('[')
  for i, x in a {
    if i > 0 {
      buf.write_char_utf8(',')
    }
    json_float(buf, x)
  }
  buf.write_char_utf8(']')
}

///|
/// x in decimal, written byte by byte.
fn json_int(buf : @buffer.Buffer, x : Int32) -> Unit {
  if x < 0 {
    buf.write_byte(b'-')
  }
  // digits of |x| as a non-negative Int64, so the lowest Int32 works too
  let v = x.to_int64().abs()
  let mut d = 1L
  while d * 10L <= v {
    d *= 10L
  }
  while d > 0L {
    buf.write_byte((48L + v / d % 10L).to_byte())
    d /= 10L
  }
}

///|
fn json_ints(buf : @buffer.Buffer, a : Array[Int32]) -> Unit {
  buf.write_char_utf8('[')
  for i, x in a {
    if i > 0 {
      buf.write_char_utf8(',')
    }
    json_int(buf, x)
  }
  buf.write_char_utf8(']')
}

///|
fn tree_json(
  buf : @buffer.Buffer,
  t : GrowTree,
  id : Int32,
  num_feature : Int32,
) -> Unit {
  let n = t.left.length()
  buf.write_string_utf8("{\"base_weights\":")
  json_floats(buf, t.base_weight)
  let cats : Array[Int32] = []
  let cat_nodes : Array[Int32] = []
  let segments : Array[Int32] = []
  let sizes : Array[Int32] = []
  for i, c in t.cats {
    if c.length() > 0 {
      cat_nodes.push(i)
      segments.push(cats.length())
      sizes.push(c.length())
      cats.append(c)
    }
  }
  buf.write_string_utf8(",\"categories\":")
  json_ints(buf, cats)
  buf.write_string_utf8(",\"categories_nodes\":")
  json_ints(buf, cat_nodes)
  buf.write_string_utf8(",\"categories_segments\":")
  json_ints(buf, segments)
  buf.write_string_utf8(",\"categories_sizes\":")
  json_ints(buf, sizes)
  buf.write_string_utf8(",\"default_left\":")
  json_ints(buf, t.default_left.map(b => if b { 1 } else { 0 }))
  buf.write_string_utf8(",\"id\":\{id},\"left_children\":")
  json_ints(buf, t.left)
  buf.write_string_utf8(",\"loss_changes\":")
  json_floats(buf, t.loss_change)
  buf.write_string_utf8(",\"parents\":")
  json_ints(buf, t.parent.map(p => if p < 0 { 2147483647 } else { p }))
  buf.write_string_utf8(",\"right_children\":")
  json_ints(buf, t.right)
  buf.write_string_utf8(",\"split_conditions\":")
  json_floats(buf, t.split_cond)
  buf.write_string_utf8(",\"split_indices\":")
  json_ints(buf, t.split_index)
  buf.write_string_utf8(",\"split_type\":")
  json_ints(buf, t.cats.map(c => if c.length() > 0 { 1 } else { 0 }))
  buf.write_string_utf8(",\"sum_hessian\":")
  json_floats(buf, t.sum_hess)
  buf.write_string_utf8(
    ",\"tree_param\":{\"num_deleted\":\"0\",\"num_feature\":\"\{num_feature}\",\"num_nodes\":\"\{n}\",\"size_leaf_vector\":\"1\"}}",
  )
}

///|
/// Softmax probabilities of row r's classes into out.
fn softmax_row(
  margin : FixedArray[Float],
  r : Int32,
  groups : Int32,
  out : FixedArray[Double],
) -> Unit {
  let a = r * groups
  let mut mx = margin[a].to_double()
  for k in 1..<groups {
    let v = margin[a + k].to_double()
    if v > mx {
      mx = v
    }
  }
  let mut sum = 0.0
  for k in 0..<groups {
    let e = train_exp(margin[a + k].to_double() - mx)
    out[a + k] = e
    sum += e
  }
  for k in 0..<groups {
    out[a + k] = out[a + k] / sum
  }
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
  @utf8.decode_lossy(train_json(data, label, params, true))
}

///|
/// `train_xgboost_json` as UTF-8 bytes, ready to write to a file: a large
/// model's document is built once, one byte a character.
pub fn train_xgboost_json_bytes(
  data : DenseMatrix,
  label : ArrayView[Float],
  params~ : TrainParams = TrainParams::default(),
) -> Bytes raise ModelError {
  train_json(data, label, params, true)
}

///|
/// Gradients of class k for rows lo..hi: stores each row's exact pair and
/// quantised pair and resets rows[r] = r; returns gsum and hsum with the
/// rows' gradients added in row order.
fn Grower::gradients(
  self : Grower,
  label : ArrayView[Float],
  margin : FixedArray[Float],
  probs : FixedArray[Double],
  groups : Int32,
  k : Int32,
  kind : Int32,
  lo : Int32,
  hi : Int32,
  gsum : Double,
  hsum : Double,
) -> (Double, Double) {
  let gh = self.gh
  let gq = self.gq
  let rows = self.rows
  let mut gsum = gsum
  let mut hsum = hsum
  if kind == 1 {
    return self.logistic_gradients(label, margin, groups, k, lo, hi, gsum, hsum)
  }
  let qa = quantile_alpha_of(self.p, k)
  for r in lo..<hi {
    let y = label[r].to_double()
    let m = margin[r * groups + k].to_double()
    let (g, h) = match kind {
      0 => (m - y, 1.0)
      1 => {
        let pr = 1.0 / (1.0 + train_exp(-m))
        let h = pr * (1.0 - pr)
        (pr - y, if h < 1.0e-16 { 1.0e-16 } else { h })
      }
      // count:poisson, with XGBoost's max_delta_step 0.7 in the hessian
      3 => {
        let e = train_exp(m)
        (e - y, e * poisson_hess_scale)
      }
      // reg:gamma
      4 => {
        let t = y * train_exp(-m)
        (1.0 - t, if t < 1.0e-16 { 1.0e-16 } else { t })
      }
      // reg:tweedie
      5 => {
        let rho = self.p.tweedie_variance_power
        let a = train_exp((1.0 - rho) * m)
        let b = train_exp((2.0 - rho) * m)
        let h = -y * (1.0 - rho) * a + (2.0 - rho) * b
        (-y * a + b, if h < 1.0e-16 { 1.0e-16 } else { h })
      }
      // reg:pseudohubererror
      6 => {
        let z = m - y
        let d = self.p.huber_slope
        let sq = (1.0 + z / d * (z / d)).sqrt()
        (z / sq, d * d / ((d * d + z * z) * sq))
      }
      // reg:absoluteerror and reg:quantileerror: the leaves are set to
      // quantiles of the residuals once the tree is grown
      8 => (if m >= y { 1.0 - qa } else { -qa }, 1.0)
      // reg:squaredlogerror
      7 => {
        let pm = if m < -1.0 + 1.0e-6 { -1.0 + 1.0e-6 } else { m }
        let lp = log1p(pm)
        let ly = log1p(y)
        let h = (ly - lp + 1.0) / ((pm + 1.0) * (pm + 1.0))
        ((lp - ly) / (pm + 1.0), if h < 1.0e-6 { 1.0e-6 } else { h })
      }
      // rank:ndcg, rank:pairwise, rank:map and survival:cox and :aft, computed beforehand
      9 | 11 | 12 => (probs[r * 2], probs[r * 2 + 1])
      // binary:hinge on labels -1 and 1: a margin short of 1 on the right
      // side pulls with weight 1, otherwise XGBoost's smallest float
      // hessian and no gradient
      10 => {
        let t = 2.0 * y - 1.0
        if m * t < 1.0 {
          (-t, 1.0)
        } else {
          (0.0, 1.1754943508222875e-38)
        }
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
    gh.unsafe_set(r * 2, Float::from_double(g))
    gh.unsafe_set(r * 2 + 1, Float::from_double(h))
    self.pack_at(gq, r, g, h)
    rows.unsafe_set(r, r)
    gsum += g
    hsum += h
  }
  (gsum, hsum)
}

///|
/// Adds the leaf values of class k to the margins of the rows src[lo:hi],
/// which reach `node` but were left out of the tree's growth: they are
/// partitioned down the finished tree on the original bins, using dst for
/// the children (the two buffers swap at each level).
fn Grower::route(
  self : Grower,
  tree : GrowTree,
  node : Int32,
  src : I32s,
  dst : I32s,
  lo : Int32,
  hi : Int32,
  margin : FixedArray[Float],
  groups : Int32,
  k : Int32,
) -> Unit {
  if lo >= hi {
    return
  }
  guard tree.splits[node] is Some(s) else {
    let v = tree.split_cond[node]
    for i in lo..<hi {
      let at = src.unsafe_get(i) * groups + k
      margin[at] = margin[at] + v
    }
    return
  }
  let column = s.feature * self.leaf_at.length()
  let mut a = lo
  let mut b = hi - 1
  // indices are in range: src holds rows, and the tables cover every bin
  if self.narrow {
    let go = self.left_table(s)
    let cb = self.by8
    for i in lo..<hi {
      let r = src.unsafe_get(i)
      let left = go.unsafe_get(cb.unsafe_get(column + r).to_int())
      dst.unsafe_set(a, r)
      dst.unsafe_set(b, r)
      a += left
      b -= 1 - left
    }
  } else {
    let go = self.bin_table(s)
    let bins = self.data.by_col
    for i in lo..<hi {
      let r = src.unsafe_get(i)
      let left = go.unsafe_get(bins.unsafe_get(column + r).to_int())
      dst.unsafe_set(a, r)
      dst.unsafe_set(b, r)
      a += left
      b -= 1 - left
    }
  }
  self.route(tree, tree.left[node], dst, src, lo, a, margin, groups, k)
  self.route(tree, tree.right[node], dst, src, a, hi, margin, groups, k)
}

///|
/// Adds each copied position's leaf value to its row's margin of class k,
/// for positions lo..hi.
fn add_leaves(
  margin : FixedArray[Float],
  orig : I32s,
  leaf_at : I32s,
  leaves : FixedArray[Float],
  groups : Int32,
  k : Int32,
  lo : Int32,
  hi : Int32,
) -> Unit {
  for i in lo..<hi {
    let at = orig.unsafe_get(i) * groups + k
    margin[at] = margin[at] + leaves[leaf_at.unsafe_get(i)]
  }
}

///|
/// The intercept of an output whose labels average `mean`, as XGBoost
/// stores it and as a margin, for the losses that allow several outputs
/// and start from the mean.
fn mean_intercept(objective : String, mean : Double) -> (Float, Double) {
  let x = if mean < 1.0e-6 {
    1.0e-6
  } else if mean > 1.0 - 1.0e-6 {
    1.0 - 1.0e-6
  } else {
    mean
  }
  match objective {
    "binary:logitraw" => {
      let b = Float::from_double(@math.ln(x / (1.0 - x)))
      (b, b.to_double())
    }
    "binary:logistic" | "reg:logistic" => {
      let pr = Float::from_double(x)
      let v = pr.to_double()
      (pr, @math.ln(v / (1.0 - v)))
    }
    _ => {
      let b = Float::from_double(mean)
      (b, b.to_double())
    }
  }
}

///|
/// The quantile output k of the absolute and quantile losses fits.
fn quantile_alpha_of(p : TrainParams, k : Int32) -> Double {
  if p.objective != "reg:quantileerror" {
    0.5
  } else if p.quantile_alphas.length() > 0 {
    p.quantile_alphas[k]
  } else {
    p.quantile_alpha
  }
}

///|
/// The alpha quantile of xs (reordered in place), interpolated as XGBoost's
/// `common::Quantile` does; NaN when xs is empty.
fn quantile(xs : FixedArray[Float], alpha : Double) -> Double {
  quantile_range(xs, 0, xs.length(), alpha)
}

///|
/// The alpha quantile of xs[lo:hi], which it reorders: the two order
/// statistics the interpolation needs are selected rather than sorted for.
fn quantile_range(
  xs : FixedArray[Float],
  lo : Int32,
  hi : Int32,
  alpha : Double,
) -> Double {
  let n = hi - lo
  if n == 0 {
    return 0.0 / 0.0
  }
  let nd = n.to_double()
  if alpha <= 1.0 / (nd + 1.0) {
    let mut v = xs[lo]
    for i in (lo + 1)..<hi {
      if xs[i] < v {
        v = xs[i]
      }
    }
    return v.to_double()
  }
  if alpha >= nd / (nd + 1.0) {
    let mut v = xs[lo]
    for i in (lo + 1)..<hi {
      if xs[i] > v {
        v = xs[i]
      }
    }
    return v.to_double()
  }
  let x = alpha * (nd + 1.0)
  let k = x.floor() - 1.0
  let d = x - 1.0 - k
  let at = lo + k.to_int()
  select_kth(xs, lo, hi, at)
  // the next order statistic: the least of what lies above it
  let mut v1 = xs[at + 1]
  for i in (at + 2)..<hi {
    if xs[i] < v1 {
      v1 = xs[i]
    }
  }
  let v0 = xs[at].to_double()
  v0 + d * (v1.to_double() - v0)
}

///|
/// Reorders xs[lo:hi] so xs[k] holds the value it would hold sorted, with
/// nothing greater before it and nothing smaller after it (quickselect,
/// falling back to sorting after too many rounds).
fn select_kth(
  xs : FixedArray[Float],
  lo : Int32,
  hi : Int32,
  k : Int32,
) -> Unit {
  let mut l = lo
  let mut h = hi - 1
  let mut budget = 64
  while h > l {
    if budget == 0 && h - l >= 16 {
      let part = FixedArray::makei(h - l + 1, i => xs[l + i])
      part.sort()
      for i, v in part {
        xs[l + i] = v
      }
      return
    }
    if h - l < 16 {
      // insertion sort of xs[l..h]
      for i in (l + 1)..<=h {
        let v = xs[i]
        let mut j = i - 1
        while j >= l && xs[j] > v {
          xs[j + 1] = xs[j]
          j -= 1
        }
        xs[j + 1] = v
      }
      return
    }
    budget -= 1
    let a = xs[l]
    let b = xs[l + (h - l) / 2]
    let c = xs[h]
    let pivot = if a < b {
      if b < c {
        b
      } else if a < c {
        c
      } else {
        a
      }
    } else if a < c {
      a
    } else if b < c {
      c
    } else {
      b
    }
    let mut i = l
    let mut j = h
    while i <= j {
      while xs[i] < pivot {
        i += 1
      }
      while xs[j] > pivot {
        j -= 1
      }
      if i <= j {
        let t = xs[i]
        xs[i] = xs[j]
        xs[j] = t
        i += 1
        j -= 1
      }
    }
    if k <= j {
      h = j
    } else if k >= i {
      l = i
    } else {
      return
    }
  }
}

///|
/// Sets each leaf of a grown tree to eta times the alpha quantile of
/// label - margin over the tree's rows that reach it (XGBoost's adaptive
/// leaves for the absolute and quantile losses).
fn quantile_leaves(
  tree : GrowTree,
  gr : Grower,
  label : ArrayView[Float],
  margin : FixedArray[Float],
  m : Int32,
  alpha : Double,
  eta : Double,
  groups : Int32,
  k : Int32,
) -> Unit {
  let nodes = tree.split_cond.length()
  // positions 0..m reach leaf leaf_at.unsafe_get(i); orig[0] maps them to rows
  let count = FixedArray::make(nodes + 1, 0)
  let leaf_at = gr.leaf_at
  let orig = gr.orig[0]
  for i in 0..<m {
    count[leaf_at.unsafe_get(i) + 1] += 1
  }
  for v in 0..<nodes {
    count[v + 1] += count[v]
  }
  let start = FixedArray::makei(nodes + 1, i => count[i])
  let res = FixedArray::make(m, Float::from_int(0))
  for i in 0..<m {
    let r = orig.unsafe_get(i)
    let v = leaf_at.unsafe_get(i)
    res[count[v]] = Float::from_double(
      label[r].to_double() - margin[r * groups + k].to_double(),
    )
    count[v] += 1
  }
  for v in 0..<nodes {
    let c = start[v + 1] - start[v]
    if c > 0 {
      tree.split_cond[v] = Float::from_double(
        quantile_range(res, start[v], start[v + 1], alpha) * eta,
      )
    }
  }
}

///|
/// `allow_narrow` false forces two-byte copied bins (for tests).
fn train_json(
  data : DenseMatrix,
  label : ArrayView[Float],
  params : TrainParams,
  allow_narrow : Bool,
) -> Bytes raise ModelError {
  let p = params
  let n = data.rows
  let nt = p.num_target
  if nt < 1 {
    raise InvalidModel("num_target must be positive")
  }
  if nt > 1 {
    match p.objective {
      "reg:squarederror"
      | "reg:pseudohubererror"
      | "reg:absoluteerror"
      | "binary:logistic"
      | "reg:logistic"
      | "binary:logitraw" => ()
      o => raise Unsupported("num_target above 1 with \{o}")
    }
  }
  if p.multi_strategy != "one_output_per_tree" &&
    p.multi_strategy != "multi_output_tree" {
    raise InvalidModel(
      "multi_strategy must be one_output_per_tree or multi_output_tree",
    )
  }
  if label.length() != n * nt {
    raise InvalidModel("\{label.length()} labels for \{n} rows")
  }
  if n == 0 {
    raise InvalidModel("no training rows")
  }
  if p.max_bin < 2 ||
    p.max_bin > 65535 ||
    p.max_depth < 0 ||
    (p.max_depth == 0 && p.grow_policy != "lossguide") ||
    (p.sampling_method != "uniform" && p.sampling_method != "gradient_based") ||
    p.num_round < 0 {
    raise InvalidModel("bad training parameters")
  }
  let groups = match p.objective {
    "reg:squarederror"
    | "binary:logistic"
    | "reg:logistic"
    | "binary:logitraw"
    | "binary:hinge"
    | "survival:cox"
    | "survival:aft"
    | "count:poisson"
    | "reg:gamma"
    | "reg:tweedie"
    | "reg:pseudohubererror"
    | "reg:squaredlogerror"
    | "reg:absoluteerror"
    | "rank:ndcg"
    | "rank:pairwise"
    | "rank:map" => 1
    "reg:quantileerror" =>
      if p.quantile_alphas.length() > 0 {
        p.quantile_alphas.length()
      } else {
        1
      }
    "multi:softprob" | "multi:softmax" => {
      if p.num_class < 2 {
        raise InvalidModel("`num_class` must be at least 2")
      }
      p.num_class
    }
    o => raise Unsupported("training objective \{o}")
  }
  let groups = if nt > 1 { nt } else { groups }
  // each output's labels, for a model with several
  let tcols : Array[Array[Float]] = Array::makei(if nt > 1 { nt } else { 0 }, k => {
    Array::makei(n, r => label[r * nt + k])
  })
  let softmax = p.objective == "multi:softprob" ||
    p.objective == "multi:softmax"
  for y in label {
    if y.is_nan() {
      raise InvalidModel("missing label")
    }
    match p.objective {
      "binary:logistic" | "reg:logistic" | "binary:logitraw" | "binary:hinge" =>
        if y < 0.0 || y > 1.0 {
          raise InvalidModel("binary labels must be in [0, 1]")
        }
      "count:poisson" | "reg:tweedie" =>
        if y < 0.0 {
          raise InvalidModel("\{p.objective} labels must be non-negative")
        }
      "reg:gamma" =>
        if y <= 0.0 {
          raise InvalidModel("reg:gamma labels must be positive")
        }
      "rank:ndcg" | "rank:pairwise" =>
        if y < 0.0 || y > 31.0 || y.to_int().to_double() != y.to_double() {
          raise InvalidModel("relevance labels must be integers in 0 .. 31")
        }
      "rank:map" =>
        if y != 0.0 && y != 1.0 {
          raise InvalidModel("rank:map labels must be 0 or 1")
        }
      "reg:squaredlogerror" =>
        if y <= -1.0 {
          raise InvalidModel("reg:squaredlogerror labels must be above -1")
        }
      _ => ()
    }
    if softmax &&
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
  if p.objective == "reg:tweedie" &&
    !(p.tweedie_variance_power > 1.0 && p.tweedie_variance_power < 2.0) {
    raise InvalidModel("tweedie_variance_power must be in (1, 2)")
  }
  if p.objective == "reg:pseudohubererror" && !(p.huber_slope > 0.0) {
    raise InvalidModel("huber_slope must be positive")
  }
  if p.objective == "reg:quantileerror" {
    for k in 0..<groups {
      let a = quantile_alpha_of(p, k)
      if !(a > 0.0 && a < 1.0) {
        raise InvalidModel("quantile_alpha must be in (0, 1)")
      }
      if k > 0 && !(quantile_alpha_of(p, k - 1) < a) {
        raise InvalidModel("quantile_alpha must be in ascending order")
      }
    }
  }
  let is_rank = p.objective == "rank:ndcg" ||
    p.objective == "rank:pairwise" ||
    p.objective == "rank:map"
  if is_rank {
    let mut total = 0
    for c in p.group {
      if c <= 0 {
        raise InvalidModel("query sizes must be positive")
      }
      total += c
    }
    if total != n {
      raise InvalidModel("query sizes add up to \{total}, not \{n} rows")
    }
  }
  let rank_starts = if is_rank {
    let st = FixedArray::make(p.group.length() + 1, 0)
    for i, c in p.group {
      st[i + 1] = st[i] + c
    }
    st
  } else {
    FixedArray::make(0, 0)
  }
  let (aft, aft_dist) = if p.objective == "survival:aft" {
    if !(p.aft_loss_distribution_scale > 0.0) {
      raise InvalidModel("aft_loss_distribution_scale must be positive")
    }
    (
      aft_rows(label, p.label_upper_bound),
      aft_distribution(p.aft_loss_distribution),
    )
  } else {
    (aft_rows(label[0:0], []), 0)
  }
  let cox_rows = if p.objective == "survival:cox" {
    cox_order(label)
  } else {
    FixedArray::make(0, 0)
  }
  let rank_inv = if is_rank {
    rank_inverse_max_dcg(
      label,
      rank_starts,
      p.objective == "rank:ndcg",
      p.rank_truncation,
    )
  } else {
    FixedArray::make(0, 0.0)
  }
  // the starting margin of each output when they differ
  let group_margins : Array[Double] = []
  let (base_text, base_margin) = match p.objective {
    "reg:squarederror" | "reg:pseudohubererror" => {
      let b = Float::from_double(mean)
      ("[" + xgb_float_text(b) + "]", b.to_double())
    }
    "reg:absoluteerror" | "reg:quantileerror" => {
      // one intercept per quantile; base_margins carries them
      let b = Array::makei(groups, k => {
        let col = if nt > 1 { tcols[k][:] } else { label }
        Float::from_double(
          quantile(FixedArray::makei(n, i => col[i]), quantile_alpha_of(p, k)),
        )
      })
      for v in b {
        group_margins.push(v.to_double())
      }
      ("[" + b.map(xgb_float_text).join(",") + "]", b[0].to_double())
    }
    "reg:squaredlogerror" => {
      // the prediction whose log1p is the mean log1p of the labels
      let mut s = 0.0
      for y in label {
        s += log1p(y.to_double())
      }
      let b = Float::from_double(train_exp(s / n.to_double()) - 1.0)
      ("[" + xgb_float_text(b) + "]", b.to_double())
    }
    "count:poisson" | "reg:gamma" | "reg:tweedie" => {
      // stored as a mean, used as its log (XGBoost's ProbToMargin)
      let b = Float::from_double(if mean < 1.0e-6 { 1.0e-6 } else { mean })
      ("[" + xgb_float_text(b) + "]", @math.ln(b.to_double()))
    }
    // the partial likelihood ignores a common shift: a hazard ratio of 1
    "survival:cox" => ("[1]", 0.0)
    "survival:aft" => {
      // the best constant: Newton steps on the likelihood from the mean
      // log time, stored as a time and used as its log (XGBoost itself
      // starts every model at a time of 0.5)
      let b = Float::from_double(
        train_exp(aft_intercept(aft, aft_dist, p.aft_loss_distribution_scale)),
      )
      ("[" + xgb_float_text(b) + "]", @math.ln(b.to_double()))
    }
    "binary:hinge" => {
      // XGBoost's intercept: one Newton step from a zero margin, where
      // every row has gradient 1 - 2y and hessian 1, put through the
      // class transform
      let b : Float = if mean > 0.5 { 1.0 } else { 0.0 }
      ("[" + xgb_float_text(b) + "]", b.to_double())
    }
    "binary:logitraw" => {
      // stored and used as a margin
      let x = if mean < 1.0e-6 {
        1.0e-6
      } else if mean > 1.0 - 1.0e-6 {
        1.0 - 1.0e-6
      } else {
        mean
      }
      let b = Float::from_double(@math.ln(x / (1.0 - x)))
      ("[" + xgb_float_text(b) + "]", b.to_double())
    }
    "binary:logistic" | "reg:logistic" => {
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
  // several outputs: each starts from its own labels' intercept
  let (base_text, base_margin) = if nt > 1 && p.objective != "reg:absoluteerror" {
    let stored = []
    for k in 0..<nt {
      let mut s = 0.0
      for y in tcols[k] {
        s += y.to_double()
      }
      let (b, m) = mean_intercept(p.objective, s / n.to_double())
      stored.push(xgb_float_text(b))
      group_margins.push(m)
    }
    ("[" + stored.join(",") + "]", group_margins[0])
  } else {
    (base_text, base_margin)
  }
  let kind = match p.objective {
    "reg:squarederror" => 0
    "binary:logistic" | "reg:logistic" | "binary:logitraw" => 1
    "count:poisson" => 3
    "reg:gamma" => 4
    "reg:tweedie" => 5
    "reg:pseudohubererror" => 6
    "reg:squaredlogerror" => 7
    "reg:absoluteerror" | "reg:quantileerror" => 8
    "rank:ndcg" | "rank:pairwise" | "rank:map" => 9
    "binary:hinge" => 10
    "survival:cox" => 11
    "survival:aft" => 12
    _ => 2
  }
  let is_cat = FixedArray::make(data.cols, false)
  for f in p.categorical {
    if f < 0 || f >= data.cols {
      raise InvalidModel("categorical column \{f} out of range")
    }
    is_cat[f] = true
    for r in 0..<n {
      let v = data.data[r * data.cols + f]
      if !v.is_nan() &&
        (v < 0.0 || v >= 16777216.0 || Float::from_int(v.to_int()) != v) {
        raise InvalidModel(
          "column \{f} is categorical but row \{r} holds \{v}, not a category code",
        )
      }
    }
  }
  if p.grow_policy != "depthwise" && p.grow_policy != "lossguide" {
    raise InvalidModel("grow_policy must be depthwise or lossguide")
  }
  // categories kept per categorical feature: the max_bin - 1 most
  // frequent (bin 0 holds missing values and the rest), fewer when the
  // histogram slots (UInt16) would run out, or up to the slots' share
  // (cat_room) when those cover under 99% of the rows
  let ncat = p.categorical.length()
  let room = 65535 - 2 * data.cols - (data.cols - ncat) * p.max_bin
  let cat_cap = if ncat == 0 {
    0
  } else if room / ncat < p.max_bin - 1 {
    room / ncat
  } else {
    p.max_bin - 1
  }
  if ncat > 0 && cat_cap < 2 {
    raise InvalidModel("too many features times bins for training")
  }
  let cat_room = if ncat == 0 { 0 } else { room / ncat }
  let binned = bin_matrix(data, p.max_bin, is_cat, cat_cap, cat_room)
  // histogram slots are stored as UInt16
  if binned.cuts.length() + 2 * data.cols > 65535 {
    raise InvalidModel("too many features times bins for training")
  }
  let hist_start = FixedArray::make(data.cols + 1, 0)
  for f in 0..<data.cols {
    hist_start[f + 1] = hist_start[f] +
      (binned.cut_start[f + 1] - binned.cut_start[f] + 2)
  }
  if p.multi_strategy == "multi_output_tree" && nt > 1 {
    match p.objective {
      "reg:squarederror"
      | "reg:pseudohubererror"
      | "binary:logistic"
      | "reg:logistic"
      | "binary:logitraw" => ()
      o => raise Unsupported("multi_output_tree with \{o}")
    }
    if ncat > 0 ||
      p.grow_policy != "depthwise" ||
      p.subsample < 1.0 ||
      p.colsample_bytree < 1.0 ||
      p.colsample_bynode < 1.0 ||
      p.min_data_in_leaf > 0.0 {
      raise Unsupported(
        "multi_output_tree takes depth-wise growth on numerical features without sampling",
      )
    }
    let margin = FixedArray::make(n * nt, (0.0 : Float))
    for r in 0..<n {
      for k in 0..<nt {
        margin[r * nt + k] = Float::from_double(group_margins[k])
      }
    }
    let trees = @buffer.Buffer()
    model_head(trees, p.num_round, 1, p)
    train_multi_trees(binned, hist_start, n, label, margin, p, kind, trees)
    return model_tail(trees, nt, false, base_text, data.cols, p)
  }
  // per-feature base of the one-byte copied bins: a feature without
  // missing values never uses its missing slot
  let cbase = FixedArray::makei(data.cols, fn(f) {
    hist_start[f] + (if binned.missing[f] { 0 } else { 1 })
  })
  let mut narrow = allow_narrow && narrow_bins
  for f in 0..<data.cols {
    if hist_start[f + 1] - 1 - cbase[f] > 255 {
      narrow = false
    }
  }
  let copies = if narrow { 0 } else { n * data.cols }
  let cols = data.cols
  let words = (cols + word_bins - 1) / word_bins
  let wn = if narrow { n * words } else { 0 }
  let bins8 = words_make(wn)
  let by8 = FixedArray::make(n * cols - copies, b'\x00')
  let by_col = binned.by_col
  if narrow {
    // one-byte bins: the bin, plus one for a feature with missing values
    // (whose missing bin is 0)
    let adjs = FixedArray::makei(cols, f => {
      if binned.missing[f] {
        0
      } else {
        1
      }
    })
    for w in 0..<words {
      for lo = 0; lo < n; lo = lo + block {
        pack_block(
          by_col,
          bins8,
          by8,
          words,
          n,
          cols,
          w,
          adjs,
          lo,
          block_end(lo, n),
        )
      }
    }
    // one-byte training reads by8 and bins8 only
    binned.by_col = u16s_make(0)
  } else {
    let bins = u16s_make(n * cols)
    for f in 0..<cols {
      for lo = 0; lo < n; lo = lo + block {
        wide_block(
          by_col,
          bins,
          cols,
          n,
          f,
          hist_start[f],
          lo,
          block_end(lo, n),
        )
      }
    }
    binned.bins = bins
  }
  // copy depths: one copy of the original arrays, and for loss-guided
  // trees with one-byte bins a fresh copy every recopy_every levels below
  // it while the tree can still grow that deep
  let compact_at = FixedArray::makei(
    if narrow && p.grow_policy == "lossguide" {
      max_copies
    } else {
      1
    },
    fn(i) { 2 + i * recopy_every },
  )
  let levels = compact_at
    .iter()
    .filter(fn(d) { p.max_depth <= 0 || d < p.max_depth })
    .count()
  let levels = if levels < 1 { 1 } else { levels }
  let qpad = narrow && compact_at.length() > 1 && qpad_ok(cols, words, n)
  let gr : Grower = {
    data: binned,
    p,
    gh: f32s_make(n * 2),
    gq: qarr_make(n),
    gs: 1.0,
    hs: 1.0,
    igs: 1.0,
    compact_at,
    narrow,
    qpad,
    bins8,
    words,
    by8,
    cbins: FixedArray::makei(levels, fn(_) { words_make(wn) }),
    cbase,
    wbase: FixedArray::makei(words * word_bins, fn(i) {
      if i < cols {
        cbase[i]
      } else {
        hist_start[cols] + i % word_bins
      }
    }),
    cbins16: FixedArray::makei(1, fn(_) { u16s_make(copies) }),
    cq: FixedArray::makei(levels, fn(_) { qarr_make(if qpad { 0 } else { n }) }),
    cgh: FixedArray::makei(1, fn(_) { f32s_make(n * 2) }),
    orig: FixedArray::makei(levels, fn(_) { i32s_make(n) }),
    ihs: 1.0,
    rows: i32s_make(n),
    scratch: i32s_make(n),
    leaf_at: i32s_make(n),
    nbins: hist_start[data.cols],
    hist_start,
    rng: p.seed.to_int64().reinterpret_as_uint64(),
    tree_feats: FixedArray::makei(data.cols, i => i),
    tree_n: data.cols,
    node_on: FixedArray::make(data.cols, true),
  }
  // margins accumulate in float32, as XGBoost predicts
  let margin = FixedArray::make(n * groups, Float::from_double(base_margin))
  if group_margins.length() > 1 {
    for r in 0..<n {
      for k in 0..<groups {
        margin[r * groups + k] = Float::from_double(group_margins[k])
      }
    }
  }
  // rows left out of a sampled tree, and a buffer for routing them
  let (out_rows, out_scratch) = if p.subsample < 1.0 {
    (i32s_make(n), i32s_make(n))
  } else {
    (i32s_make(0), i32s_make(0))
  }
  let trees = @buffer.Buffer()
  model_head(trees, p.num_round * groups, groups, p)
  let mut num_trees = 0
  let prob_buf = FixedArray::make(if softmax { n * groups } else { 0 }, 0.0)
  for _ in 0..<p.num_round {
    // softmax probabilities are computed once per round for all classes
    let probs = if softmax {
      softmax_probs(margin, n, groups, prob_buf)
      prob_buf
    } else if kind == 11 {
      cox_gradients(label, margin, cox_rows)
    } else if kind == 12 {
      aft_gradients(aft, margin, aft_dist, p.aft_loss_distribution_scale)
    } else if kind == 9 {
      rank_gradients(
        label,
        margin,
        rank_starts,
        rank_inv,
        p.objective == "rank:ndcg",
        p.rank_truncation,
        map=p.objective == "rank:map",
      )
    } else {
      FixedArray::make(0, 0.0)
    }
    for k in 0..<groups {
      // this output's labels
      let label = if nt > 1 { tcols[k][:] } else { label }
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
        2 => gr.set_scales(n, 1.0, 0.5)
        // measured after the gradients are computed
        _ => gr.set_scales(n, 1.0, 1.0)
      }
      for lo = 0; lo < n; lo = lo + block {
        let (g, h) = gr.gradients(
          label,
          margin,
          probs,
          groups,
          k,
          kind,
          lo,
          block_end(lo, n),
          gsum,
          hsum,
        )
        gsum = g
        hsum = h
      }
      if kind >= 3 {
        gr.requantize(n)
      }
      // with row sampling the tree grows on the kept rows alone, rows[0:m];
      // the rest (out_rows[0:n - m]) are routed down it afterwards
      let m = if p.subsample < 1.0 && p.sampling_method == "gradient_based" {
        let (g, h, m) = gr.mvs_sample(n, p.subsample, out_rows)
        gsum = g
        hsum = h
        m
      } else if p.subsample < 1.0 {
        // without branches: each row is written to both lists and only
        // its own list's end advances
        let gh = gr.gh
        let rows = gr.rows
        let sub = p.subsample
        let mut m = 0
        let mut u = 0
        for r in 0..<n {
          let keep = if gr.uniform() < sub { 1 } else { 0 }
          let drop = (1 - keep).to_double()
          gsum -= gh.unsafe_get(r * 2).to_double() * drop
          hsum -= gh.unsafe_get(r * 2 + 1).to_double() * drop
          rows.unsafe_set(m, r)
          out_rows.unsafe_set(u, r)
          m += keep
          u += 1 - keep
        }
        m
      } else {
        n
      }
      gr.sample_tree()
      let tree = GrowTree::new()
      let root = tree.add(-1, gr.weight(gsum, hsum), hsum)
      let hist = gr.histogram(0, m, 0, 0)
      // every row is in exactly one slot of feature 0, so those slots sum
      // to the quantised totals
      let mut qg = 0.0
      let mut qh = 0.0
      for slot in hist_start[0]..<hist_start[1] {
        qg = qg + hist_g(hist, slot)
        qh = qh + hist_h(hist, slot)
      }
      if p.grow_policy == "lossguide" {
        gr.grow_lossguide(tree, m, hist, qg * gr.igs, qh * gr.ihs)
      } else {
        gr.grow(tree, root, 0, m, hist, qg * gr.igs, qh * gr.ihs, 0, 0)
      }
      if kind == 8 {
        quantile_leaves(
          tree,
          gr,
          label,
          margin,
          m,
          quantile_alpha_of(p, k),
          p.eta,
          groups,
          k,
        )
      }
      let leaves = FixedArray::from_array(tree.split_cond)
      if m < n {
        gr.route(tree, 0, out_rows, out_scratch, 0, n - m, margin, groups, k)
      }
      for lo = 0; lo < m; lo = lo + block {
        add_leaves(
          margin,
          gr.orig[0],
          gr.leaf_at,
          leaves,
          groups,
          k,
          lo,
          block_end(lo, m),
        )
      }
      if num_trees > 0 {
        trees.write_char_utf8(',')
      }
      tree_json(trees, tree, num_trees, data.cols)
      num_trees += 1
    }
  }
  model_tail(trees, groups, softmax, base_text, data.cols, p)
}

///|
/// The XGBoost JSON document up to its trees' JSON: `num_trees` trees,
/// `per_round` a round, tree i for output i % per_round. The trees and
/// `model_tail` follow in the same buffer, so the document is built once.
fn model_head(
  buf : @buffer.Buffer,
  num_trees : Int32,
  per_round : Int32,
  p : TrainParams,
) -> Unit {
  buf.write_string_utf8(
    "{\"learner\":{\"attributes\":{},\"feature_names\":[],\"feature_types\":[],\"gradient_booster\":{\"model\":{\"cats\":{\"enc\":[],\"feature_segments\":[],\"sorted_idx\":[]},",
  )
  buf.write_string_utf8(
    "\"gbtree_model_param\":{\"num_parallel_tree\":\"1\",\"num_trees\":\"\{num_trees}\"},\"iteration_indptr\":",
  )
  json_ints(buf, Array::makei(p.num_round + 1, i => i * per_round))
  buf.write_string_utf8(",\"tree_info\":")
  json_ints(buf, Array::makei(num_trees, i => i % per_round))
  buf.write_string_utf8(",\"trees\":[")
}

///|
/// The rest of the XGBoost JSON document after the trees' JSON.
fn model_tail(
  buf : @buffer.Buffer,
  groups : Int32,
  softmax : Bool,
  base_text : String,
  num_feature : Int32,
  p : TrainParams,
) -> Bytes {
  buf.write_string_utf8("]},\"name\":\"gbtree\"},\"learner_model_param\":{")
  let num_class = if softmax { groups } else { 0 }
  let num_target = if softmax { 1 } else { groups }
  buf.write_string_utf8(
    "\"base_score\":\"\{base_text}\",\"boost_from_average\":\"1\",\"num_class\":\"\{num_class}\",\"num_feature\":\"\{num_feature}\",\"num_target\":\"\{num_target}\"},",
  )
  if softmax {
    buf.write_string_utf8(
      "\"objective\":{\"name\":\"\{p.objective}\",\"softmax_multiclass_param\":{\"num_class\":\"\{groups}\"}}",
    )
  } else {
    let param = match p.objective {
      "count:poisson" =>
        ",\"poisson_regression_param\":{\"max_delta_step\":\"0.699999988\"}"
      "reg:tweedie" =>
        ",\"tweedie_regression_param\":{\"tweedie_variance_power\":\"\{p.tweedie_variance_power}\"}"
      "reg:pseudohubererror" =>
        ",\"pseudo_huber_param\":{\"huber_slope\":\"\{p.huber_slope}\"}"
      "reg:squaredlogerror" | "reg:absoluteerror" => ""
      "rank:ndcg" | "rank:pairwise" | "rank:map" =>
        ",\"lambdarank_param\":{\"lambdarank_bias_norm\":\"1\",\"lambdarank_normalization\":\"1\",\"lambdarank_num_pair_per_sample\":\"4294967295\",\"lambdarank_pair_method\":\"topk\",\"lambdarank_score_normalization\":\"1\",\"lambdarank_unbiased\":\"0\",\"ndcg_exp_gain\":\"1\"}"
      "survival:aft" =>
        ",\"aft_loss_param\":{\"aft_loss_distribution\":\"\{p.aft_loss_distribution}\",\"aft_loss_distribution_scale\":\"\{Float::from_double(p.aft_loss_distribution_scale)}\"}"
      "reg:quantileerror" =>
        ",\"quantile_loss_param\":{\"quantile_alpha\":\"[" +
        Array::makei(groups, k => quantile_alpha_of(p, k).to_string()).join(",") +
        "]\"}"
      _ => ",\"reg_loss_param\":{\"scale_pos_weight\":\"1\"}"
    }
    buf.write_string_utf8("\"objective\":{\"name\":\"\{p.objective}\"\{param}}")
  }
  buf.write_string_utf8("},\"version\":[3,4,1]}")
  buf.to_bytes()
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
