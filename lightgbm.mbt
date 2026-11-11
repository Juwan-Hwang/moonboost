///|
/// One tree of a LightGBM model, kept in LightGBM's own form: double
/// thresholds and leaf values, `decision_type` flags, and children where a
/// negative index `~i` is leaf `i`.
pub struct LightGBMTree {
  split_feature : Array[Int32]
  threshold : Array[Double]
  decision_type : Array[Int32]
  left : Array[Int32]
  right : Array[Int32]
  leaf_value : Array[Double]
  cat_boundaries : Array[Int32]
  cat_threshold : Array[UInt32]
}

///|
/// A model saved by LightGBM (`Booster.save_model("model.txt")`, the text
/// format). Predictions follow LightGBM's CPU predictor: every value is a
/// double, trees are added in order, and the result is bit-for-bit what
/// LightGBM's `predict` returns.
pub struct LightGBMModel {
  trees : Array[LightGBMTree]
  num_class : Int32
  /// Trees per boosting round: one per class for `multiclass`, else 1.
  num_tree_per_iteration : Int32
  num_feature : Int32
  /// The objective line as saved, e.g. `binary sigmoid:1`.
  objective : String
  feature_names : Array[String]
  average_output : Bool
  priv fast : LgbFast?
  // the objective's name and `sigmoid:` parameter, parsed once
  priv obj_name : String
  priv sigmoid : Double
}

///|
/// LightGBM's `kZeroThreshold` (`1e-35f`): inputs this close to zero are
/// read as zero.
let lgb_zero : Double = Float::from_double(1.0e-35).to_double()

///|
/// Tree fields that loading reads; the rest (gains, counts, weights) are
/// skipped.
fn lgb_tree_key(key : StringView) -> Bool {
  match key {
    "num_leaves"
    | "num_cat"
    | "split_feature"
    | "threshold"
    | "decision_type"
    | "left_child"
    | "right_child"
    | "leaf_value"
    | "cat_boundaries"
    | "cat_threshold"
    | "is_linear" => true
    _ => false
  }
}

///|
fn lgb_values(
  fields : Map[String, StringView],
  key : String,
) -> Array[StringView] raise ModelError {
  match fields.get(key) {
    Some(s) => s.split(" ").filter(x => x != "").to_array()
    None => raise InvalidModel("LightGBM tree is missing `\{key}`")
  }
}

///|
fn lgb_ints(
  fields : Map[String, StringView],
  key : String,
) -> Array[Int32] raise ModelError {
  let s = match fields.get(key) {
    Some(s) => s
    None => raise InvalidModel("LightGBM tree is missing `\{key}`")
  }
  let out = []
  let n = s.length()
  let mut i = 0
  while i < n {
    while i < n && s.unsafe_get(i) == ' ' {
      i += 1
    }
    let start = i
    let neg = i < n && s.unsafe_get(i) == '-'
    if neg {
      i += 1
    }
    let mut v = 0
    let mut ok = i < n
    while i < n && s.unsafe_get(i) != ' ' {
      let c = s.unsafe_get(i).to_int() - '0'.to_int()
      if c < 0 || c > 9 || v > 214748364 {
        ok = false
      }
      v = v * 10 + c
      i += 1
    }
    if start == i {
      break
    }
    if i == start + (if neg { 1 } else { 0 }) {
      ok = false
    }
    if !ok {
      // not a plain decimal integer: let the general parser judge it
      let t = s.unsafe_view(start~, end=i)
      out.push(
        @string.parse_int(t) catch {
          _ => raise InvalidModel("bad integer in `\{key}`: \{t}")
        },
      )
    } else {
      out.push(if neg { -v } else { v })
    }
  }
  out
}

///|
fn lgb_doubles(
  fields : Map[String, StringView],
  key : String,
) -> Array[Double] raise ModelError {
  let s = match fields.get(key) {
    Some(s) => s
    None => raise InvalidModel("LightGBM tree is missing `\{key}`")
  }
  let out = []
  let n = s.length()
  let mut i = 0
  while i < n {
    while i < n && s.unsafe_get(i) == ' ' {
      i += 1
    }
    let start = i
    while i < n && s.unsafe_get(i) != ' ' {
      i += 1
    }
    if start == i {
      break
    }
    out.push(
      lgb_parse_double(s, start, i) catch {
        _ => {
          let t = s.unsafe_view(start~, end=i)
          raise InvalidModel("bad number in `\{key}`: \{t}")
        }
      },
    )
  }
  out
}

///|
fn lgb_tree(fields : Map[String, StringView]) -> LightGBMTree raise ModelError {
  let num_leaves = lgb_ints(fields, "num_leaves")
  if num_leaves.length() != 1 || num_leaves[0] < 1 {
    raise InvalidModel("bad `num_leaves`")
  }
  let n = num_leaves[0]
  if fields.get("is_linear") is Some(v) && v.trim(chars=" ") != "0" {
    raise Unsupported("LightGBM linear trees")
  }
  let leaf_value = lgb_doubles(fields, "leaf_value")
  if leaf_value.length() != n {
    raise InvalidModel("\{leaf_value.length()} leaf values for \{n} leaves")
  }
  if n == 1 {
    return {
      split_feature: [],
      threshold: [],
      decision_type: [],
      left: [],
      right: [],
      leaf_value,
      cat_boundaries: [],
      cat_threshold: [],
    }
  }
  let split_feature = lgb_ints(fields, "split_feature")
  let threshold = lgb_doubles(fields, "threshold")
  let decision_type = lgb_ints(fields, "decision_type")
  let left = lgb_ints(fields, "left_child")
  let right = lgb_ints(fields, "right_child")
  let inner = n - 1
  for
    a in [
      split_feature.length(),
      threshold.length(),
      decision_type.length(),
      left.length(),
      right.length(),
    ] {
    if a != inner {
      raise InvalidModel("tree arrays should have \{inner} entries, got \{a}")
    }
  }
  for c in left + right {
    // an internal node index, or ~leaf for a leaf
    if c >= inner || c < -n {
      raise InvalidModel("child \{c} out of range")
    }
  }
  let num_cat = match fields.get("num_cat") {
    Some(_) => lgb_ints(fields, "num_cat")
    None => [0]
  }
  let (cat_boundaries, cat_threshold) = if num_cat.length() == 1 &&
    num_cat[0] > 0 {
    let b = lgb_ints(fields, "cat_boundaries")
    let words = []
    for v in lgb_values(fields, "cat_threshold") {
      let d = @string.parse_double(v) catch {
        _ => raise InvalidModel("bad category bitset word: \{v}")
      }
      words.push(d.to_uint())
    }
    if b.length() != num_cat[0] + 1 {
      raise InvalidModel(
        "`cat_boundaries` should have \{num_cat[0] + 1} entries",
      )
    }
    for i in 0..<num_cat[0] {
      if b[i] < 0 || b[i] > b[i + 1] || b[i + 1] > words.length() {
        raise InvalidModel("`cat_boundaries` out of range")
      }
    }
    (b, words)
  } else {
    ([], [])
  }
  for i in 0..<inner {
    if decision_type[i] % 2 == 1 {
      let k = threshold[i].to_int()
      if k < 0 || k + 1 >= cat_boundaries.length() {
        raise InvalidModel("categorical split \{i} has no category set")
      }
    }
    if split_feature[i] < 0 {
      raise InvalidModel("negative split feature")
    }
  }
  {
    split_feature,
    threshold,
    decision_type,
    left,
    right,
    leaf_value,
    cat_boundaries,
    cat_threshold,
  }
}

///|
/// Load a LightGBM text model (`Booster.save_model` or `model_to_string`).
/// Numerical and categorical splits, all three missing-value modes, and every
/// built-in objective's output transform are supported; linear trees are not.
pub fn LightGBMModel::from_text(
  text : StringView,
) -> LightGBMModel raise ModelError {
  let header : Map[String, String] = Map([])
  let blocks : Array[Map[String, StringView]] = []
  let mut cur : Map[String, StringView]? = None
  for raw in text.split("\n") {
    let line = raw.trim(chars="\r ")
    if line.has_prefix("end of trees") {
      break
    }
    if line.has_prefix("Tree=") {
      let m : Map[String, StringView] = Map([])
      blocks.push(m)
      cur = Some(m)
      continue
    }
    match line.find("=") {
      Some(i) => {
        let key = line[:i]
        match cur {
          // a tree keeps only the fields prediction reads, without copying
          Some(m) => if lgb_tree_key(key) { m[key.to_owned()] = line[i + 1:] }
          None => header[key.to_owned()] = line[i + 1:].to_owned()
        }
      }
      // a bare flag such as `average_output`
      None => if cur is None && line != "" { header[line.to_owned()] = "" }
    }
  }
  if header.get("version") is None {
    raise InvalidModel("not a LightGBM text model (no `version` line)")
  }
  let get_int = fn(key : String) -> Int32 raise ModelError {
    match header.get(key) {
      Some(v) =>
        @string.parse_int(v.trim(chars=" ")) catch {
          _ => raise InvalidModel("`\{key}` is not an integer: \{v}")
        }
      None => raise InvalidModel("missing `\{key}`")
    }
  }
  let num_class = get_int("num_class")
  let per_iter = get_int("num_tree_per_iteration")
  let num_feature = get_int("max_feature_idx") + 1
  if num_class < 1 || per_iter < 1 || num_feature < 1 {
    raise InvalidModel("bad model header")
  }
  let trees = []
  for b in blocks {
    trees.push(lgb_tree(b))
  }
  if trees.length() % per_iter != 0 {
    raise InvalidModel(
      "\{trees.length()} trees is not a multiple of \{per_iter} per iteration",
    )
  }
  for t in trees {
    for f in t.split_feature {
      if f >= num_feature {
        raise InvalidModel(
          "tree uses feature \{f} but model has \{num_feature}",
        )
      }
    }
  }
  let objective = header.get("objective").unwrap_or("")
  let feature_names = match header.get("feature_names") {
    Some(s) =>
      s.split(" ").filter(x => x != "").map(x => x.to_owned()).to_array()
    None => []
  }
  {
    trees,
    num_class,
    num_tree_per_iteration: per_iter,
    num_feature,
    objective,
    feature_names,
    average_output: header.contains("average_output"),
    fast: LgbFast::build(trees, num_feature, per_iter),
    obj_name: objective.split(" ").next().map(x => x.to_owned()).unwrap_or(""),
    sigmoid: lgb_param(objective, "sigmoid", 1.0),
  }
}

///|
/// Leaf reached by `row` (already cleaned by `lgb_row`), as LightGBM's
/// `Tree::GetLeaf` walks it.
fn LightGBMTree::leaf(self : LightGBMTree, row : FixedArray[Double]) -> Int32 {
  if self.left.length() == 0 {
    return 0
  }
  let mut node = 0
  while node >= 0 {
    let d = self.decision_type[node]
    let x = row[self.split_feature[node]]
    node = if d % 2 == 1 {
      // categorical: NaN and negative codes go right; otherwise left when the
      // code is in the node's bitset
      if x.is_nan() || x.to_int() < 0 {
        self.right[node]
      } else {
        let code = x.to_int()
        let k = self.threshold[node].to_int()
        let start = self.cat_boundaries[k]
        let words = self.cat_boundaries[k + 1] - start
        let w = code / 32
        if w < words &&
          ((self.cat_threshold[start + w] >> (code % 32)) & 1) == 1 {
          self.left[node]
        } else {
          self.right[node]
        }
      }
    } else {
      let missing = (d >> 2) & 3
      let v = if x.is_nan() && missing != 2 { 0.0 } else { x }
      if (missing == 1 && v >= -lgb_zero && v <= lgb_zero) ||
        (missing == 2 && v.is_nan()) {
        if (d & 2) != 0 {
          self.left[node]
        } else {
          self.right[node]
        }
      } else if v <= self.threshold[node] {
        self.left[node]
      } else {
        self.right[node]
      }
    }
  }
  -node - 1
}

///|
/// LightGBM's predictor reads a dense row through its sparse path: values
/// within `kZeroThreshold` of zero become 0, and absent features are 0.
fn LightGBMModel::lgb_row(
  self : LightGBMModel,
  row : ArrayView[Double],
) -> FixedArray[Double] {
  let out = FixedArray::make(self.num_feature, 0.0)
  let n = if row.length() < self.num_feature {
    row.length()
  } else {
    self.num_feature
  }
  for i in 0..<n {
    let x = row[i]
    if x.is_nan() || x.abs() > lgb_zero {
      out[i] = x
    }
  }
  out
}

///|
/// Raw scores, one per class (`predict(raw_score=True)`): the sum of the
/// trees. Like LightGBM, a random forest's raw score is not averaged.
pub fn LightGBMModel::predict_raw(
  self : LightGBMModel,
  row : ArrayView[Double],
) -> Array[Double] {
  let k = self.num_tree_per_iteration
  if self.fast is Some(f) {
    let w = f.width()
    let ranks = FixedArray::make(if f.by_value { 0 } else { w }, 0)
    let values = FixedArray::make(if f.by_value { w } else { 0 }, 0.0)
    f.slots_of_doubles(row, ranks, values)
    let m = FixedArray::make(k, 0.0)
    f.accumulate(ranks, values, 1, m)
    return Array::makei(k, i => m[i])
  }
  let x = self.lgb_row(row)
  let out = Array::make(k, 0.0)
  for i, t in self.trees {
    out[i % k] = out[i % k] + t.leaf_value[t.leaf(x)]
  }
  out
}

///|
fn lgb_param(objective : String, key : String, default : Double) -> Double {
  for part in objective.split(" ") {
    if part.has_prefix(key + ":") {
      return @string.parse_double(part[key.length() + 1:]) catch {
        _ => default
      }
    }
  }
  default
}

///|
/// Final predictions (`predict()`): the raw scores, averaged over the rounds
/// for a random forest, then the objective's output transform.
pub fn LightGBMModel::predict(
  self : LightGBMModel,
  row : ArrayView[Double],
) -> Array[Double] {
  let out = self.predict_raw(row)
  self.transform(out)
  out
}

///|
/// Raw scores to predictions, in place: the average over rounds for a
/// random forest, then the objective's output transform.
fn LightGBMModel::transform(self : LightGBMModel, out : Array[Double]) -> Unit {
  let k = self.num_tree_per_iteration
  if self.average_output && self.trees.length() > 0 {
    let rounds = (self.trees.length() / k).to_double()
    for g in 0..<k {
      out[g] = out[g] / rounds
    }
  }
  let sigmoid = self.sigmoid
  match self.obj_name {
    "binary" | "multiclassova" =>
      for i in 0..<out.length() {
        out[i] = 1.0 / (1.0 + lgb_exp(-sigmoid * out[i]))
      }
    "cross_entropy" => out[0] = 1.0 / (1.0 + lgb_exp(-out[0]))
    "cross_entropy_lambda" => out[0] = @math.ln_1p(lgb_exp(out[0]))
    "poisson" | "gamma" | "tweedie" => out[0] = lgb_exp(out[0])
    "multiclass" => {
      let mut wmax = out[0]
      for v in out {
        if v > wmax {
          wmax = v
        }
      }
      let mut wsum = 0.0
      for i in 0..<out.length() {
        out[i] = lgb_exp(out[i] - wmax)
        wsum = wsum + out[i]
      }
      for i in 0..<out.length() {
        out[i] = out[i] / wsum
      }
    }
    "regression" if self.objective.contains("sqrt") => {
      let v = out[0]
      out[0] = if v > 0.0 { v * v } else if v < 0.0 { -(v * v) } else { 0.0 }
    }
    _ => ()
  }
}

///|
/// Predictions for every row of `data`, `num_tree_per_iteration` values per
/// row (row-major). Same results as `predict` on each row.
pub fn LightGBMModel::predict_batch(
  self : LightGBMModel,
  data : DenseMatrix,
) -> Array[Double] {
  let k = self.num_tree_per_iteration
  let out = Array::make(data.rows * k, 0.0)
  match self.fast {
    Some(f) if f.heap_depth > 0 && k == 1 => {
      let block = 256
      let vals = FixedArray::make(block * f.width(), (0.0 : Float))
      let m = FixedArray::make(block, 0.0)
      let row = [0.0]
      let mut start = 0
      while start < data.rows {
        let n = if data.rows - start < block {
          data.rows - start
        } else {
          block
        }
        f.float_slots(data.data, data.cols, start, n, vals)
        m.fill(0.0)
        f.accumulate_fheap(vals, n, m)
        for r in 0..<n {
          row[0] = m[r]
          self.transform(row)
          out[start + r] = row[0]
        }
        start += n
      }
    }
    Some(f) => {
      // rank blocks of 256 rows reuse each tree from cache four times as
      // long; value blocks are eight bytes a slot and stay at 64 rows
      let block = if f.by_value { 64 } else { 256 }
      let w = f.width()
      let ranks = FixedArray::make(if f.by_value { 0 } else { block * w }, 0)
      let values = FixedArray::make(if f.by_value { block * w } else { 0 }, 0.0)
      let m = FixedArray::make(block * k, 0.0)
      let cols = data.cols
      let mut start = 0
      while start < data.rows {
        let n = if data.rows - start < block {
          data.rows - start
        } else {
          block
        }
        for r in 0..<n {
          f.slots_of_floats(data.data, start + r, cols, ranks, values, r * w)
        }
        m.fill(0.0)
        f.accumulate(ranks, values, n, m)
        let row = Array::make(k, 0.0)
        for r in 0..<n {
          for g in 0..<k {
            row[g] = m[r * k + g]
          }
          self.transform(row)
          for g in 0..<k {
            out[(start + r) * k + g] = row[g]
          }
        }
        start += n
      }
    }
    None =>
      for r in 0..<data.rows {
        let p = self.predict(data.row(r).map(x => x.to_double()))
        for g in 0..<k {
          out[r * k + g] = p[g]
        }
      }
  }
  out
}

///|
/// Leaf index reached in every tree (`predict(pred_leaf=True)`).
pub fn LightGBMModel::predict_leaf(
  self : LightGBMModel,
  row : ArrayView[Double],
) -> Array[Int32] {
  let x = self.lgb_row(row)
  self.trees.map(t => t.leaf(x))
}

///|
/// Number of trees in the model.
pub fn LightGBMModel::num_trees(self : LightGBMModel) -> Int32 {
  self.trees.length()
}
