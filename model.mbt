///|
/// A boosted tree ensemble.
///
/// `tree_group[i]` is the output group (class) that tree `i` contributes to.
/// Single-output models have one group; `multi:*` models have one per class.
/// `base_margin[g]` is the starting score of group `g`, already in margin
/// space. A linear model (`gblinear`) has no trees and keeps its weights in
/// `linear`: the weight of feature `f` for group `g` at `f * num_group + g`,
/// then one bias per group.
pub struct Model {
  trees : Array[Tree]
  tree_group : Array[Int32]
  num_group : Int32
  num_feature : Int32
  base_margin : Array[Float]
  linear : Array[Float]
  objective : Objective
  feature_names : Array[String]
  priv compiled : Compiled
  priv fast : Lockstep?
}

///|
pub fn Model::new(
  trees~ : Array[Tree],
  tree_group~ : Array[Int32],
  num_group~ : Int32,
  num_feature~ : Int32,
  base_score~ : Float,
  objective~ : Objective,
  feature_names~ : Array[String] = [],
  base_scores~ : Array[Float] = [],
) -> Model raise ModelError {
  if base_scores.length() != 0 && base_scores.length() != num_group {
    raise InvalidModel(
      "\{base_scores.length()} base scores for \{num_group} groups",
    )
  }
  if trees.length() != tree_group.length() {
    raise InvalidModel(
      "\{trees.length()} trees but \{tree_group.length()} group entries",
    )
  }
  if num_group < 1 {
    raise InvalidModel("num_group must be positive, got \{num_group}")
  }
  for g in tree_group {
    if g < 0 || g >= num_group {
      raise InvalidModel("tree group \{g} out of range 0..<\{num_group}")
    }
  }
  for t in trees {
    if t.max_feature() >= num_feature {
      raise InvalidModel(
        "tree uses feature \{t.max_feature()} but model has \{num_feature}",
      )
    }
  }
  {
    trees,
    tree_group,
    num_group,
    num_feature,
    base_margin: Array::makei(num_group, g => {
      objective.base_margin(
        if base_scores.length() == 0 {
          base_score
        } else {
          base_scores[g]
        },
      )
    }),
    linear: [],
    objective,
    feature_names,
    compiled: compile(trees, num_feature),
    fast: Lockstep::build(trees, num_feature),
  }
}

///|
/// Number of trees in the ensemble.
pub fn Model::num_trees(self : Model) -> Int32 {
  self.trees.length()
}

///|
/// Raw scores (one per output group) before the objective's link function.
/// Accumulates in tree order in 32-bit floats, as XGBoost's CPU predictor does.
pub fn Model::predict_margin(
  self : Model,
  row : ArrayView[Float],
) -> Array[Float] {
  if self.fast is Some(fast) {
    let ranks = Buf::new(2 * fast.num_feature)
    fast.ranks(row, ranks, 0)
    let margin = FixedArray::makei(self.num_group, g => self.base_margin[g])
    fast.accumulate(
      self.tree_group,
      self.num_group,
      ranks,
      1,
      Buf::new(2),
      margin,
    )
    return Array::makei(self.num_group, g => margin[g])
  }
  if self.linear.length() != 0 {
    return Array::makei(self.num_group, g => self.linear_margin(row, g))
  }
  let out = self.base_margin.copy()
  let c = self.compiled
  let keys = c.keys(row)
  for t in 0..<self.tree_group.length() {
    let g = self.tree_group[t]
    out[g] = out[g] + c.leaf_value[c.leaf(t, keys)]
  }
  out
}

///|
/// Final predictions for one row: probabilities for classifiers, values for
/// regressors, or the class index for `multi:softmax`.
pub fn Model::predict(self : Model, row : ArrayView[Float]) -> Array[Float] {
  let out = self.predict_margin(row)
  self.objective.transform(out)
  out
}

///|
/// Single-output convenience: the first (and only) prediction for `row`.
pub fn Model::predict_one(self : Model, row : ArrayView[Float]) -> Float {
  self.predict(row)[0]
}

///|
/// Leaf index reached in every tree, like XGBoost's `pred_leaf=True`.
pub fn Model::predict_leaf(
  self : Model,
  row : ArrayView[Float],
) -> Array[Int32] {
  let c = self.compiled
  let keys = c.keys(row)
  Array::makei(self.tree_group.length(), t => c.leaf(t, keys) - c.offsets[t])
}

///|
/// A linear model, like XGBoost's `gblinear` booster. `weights` holds the
/// weight of feature `f` for group `g` at `f * num_group + g`, followed by one
/// bias per group. Every group starts from `base_score` (XGBoost uses the
/// first entry of its base score for all groups of a linear model).
pub fn Model::linear(
  weights~ : Array[Float],
  num_group~ : Int32,
  num_feature~ : Int32,
  base_score~ : Float,
  objective~ : Objective,
  feature_names~ : Array[String] = [],
) -> Model raise ModelError {
  if num_group < 1 {
    raise InvalidModel("num_group must be positive, got \{num_group}")
  }
  if num_feature < 0 || weights.length() != (num_feature + 1) * num_group {
    raise InvalidModel(
      "\{weights.length()} linear weights for \{num_feature} features and \{num_group} groups",
    )
  }
  {
    trees: [],
    tree_group: [],
    num_group,
    num_feature,
    base_margin: Array::make(num_group, objective.base_margin(base_score)),
    linear: weights,
    objective,
    feature_names,
    compiled: compile([], num_feature),
    fast: None,
  }
}

///|
/// Margin of group `g` for a linear model: bias plus base, then each present
/// feature's `x * w` added in feature order in 32-bit floats, as XGBoost does
/// (the product is rounded before the sum). Missing (NaN) features and
/// features beyond the model's count are skipped.
fn Model::linear_margin(
  self : Model,
  row : ArrayView[Float],
  g : Int32,
) -> Float {
  let groups = self.num_group
  let nf = self.num_feature
  let mut sum = self.linear[nf * groups + g] + self.base_margin[g]
  let n = if row.length() < nf { row.length() } else { nf }
  for f in 0..<n {
    let x = row[f]
    if !x.is_nan() {
      // the product of two floats is exact in a double; rounding it back
      // keeps a C compiler from fusing it into the sum (XGBoost's x86-64
      // build rounds the product, and so does every MoonBit target)
      let p = Float::from_double(
        x.to_double() * self.linear[f * groups + g].to_double(),
      )
      sum = sum + p
    }
  }
  sum
}
