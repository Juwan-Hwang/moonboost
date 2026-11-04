///|
/// A boosted tree ensemble.
///
/// `tree_group[i]` is the output group (class) that tree `i` contributes to.
/// Single-output models have one group; `multi:*` models have one per class.
pub struct Model {
  trees : Array[Tree]
  tree_group : Array[Int32]
  num_group : Int32
  num_feature : Int32
  base_margin : Float
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
) -> Model raise ModelError {
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
    base_margin: objective.base_margin(base_score),
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
    let margin = FixedArray::make(self.num_group, self.base_margin)
    fast.accumulate(
      self.tree_group,
      self.num_group,
      ranks,
      1,
      Buf::new(1),
      margin,
    )
    return Array::makei(self.num_group, g => margin[g])
  }
  let out = Array::make(self.num_group, self.base_margin)
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
