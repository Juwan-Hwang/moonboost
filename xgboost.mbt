///|
/// XGBoost stores most scalar parameters as strings, e.g. `"num_class": "3"`.
fn param_int(obj : Src, key : String) -> Int32 raise ModelError {
  let s = obj.field(key).string(key)
  @string.parse_int(s) catch {
    _ => raise InvalidModel("`\{key}` is not an integer: \{s}")
  }
}

///|
/// `base_score` is `"0.5"` in older models and `"[5E-1]"` (a vector) in 3.x.
fn parse_base_score(s : String) -> Float raise ModelError {
  let body = s.trim(chars="[] ")
  let first = match body.to_owned().split_once(",") {
    Some((a, _)) => a
    None => body
  }
  let d = @string.parse_double(first) catch {
    _ => raise InvalidModel("bad base_score: \{s}")
  }
  Float::from_double(d)
}

///|
fn parse_tree(t : Src) -> Tree raise ModelError {
  let left = t.field("left_children").ints("left_children")
  let split_type = t.field("split_type").ints("split_type")
  let categories : Array[Array[Int32]?] = Array::make(left.length(), None)
  if split_type.iter().any(k => k != 0) {
    let cats = t.field("categories").ints("categories")
    let nodes = t.field("categories_nodes").ints("categories_nodes")
    let segs = t.field("categories_segments").ints("categories_segments")
    let sizes = t.field("categories_sizes").ints("categories_sizes")
    if nodes.length() != segs.length() || nodes.length() != sizes.length() {
      raise InvalidModel("categorical split arrays have inconsistent lengths")
    }
    for i, node in nodes {
      let start = segs[i]
      let end = start + sizes[i]
      if node < 0 || node >= left.length() || start < 0 || end > cats.length() {
        raise InvalidModel("categorical split \{i} is out of range")
      }
      categories[node] = Some(cats[start:end].to_owned())
    }
    for i, k in split_type {
      if k != 0 && categories[i] is None {
        raise InvalidModel("node \{i} is categorical but has no categories")
      }
    }
  }
  Tree::new(
    left~,
    right=t.field("right_children").ints("right_children"),
    feature=t.field("split_indices").ints("split_indices"),
    value=t.field("split_conditions").floats("split_conditions"),
    default_left=t.field("default_left").ints("default_left").map(x => x != 0),
    categories~,
  )
}

///|
/// Load a model saved by XGBoost's `Booster.save_model("model.json")`.
///
/// Supports the `gbtree` booster with numerical and categorical splits.
/// Categorical features are passed as their integer codes. Raises
/// `Unsupported` for `gblinear` and `dart`.
pub fn Model::from_xgboost_json(text : StringView) -> Model raise ModelError {
  let root = Src::parse(text)
  let learner = root.field("learner")
  let lmp = learner.field("learner_model_param")
  let num_feature = param_int(lmp, "num_feature")
  let num_class = param_int(lmp, "num_class")
  let base_score = parse_base_score(
    lmp.field("base_score").string("base_score"),
  )
  let objective = Objective::from_name(
    learner.field("objective").field("name").string("objective.name"),
    num_class~,
  )
  let gb = learner.field("gradient_booster")
  let booster = gb.field("name").string("gradient_booster.name")
  if booster != "gbtree" {
    raise Unsupported("booster \{booster}")
  }
  let model = gb.field("model")
  let trees = model.field("trees").items("trees").map(parse_tree)
  let tree_group = model.field("tree_info").ints("tree_info")
  // optional; anything but an array of strings is ignored as before
  let names = if learner.has("feature_names") {
    learner.field("feature_names").items("feature_names") catch {
      _ => []
    }
  } else {
    []
  }
  let feature_names = names.map(v => v.string("feature_names"))
  Model::new(
    trees~,
    tree_group~,
    num_group=if num_class > 1 { num_class } else { 1 },
    num_feature~,
    base_score~,
    objective~,
    feature_names~,
  )
}
