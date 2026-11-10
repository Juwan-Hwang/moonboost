///|
/// XGBoost stores most scalar parameters as strings, e.g. `"num_class": "3"`.
fn param_int(obj : Src, key : String) -> Int32 raise ModelError {
  let s = obj.field(key).string(key)
  @string.parse_int(s) catch {
    _ => raise InvalidModel("`\{key}` is not an integer: \{s}")
  }
}

///|
/// `base_score` is `"0.5"` in older models and a vector such as `"[5E-1]"`
/// in 3.x, where multiclass models keep one entry per class.
fn parse_base_score(s : String) -> Array[Float] raise ModelError {
  let out = []
  for part in s.trim(chars="[] ").split(",") {
    let d = @string.parse_double(part.trim(chars=" ")) catch {
      _ => raise InvalidModel("bad base_score: \{s}")
    }
    out.push(Float::from_double(d))
  }
  out
}

///|
/// Parse one tree. `weight` scales its leaves (DART's tree weights): XGBoost
/// adds `leaf * weight` rounded to float32, so the scaled leaf is exactly the
/// term it adds.
fn parse_tree(t : Src, weight : Float) -> Tree raise ModelError {
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
  let value = t.field("split_conditions").floats("split_conditions")
  if value.length() != left.length() {
    raise InvalidModel("tree arrays have inconsistent lengths")
  }
  Tree::new(
    left~,
    right=t.field("right_children").ints("right_children"),
    feature=t.field("split_indices").ints("split_indices"),
    value=if weight == 1.0 {
      value
    } else {
      Array::makei(value.length(), i => {
        if left[i] == -1 {
          value[i] * weight
        } else {
          value[i]
        }
      })
    },
    default_left=t.field("default_left").ints("default_left").map(x => x != 0),
    categories~,
  )
}

///|
/// Load a model saved by XGBoost's `Booster.save_model("model.json")`.
///
/// Supports the `gbtree` booster with numerical and categorical splits, DART
/// models (tree weights, in both the 3.x layout and the older `dart` booster
/// object) and the `gblinear` booster. Categorical features are passed as
/// their integer codes.
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
  let num_group = if num_class > 1 { num_class } else { 1 }
  let gb = learner.field("gradient_booster")
  let booster = gb.field("name").string("gradient_booster.name")
  // optional; anything but an array of strings is ignored as before
  let names = if learner.has("feature_names") {
    learner.field("feature_names").items("feature_names") catch {
      _ => []
    }
  } else {
    []
  }
  let feature_names = names.map(v => v.string("feature_names"))
  if booster == "gblinear" {
    return Model::linear(
      weights=gb.field("model").field("weights").floats("weights"),
      num_group~,
      num_feature~,
      base_score=base_score[0],
      objective~,
      feature_names~,
    )
  }
  // DART: 3.x saves a `gbtree` whose model carries `weight_drop`; older
  // releases save a `dart` booster wrapping the gbtree, weights beside it.
  let (model, weights) = match booster {
    "gbtree" => {
      let model = gb.field("model")
      let w = if model.has("weight_drop") {
        model.field("weight_drop").floats("weight_drop")
      } else {
        []
      }
      (model, w)
    }
    "dart" =>
      (
        gb.field("gbtree").field("model"),
        gb.field("weight_drop").floats("weight_drop"),
      )
    _ => raise Unsupported("booster \{booster}")
  }
  let items = model.field("trees").items("trees")
  if weights.length() > items.length() {
    raise InvalidModel(
      "\{weights.length()} tree weights for \{items.length()} trees",
    )
  }
  let trees = items.mapi((i, t) => {
    parse_tree(t, if i < weights.length() { weights[i] } else { 1.0 })
  })
  let tree_group = model.field("tree_info").ints("tree_info")
  if base_score.length() != 1 && base_score.length() != num_group {
    raise InvalidModel(
      "\{base_score.length()} base scores for \{num_group} groups",
    )
  }
  Model::new(
    trees~,
    tree_group~,
    num_group~,
    num_feature~,
    base_score=base_score[0],
    objective~,
    feature_names~,
    base_scores=if base_score.length() == num_group { base_score } else { [] },
  )
}
