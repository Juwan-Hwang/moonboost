///|
fn field(obj : Map[String, Json], key : String) -> Json raise ModelError {
  match obj.get(key) {
    Some(v) => v
    None => raise InvalidModel("missing field `\{key}`")
  }
}

///|
fn as_object(j : Json, what : String) -> Map[String, Json] raise ModelError {
  match j {
    Object(m) => m
    _ => raise InvalidModel("`\{what}` should be an object")
  }
}

///|
fn as_array(j : Json, what : String) -> Array[Json] raise ModelError {
  match j {
    Array(a) => a
    _ => raise InvalidModel("`\{what}` should be an array")
  }
}

///|
fn as_string(j : Json, what : String) -> String raise ModelError {
  match j {
    String(s) => s
    _ => raise InvalidModel("`\{what}` should be a string")
  }
}

///|
fn as_number(j : Json, what : String) -> Double raise ModelError {
  match j {
    Number(d, ..) => d
    _ => raise InvalidModel("`\{what}` should be a number")
  }
}

///|
/// XGBoost stores most scalar parameters as strings, e.g. `"num_class": "3"`.
fn param_int(obj : Map[String, Json], key : String) -> Int32 raise ModelError {
  let s = as_string(field(obj, key), key)
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
fn int_array(j : Json, what : String) -> Array[Int32] raise ModelError {
  as_array(j, what).map(v => as_number(v, what).to_int())
}

///|
fn float_array(j : Json, what : String) -> Array[Float] raise ModelError {
  as_array(j, what).map(v => Float::from_double(as_number(v, what)))
}

///|
fn parse_tree(j : Json) -> Tree raise ModelError {
  let t = as_object(j, "tree")
  let split_type = int_array(field(t, "split_type"), "split_type")
  for k in split_type {
    if k != 0 {
      raise Unsupported("categorical splits")
    }
  }
  Tree::new(
    left=int_array(field(t, "left_children"), "left_children"),
    right=int_array(field(t, "right_children"), "right_children"),
    feature=int_array(field(t, "split_indices"), "split_indices"),
    value=float_array(field(t, "split_conditions"), "split_conditions"),
    default_left=int_array(field(t, "default_left"), "default_left").map(x => {
      x != 0
    }),
  )
}

///|
/// Load a model saved by XGBoost's `Booster.save_model("model.json")`.
///
/// Supports the `gbtree` booster with numerical splits. Raises `Unsupported`
/// for `gblinear`, `dart` and categorical splits.
pub fn Model::from_xgboost_json(text : StringView) -> Model raise ModelError {
  let root = @json.parse(text) catch {
    e => raise InvalidModel("invalid JSON: \{e}")
  }
  let learner = as_object(field(as_object(root, "root"), "learner"), "learner")
  let lmp = as_object(
    field(learner, "learner_model_param"),
    "learner_model_param",
  )
  let num_feature = param_int(lmp, "num_feature")
  let num_class = param_int(lmp, "num_class")
  let base_score = parse_base_score(
    as_string(field(lmp, "base_score"), "base_score"),
  )
  let obj = as_object(field(learner, "objective"), "objective")
  let objective = Objective::from_name(
    as_string(field(obj, "name"), "objective.name"),
    num_class~,
  )
  let gb = as_object(field(learner, "gradient_booster"), "gradient_booster")
  let booster = as_string(field(gb, "name"), "gradient_booster.name")
  if booster != "gbtree" {
    raise Unsupported("booster \{booster}")
  }
  let model = as_object(field(gb, "model"), "model")
  let trees = as_array(field(model, "trees"), "trees").map(parse_tree)
  let tree_group = int_array(field(model, "tree_info"), "tree_info")
  let feature_names = match learner.get("feature_names") {
    Some(Array(a)) => a.map(v => as_string(v, "feature_names"))
    _ => []
  }
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
