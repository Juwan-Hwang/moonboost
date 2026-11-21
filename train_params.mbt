///|
/// Training settings from a JSON object of XGBoost parameter names, as
/// passed to `xgboost.train` (`{"objective": "binary:logistic", "eta": 0.1,
/// "max_depth": 8}`); unnamed settings keep `TrainParams::default()`.
/// Numbers may also be written as strings. Besides the fields of
/// `TrainParams`, it accepts XGBoost's aliases `learning_rate`,
/// `reg_lambda`, `min_split_loss`, `num_boost_round` and `n_estimators`,
/// and `lambdarank_num_pair_per_sample` for `rank_truncation`.
pub fn TrainParams::from_json(
  text : StringView,
) -> TrainParams raise ModelError {
  let j = @json.parse(text) catch {
    e => raise InvalidModel("invalid JSON: \{e}")
  }
  guard j is Object(m) else {
    raise InvalidModel("training parameters should be a JSON object")
  }
  let mut p = TrainParams::default()
  for key, v in m {
    p = p.with_setting(key, v)
  }
  p
}

///|
fn setting_number(key : String, v : Json) -> Double raise ModelError {
  match v {
    Number(d, ..) => d
    String(s) =>
      @string.parse_double(s) catch {
        _ => raise InvalidModel("`\{key}` should be a number")
      }
    True => 1.0
    False => 0.0
    _ => raise InvalidModel("`\{key}` should be a number")
  }
}

///|
fn setting_int(key : String, v : Json) -> Int32 raise ModelError {
  let d = setting_number(key, v)
  if d != d.floor() || d.abs() > 2147483647.0 {
    raise InvalidModel("`\{key}` should be an integer")
  }
  d.to_int()
}

///|
fn setting_string(key : String, v : Json) -> String raise ModelError {
  match v {
    String(s) => s
    _ => raise InvalidModel("`\{key}` should be a string")
  }
}

///|
/// A list of integers: a JSON array, or a string `"0,3,4"`.
fn setting_ints(key : String, v : Json) -> Array[Int32] raise ModelError {
  match v {
    Array(a) => a.map(x => setting_int(key, x))
    String(s) => {
      let out = []
      for part in s.split(",") {
        let t = part.trim()
        if t.length() > 0 {
          out.push(setting_int(key, Json::string(t.to_owned())))
        }
      }
      out
    }
    _ => raise InvalidModel("`\{key}` should be an array of integers")
  }
}

///|
fn TrainParams::with_setting(
  self : TrainParams,
  key : String,
  v : Json,
) -> TrainParams raise ModelError {
  let p = self
  match key {
    "objective" => { ..p, objective: setting_string(key, v), }
    "num_class" => { ..p, num_class: setting_int(key, v), }
    "num_round" | "num_boost_round" | "n_estimators" =>
      { ..p, num_round: setting_int(key, v), }
    "max_depth" => { ..p, max_depth: setting_int(key, v), }
    "eta" | "learning_rate" => { ..p, eta: setting_number(key, v), }
    "lambda" | "reg_lambda" => { ..p, lambda: setting_number(key, v), }
    "gamma" | "min_split_loss" => { ..p, gamma: setting_number(key, v), }
    "min_child_weight" => { ..p, min_child_weight: setting_number(key, v), }
    "max_bin" => { ..p, max_bin: setting_int(key, v), }
    "grow_policy" => { ..p, grow_policy: setting_string(key, v), }
    "max_leaves" => { ..p, max_leaves: setting_int(key, v), }
    "categorical" => { ..p, categorical: setting_ints(key, v), }
    "max_cat_to_onehot" => { ..p, max_cat_to_onehot: setting_int(key, v), }
    "max_cat_threshold" => { ..p, max_cat_threshold: setting_int(key, v), }
    "cat_smooth" => { ..p, cat_smooth: setting_number(key, v), }
    "cat_l2" => { ..p, cat_l2: setting_number(key, v), }
    "min_data_per_group" => { ..p, min_data_per_group: setting_number(key, v), }
    "subsample" => { ..p, subsample: setting_number(key, v), }
    "sampling_method" => { ..p, sampling_method: setting_string(key, v), }
    "colsample_bytree" => { ..p, colsample_bytree: setting_number(key, v), }
    "colsample_bynode" => { ..p, colsample_bynode: setting_number(key, v), }
    "seed" => { ..p, seed: setting_int(key, v), }
    "min_data_in_leaf" => { ..p, min_data_in_leaf: setting_number(key, v), }
    "tweedie_variance_power" =>
      { ..p, tweedie_variance_power: setting_number(key, v), }
    "huber_slope" => { ..p, huber_slope: setting_number(key, v), }
    "quantile_alpha" => { ..p, quantile_alpha: setting_number(key, v), }
    "group" => { ..p, group: setting_ints(key, v), }
    "rank_truncation" | "lambdarank_num_pair_per_sample" =>
      { ..p, rank_truncation: setting_int(key, v), }
    _ => raise Unsupported("training parameter `\{key}`")
  }
}
