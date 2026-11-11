///|
/// JavaScript entry points. Build with
/// `moon build --target js --release` and import
/// `_build/js/release/build/web/web.js` as an ES module.
///
/// ```js
/// import { load_model, predict, predict_batch } from "./web.js";
/// const model = load_model(jsonText);
/// predict(model, [5.1, 3.5, 1.4, 0.2]); // [0.99, 0.005, 0.003]
/// ```
///
/// Missing values are `NaN`.

///|
/// Parse an XGBoost JSON model. Returns `undefined` if it cannot be loaded;
/// `last_error()` then says why.
#export_name("load_model")
pub fn load_model(json : String) -> @moonboost.Model? {
  try {
    let m = @moonboost.Model::from_xgboost_json(json)
    error.val = ""
    Some(m)
  } catch {
    @moonboost.InvalidModel(msg) => {
      error.val = "invalid model: " + msg
      None
    }
    @moonboost.Unsupported(msg) => {
      error.val = "unsupported: " + msg
      None
    }
  }
}

///|
/// Parse an XGBoost UBJSON model (`.ubj`) from a `Uint8Array`. Returns
/// `undefined` if it cannot be loaded; `last_error()` then says why.
#export_name("load_model_ubjson")
pub fn load_model_ubjson(data : Bytes) -> @moonboost.Model? {
  try {
    let m = @moonboost.Model::from_xgboost_ubjson(data)
    error.val = ""
    Some(m)
  } catch {
    @moonboost.InvalidModel(msg) => {
      error.val = "invalid model: " + msg
      None
    }
    @moonboost.Unsupported(msg) => {
      error.val = "unsupported: " + msg
      None
    }
  }
}

///|
let error : Ref[String] = Ref("")

///|
/// Why the last `load_model` call failed, or "" if it succeeded.
#export_name("last_error")
pub fn last_error() -> String {
  error.val
}

///|
fn to_floats(xs : FixedArray[Double]) -> Array[Float] {
  Array::makei(xs.length(), i => Float::from_double(xs[i]))
}

///|
fn to_doubles(xs : Array[Float]) -> FixedArray[Double] {
  FixedArray::makei(xs.length(), i => xs[i].to_double())
}

///|
/// Predictions for one row, as `predict` in XGBoost would return them.
#export_name("predict")
pub fn predict(
  model : @moonboost.Model,
  row : FixedArray[Double],
) -> FixedArray[Double] {
  to_doubles(model.predict(to_floats(row)))
}

///|
/// Raw scores for one row, before the objective's link function.
#export_name("predict_margin")
pub fn predict_margin(
  model : @moonboost.Model,
  row : FixedArray[Double],
) -> FixedArray[Double] {
  to_doubles(model.predict_margin(to_floats(row)))
}

///|
/// Predictions for `data.length / cols` rows stored row-major in `data`.
/// Returns `output_width(model)` values per row, or an empty array if
/// `data.length` is not a multiple of `cols`.
#export_name("predict_batch")
pub fn predict_batch(
  model : @moonboost.Model,
  data : FixedArray[Double],
  cols : Int32,
) -> FixedArray[Double] {
  let m = @moonboost.DenseMatrix::from_fixed(to_float_matrix(data), cols~) catch {
    _ => return []
  }
  to_doubles(model.predict_batch(m))
}

///|
/// Number of values `predict` returns per row.
#export_name("output_width")
pub fn output_width(model : @moonboost.Model) -> Int32 {
  model.output_width()
}

///|
/// Number of input features the model expects.
#export_name("num_features")
pub fn num_features(model : @moonboost.Model) -> Int32 {
  model.num_feature
}

///|
/// Parse a LightGBM text model. Returns `undefined` if it cannot be loaded;
/// `last_error()` then says why.
#export_name("load_lightgbm")
pub fn load_lightgbm(text : String) -> @moonboost.LightGBMModel? {
  try {
    let m = @moonboost.LightGBMModel::from_text(text)
    error.val = ""
    Some(m)
  } catch {
    @moonboost.InvalidModel(msg) => {
      error.val = "invalid model: " + msg
      None
    }
    @moonboost.Unsupported(msg) => {
      error.val = "unsupported: " + msg
      None
    }
  }
}

///|
/// LightGBM predictions for one row, as its `predict` returns them. LightGBM
/// works in doubles, so the row is used as given.
#export_name("predict_lightgbm")
pub fn predict_lightgbm(
  model : @moonboost.LightGBMModel,
  row : FixedArray[Double],
) -> FixedArray[Double] {
  let out = model.predict(row[:])
  FixedArray::makei(out.length(), i => out[i])
}

///|
/// LightGBM raw scores for one row (`predict(raw_score=True)`).
#export_name("predict_lightgbm_raw")
pub fn predict_lightgbm_raw(
  model : @moonboost.LightGBMModel,
  row : FixedArray[Double],
) -> FixedArray[Double] {
  let out = model.predict_raw(row[:])
  FixedArray::makei(out.length(), i => out[i])
}

///|
/// Train a model in the page. `data` holds the rows one after another
/// (an array of numbers or a `Float32Array`, `NaN` for missing), `labels` one
/// label per row. The settings are XGBoost's `objective`, `num_class`,
/// `num_round`, `max_depth` and `eta`; pass `-1` (or `""` for `objective`)
/// to keep XGBoost's default.
/// Returns the model as XGBoost JSON text (load it with `load_model`, or in
/// XGBoost), or `undefined` with `last_error()` set.
#export_name("train")
pub fn train(
  data : FixedArray[Double],
  cols : Int32,
  labels : FixedArray[Double],
  objective : String,
  num_class : Int32,
  num_round : Int32,
  max_depth : Int32,
  eta : Double,
) -> String? {
  let d = @moonboost.TrainParams::default()
  let params : @moonboost.TrainParams = {
    ..d,
    objective: if objective == "" {
      d.objective
    } else {
      objective
    },
    num_class: if num_class < 0 {
      d.num_class
    } else {
      num_class
    },
    num_round: if num_round < 0 {
      d.num_round
    } else {
      num_round
    },
    max_depth: if max_depth < 0 {
      d.max_depth
    } else {
      max_depth
    },
    eta: if eta < 0.0 {
      d.eta
    } else {
      eta
    },
  }
  try {
    let m = @moonboost.DenseMatrix::from_fixed(to_float_matrix(data), cols~)
    let y = to_float_matrix(labels)
    let json = @moonboost.train_xgboost_json(
      m,
      Array::makei(y.length(), i => y[i]),
      params~,
    )
    error.val = ""
    Some(json)
  } catch {
    @moonboost.InvalidModel(msg) => {
      error.val = "invalid input: " + msg
      None
    }
    @moonboost.Unsupported(msg) => {
      error.val = "unsupported: " + msg
      None
    }
  }
}
