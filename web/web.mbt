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
