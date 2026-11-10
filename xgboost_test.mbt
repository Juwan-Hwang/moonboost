///|
/// A model trained by XGBoost plus its reference outputs, as float32 bits.
struct Fixture {
  name : String
  model : String
  num_feature : Int32
  num_group : Int32
  pred_width : Int32
  x : Array[UInt32]
  margin : Array[UInt32]
  pred : Array[UInt32]
}

///|
fn to_floats(bits : ArrayView[UInt32]) -> Array[Float] {
  bits.map(Float::reinterpret_from_uint)
}

///|
/// Margins must match XGBoost bit for bit. Final predictions go through
/// libm `exp`, which may differ in the last ulp across platforms, so they are
/// compared within 2 ulp.
fn check(f : Fixture) -> Unit raise {
  let model = @moonboost.Model::from_xgboost_json(f.model)
  let x = to_floats(f.x)
  let rows = x.length() / f.num_feature
  let mut max_ulp = 0
  for r in 0..<rows {
    let row = x[r * f.num_feature:(r + 1) * f.num_feature]
    let m = model.predict_margin(row)
    for g in 0..<f.num_group {
      let want = f.margin[r * f.num_group + g]
      let got = m[g].reinterpret_as_uint()
      if got != want {
        fail(
          "\{f.name}: row \{r} group \{g} margin \{m[g]} != \{Float::reinterpret_from_uint(want)}",
        )
      }
    }
    let p = model.predict(row)
    for k in 0..<f.pred_width {
      let want = f.pred[r * f.pred_width + k].reinterpret_as_int()
      let got = p[k].reinterpret_as_int()
      let ulp = (got - want).abs()
      if ulp > max_ulp {
        max_ulp = ulp
      }
    }
  }
  if max_ulp > 2 {
    fail("\{f.name}: predictions differ by \{max_ulp} ulp")
  }
  // The blocked batch path must agree with `predict` bit for bit.
  let batch = model.predict_batch(
    @moonboost.DenseMatrix::new(x, cols=f.num_feature),
  )
  for r in 0..<rows {
    let p = model.predict(x[r * f.num_feature:(r + 1) * f.num_feature])
    for k in 0..<f.pred_width {
      if batch[r * f.pred_width + k].reinterpret_as_int() !=
        p[k].reinterpret_as_int() {
        fail("\{f.name}: predict_batch differs from predict at row \{r}")
      }
    }
  }
}

///|
test "xgboost regression (reg:squarederror)" {
  check(fixture_regression)
}

///|
test "xgboost binary classification with missing values" {
  check(fixture_binary_logistic)
}

///|
test "xgboost multiclass softprob" {
  check(fixture_multi_softprob)
}

///|
test "xgboost multiclass softmax" {
  check(fixture_multi_softmax)
}

///|
test "xgboost poisson regression" {
  check(fixture_poisson)
}

///|
test "rejects unknown boosters" {
  let json =
    #|{"learner":{"learner_model_param":{"base_score":"[5E-1]","num_class":"0","num_feature":"1","num_target":"1"},
    #|"objective":{"name":"reg:squarederror"},"gradient_booster":{"name":"gbforest","model":{}}}}
  let r = Ok(@moonboost.Model::from_xgboost_json(json)) catch { e => Err(e) }
  inspect(r is Err(Unsupported(_)), content="true")
}

///|
test "xgboost categorical splits" {
  check(fixture_categorical)
}

///|
test "xgboost gamma regression" {
  check(fixture_gamma)
}

///|
test "xgboost tweedie regression" {
  check(fixture_tweedie)
}

///|
test "xgboost binary logitraw" {
  check(fixture_binary_logitraw)
}

///|
test "xgboost multiclass with one base score per class" {
  check(fixture_multi_base_vector)
}

///|
test "xgboost dart binary classification" {
  check(fixture_dart_binary)
}

///|
test "xgboost dart multiclass" {
  check(fixture_dart_multi)
}

///|
test "xgboost dart in the pre-3.0 layout" {
  check(fixture_dart_legacy_layout)
}

///|
test "xgboost gblinear regression" {
  check(fixture_gblinear_regression)
}

///|
test "xgboost gblinear logistic" {
  check(fixture_gblinear_logistic)
}

///|
test "xgboost gblinear multiclass" {
  check(fixture_gblinear_multi)
}
