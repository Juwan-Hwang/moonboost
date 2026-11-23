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

///|
test "XGBoost vector-leaf trees load as one tree per output" {
  // trained by XGBoost 3.4.1: multi_strategy multi_output_tree, three
  // targets, depth 2, two rounds; its margins for the rows below
  let text =
    #|{"learner":{"attributes":{},"feature_names":[],"feature_types":[],
    #|"gradient_booster":{"model":{"cats":{"enc":[],"feature_segments":[],"sorted_idx":[]},
    #|"gbtree_model_param":{"num_parallel_tree":"1","num_trees":"2"},"iteration_indptr":[0,1,2],
    #|"tree_info":[0,0],"trees":[{"base_weights":[-1.2995198e-09,-1.073651e-09,-6.5594813e-10,
    #|-0.02269313,-1.0385689,-0.016445188,0.04018141,1.8389338,0.02911854,-0.0029408846,
    #|-0.77636445,0.05404205,-0.00828909,-0.12541793,-0.028233733,-0.020704411,0.32017106,
    #|0.03278589,0.059104864,0.8738227,-0.026197916],"categories":[],"categories_nodes":[],
    #|"categories_segments":[],"categories_sizes":[],"default_left":[1,0,0,0,0,0,0],"id":0,
    #|"leaf_weights":[-0.0029408846,-0.77636445,0.05404205,-0.00828909,-0.12541793,-0.028233733,
    #|-0.020704411,0.32017106,0.03278589,0.059104864,0.8738227,-0.026197916],"left_children":[1,
    #|3,5,-1,-1,-1,-1],"loss_changes":[577.1975,187.43683,90.122154,0.0,0.0,0.0,0.0],
    #|"parents":[-1,0,0,1,1,2,2],"right_children":[2,4,6,0,1,2,3],
    #|"split_conditions":[0.21465912,-0.75220436,0.94708097,1e-45,1e-45,1e-45,1e-45],
    #|"split_indices":[1,1,1,0,0,0,0],"split_type":[0,0,0,0,0,0,0],"sum_hessian":[900.0,576.0,
    #|324.0,162.0,414.0,192.0,132.0],"tree_param":{"num_deleted":"0","num_feature":"3",
    #|"num_nodes":"7","size_leaf_vector":"3"}},{"base_weights":[9.026867e-05,0.0009707898,
    #|0.00010762551,-0.0901893,-1.2865697,0.003567023,0.05798646,0.8266708,-0.0021114862,
    #|-0.03754045,-0.9228248,0.07344497,-0.025044253,-0.29271126,-0.011002862,-0.14479265,
    #|0.26775587,0.0063332436,0.3237619,0.20677772,-0.013794946],"categories":[],
    #|"categories_nodes":[],"categories_segments":[],"categories_sizes":[],"default_left":[0,0,
    #|1,0,0,0,0],"id":1,"leaf_weights":[-0.03754045,-0.9228248,0.07344497,-0.025044253,
    #|-0.29271126,-0.011002862,-0.14479265,0.26775587,0.0063332436,0.3237619,0.20677772,
    #|-0.013794946],"left_children":[1,3,5,-1,-1,-1,-1],"loss_changes":[322.6442,63.813625,
    #|103.34554,0.0,0.0,0.0,0.0],"parents":[-1,0,0,1,1,2,2],"right_children":[2,4,6,0,1,2,3],
    #|"split_conditions":[-0.15219297,-1.3648796,0.28245184,1e-45,1e-45,1e-45,1e-45],
    #|"split_indices":[1,1,0,0,0,0,0],"split_type":[0,0,0,0,0,0,0],"sum_hessian":[900.0,351.0,
    #|549.0,48.0,303.0,360.0,189.0],"tree_param":{"num_deleted":"0","num_feature":"3",
    #|"num_nodes":"7","size_leaf_vector":"3"}}]},"name":"gbtree"},
    #|"learner_model_param":{"base_score":"[-1.1061978E-1,3.2315243E-3,-3.044113E-2]",
    #|"boost_from_average":"1","num_class":"0","num_feature":"3","num_target":"3"},
    #|"objective":{"name":"reg:squarederror","reg_loss_param":{"scale_pos_weight":"1"}}},
    #|"version":[3,4,1]}

  let m = @moonboost.Model::from_xgboost_json(text)
  let zero : Float = 0.0
  let nan = zero / zero
  let rows : Array[Array[Float]] = [
    [0.5, -1.0, 2.0],
    [nan, 0.3, -0.2],
    [-2.0, nan, 1.0],
    [0.1, 0.1, nan],
  ]
  let want : Array[UInt32] = [
    0xBE0DEE73U, 0xBF886D95U, 0x3C4E6815U, 0xBE8D5F30U, 0x3F175629U, 0x3C0E2E2EU,
    0xBE8703E3U, 0x3E151029U, 0xBD56642BU, 0xBE8703E3U, 0x3E151029U, 0xBD56642BU,
  ]
  for r, row in rows {
    let p = m.predict(row[:])
    assert_eq(p.length(), 3)
    for k in 0..<3 {
      assert_eq(
        p[k].reinterpret_as_uint(),
        want[r * 3 + k],
        msg="row \{r} output \{k}",
      )
    }
  }
}
