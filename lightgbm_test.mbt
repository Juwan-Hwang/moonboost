///|
/// A LightGBM model with inputs and LightGBM's own outputs, as float64 bits
/// (see tools/gen_lgb_fixtures.py).
struct LgbFixture {
  name : String
  model : String
  num_feature : Int32
  width : Int32
  num_trees : Int32
  x : Array[UInt64]
  raw : Array[UInt64]
  pred : Array[UInt64]
  leaf : Array[Int32]
}

///|
/// Raw scores, final predictions and leaf indices must all match LightGBM
/// bit for bit (predictions go through glibc's `exp`, which `lgb_exp`
/// reproduces). `predict_batch` must agree with `predict` on the same rows
/// read as floats.
fn check_lgb(f : LgbFixture) -> Unit raise {
  let model = @moonboost.LightGBMModel::from_text(f.model)
  let x = f.x.map(b => b.reinterpret_as_double())
  let rows = x.length() / f.num_feature
  for r in 0..<rows {
    let row = x[r * f.num_feature:(r + 1) * f.num_feature]
    let raw = model.predict_raw(row)
    for k in 0..<f.width {
      let want = f.raw[r * f.width + k]
      if raw[k].reinterpret_as_uint64() != want {
        fail(
          "\{f.name}: row \{r} class \{k} raw \{raw[k]} != \{want.reinterpret_as_double()}",
        )
      }
    }
    let p = model.predict(row)
    for k in 0..<f.width {
      let want = f.pred[r * f.width + k]
      if p[k].reinterpret_as_uint64() != want {
        fail(
          "\{f.name}: row \{r} class \{k} prediction \{p[k]} != \{want.reinterpret_as_double()}",
        )
      }
    }
    let leaf = model.predict_leaf(row)
    for t in 0..<f.num_trees {
      if leaf[t] != f.leaf[r * f.num_trees + t] {
        fail("\{f.name}: row \{r} tree \{t} leaf \{leaf[t]}")
      }
    }
  }
  // the batch path, on the inputs rounded to float
  let xf = x.map(v => Float::from_double(v))
  let batch = model.predict_batch(
    @moonboost.DenseMatrix::new(xf, cols=f.num_feature),
  )
  for r in 0..<rows {
    let row = xf[r * f.num_feature:(r + 1) * f.num_feature].map(v => {
      v.to_double()
    })
    let p = model.predict(row)
    for k in 0..<f.width {
      if batch[r * f.width + k].reinterpret_as_uint64() !=
        p[k].reinterpret_as_uint64() {
        fail(
          "\{f.name}: row \{r} class \{k} batch \{batch[r * f.width + k]} != \{p[k]}",
        )
      }
    }
  }
}

///|
test "lightgbm regression" {
  check_lgb(lgb_fixture_regression)
}

///|
test "lightgbm regression with reg_sqrt" {
  check_lgb(lgb_fixture_regression_sqrt)
}

///|
test "lightgbm binary" {
  check_lgb(lgb_fixture_binary)
}

///|
test "lightgbm binary with a sigmoid parameter" {
  check_lgb(lgb_fixture_binary_sigmoid)
}

///|
test "lightgbm multiclass" {
  check_lgb(lgb_fixture_multiclass)
}

///|
test "lightgbm multiclassova" {
  check_lgb(lgb_fixture_multiclassova)
}

///|
test "lightgbm poisson, gamma and tweedie" {
  check_lgb(lgb_fixture_poisson)
  check_lgb(lgb_fixture_gamma)
  check_lgb(lgb_fixture_tweedie)
}

///|
test "lightgbm cross entropy" {
  check_lgb(lgb_fixture_cross_entropy)
  check_lgb(lgb_fixture_cross_entropy_lambda)
}

///|
test "lightgbm zero as missing and no missing handling" {
  check_lgb(lgb_fixture_zero_as_missing)
  check_lgb(lgb_fixture_no_missing_handling)
}

///|
test "lightgbm categorical splits" {
  check_lgb(lgb_fixture_categorical)
}

///|
test "lightgbm random forest averages its trees" {
  check_lgb(lgb_fixture_random_forest)
}

///|
test "lightgbm rejects text that is not a model" {
  let r = Ok(@moonboost.LightGBMModel::from_text("hello")) catch { e => Err(e) }
  inspect(r is Err(_), content="true")
}
