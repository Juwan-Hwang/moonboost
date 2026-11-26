///|
/// Training for browsers as a WebAssembly GC module, which runs about
/// twice as fast as the JS build. Only numbers cross the boundary: the
/// page fills the rows and labels value by value, trains, and reads the
/// model text back one UTF-16 code unit at a time (`web/train_wasm.mjs`
/// wraps this).
let data : Ref[FixedArray[Float]] = Ref(FixedArray::make(0, 0.0))

///|
let labels : Ref[FixedArray[Float]] = Ref(FixedArray::make(0, 0.0))

///|
let columns : Ref[Int32] = Ref(0)

///|
let text : Ref[String] = Ref("")

///|
let error : Ref[String] = Ref("")

///|
/// Makes room for `rows` rows of `cols` values.
#export_name("train_alloc")
pub fn train_alloc(rows : Int32, cols : Int32) -> Unit {
  data.val = FixedArray::make(rows * cols, 0.0)
  labels.val = FixedArray::make(rows, 0.0)
  columns.val = cols
}

///|
/// Sets value `i` of the row-major data (`NaN` for missing).
#export_name("train_set")
pub fn train_set(i : Int32, v : Double) -> Unit {
  data.val[i] = Float::from_double(v)
}

///|
/// Sets the label of row `i`.
#export_name("train_label")
pub fn train_label(i : Int32, v : Double) -> Unit {
  labels.val[i] = Float::from_double(v)
}

///|
/// Trains on the rows set so far. `objective` is 0 for
/// `reg:squarederror`, 1 `binary:logistic`, 2 `multi:softprob` and
/// 3 `multi:softmax`; a negative setting keeps XGBoost's default. Returns
/// the length of the model's XGBoost JSON text, or -1 with the message's
/// length available through `train_error_char(-1)`.
#export_name("train_run")
pub fn train_run(
  objective : Int32,
  num_class : Int32,
  num_round : Int32,
  max_depth : Int32,
  eta : Double,
) -> Int32 {
  let d = @moonboost.TrainParams::default()
  let params : @moonboost.TrainParams = {
    ..d,
    objective: match objective {
      0 => "reg:squarederror"
      1 => "binary:logistic"
      2 => "multi:softprob"
      3 => "multi:softmax"
      _ => d.objective
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
  run(params)
}

///|
/// Code unit `i` of the trained model's text.
#export_name("train_char")
pub fn train_char(i : Int32) -> Int32 {
  text.val.code_unit_at(i).to_int()
}

///|
/// Code unit `i` of the last error message, or its length for `i = -1`.
#export_name("train_error_char")
pub fn train_error_char(i : Int32) -> Int32 {
  if i < 0 {
    error.val.length()
  } else {
    error.val.code_unit_at(i).to_int()
  }
}

///|
let param_text : Ref[FixedArray[Int32]] = Ref(FixedArray::make(0, 0))

///|
/// Makes room for training parameters given as `n` UTF-16 code units of
/// JSON (XGBoost's names, as for `train_with` in the ES module).
#export_name("train_params_alloc")
pub fn train_params_alloc(n : Int32) -> Unit {
  param_text.val = FixedArray::make(n, 0)
}

///|
/// Sets code unit `i` of the parameters' JSON text.
#export_name("train_params_char")
pub fn train_params_char(i : Int32, c : Int32) -> Unit {
  param_text.val[i] = c
}

///|
/// Trains on the rows set so far with the parameters' JSON text; returns
/// as `train_run` does.
#export_name("train_run_params")
pub fn train_run_params() -> Int32 {
  let units = param_text.val
  let buf = StringBuilder(capacity=units.length())
  for c in units {
    buf.write_char(Int32::unsafe_to_char(c))
  }
  let p = @moonboost.TrainParams::from_json(buf.to_string()) catch {
    @moonboost.InvalidModel(msg) => {
      error.val = "invalid input: " + msg
      return -1
    }
    @moonboost.Unsupported(msg) => {
      error.val = "unsupported: " + msg
      return -1
    }
  }
  run(p)
}

///|
fn run(params : @moonboost.TrainParams) -> Int32 {
  text.val = ""
  try {
    let m = @moonboost.DenseMatrix::from_fixed(data.val, cols=columns.val)
    let y = labels.val
    text.val = @moonboost.train_xgboost_json(
      m,
      Array::makei(y.length(), i => y[i]),
      params~,
    )
    error.val = ""
    text.val.length()
  } catch {
    @moonboost.InvalidModel(msg) => {
      error.val = "invalid input: " + msg
      -1
    }
    @moonboost.Unsupported(msg) => {
      error.val = "unsupported: " + msg
      -1
    }
  }
}
