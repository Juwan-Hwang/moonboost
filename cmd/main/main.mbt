///|
const USAGE : String =
  #|moonboost - gradient boosted trees in pure MoonBit
  #|
  #|USAGE:
  #|  moonboost info    <model.json>
  #|  moonboost predict <model.json> <data.csv> [--header] [--margin] [--leaf]
  #|  moonboost bench   <model.json> <data.csv> [--header] [--repeat N]
  #|
  #|predict prints one CSV line per input row:
  #|  default   final predictions (probabilities, values or class index)
  #|  --margin  raw scores before the link function
  #|  --leaf    leaf index reached in every tree
  #|
  #|bench times batch prediction and prints rows per second and the
  #|average latency per row, then the latency of one row at a time.
  #|
  #|Models are XGBoost JSON files (Booster.save_model("model.json")) or
  #|LightGBM text files (Booster.save_model("model.txt")).
  #|Empty cells, NA and NaN in the CSV are treated as missing.

///|
let commands : Array[String] = [
  "info", "predict", "bench", "help", "--help", "-h",
]

///|
suberror CliError {
  CliError(String)
}

///|
fn load_model(path : String) -> @moonboost.Model raise {
  @moonboost.Model::from_xgboost_json(@fs.read_file_to_string(path))
}

///|
/// A LightGBM text model starts with the line `tree`.
fn is_lightgbm(text : String) -> Bool {
  text.trim_start(chars=" \t\r\n").has_prefix("tree")
}

///|
fn lgb_info(m : @moonboost.LightGBMModel) -> Unit {
  println("format:      LightGBM")
  println("objective:   \{m.objective}")
  println("trees:       \{m.num_trees()}")
  println("outputs:     \{m.num_tree_per_iteration}")
  println("features:    \{m.num_feature}")
  if m.feature_names.length() > 0 {
    println("names:       \{m.feature_names.join(",")}")
  }
}

///|
fn info(path : String) -> Unit raise {
  let text = @fs.read_file_to_string(path)
  if is_lightgbm(text) {
    lgb_info(@moonboost.LightGBMModel::from_text(text))
    return
  }
  let m = @moonboost.Model::from_xgboost_json(text)
  println("objective:   \{m.objective.name()}")
  if m.linear.length() > 0 {
    println("booster:     gblinear")
  } else {
    println("trees:       \{m.num_trees()}")
  }
  println("outputs:     \{m.num_group}")
  println("features:    \{m.num_feature}")
  let nodes = m.trees.fold(init=0, (acc, t) => acc + t.num_nodes())
  println("total nodes: \{nodes}")
  if m.feature_names.length() > 0 {
    println("names:       \{m.feature_names.join(",")}")
  }
}

///|
fn join_floats(xs : ArrayView[Float]) -> String {
  xs.map(x => x.to_string()).join(",")
}

///|
fn predict(args : ArrayView[String]) -> Unit raise {
  let files = args.filter(a => !a.has_prefix("--"))
  if files.length() != 2 {
    raise CliError("predict needs <model.json> <data.csv>")
  }
  let header = args.contains("--header")
  let text = @fs.read_file_to_string(files[0])
  let data = @moonboost.DenseMatrix::from_csv(
    @fs.read_file_to_string(files[1]),
    header~,
  )
  if is_lightgbm(text) {
    lgb_predict(@moonboost.LightGBMModel::from_text(text), data, args)
    return
  }
  let model = @moonboost.Model::from_xgboost_json(text)
  if data.cols != model.num_feature {
    raise CliError(
      "data has \{data.cols} columns but model expects \{model.num_feature}",
    )
  }
  for i in 0..<data.rows {
    let row = data.row(i)
    let line = if args.contains("--leaf") {
      model.predict_leaf(row).map(x => x.to_string()).join(",")
    } else if args.contains("--margin") {
      join_floats(model.predict_margin(row))
    } else {
      join_floats(model.predict(row))
    }
    println(line)
  }
}

///|
/// LightGBM reads inputs as doubles. CSV values are parsed into float32
/// cells, so a value with more digits than float32 holds is rounded first.
fn lgb_predict(
  model : @moonboost.LightGBMModel,
  data : @moonboost.DenseMatrix,
  args : ArrayView[String],
) -> Unit raise {
  if data.cols != model.num_feature {
    raise CliError(
      "data has \{data.cols} columns but model expects \{model.num_feature}",
    )
  }
  for i in 0..<data.rows {
    let row = data.row(i).map(x => x.to_double())
    let line = if args.contains("--leaf") {
      model.predict_leaf(row).map(x => x.to_string()).join(",")
    } else if args.contains("--margin") {
      model.predict_raw(row).map(x => x.to_string()).join(",")
    } else {
      model.predict(row).map(x => x.to_string()).join(",")
    }
    println(line)
  }
}

///|
fn bench(args : ArrayView[String]) -> Unit raise {
  let mut repeat = 10
  let files = []
  let mut i = 0
  while i < args.length() {
    let a = args[i]
    if a == "--repeat" && i + 1 < args.length() {
      repeat = @string.parse_int(args[i + 1])
      i += 2
      continue
    }
    if !a.has_prefix("--") {
      files.push(a)
    }
    i += 1
  }
  if files.length() != 2 {
    raise CliError("bench needs <model.json> <data.csv>")
  }
  let tl = @env.now()
  let text0 = @fs.read_file_to_string(files[0])
  if is_lightgbm(text0) {
    let m = @moonboost.LightGBMModel::from_text(text0)
    println("load:       \{(@env.now() - tl).to_double()} ms")
    let data = @moonboost.DenseMatrix::from_csv(
      @fs.read_file_to_string(files[1]),
      header=args.contains("--header"),
    )
    let rowsd = Array::makei(data.rows, r => data.row(r).map(x => x.to_double()))
    let mut check = m.predict_batch(data)[0]
    let t2 = @env.now()
    for _ in 0..<repeat {
      check = check + m.predict_batch(data)[0]
    }
    let t3 = @env.now()
    for r in rowsd {
      check = check + m.predict(r)[0]
    }
    let t4 = @env.now()
    let n = data.rows.to_double() * repeat.to_double()
    println("latency:    \{(t3 - t2).to_double() * 1000.0 / n} us/row")
    println(
      "single:     \{(t4 - t3).to_double() * 1000.0 / data.rows.to_double()} us/row",
    )
    println("checksum:   \{check}")
    return
  }
  let t0 = @env.now()
  let model = load_model(files[0])
  let t1 = @env.now()
  let data = @moonboost.DenseMatrix::from_csv(
    @fs.read_file_to_string(files[1]),
    header=args.contains("--header"),
  )
  // warm up, and keep the result alive so the work is not optimised away
  let mut check = model.predict_batch(data)[0]
  let t2 = @env.now()
  for _ in 0..<repeat {
    check = check + model.predict_batch(data)[0]
  }
  let t3 = @env.now()
  let ms = (t3 - t2).to_double()
  let rows = data.rows.to_double() * repeat.to_double()
  println("load:       \{(t1 - t0).to_double()} ms")
  println("rows:       \{data.rows} x \{repeat}")
  println("throughput: \{(rows / ms * 1000.0).to_int64()} rows/s")
  println("latency:    \{ms * 1000.0 / rows} us/row")
  // one row at a time, as an online caller would
  let t4 = @env.now()
  for r in 0..<data.rows {
    check = check + model.predict(data.row(r))[0]
  }
  let t5 = @env.now()
  println(
    "single:     \{(t5 - t4).to_double() * 1000.0 / data.rows.to_double()} us/row",
  )
  println("checksum:   \{check}")
}

///|
fn main {
  let args = @env.args()
  // The runtime may prepend its own entries (e.g. `node main.js`), so start
  // at the first recognised command.
  let start = args.search_by(a => commands.contains(a)).unwrap_or(args.length())
  let rest = args[start:]
  if rest.length() == 0 {
    println(USAGE)
    return
  }
  let result = try {
    match rest[0] {
      "info" if rest.length() == 2 => info(rest[1])
      "predict" => predict(rest[1:])
      "bench" => bench(rest[1:])
      "help" | "--help" | "-h" => println(USAGE)
      _ => raise CliError("unknown command; run `moonboost help`")
    }
    Ok(())
  } catch {
    e => Err(e)
  }
  if result is Err(e) {
    let msg = match e {
      CliError(m) => m
      @moonboost.InvalidModel(m) => "invalid model or data: \{m}"
      @moonboost.Unsupported(m) => "unsupported: \{m}"
      _ => e.to_string()
    }
    println("error: \{msg}")
    @env.exit(1)
  }
}
