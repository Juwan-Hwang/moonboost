// Times ml-xgboost (npm, XGBoost compiled to WebAssembly with Emscripten)
// on the same data as bench/train_wasm_bench.mjs:
//   npm i ml-xgboost@1.1.2
//   node bench/ml_xgboost_bench.cjs X.csv y.csv rounds rows
// Its loader copies the data onto a 5 MB stack, so about 150k rows of 8
// features is past its limit.
const fs = require("fs");
const [xf, yf, rounds = "30", nrows = "100000"] = process.argv.slice(2);
const lines = fs.readFileSync(xf, "utf8").trim().split("\n").slice(0, +nrows);
const cols = lines[0].split(",").length;
const X = new Float32Array(lines.length * cols);
lines.forEach((l, r) => l.split(",").forEach((v, c) => { X[r * cols + c] = v === "" ? NaN : +v; }));
const y = Float32Array.from(fs.readFileSync(yf, "utf8").trim().split("\n").slice(0, +nrows), Number);
require("ml-xgboost/dist/wasm/xgboost").isReady.then((xg) => {
  const create = xg.cwrap("create_model", "number", ["array", "array", "number", "number"]);
  const setp = xg.cwrap("set_param", null, ["number", "string", "string"]);
  const train = xg.cwrap("train_full_model", null, ["number", "number"]);
  const t0 = performance.now();
  const m = create(new Uint8Array(X.buffer), new Uint8Array(y.buffer), lines.length, cols);
  const p = { booster: "gbtree", objective: "binary:logistic", max_depth: "10", eta: "0.1",
    subsample: "1", silent: "1", tree_method: "hist", max_bin: "256", nthread: "1" };
  for (const k in p) setp(m, k, p[k]);
  train(m, +rounds);
  console.log(`ml-xgboost ${lines.length} rows ${rounds} rounds: ${Math.round(performance.now() - t0)} ms`);
});
