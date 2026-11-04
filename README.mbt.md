# moonboost

[![CI](https://github.com/Juwan-Hwang/moonboost/actions/workflows/ci.yml/badge.svg)](https://github.com/Juwan-Hwang/moonboost/actions/workflows/ci.yml)

Gradient boosted decision trees (GBDT) in pure MoonBit.

moonboost loads models trained by [XGBoost](https://github.com/dmlc/xgboost)
and reproduces their predictions **bit for bit**, from one codebase that runs
on the `native`, `wasm`, `wasm-gc` and `js` backends. No Python, no C++
runtime, no ONNX conversion step.

## Why

Most GBDT models are trained in Python but need to run somewhere else: a
browser, an edge function, a mobile app, a game server. Today that usually
means shipping a multi-megabyte runtime or re-implementing tree walking by
hand. moonboost is a small, dependency-free predictor whose output you can
trust because it is tested against XGBoost itself.

## Status

| Feature | Status |
|---|---|
| XGBoost JSON models (`gbtree`) | done |
| Objectives: `reg:*`, `binary:logistic`, `binary:logitraw`, `count:poisson`, `reg:gamma`, `reg:tweedie`, `multi:softprob`, `multi:softmax` | done |
| Missing values (NaN, absent features) | done |
| Categorical splits | done |
| Leaf indices (`pred_leaf`) | done |
| Batch prediction from CSV | done |
| CLI | done |
| `dart`, `gblinear` boosters | planned |
| LightGBM text models | planned |
| Verified tree-walking kernel (`moon prove`) | done |
| Blocked batch prediction | done |
| ES module for browsers and Node (`web/`) | done |
| XGBoost UBJSON models | planned |
| Training | planned |

## Correctness

Every test model is trained by XGBoost 3.4.1 (`tools/gen_fixtures.py`), and
the expected outputs are stored as raw float32 bit patterns. `moon test`
checks that:

- raw margins are **identical bit for bit** to XGBoost's
  `predict(output_margin=True)`, on every backend;
- final predictions are within 2 ulp (they pass through `exp`, whose last bit
  may differ between libm implementations).

Covered today: regression, binary classification with 10% missing values,
multiclass `softprob` and `softmax`, Poisson regression, and a model with two
categorical features. CI regenerates the fixtures from the pinned XGBoost
release and fails if anything drifts.

On the three benchmark models below, all 30,000 final predictions match
XGBoost exactly on both the native and the JS build.

The trick is to do what XGBoost does: keep every value in 32-bit floats,
compare `x < threshold` in float32, start from the objective's base margin,
and accumulate trees in order.

## Verified kernel

Tree walking lives in its own package, `kernel/`, written so that
`moon prove` (Why3 + Z3) can check it. CI runs the proofs on every push.
What is proved:

- `Forest::new` returns a forest only if every split node points strictly
  forward to in-range children, every root is in range, and every
  categorical split refers to a valid slice of the category table;
- for such a forest, `Forest::leaf` always terminates, never indexes out of
  bounds (nodes, row, category table) and returns a leaf.

So a malformed or malicious model file can be rejected, but cannot make
prediction loop forever or read out of bounds. To keep floating point out of
the proof, the kernel compares integer keys: `float_key` maps every non-NaN
float32 to an `Int` with the same order (tested over random bit patterns,
zeros, subnormals and infinities). Which way a split goes is not part of the
proof; that is covered by the differential tests against XGBoost.

## Performance

Batch prediction, 10,000 rows of 32 features (5% missing), single thread,
measured with `bench/` on an aarch64 Linux VM. Lower is better.

| Model | moonboost native | moonboost JS (Node 22) | XGBoost 3.4.1 C++ |
|---|---|---|---|
| 100 trees, depth 4 | 2.1 µs/row | 11.5 µs/row | 1.1 µs/row |
| 500 trees, depth 8 | 25.9 µs/row | 81.8 µs/row | 13.4 µs/row |
| 1000 trees, depth 10 | 64.0 µs/row | 185.6 µs/row | 39.0 µs/row |

The native build is within 2x of XGBoost's own C++ predictor; closing that
gap (and the JS one) is the next milestone. The minified JS bundle is
124 KB, 41.6 KB gzipped, with no other runtime needed. Reproduce with
`python3 bench/gen_models.py`, then `moonboost bench`, `node
bench/node_bench.mjs` and `python3 bench/xgb_bench.py`.

## Usage

Add the package:

```
moon add Juwan-Hwang/moonboost
```

Load a model saved with `booster.save_model("model.json")` and predict:

```moonbit nocheck
///|
fn predict_iris(json_text : String) -> Unit raise {
  let model = @moonboost.Model::from_xgboost_json(json_text)
  let row : Array[Float] = [5.1, 3.5, 1.4, 0.2]
  let probs = model.predict(row) // class probabilities
  let margin = model.predict_margin(row) // raw scores
  let leaves = model.predict_leaf(row) // leaf index per tree
  println("\{probs} \{margin} \{leaves}")
}
```

Missing values are `NaN`. Categorical features are passed as their integer
category codes, as XGBoost sees them.

Batch prediction:

```moonbit nocheck
///|
fn predict_csv(
  model : @moonboost.Model,
  csv_text : String,
) -> Array[Float] raise {
  let data = @moonboost.DenseMatrix::from_csv(csv_text, header=true)
  model.predict_batch(data) // row-major, model.output_width() values per row
}
```

## JavaScript

```
moon build --target js --release
```

```js
import { load_model, predict, predict_batch } from "./_build/js/release/build/web/web.js";
const model = load_model(jsonText);
predict(model, [5.1, 3.5, 1.4, 0.2]);
```

See `examples/node/iris.mjs`.

## CLI

```
moon run cmd/main --target js -- info examples/iris/model.json
moon run cmd/main --target js -- predict examples/iris/model.json examples/iris/data.csv --header
```

```
0.99185115,0.005436964,0.0027118726
0.992132,0.0047212564,0.003146739
...
```

`--margin` prints raw scores and `--leaf` prints the leaf reached in each tree.

## Development

```
moon test --target all        # unit + differential tests
python3 tools/gen_fixtures.py && moon fmt   # regenerate fixtures
```

## License and attribution

Apache-2.0. moonboost reads the XGBoost JSON model format and follows
XGBoost's prediction semantics; no XGBoost source code is included. XGBoost is
Apache-2.0, © the XGBoost contributors.
