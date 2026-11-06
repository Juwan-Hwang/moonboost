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

Batch prediction, 10,000 rows of 32 features (5% missing), single thread, on
an aarch64 Linux VM (Neoverse-V2). µs per row, best of several runs, lower is
better. XGBoost is timed through its C API (`XGBoosterPredictFromDense`,
`nthread=1`), so no Python is in its loop.

| Model | moonboost native | moonboost JS (Node 22) | moonboost wasm / wasm-gc (moonrun) | XGBoost 3.4.1 C API |
|---|---|---|---|---|
| 100 trees, depth 4 | 0.46 | 2.5 | 0.9 / 1.1 | 1.00 |
| 500 trees, depth 8 | 3.32 | 11.7 | 6.2 / 8.5 | 12.9 |
| 1000 trees, depth 10 | 8.9 | 28.4 | 15.5 / 21.7 | 35.0 |

One row per call (rows predicted one at a time, µs per row), which is what
an online caller pays. For XGBoost this is one `XGBoosterPredictFromDense`
call per row, so its per-call overhead is included.

| Model | moonboost native | moonboost JS (Node 22) | moonboost wasm / wasm-gc (moonrun) | XGBoost 3.4.1 C API |
|---|---|---|---|---|
| 100 trees, depth 4 | 0.8 | 3.6 | 2.5 / 1.9 | 12.5 |
| 500 trees, depth 8 | 5.4 | 13.3 | 14.6 / 9.8 | 106 |
| 1000 trees, depth 10 | 18.9 | 72 | 52 / 26 | 439 |

Loading the 1000-tree model (a 19 MB JSON file) takes 325 ms native and
360 ms in Node, where the JS build parses with the engine's own
`JSON.parse`; XGBoost's `Booster.load_model` takes about 580 ms on the same
file.

All 30,000 predictions are bit-for-bit identical to XGBoost's. Reproduce with
`python3 bench/gen_models.py`, then `moonboost bench`, `node
bench/node_bench.mjs`, and `bench/xgb_capi_bench.c` (build line at its top).

How the batch path works (`lockstep.mbt`):

- **Ranks instead of floats.** The distinct split thresholds of each feature
  are sorted once. A row value is replaced by its rank among them, found by a
  branch-free search, and a split on the `k`-th threshold (from 0) goes
  right when `rank > k`. Exact, because `x < t_k` holds exactly when at most
  `k` thresholds are `<= x`.
- **Rank tables (JS and wasm).** Each feature's thresholds are bucketed by
  value, `b = floor((x - lo) * scale)`. That map is monotone in `x`, so every
  threshold in an earlier bucket is `<= x` and every one in a later bucket is
  `> x`; a rank is one table lookup plus at most a few compares inside the
  bucket. Features whose thresholds crowd into one bucket keep the search.
  Ranks are computed one feature at a time across a block of rows.
- **Missing values cost nothing.** Each feature has two rank slots, one
  holding 0 and one holding a value above every threshold when the input is
  missing. A split reads whichever slot matches its default direction.
- **Self-looping leaves.** Children sit side by side, so the next node is
  `left + go_right`, and a leaf is its own left child with a threshold no rank
  reaches. Every row takes the same number of steps through a tree, so 64 rows
  walk it in lockstep with no branches.
- **One row, many trees.** A single-row call walks sixteen trees to their
  leaves side by side (four on plain wasm, where more spill), so their
  cache misses overlap, instead of using the block machinery.
- **Pointer-free trees on JS.** The JS build pads every tree to the forest's
  depth and stores it as an implicit heap (children of node `h` at `2h` and
  `2h + 1`; a shallow leaf is extended by nodes that always go right), with
  split words in an `Int32Array` and leaves in a `Float32Array`. A step then
  reads no child index, which V8 rewards; native and wasm keep the compact
  layout, which is faster there. A heap forest over 4 MB is walked 256 rows
  at a time instead of 64, so it streams through the cache fewer times.

Leaves are still added in tree order in float32, which is what keeps the
results identical. Models with categorical splits use the verified kernel.
The JS bundle minified with esbuild is 146 KB, 47.0 KB gzipped, with no other runtime.

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
