# moonboost

[![CI](https://github.com/Juwan-Hwang/moonboost/actions/workflows/ci.yml/badge.svg)](https://github.com/Juwan-Hwang/moonboost/actions/workflows/ci.yml)

Gradient boosted decision trees (GBDT) in pure MoonBit.

moonboost loads models trained by [XGBoost](https://github.com/dmlc/xgboost)
or [LightGBM](https://github.com/microsoft/LightGBM) and reproduces their
predictions **bit for bit**, from one codebase that runs
on the `native`, `wasm`, `wasm-gc` and `js` backends. No Python, no C++
runtime, no ONNX conversion step.

**[Try it in your browser](https://juwan-hwang.github.io/moonboost/)**: a
200-tree XGBoost model scoring California housing blocks in the page, a
bit-for-bit check against XGBoost on 2,000 rows, and a race against
onnxruntime-web on the same model (source in `docs/`).

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
| `dart` (tree weights) and `gblinear` boosters | done |
| Multiclass models with one base score per class (XGBoost 3.x) | done |
| LightGBM text models (numerical and categorical splits, all missing-value modes, every built-in objective) | done |
| Verified tree-walking kernel (`moon prove`) | done |
| Blocked batch prediction | done |
| ES module for browsers and Node (`web/`) | done |
| Browser demo (`docs/`, GitHub Pages) | done |
| XGBoost UBJSON models | planned |
| Training | planned |

## Correctness

Every XGBoost test model is trained by XGBoost 3.4.1
(`tools/gen_fixtures.py`), and the expected outputs are stored as raw float32
bit patterns. `moon test`
checks that:

- raw margins are **identical bit for bit** to XGBoost's
  `predict(output_margin=True)`, on every backend;
- final predictions are within 2 ulp (they pass through `exp`, whose last bit
  may differ between libm implementations).

Covered today: regression, binary classification with 10% missing values,
multiclass `softprob` and `softmax` (including unbalanced classes, where
XGBoost 3.x keeps one base score per class), Poisson, Gamma and Tweedie
regression, `binary:logitraw`, a model with two categorical features, DART
models (in the 3.x layout and the older `dart` booster layout), and
`gblinear` regression, logistic and multiclass models. CI regenerates the fixtures from the pinned XGBoost
release and fails if anything drifts.

On the three benchmark models below, all 30,000 final predictions match
XGBoost exactly on both the native and the JS build.

The trick is to do what XGBoost does: keep every value in 32-bit floats,
compare `x < threshold` in float32, start from the objective's base margin,
and accumulate trees in order.

DART and `gblinear` add a product (`leaf * weight`, `x * w`) to a running
sum. XGBoost's x86-64 build rounds the product first, and so does
moonboost on every backend; the reference fixtures come from x86-64. Builds
of XGBoost for aarch64 fuse that multiply-add into one instruction, so for
these two boosters an aarch64 XGBoost can differ from x86-64 XGBoost (and
from moonboost) in the last bit.

LightGBM models are checked the same way against LightGBM 4.7.0
(`tools/gen_lgb_fixtures.py`, float64 bit patterns): raw scores and leaf
indices bit for bit, final predictions within 2 ulp. The cases cover
regression (also with `reg_sqrt`), binary (also with a `sigmoid` parameter),
`multiclass`, `multiclassova`, Poisson, Gamma, Tweedie, both cross-entropy
objectives, zeros as missing values, `use_missing=false`, categorical
features with missing and negative codes, and random forest mode. LightGBM
compares doubles (`x <= threshold`), turns NaN into 0 unless the split
handles NaN, reads values within 1e-35 of zero as zero, and adds trees in
double; moonboost does the same.

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
`nthread=1`), so no Python is in its loop. All numbers in a table come from
one session on the same VM; between sessions they move by up to about 25%.

| Model | moonboost native | moonboost JS (Node 22) | moonboost wasm / wasm-gc (moonrun) | XGBoost 3.4.1 C API |
|---|---|---|---|---|
| 100 trees, depth 4 | 0.38 | 0.91 | 0.70 / 0.77 | 1.00 |
| 500 trees, depth 8 | 3.3 | 5.8 | 4.5 / 5.6 | 12.4 |
| 1000 trees, depth 10 | 8.5 | 15.7 | 11.0 / 13.6 | 34.2 |

One row per call (rows predicted one at a time, µs per row), which is what
an online caller pays. For XGBoost this is one `XGBoosterPredictFromDense`
call per row, so its per-call overhead is included.

| Model | moonboost native | moonboost JS (Node 22) | moonboost wasm / wasm-gc (moonrun) | XGBoost 3.4.1 C API |
|---|---|---|---|---|
| 100 trees, depth 4 | 0.7 | 3.1 | 2.2 / 1.7 | 12.2 |
| 500 trees, depth 8 | 5.2 | 12.4 | 12.5 / 9.1 | 91 |
| 1000 trees, depth 10 | 17.5 | 69 | 45 / 27 | 316 |

Loading the 1000-tree model (a 19 MB JSON file) takes 325 ms native and
360 ms in Node, where the JS build parses with the engine's own
`JSON.parse`; XGBoost's `Booster.load_model` takes about 580 ms on the same
file.

All 30,000 predictions are bit-for-bit identical to XGBoost's. Reproduce with
`python3 bench/gen_models.py`, then `moonboost bench`, `node
bench/node_bench.mjs`, and `bench/xgb_capi_bench.c` (build line at its top).

### Against onnxruntime-web

The usual way to run an XGBoost model in a browser is to convert it to ONNX
(`onnxmltools`) and load it with onnxruntime-web. `bench/web_bench.mjs`
compares that with moonboost's JS build in Node 22, same machine, same
10,000 rows, one thread each (`ort.env.wasm.numThreads = 1`, WebAssembly
backend), each engine in a fresh process. Means of two runs:

| Model | | moonboost (JS) | onnxruntime-web 1.31.1 (wasm) |
|---|---|---|---|
| 100 trees, depth 4 | cold start | 52 ms | 312 ms |
| | batch, µs/row | 0.92 | 3.4 |
| | one row per call, µs | 3.6 | 39 |
| | probabilities equal to XGBoost's | 10,000 / 10,000 | 5,776 / 10,000 |
| 500 trees, depth 8 | cold start | 272 ms | 573 ms |
| | batch, µs/row | 5.8 | 28.6 |
| | one row per call, µs | 12.6 | 117 |
| | probabilities equal to XGBoost's | 10,000 / 10,000 | 5,037 / 10,000 |
| 1000 trees, depth 10 | cold start | 431 ms | 1,191 ms |
| | batch, µs/row | 15.3 | 65.2 |
| | one row per call, µs | 62 | 276 |
| | probabilities equal to XGBoost's | 10,000 / 10,000 | 5,053 / 10,000 |

Cold start is import, model load and the first prediction. Where
onnxruntime-web differs from XGBoost, it is by at most 1.2e-7, which is
enough to reorder near-ties in a ranking. One-row-per-call times move by up
to about 40% between runs on this VM; the other columns by a few percent. What ships to the
browser: moonboost is one 181 KB module (56 KB gzipped, LightGBM support
included); onnxruntime-web's
WebAssembly build is `ort.wasm.min.mjs` (54 KB) plus
`ort-wasm-simd-threaded.mjs` (25 KB) plus a 14.8 MB `.wasm` file (3.8 MB
gzipped). Model files are smaller in ONNX (97 KB against 223 KB of JSON for
the small model).

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
  walk it in lockstep with no branches. On JS and wasm, four rows walk two
  trees to their leaves at once in locals: eight independent chains of loads,
  and no node index written back between levels.
- **One row, many trees.** A single-row call walks sixteen trees to their
  leaves side by side (four on plain wasm, where more spill), so their
  cache misses overlap, instead of using the block machinery.
- **Pointer-free trees on JS.** The JS build pads every tree to the forest's
  depth and stores it as an implicit heap (children of node `h` at `2h` and
  `2h + 1`; a shallow leaf is extended by nodes that always go left), with
  split words in an `Int32Array` and leaves in a `Float32Array`. A step then
  reads no child index, which V8 rewards; native and wasm keep the compact
  layout, which is faster there.
- **Tree chunks, margins in registers (JS).** For single-output models the
  heap forest is cut into chunks of about 32 KB. Four rows walk four trees of
  a chunk at once and carry on through the whole chunk with their running
  sums in locals, so the margin never goes back to memory and the chunk
  stays in L1 while the 256 rows of a block pass through it.
- **Values instead of ranks for short forests.** When the forest takes
  few steps per feature (at most 32 tree levels per feature, like the
  100-tree model), ranking every feature costs more than the walk. Such
  forests compare the float32 row value with the split value directly. A
  missing value is stored as -inf in the slot read by default-left splits and
  +inf in the default-right slot, and padded heap nodes split on NaN, so they
  always go left. The comparison becomes a 0/1 value with `+(a >= b)`,
  which V8 compiles without a branch. Native and wasm build the same padded
  heap with each split packed into one 64-bit word, so a step is a single
  load: native packs the split value's bits above the slot and compares
  floats; wasm packs the value's `float_key` and turns the integer
  difference's sign into the step (`((k - t) >> 63) + 1`), since a float
  compare there costs a branch.

Leaves are still added in tree order in float32, which is what keeps the
results identical. Models with categorical splits use the verified kernel.
The JS bundle minified with esbuild is 181 KB, 56 KB gzipped, with no other runtime.

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

LightGBM text models (`booster.save_model("model.txt")`) load into a
`LightGBMModel`. LightGBM works in doubles throughout, so rows are
`Double`s and so are the results:

```moonbit nocheck
///|
fn predict_lightgbm(text : String) -> Unit raise {
  let model = @moonboost.LightGBMModel::from_text(text)
  let row : Array[Double] = [5.1, 3.5, 1.4, 0.2]
  let probs = model.predict(row) // what LightGBM's predict() returns
  let raw = model.predict_raw(row) // predict(raw_score=True)
  let leaves = model.predict_leaf(row) // predict(pred_leaf=True)
  println("\{probs} \{raw} \{leaves}")
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
// rows may be a Float32Array (used as is) or any array of numbers
predict_batch(model, new Float32Array(rowMajorValues), 4); // 4 columns
```

LightGBM models use `load_lightgbm(text)`, `predict_lightgbm(model, row)`
and `predict_lightgbm_raw(model, row)`.

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
LightGBM text models work the same way; the CLI tells the formats apart by
content.

## Development

```
moon test --target all        # unit + differential tests
python3 tools/gen_fixtures.py && moon fmt       # regenerate XGBoost fixtures
python3 tools/gen_lgb_fixtures.py && moon fmt   # regenerate LightGBM fixtures
python3 tools/gen_demo.py && sh tools/build_demo.sh   # rebuild the browser demo
```

## License and attribution

Apache-2.0. moonboost reads the XGBoost JSON and LightGBM text model formats
and follows their prediction semantics; no XGBoost or LightGBM source code is
included. XGBoost is Apache-2.0, © the XGBoost contributors. LightGBM is MIT,
© Microsoft Corporation and the LightGBM developers.
