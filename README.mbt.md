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
| Objectives: `reg:*` (including `reg:absoluteerror` and `reg:quantileerror`), `binary:logistic`, `binary:logitraw`, `binary:hinge`, `count:poisson`, `reg:gamma`, `reg:tweedie`, `survival:cox`, `survival:aft`, `multi:softprob`, `multi:softmax`, `rank:ndcg`, `rank:pairwise`, `rank:map` | done |
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
| XGBoost UBJSON models (`.ubj`) | done |
| Training: histogram trees for `reg:squarederror`, `reg:pseudohubererror`, `reg:squaredlogerror`, `binary:logistic`, `reg:logistic`, `binary:logitraw`, `binary:hinge`, `count:poisson`, `reg:gamma`, `reg:tweedie`, `survival:cox`, `survival:aft`, `multi:softprob`, `multi:softmax`, written as XGBoost JSON | done |
| Training: row and feature sampling (`subsample` with `uniform` or `gradient_based` rows, `colsample_bytree`, `colsample_bynode`), `min_data_in_leaf` | done |
| Training: categorical features (one category against the rest, or a set of categories) | done |
| Training: loss-guided growth (`grow_policy: "lossguide"`, `max_leaves`) | done |
| Training in browsers: JS, and a WebAssembly GC module (`webgc/`) | done |

Later: multithreaded training and batch prediction, once MoonBit ships
official support for parallelism. Everything here runs on one thread today.

## Training

```moonbit nocheck
///|
fn train_example(x : DenseMatrix, y : Array[Float]) -> Unit raise {
  let params = {
    ..TrainParams::default(),
    objective: "binary:logistic",
    num_round: 100,
  }
  let json = train_xgboost_json(x, y, params~) // save it; XGBoost loads it too
  let model = Model::from_xgboost_json(json)
  ignore(model.predict_batch(x))
}
```

Features are cut into at most `max_bin` bins at observed values, trees grow
depth-wise on gradient histograms (the smaller child is built, the larger is
the parent minus it), and each split learns where missing values go.
Gradient pairs are quantised per tree to integers packed into one 64-bit
word, so a histogram slot is one integer add and the subtraction is exact
(leaf values still use the exact gradient sums). Below depth 2 each subtree
copies its rows' bins (one byte each when they fit) and gradients into a
contiguous buffer, so deep, scattered nodes stay in cache. The
result is an ordinary XGBoost JSON model: XGBoost loads it and predicts the
same float32 bits as moonboost. Training is single-threaded.

Against XGBoost 3.4.1 (`tree_method="hist"`, one thread, same depth, eta and
bins), 100 rounds on 15,000 synthetic rows with 8 features and 5% missing
values (`bench/train_data.py`, `bench/train_compare.py`; ARM sandbox):

| Task | moonboost native | XGBoost | Test metric, moonboost / XGBoost | XGBoost on the moonboost model |
|---|---|---|---|---|
| Regression | 251 ms | 315 ms | RMSE 0.4327 / 0.4304 | 5,000 / 5,000 bit-identical |
| Binary | 229 ms | 267 ms | log loss 0.1628 / 0.1632 | 5,000 / 5,000 |
| 3 classes | 705 ms | 817 ms | log loss 0.2740 / 0.2747 | 15,000 / 15,000 |

On the [GBM-perf](https://github.com/szilard/GBM-perf) airline data (1M
training rows, 100k test rows, 100 rounds, depth 10, eta 0.1, one thread
pinned to one core of an otherwise idle 4-vCPU x86 AMD EPYC VM; the 8
categorical columns are label-encoded, which differs from GBM-perf's own
setup), two runs each (JS once):

| Library | Training time | Test AUC |
|---|---|---|
| moonboost native | 3.7 s, 3.8 s | 0.7537 |
| moonboost wasm-gc (Node 26) | 5.6 s, 5.8 s | 0.7537 |
| moonboost JS (Node 26) | 12.0 s | 0.7537 |
| XGBoost 3.4.1 `hist` | 7.9 s, 7.8 s | 0.7533 |
| LightGBM 4.7.0 (`num_leaves=1024`, `max_depth=10`) | 6.2 s, 6.2 s | 0.7545 |

moonboost native takes under half the time of XGBoost and about 40% less
than LightGBM here (`bench/gbmperf_compare.py`; all in the same session).
On native, wasm and wasm-gc the logistic and softmax gradients run two rows
at a time in 128-bit SIMD lanes (`moonbitlang/core/v128`), with the same
operations in the same order as the scalar code, so every build still makes
the same model bit for bit. moonboost's times leave out reading the CSV.

### Categorical features and loss-guided growth

Columns listed in `categorical` hold category codes (non-negative
integers). A split on one sends one category against the rest when the
node holds at most `max_cat_to_onehot` categories, and otherwise a set of
categories taken from either end of their order by mean gradient
(`cat_smooth`, `cat_l2`, `min_data_per_group` and `max_cat_threshold` as in
LightGBM). Each categorical column keeps its `max_bin - 1` most frequent
categories, or, when those cover under 99% of the rows, as many as the
histogram has room for; the rest share a bin with missing values, and both
go left, as unlisted categories do in the model. `grow_policy: "lossguide"` splits the
leaf with the largest gain first, up to `max_leaves` leaves (`max_depth: 0`
for no depth limit). The models are XGBoost JSON with categorical splits:
XGBoost loads them and predicts the same bits.

```
moonboost train rows.csv labels.csv -o model.json --objective binary:logistic \
  --rounds 100 --eta 0.1 --categorical 0,1,2,5,6,7 --max-cat-to-onehot 32 \
  --grow-policy lossguide --depth 0 --max-leaves 1023
```

On the airline data with its six categorical columns as categories (and
the departure hour and minute as two more columns), 100 rounds, eta 0.1,
`max_cat_to_onehot` 32, the same number of leaves for every library, one
thread on one core of the VM above (`bench/airline_categorical.py`):

| Leaves | moonboost | LightGBM 4.7.0 | XGBoost 3.4.1 `hist` |
|---|---|---|---|
| 255 | 6.9 s, AUC 0.7629 | 6.5 s, AUC 0.7612 | 20.5 s, AUC 0.7547 |
| 1023 | 8.8 s, AUC 0.7779 | 9.2 s, AUC 0.7761 | 32.8 s, AUC 0.7638 |
| 4095 | 16.1 s, AUC 0.7868 | 19.8 s, AUC 0.7829 | 53.9 s, AUC 0.7655 |

At every size moonboost's model is the most accurate. At 4095 leaves it
trains 19% faster than LightGBM and at 1023 leaves 4% faster; at 255
leaves LightGBM is about as fast (moonboost took 6.4 s in another run). With the categories label-encoded
instead, moonboost reaches 0.7520 at depth 10.

### Sampling and other objectives

`subsample` trains each tree on a share of the rows, drawn anew per tree
with `seed`. With `sampling_method: "gradient_based"` (minimal variance
sampling; XGBoost offers it only on GPUs) a row is kept with a chance
proportional to `sqrt(g^2 + lambda h^2)`, capped at 1, and a kept row's
gradient pair is divided by that chance. On ten OpenML data sets, three
random splits each, at `subsample: 0.5` it beat uniform sampling on adult
(+0.0030 AUC), click (+0.0030), electricity (+0.0053) and kdd_upselling
(+0.0075), by less on four more, and lost on credit (-0.0030); its AUC
there matched training on all rows.

A sampled tree is grown on the kept rows only; the others are passed down
the finished tree afterwards. On airline (1M rows, categories native, 100
rounds, eta 0.1, one thread) at `subsample: 0.5`, against LightGBM with
`bagging_fraction: 0.5, bagging_freq: 1` and XGBoost with `subsample: 0.5`:

| Leaves | moonboost | LightGBM 4.7.0 | XGBoost 3.4.1 `hist` |
|---|---|---|---|
| 255 | 4.0 s, AUC 0.7619 | 11.3 s, AUC 0.7596 | 22.2 s, AUC 0.7471 |
| 1023 | 6.1 s, AUC 0.7731 | 14.3 s, AUC 0.7716 | 33.6 s, AUC 0.7463 |

Without sampling LightGBM takes 6.4 s and 9.0 s there, so sampling makes
moonboost faster and LightGBM slower. Gradient-based sampling finds its
threshold exactly in two passes over the rows (255 leaves, categories as
numbers: 6.7 s with a bisection, 4.8 s now, same predictions).

The regression objectives use XGBoost's gradients (`count:poisson` with its
`max_delta_step` of 0.7, `tweedie_variance_power` and `huber_slope`
settable). On 16,000 synthetic rows, 100 rounds, depth 6, eta 0.1
(`bench/train_objectives.py`), XGBoost predicts the same bits as moonboost
on the moonboost model for every one, and the held-out loss is close to
XGBoost's with the same settings:

| Objective | moonboost | XGBoost 3.4.1 `hist` |
|---|---|---|
| `count:poisson` (Poisson deviance) | 0.14831 | 0.14629 |
| `reg:gamma` | 1.26533 | 1.26800 |
| `reg:tweedie` | 4.93330 | 4.93413 |
| `reg:pseudohubererror` | 0.98422 | 0.99400 |
| `reg:squaredlogerror` (RMSLE) | 0.18395 | 0.18536 |
| `reg:logistic` (log loss) | 0.62110 | 0.62099 |
| `binary:logitraw` (log loss) | 0.64138 | 0.64120 |

`binary:hinge` predicts class 0 or 1 from the margin's sign. On the
GBM-perf airline data (800,000 rows to train, the remaining 200,000 to
test), one thread, 100 rounds, eta 0.1, XGBoost predicts the same bits on
the moonboost model:

| Depth | moonboost time | accuracy | XGBoost 3.4.1 time | accuracy |
|---|---|---|---|---|
| 6 | 2.2 s | 0.8144 | 4.7 s | 0.8145 |
| 10 | 3.2 s | 0.8237 | 6.7 s | 0.8229 |

`survival:cox` fits Cox proportional hazards with XGBoost's gradients
(Breslow ties; a negative label is a time censored at its absolute
value). On 250,000 synthetic rows with 38% censored (200,000 to train),
one thread, 200 rounds, eta 0.05 (`bench/train_cox.py`), XGBoost predicts
the same bits on the moonboost model; concordance on held-out rows:

| Depth | moonboost time | C-index | XGBoost 3.4.1 time | C-index |
|---|---|---|---|---|
| 4 | 1.3 s | 0.7515 | 1.8 s | 0.7518 |
| 6 | 1.6 s | 0.7565 | 2.5 s | 0.7559 |

`survival:aft` is the accelerated failure time model on interval
censored times: `label` is each time's lower bound and
`label_upper_bound` its upper (equal when observed, +inf when right
censored, a lower bound of 0 when left censored), with XGBoost's normal,
logistic and extreme distributions, gradients, clipping and limits.
Instead of XGBoost's fixed starting time of 0.5, the model starts at the
constant of highest likelihood. On 250,000 synthetic rows (50% observed,
25% right, 10% left and 15% interval censored; 200,000 to train), one
thread, 200 rounds, eta 0.05 (`bench/train_aft.py`), XGBoost predicts the
same bits on the moonboost model; held-out `aft-nloglik` as XGBoost
computes it:

| Distribution, scale | Depth | moonboost time | nloglik | XGBoost 3.4.1 time | nloglik |
|---|---|---|---|---|---|
| normal, 1 | 4 | 1.9 s | 1.4141 | 4.5 s | 1.4175 |
| normal, 1 | 6 | 2.1 s | 1.3956 | 4.9 s | 1.3955 |
| logistic, 0.5 | 4 | 2.2 s | 1.5643 | 3.5 s | 1.5595 |
| logistic, 0.5 | 6 | 2.5 s | 1.5553 | 4.0 s | 1.5549 |
| extreme, 0.8 | 4 | 3.1 s | 1.4620 | 4.4 s | 1.4735 |
| extreme, 0.8 | 6 | 3.3 s | 1.4571 | 5.0 s | 1.4572 |

Over three more data seeds at depth 4, the extreme distribution's loss
is 0.012 to 0.022 below XGBoost's, the normal's between 0.006 below and
0.003 above, and the logistic's between 0.0004 below and 0.007 above:
there the better start does not help.

`reg:absoluteerror` and `reg:quantileerror` (`quantile_alpha`) grow each
tree on the loss's gradients and then set every leaf to eta times the
alpha quantile of its rows' residuals. On 100,000 synthetic rows with
heavy-tailed noise (80,000 to train), four seeds, 100 rounds, depth 6
(`bench/train_quantile.py`), XGBoost again predicts the same bits; mean
held-out pinball loss with `min_data_in_leaf: 20` (LightGBM's default):

| Objective | moonboost | XGBoost 3.4.1 `hist` | LightGBM 4.7.0 |
|---|---|---|---|
| `reg:absoluteerror` | 0.67929 | 0.68250 | 0.67902 |
| `reg:quantileerror`, alpha 0.1 | 0.41051 | 0.41452 | 0.41021 |
| `reg:quantileerror`, alpha 0.9 | 0.41214 | 0.41581 | 0.41162 |

moonboost's loss is about 0.5% below XGBoost's (five to ten standard
errors per seed) and 0.04 to 0.13% above LightGBM's (about one standard
error per seed).
The leaf values need only two order statistics of each leaf's residuals,
which are selected rather than sorted for: one quantile model trains in
0.35 to 0.42 s against XGBoost's 0.74 to 0.76 s and LightGBM's 0.44 to
0.51 s (seed 0, alpha 0.5 and 0.9, one thread).

Several quantiles fit in one model: `quantile_alphas: [0.1, 0.5, 0.9]`
(or `"quantile_alpha": [0.1, 0.5, 0.9]` in `TrainParams::from_json`, or
`--quantile-alpha 0.1,0.5,0.9`) grows one tree per quantile each round
and saves XGBoost's multi-target model, `num_target` 3; predictions come
sorted per row, as XGBoost returns them, so the quantiles never cross.
On the same data, four seeds, 100 rounds, depth 6, one thread
(`bench/train_multiquantile.py`, data loading included), XGBoost
predicts the same bits from our model; LightGBM has no multi-quantile
objective, so it trains three models. Mean held-out pinball loss with
`min_data_in_leaf: 20`:

| | moonboost | XGBoost 3.4.1 `hist` | LightGBM 4.7.0 (3 models) |
|---|---|---|---|
| Training time | 1.07 s | 2.10 s | 1.44 s |
| alpha 0.1 | 0.42260 | 0.42432 | 0.42238 |
| alpha 0.5 | 0.69731 | 0.69900 | 0.69703 |
| alpha 0.9 | 0.42247 | 0.42646 | 0.42237 |

The loss is below XGBoost's for every seed and quantile, and within
about one standard error of LightGBM's (above it in 9 of 12 cases).

### Several outputs

`num_target: K` trains one model with K outputs for `reg:squarederror`,
`reg:pseudohubererror`, `reg:absoluteerror`, `binary:logistic`,
`reg:logistic` and `binary:logitraw`: the label holds K values per row,
row by row (on the command line, a labels file with K comma-separated
columns), each output starts from its own labels' intercept, and each
round grows one tree per output, which is XGBoost's multi-target model
with its default `one_output_per_tree`. Each output's trees are the ones
a model trained on that column alone would grow. On 250,000 synthetic
rows with 10 features (200,000 to train), three targets, three seeds,
100 rounds, depth 6, one thread (`bench/train_multioutput.py`, data
loading included), XGBoost predicts the same bits from our models;
LightGBM trains three models:

| Three outputs | moonboost | XGBoost 3.4.1 `hist` | LightGBM 4.7.0 (3 models) |
|---|---|---|---|
| `reg:squarederror`, time | 1.88 to 1.95 s | 3.53 to 3.68 s | 3.08 to 3.13 s |
| held-out RMSE, mean of the three | 0.50209 | 0.50215 | 0.50231 |
| `binary:logistic`, time | 2.19 to 2.22 s | 4.12 to 4.30 s | 3.75 to 3.87 s |
| held-out log loss, mean of the three | 0.38527 | 0.38534 | 0.38548 |

Against XGBoost the per-output, per-seed losses differ by at most about
three standard errors, either way (lower in 10 of 18); moonboost's is
lower than LightGBM's in 13 of 18.

`multi_strategy: "multi_output_tree"` (`--multi-strategy` on the command
line) grows one tree a round instead, whose splits serve every output and
whose leaves hold one value per output (squared, pseudo-Huber and
logistic losses; depth-wise, numerical features, no sampling). A split's
gain is the sum of each output's gain, from histograms that keep one
packed integer gradient pair per output in each bin; leaf values come
from the rows' exact gradients. The trees are XGBoost's vector-leaf trees
(`size_leaf_vector`), which XGBoost loads and predicts the same bits
from, and moonboost loads XGBoost's. Same data and settings
(`bench/train_multitree.py`), against XGBoost's `multi_output_tree`:

| Three outputs, vector leaves | moonboost | XGBoost 3.4.1 `hist` | LightGBM 4.7.0 (3 separate models) |
|---|---|---|---|
| `reg:squarederror`, time | 1.37 to 1.42 s | 2.71 to 2.85 s | 3.45 to 3.47 s |
| held-out RMSE, mean of the three | 0.50557 | 0.50549 | 0.50231 |
| `binary:logistic`, time | 1.66 to 1.71 s | 3.33 to 3.36 s | 4.09 to 4.16 s |
| held-out log loss, mean of the three | 0.38825 | 0.38825 | 0.38548 |

moonboost and XGBoost differ per output and seed by at most about three
standard errors, either way (moonboost lower in 7 of 18). On this data
vector leaves fit worse than one tree per output, for both libraries:
LightGBM's three separate models are better on all 18.

### Ranking

`rank:ndcg` is LambdaRank: within each query (`group`, the query sizes in
row order, as XGBoost's), every pair of documents with different labels of
which one is in the current top `rank_truncation` (30, LightGBM's
default; 0 for all pairs) is weighted by the change in NDCG of swapping
them, divided by 0.01 plus their score gap, and each query's gradients are
scaled by log2(1 + s) / s. `rank:pairwise` is RankNet on the same pairs.
The model is an XGBoost `rank:ndcg` model; XGBoost predicts the same bits.

MSLR-WEB10K fold 1 (723,412 rows in 6,000 queries, 136 features), one
thread, 100 rounds, eta 0.1, 255 leaves, `min_child_weight` 0.001,
NDCG@10 on the 2,000 test queries (`bench/train_rank.py`); times include
binning:

| | moonboost | XGBoost 3.4.1 `hist` | LightGBM 4.7.0 |
|---|---|---|---|
| Training time | 38.4 s | 59.4 s | 42.0 s |
| NDCG@10 | 0.5051 | 0.5042 | 0.5042 |

The NDCG difference (+0.0010, standard error 0.0019 over queries) is not
significant. With all pairs (`rank_truncation: 0`) moonboost takes 46.0 s
for 0.5020.

`rank:pairwise` on the same data: moonboost 38.9 to 39.4 s for 0.4915,
XGBoost 53.6 s for 0.4923 with its default pairs and 54.1 s for 0.4875 with
`lambdarank_pair_method: topk` and 30 pairs per document; neither NDCG
difference is significant (-0.0008 and +0.0040, standard errors 0.0023 and
0.0022). LightGBM has no pairwise objective.

`rank:map` takes 0/1 labels and weights each pair by XGBoost's change in
average precision of swapping them, on the same pairs and with the same
normalisation. MSLR-WEB10K as above with relevance of 1 or more counted as
relevant, MAP on the test queries that have a relevant document
(`bench/train_map.py`; LightGBM, which has no MAP objective, runs
`lambdarank` on the same labels):

| | moonboost | XGBoost 3.4.1 `hist` | LightGBM 4.7.0 `lambdarank` |
|---|---|---|---|
| Training time | 37.5 to 38.3 s | 55.0 to 56.6 s | 38.9 to 39.1 s |
| MAP | 0.6443 | 0.6419 | 0.6427 |

Against XGBoost the MAP difference is +0.0024 (standard error 0.0009 over
queries); against LightGBM +0.0016 (0.0011), not significant. XGBoost
predicts the same bits from the moonboost model.

Training also runs in the browser, and every build makes the same model bit
for bit. `webgc/` is a 37 KB WebAssembly GC module with a loader,
`web/train_wasm.mjs` (`bench/train_wasm_bench.mjs` times it); in Node it
trains the airline model above faster than native XGBoost and LightGBM.
The ES module's `train` does the same in plain JS, at two to three times
the wasm-gc time: there the quantised sums are Doubles (Int64 would be a
BigInt) and the buffers typed arrays.

The only other GBDT trainer for browsers we found is
[ml-xgboost](https://www.npmjs.com/package/ml-xgboost) 1.1.2 (XGBoost
compiled to WebAssembly with Emscripten, 2018). On the first 100k rows of
the airline data, 30 rounds, depth 10, `tree_method=hist`, in Node 26: 1.01 s
against 0.30 s for moonboost wasm-gc and 0.79 s for moonboost JS
(`bench/ml_xgboost_bench.cjs`). Past about 150k rows ml-xgboost stops: its
loader copies the data onto a 5 MB stack.

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
indices bit for bit, and final predictions bit for bit too. The cases cover
regression (also with `reg_sqrt`), binary (also with a `sigmoid` parameter),
`multiclass`, `multiclassova`, Poisson, Gamma, Tweedie, both cross-entropy
objectives, zeros as missing values, `use_missing=false`, categorical
features with missing and negative codes, and random forest mode. LightGBM
compares doubles (`x <= threshold`), turns NaN into 0 unless the split
handles NaN, reads values within 1e-35 of zero as zero, and adds trees in
double; moonboost does the same. Its sigmoid, softmax and exponential
outputs call the C library's `exp`, which is not the same function on
every platform; moonboost calls the C `exp` on native builds and, on the
web targets, a port of glibc's (the x86-64 FMA build), checked bit for bit
against glibc on 2,000 inputs including the overflow and subnormal ranges.

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
to about 40% between runs on this VM; the other columns by a few percent.
What ships to the browser: moonboost is one 270 KB module (77 KB gzipped, LightGBM support
and training included); onnxruntime-web's
WebAssembly build is `ort.wasm.min.mjs` (54 KB) plus
`ort-wasm-simd-threaded.mjs` (25 KB) plus a 14.8 MB `.wasm` file (3.8 MB
gzipped). Model files are smaller in ONNX (97 KB against 223 KB of JSON for
the small model).

The same comparison in a browser, `bench/browser_bench.mjs`: headless
Chromium 141 through Playwright, the files served from localhost, a fresh
browser context for every run, one thread each. Cold start here also
includes fetching the model. Medians of three runs:

| Model | | moonboost (JS) | onnxruntime-web 1.31.1 (wasm) |
|---|---|---|---|
| 100 trees, depth 4 | cold start | 59 ms | 466 ms |
| | batch, µs/row | 0.99 | 3.1 |
| | one row per call, µs | 4.7 | 34 |
| | probabilities equal to XGBoost's | 10,000 / 10,000 | 5,776 / 10,000 |
| 500 trees, depth 8 | cold start | 331 ms | 792 ms |
| | batch, µs/row | 6.1 | 27.6 |
| | one row per call, µs | 14.4 | 121 |
| | probabilities equal to XGBoost's | 10,000 / 10,000 | 5,037 / 10,000 |
| 1000 trees, depth 10 | cold start | 526 ms | 1,536 ms |
| | batch, µs/row | 16.0 | 61.6 |
| | one row per call, µs | 75 | 336 |
| | probabilities equal to XGBoost's | 10,000 / 10,000 | 5,053 / 10,000 |

XGBoost has no browser build; it is compared natively in the tables above.

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
The JS bundle minified with esbuild is 270 KB, 77 KB gzipped, training included, with no other runtime.

### Against compiled engines

TL2cgen (Treelite's C code generator) and lleaves (LLVM-compiled LightGBM)
turn a model into machine code ahead of time. Same VM, one thread, 10,000
rows, µs per row in batch; each engine's predictions compared bit for bit
with the library that trained the model.

XGBoost models (the ones above). TL2cgen 1.1.0 C code built with `gcc -O3
-march=native` in 64 translation units, best of `quantize` on and off;
`bench/tl2cgen_bench.sh`:

| Model | moonboost native | TL2cgen | XGBoost 3.4.1 C API |
|---|---|---|---|
| 100 trees, depth 4 | 0.60 | 1.68 | 1.03 |
| 500 trees, depth 8 | 3.5 | 23.5 | 12.5 |
| 1000 trees, depth 10 | 8.2 | 104 | 34.8 |
| equal to XGBoost | 10,000 / 10,000 | 6,355 to 7,121 / 10,000 | |

The generated C for the two larger models takes about 1 and 3 to 4 minutes
to compile on this 2-core VM.

LightGBM models of the same shape (binary, 32 features, 5% missing,
`num_leaves = 2^depth`; `bench/lgb_models.py`). lleaves 1.3.0 is timed
through its Python `predict` on the whole batch (`n_jobs=1`);
`bench/lleaves_bench.py`:

| Model | moonboost native | lleaves |
|---|---|---|
| 100 trees, depth 4 | 0.38 | 0.96 |
| 500 trees, depth 8 | 3.7 | 12.9 |
| 1000 trees, depth 10 | 13.5 | not measured (compiling took over 20 minutes) |
| equal to LightGBM | 10,000 / 10,000 | 5,262 and 5,284 / 10,000 |

moonboost needs no compile step: a LightGBM model loads into the same
branch-free walk as XGBoost models, with doubles where LightGBM uses them.
Each feature value is turned once per row into the number every split on it
compares (a missing value or zero that a split sends one way by rule
becomes ±infinity), trees up to 10 levels are laid out as implicit heaps,
and float inputs walk a packed copy where each double threshold is replaced
by the largest float not above it, which decides `x <= t` exactly for any
float `x`. Larger forests compare integer ranks of the values instead, each
node packed into one 64-bit word (child, slot and threshold rank), eight
rows walking each tree side by side over blocks of 256 rows.

### On x86-64

The same benchmarks on a 4-vCPU AMD EPYC VM (Debian 13, glibc 2.41, gcc
14.2), every engine pinned to one core in one session, best of three. This
machine also runs Intel oneDAL (scikit-learn-intelex 2026.2,
`daal4py.mb.convert_model`), which has no ARM build; it is timed through its
Python `predict_proba` on the whole batch (`bench/onedal_bench.py`), as LightGBM and lleaves are through their `predict`.
µs per row in batch:

| XGBoost model | moonboost native | oneDAL | XGBoost 3.4.1 C API | TL2cgen |
|---|---|---|---|---|
| 100 trees, depth 4 | 0.30 | 0.52 | 0.86 | 1.60 |
| 500 trees, depth 8 | 3.3 | 5.1 | 8.8 | 28.7 |
| 1000 trees, depth 10 | 7.6 | 13.2 | 25.7 | not measured |
| equal to XGBoost | 10,000 / 10,000 | 5,156 to 5,852 / 10,000 | | 6,355 to 7,121 / 10,000 |

| LightGBM model | moonboost native | oneDAL | LightGBM 4.7.0 | lleaves |
|---|---|---|---|---|
| 100 trees, depth 4 | 0.34 | 0.53 | 3.1 | 0.72 |
| 500 trees, depth 8 | 3.5 | 5.1 | 39.3 | 14.0 |
| 1000 trees, depth 10 | 10.6 | 14.5 | 127.5 | not measured |
| equal to LightGBM | 10,000 / 10,000 | 3,911 to 4,972 / 10,000 | | 5,262 and 5,284 / 10,000 |

oneDAL's differences from the original libraries are at most 2.2e-7. Every
moonboost prediction on this machine matched XGBoost and LightGBM bit for
bit, as on ARM. lleaves took over nine minutes to compile the 500-tree model.

Loading, in ms (read the file, parse, build the prediction form, first load
in a fresh process; oneDAL's figure is its conversion from an already loaded
booster):

| Model file | moonboost native | XGBoost / LightGBM | oneDAL conversion |
|---|---|---|---|
| XGBoost, 500 trees (9.1 MB JSON) | 127 | 159 | 604 |
| XGBoost, 1000 trees (19 MB JSON) | 269 | 370 | 1,927 |
| LightGBM, 500 trees (text) | 38 | 44 | 2,048 |
| LightGBM, 1000 trees (38 MB text) | 121 | 149 | 7,865 |

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

`DenseMatrix::from_csv_bytes` reads the file's UTF-8 bytes directly, with
no string in between. Plain decimals are parsed by a fast path that gives
the same bits as `parse_double`: the CLI loads the 723,412 x 136 MSLR
training file (579 MB) about 5 s faster than before, and 1M x 8 GBM-perf
in 0.12 s instead of 0.62 s.

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
  // many float rows at once, row-major, num_tree_per_iteration values per row
  let data = @moonboost.DenseMatrix::new([5.1, 3.5, 1.4, 0.2], cols=4)
  println(model.predict_batch(data))
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
and `predict_lightgbm_raw(model, row)`. An XGBoost `.ubj` file loads with
`load_model_ubjson(bytes)`, where `bytes` is a `Uint8Array`.
`train(rows, cols, labels, objective, num_class, num_round, max_depth, eta)`
trains in the page and returns XGBoost JSON text (`-1` or `""` keeps a
default). `train_with(rows, cols, labels, JSON.stringify(params))` takes
XGBoost's parameter names instead, with every training setting: any
objective, `group` for ranking, `categorical`, `grow_policy`,
`subsample`, `quantile_alpha` and the rest
(`TrainParams::from_json` in MoonBit). For two to three times the speed, train with the WebAssembly GC module
(build it with `moon build --target wasm-gc --release`; it is
`_build/wasm-gc/release/build/webgc/webgc.wasm`):

```js
import { loadTrainer } from "./train_wasm.mjs";
const train = await loadTrainer("webgc.wasm");
const json = train(rows, cols, labels, "binary:logistic", -1, 100, 6, 0.3);
// or a parameter object, as for train_with
const ranker = train(rows, cols, labels, { objective: "rank:ndcg", group, eta: 0.1 });
```

See `examples/node/iris.mjs`.

Each [GitHub release](https://github.com/Juwan-Hwang/moonboost/releases)
carries these files prebuilt: `moonboost.js` (the ES module above),
`webgc.wasm` with `train_wasm.mjs`, and the CLI for Linux x86-64 and arm64.

## CLI

```
moon run cmd/main --target js -- info examples/iris/model.json
moon run cmd/main --target js -- predict examples/iris/model.json examples/iris/data.csv --header
moon run cmd/main --target native -- train rows.csv labels.csv -o model.json --objective binary:logistic --rounds 100
```

```
0.99185115,0.005436964,0.0027118726
0.992132,0.0047212564,0.003146739
...
```

`--margin` prints raw scores and `--leaf` prints the leaf reached in each tree.
LightGBM text models work the same way; the CLI tells the formats apart by
content. XGBoost UBJSON models are read when the file name ends in `.ubj`.
The release binaries (`moonboost-linux-x86_64`, `moonboost-linux-arm64`) take
the same arguments: `./moonboost-linux-x86_64 info model.json`.

Load errors are `ModelError`, which prints as `InvalidModel: ...` or
`Unsupported: ...`.

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
© Microsoft Corporation and the LightGBM developers. `lgb_exp.mbt` ports
the double-precision `exp` of Arm's optimized-routines (as used by glibc),
MIT OR Apache-2.0 WITH LLVM-exception, © Arm Limited; its table and
coefficients are copied from glibc 2.41's x86-64 `libm`.
