# Changelog

## 0.2.0 (2026-11-25)

Training grows from four objectives to all 19 of XGBoost's, with categorical
features, loss-guided growth, sampling and several outputs. Models are still
XGBoost JSON that XGBoost loads and predicts bit for bit. Prediction is
unchanged from 0.1.1.

### New in training

- Objectives: `reg:absoluteerror` and `reg:quantileerror` (one quantile or
  several, `quantile_alpha` as a list), `reg:logistic`, `binary:logitraw`,
  `binary:hinge`, `count:poisson`, `reg:gamma`, `reg:tweedie`,
  `reg:pseudohubererror`, `reg:squaredlogerror`, `survival:cox`,
  `survival:aft` (normal, logistic and extreme distributions), `rank:ndcg`,
  `rank:pairwise` and `rank:map` with query groups.
- Categorical features: one category against the rest up to
  `max_cat_to_onehot`, else a set of categories by sorted gradient ratio
  (`cat_smooth`, `cat_l2`, `min_data_per_group`, `max_cat_threshold`).
- Loss-guided growth: `grow_policy: "lossguide"` with `max_leaves`.
- Sampling: `subsample` with `uniform` or `gradient_based` (minimal variance)
  rows, `colsample_bytree`, `colsample_bynode`, `seed`; `min_data_in_leaf`.
- Several outputs (`num_target`): one tree per output, or vector leaves with
  `multi_strategy: "multi_output_tree"`.
- Parameters as XGBoost-style JSON: `TrainParams::from_json`, CLI `--params`,
  web `train_with`, webgc `train_run_params`.
- `train_xgboost_json_bytes` returns the model as UTF-8 bytes; the CLI writes
  them directly. `DenseMatrix::from_csv_bytes` parses CSV in place.

### Faster and smaller

- Airline (1M rows, six categorical columns, one thread), against
  LightGBM 4.7.0 in one session: 21%, 25% and 41% less time at 255, 1023 and
  4095 leaves, with the highest AUC at each size. Loss-guided trees copy a
  node's rows every three levels so deep nodes stay contiguous; with 9 to 12
  features the quantised gradient pairs ride in the bin words.
- Peak memory of the CLI: 273 / 306 / 405 MB at 255 / 1023 / 4095 leaves
  (LightGBM 320 / 378 / 621 MB and XGBoost 477 / 637 / 1075 MB as Python
  processes); 128 MB on GBM-perf at depth 10.
- SIMD (`moonbitlang/core/v128`) logistic gradients and softmax probabilities
  on native, wasm and wasm-gc.
- Row sampling builds trees on the kept rows only; CSV loading about twice as
  fast; binning in fewer passes.

### Limits

- One thread; no GPU.
- Benchmarks are single-machine, single-session runs (see README for the
  setup of each). Ranking was measured on one dataset (MSLR-WEB10K, one
  fold); survival, quantile, several-output and vector-leaf results on one
  dataset or synthetic data.
- The SIMD paths use MoonBit's experimental `v128` API.
- Loss-guided training trades memory for speed: at 255 and 1023 leaves it
  needs more than LightGBM net of the Python interpreter.
