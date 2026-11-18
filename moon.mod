// Learn more about moon.mod configuration:
// https://docs.moonbitlang.com/en/latest/toolchain/moon/module.html
//
// To add a dependency, run this command in your terminal:
//   moon add moonbitlang/x
//
// Or manually declare it in `import`, for example:
// import {
//   "moonbitlang/x@0.4.6",
// }

name = "Juwan-Hwang/moonboost"

version = "0.1.1"

readme = "README.mbt.md"

repository = "https://github.com/Juwan-Hwang/moonboost"

license = "Apache-2.0"

keywords = [
  "gbdt",
  "xgboost",
  "machine-learning",
  "gradient-boosting",
  "decision-tree",
  "lightgbm",
  "wasm",
]

preferred_target = "wasm"

description = "Gradient boosted decision trees in pure MoonBit: predict XGBoost and LightGBM models bit for bit, and train histogram trees, on native, wasm, wasm-gc and js."

import {
  "moonbitlang/x@0.5.7",
}
