"""Build the browser demo's model and data (docs/).

Trains an XGBoost regressor on the California housing data, saves it as
JSON (for moonboost) and ONNX (for the onnxruntime-web comparison), and
writes 2,000 held-out rows with XGBoost's own predictions as float32 bits.

Usage: python3 tools/gen_demo.py && sh tools/build_demo.sh
       (needs xgboost, scikit-learn, onnxmltools, onnx)
"""
import json
import os

import numpy as np
import onnx
import onnxmltools
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.datasets import fetch_california_housing

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")


def main():
    d = fetch_california_housing()
    X = d.data.astype(np.float32)
    y = d.target
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    test, train = idx[:2000], idx[2000:]
    params = {"objective": "reg:squarederror", "max_depth": 6, "eta": 0.1,
              "seed": 0, "nthread": 1}
    # no feature names in the model: the ONNX converter wants f0, f1, ...
    b = xgb.train(params, xgb.DMatrix(X[train], label=y[train]), 200)
    b.save_model(os.path.join(OUT, "model.json"))
    rows = X[test]
    pred = b.predict(xgb.DMatrix(rows))
    reg = xgb.XGBRegressor()
    reg.load_model(os.path.join(OUT, "model.json"))
    m = onnxmltools.convert_xgboost(
        reg, initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
        target_opset=15)
    onnx.save(m, os.path.join(OUT, "model.onnx"))
    with open(os.path.join(OUT, "rows.json"), "w") as f:
        json.dump({
            "features": d.feature_names,
            "x": [float(v) for v in rows.ravel()],
            "expected_bits": [int(v) for v in pred.astype(np.float32).view(np.uint32)],
            "medians": [float(v) for v in np.median(X, axis=0)],
            "lo": [float(v) for v in np.percentile(X, 1, axis=0)],
            "hi": [float(v) for v in np.percentile(X, 99, axis=0)],
        }, f)
    print("demo data written")


if __name__ == "__main__":
    main()
