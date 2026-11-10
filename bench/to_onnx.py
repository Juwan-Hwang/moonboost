"""Convert the benchmark models to ONNX for bench/web_bench.mjs.

Usage: python3 bench/to_onnx.py   (needs xgboost, onnxmltools, onnx)
"""
import os

import onnx
import onnxmltools
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

for name in ["small", "medium", "large"]:
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(DATA, f"{name}.json"))
    m = onnxmltools.convert_xgboost(
        clf, initial_types=[("input", FloatTensorType([None, 32]))], target_opset=15)
    onnx.save(m, os.path.join(DATA, f"{name}.onnx"))
    print(name, "done")
