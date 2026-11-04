"""Time XGBoost's own CPU predictor on the benchmark inputs (1 thread).

Usage: python3 bench/xgb_bench.py
"""

import os
import time

import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

X = np.loadtxt(os.path.join(DATA, "input.csv"), delimiter=",", dtype=np.float32)
for name in ["small", "medium", "large"]:
    b = xgb.Booster(model_file=os.path.join(DATA, f"{name}.json"))
    b.set_param({"nthread": 1})
    b.inplace_predict(X)
    reps = 3
    t = time.perf_counter()
    for _ in range(reps):
        b.inplace_predict(X)
    batch = (time.perf_counter() - t) / (reps * len(X))
    row = X[:1]
    t = time.perf_counter()
    for _ in range(2000):
        b.inplace_predict(row)
    single = (time.perf_counter() - t) / 2000
    print(f"{name:7s} batch {batch * 1e6:8.2f} us/row  "
          f"single-row call {single * 1e6:8.1f} us")
