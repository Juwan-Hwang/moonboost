"""Intel oneDAL (daal4py) on the benchmark models, one thread: batch and
single-row latency and how many predictions match the original library bit
for bit. daal4py ships for x86-64 only.

Run from bench/ after gen_models.py and lgb_models.py:
  python onedal_bench.py
"""
import time
import numpy as np
import daal4py as d4p
import xgboost as xgb
import lightgbm as lgb

d4p.daalinit(1)


def load_csv(path):
    rows = [[float(v) if v else np.nan for v in l.rstrip("\n").split(",")]
            for l in open(path)]
    return np.array(rows, dtype=np.float32)


def bench(name, model, X, ref):
    pred = model.predict_proba(X)[:, 1]
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        model.predict_proba(X)
        best = min(best, time.perf_counter() - t0)
    n = 1000
    t0 = time.perf_counter()
    for i in range(n):
        model.predict_proba(X[i:i + 1])
    single = (time.perf_counter() - t0) / n
    exact = int((pred.astype(np.float32) == ref.astype(np.float32)).sum())
    print(f"{name}: batch {best / len(X) * 1e6:.2f} us/row, "
          f"single {single * 1e6:.2f} us, exact {exact}/{len(X)}, "
          f"max diff {np.abs(pred - ref).max():.3g}")


X = load_csv("data/input.csv")
for m in ["small", "medium", "large"]:
    booster = xgb.Booster(model_file=f"data/{m}.json")
    ref = np.loadtxt(f"data/{m}.expected.csv")
    bench(f"xgboost {m}", d4p.mb.convert_model(booster), X, ref)

X = load_csv("data/lgb_input.csv")
for m in ["small", "medium", "large"]:
    booster = lgb.Booster(model_file=f"data/lgb_{m}.txt")
    ref = np.loadtxt(f"data/lgb_{m}.expected.csv")
    bench(f"lightgbm {m}", d4p.mb.convert_model(booster), X, ref)
