"""Generate the benchmark models and inputs.

Three sizes, all binary:logistic on 32 dense features (5% missing):
  small   100 trees, depth 4
  medium  500 trees, depth 8
  large  1000 trees, depth 10

Writes bench/data/<size>.json, bench/data/input.csv (10,000 rows) and
bench/data/<size>.expected.csv (XGBoost's predictions for the same rows).

Usage: python3 bench/gen_models.py
"""

import os

import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
SIZES = {"small": (100, 4), "medium": (500, 8), "large": (1000, 10)}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(42)
    n, f = 50_000, 32
    X = rng.normal(size=(n, f)).astype(np.float32)
    w = rng.normal(size=f)
    logits = X @ w + 0.5 * np.sin(3 * X[:, 0]) * X[:, 1]
    y = (logits + rng.normal(size=n) > 0).astype(int)
    X[rng.random(X.shape) < 0.05] = np.nan
    train = xgb.DMatrix(X, label=y)

    test = X[:10_000]
    np.savetxt(os.path.join(OUT, "input.csv"), test, fmt="%.9g", delimiter=",")
    with open(os.path.join(OUT, "input.csv")) as fh:
        text = fh.read().replace("nan", "NaN")
    with open(os.path.join(OUT, "input.csv"), "w") as fh:
        fh.write(text)
    for name, (trees, depth) in SIZES.items():
        b = xgb.train(
            {"objective": "binary:logistic", "max_depth": depth, "eta": 0.1,
             "seed": 0, "nthread": 1},
            train, trees)
        b.save_model(os.path.join(OUT, f"{name}.json"))
        p = b.predict(xgb.DMatrix(test))
        np.savetxt(os.path.join(OUT, f"{name}.expected.csv"), p, fmt="%.9g")
        print(name, "done")


if __name__ == "__main__":
    main()
