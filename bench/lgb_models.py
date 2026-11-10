"""Three LightGBM benchmark models (binary, 32 features, 5% missing) and
their reference predictions, written to bench/data/:

  lgb_{small,medium,large}.txt   100/500/1000 trees, depth 4/8/10, 2^depth leaves
  lgb_input.csv                  10000 rows (float32 values, empty = missing)
  lgb_{name}.expected.csv        LightGBM predict(), one thread, 17 digits

Run from bench/:  python lgb_models.py
"""
import os
import numpy as np
import lightgbm as lgb

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(42)
n, f = 50_000, 32
X = rng.normal(size=(n, f)).astype(np.float32)
w = rng.normal(size=f)
logits = X @ w + 0.5 * np.sin(3 * X[:, 0]) * X[:, 1]
y = (logits + rng.normal(size=n) > 0).astype(int)
X[rng.random(X.shape) < 0.05] = np.nan
test = X[:10000]
with open("data/lgb_input.csv", "w") as out:
    for row in test:
        out.write(",".join("" if np.isnan(v) else repr(float(v)) for v in row) + "\n")
np.save("data/lgb_input.npy", test)
for name, (trees, depth) in {"small": (100, 4), "medium": (500, 8), "large": (1000, 10)}.items():
    params = {"objective": "binary", "max_depth": depth, "num_leaves": 2**depth,
              "learning_rate": 0.1, "num_threads": 1, "verbose": -1, "min_data_in_leaf": 5}
    booster = lgb.train(params, lgb.Dataset(X, y), trees)
    booster.save_model(f"data/lgb_{name}.txt")
    pred = booster.predict(test.astype(np.float64), num_threads=1)
    np.savetxt(f"data/lgb_{name}.expected.csv", pred, fmt="%.17g")
    print(name, booster.num_trees())
