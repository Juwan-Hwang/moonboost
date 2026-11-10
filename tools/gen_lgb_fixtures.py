"""Generate differential-test fixtures from the reference LightGBM implementation.

Every case trains a small LightGBM model, keeps its text form, and records the
exact float64 bit patterns of the inputs, the raw scores, the final
predictions and the leaf indices. `moon test` checks moonboost against them.

Usage: python3 tools/gen_lgb_fixtures.py && moon fmt
       (needs lightgbm, scikit-learn, numpy)
"""
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

ROWS = 40
rng = np.random.default_rng(7)


def bits(a):
    return [int(v) for v in np.asarray(a, dtype=np.float64).view(np.uint64).ravel()]


def with_missing(X, frac=0.1):
    X = X.astype(np.float64).copy()
    X[rng.random(X.shape) < frac] = np.nan
    return X


def case(name, X, y, params, rounds, categorical=(), missing=True):
    if missing:
        X = with_missing(X)
    d = lgb.Dataset(X, y, categorical_feature=list(categorical) or "auto",
                    free_raw_data=False)
    p = {"verbose": -1, "num_threads": 1, "deterministic": True, "seed": 0,
         "force_col_wise": True} | params
    b = lgb.train(p, d, rounds)
    text = b.model_to_string()
    # predict from the saved text, as a user of the file would
    b = lgb.Booster(model_str=text)
    rows = X[:ROWS]
    raw = b.predict(rows, raw_score=True)
    pred = b.predict(rows)
    leaf = b.predict(rows, pred_leaf=True)
    width = 1 if raw.ndim == 1 else raw.shape[1]
    return {
        "name": name,
        "model": text,
        "num_feature": X.shape[1],
        "width": width,
        "x": bits(rows),
        "raw": bits(raw),
        "pred": bits(pred),
        "leaf": [int(v) for v in np.asarray(leaf).ravel()],
        "num_trees": int(np.asarray(leaf).reshape(len(rows), -1).shape[1]),
    }


def main():
    cases = []
    Xd, yd = load_diabetes(return_X_y=True)
    Xb, yb = load_breast_cancer(return_X_y=True)
    Xi, yi = load_iris(return_X_y=True)
    cases.append(case("regression", Xd, yd,
                      {"objective": "regression", "num_leaves": 15}, 30))
    cases.append(case("regression_sqrt", Xd, yd,
                      {"objective": "regression", "reg_sqrt": True, "num_leaves": 7}, 20))
    cases.append(case("binary", Xb, yb,
                      {"objective": "binary", "num_leaves": 7}, 30))
    cases.append(case("binary_sigmoid", Xb, yb,
                      {"objective": "binary", "sigmoid": 0.7, "num_leaves": 7}, 20))
    cases.append(case("multiclass", Xi, yi,
                      {"objective": "multiclass", "num_class": 3, "num_leaves": 5,
                       "min_data_in_leaf": 5}, 15))
    cases.append(case("multiclassova", Xi, yi,
                      {"objective": "multiclassova", "num_class": 3, "num_leaves": 5,
                       "min_data_in_leaf": 5}, 15))
    yp = rng.poisson(np.exp(Xd[:, 2] * 5 + 1))
    cases.append(case("poisson", Xd, yp,
                      {"objective": "poisson", "num_leaves": 7}, 30))
    yg = rng.gamma(2.0, np.exp(Xd[:, 2] * 4 + 0.5))
    cases.append(case("gamma", Xd, yg, {"objective": "gamma", "num_leaves": 7}, 20))
    cases.append(case("tweedie", Xd, np.where(rng.random(len(yg)) < 0.3, 0.0, yg),
                      {"objective": "tweedie", "num_leaves": 7}, 20))
    cases.append(case("cross_entropy", Xb, yb * 0.8 + 0.1,
                      {"objective": "cross_entropy", "num_leaves": 7}, 20))
    cases.append(case("cross_entropy_lambda", Xb, yb * 0.8 + 0.1,
                      {"objective": "cross_entropy_lambda", "num_leaves": 7}, 20))
    # Zeros as the missing value (missing type Zero), and no missing handling
    # at all (type None, NaN read as 0).
    Xz = Xd.copy()
    Xz[rng.random(Xz.shape) < 0.2] = 0.0
    cases.append(case("zero_as_missing", Xz, yd,
                      {"objective": "regression", "zero_as_missing": True,
                       "num_leaves": 7}, 20, missing=False))
    cases.append(case("no_missing_handling", Xd, yd,
                      {"objective": "regression", "use_missing": False,
                       "num_leaves": 7}, 20))
    # Categorical features (codes 0..19 and 0..4), with missing and negative codes.
    n = 3000
    c1 = rng.integers(0, 20, n).astype(float)
    c2 = rng.integers(0, 5, n).astype(float)
    z = rng.normal(size=(n, 2))
    yc = (np.isin(c1, [1, 4, 7, 13, 18]) * 1.5 + (c2 == 3) - z[:, 0]
          + rng.normal(scale=0.5, size=n) > 0.8).astype(int)
    Xc = with_missing(np.column_stack([c1, z[:, 0], c2, z[:, 1]]))
    Xc[:40:7, 0] = -1.0
    cases.append(case("categorical", Xc, yc,
                      {"objective": "binary", "num_leaves": 15, "max_cat_to_onehot": 2,
                       "min_data_per_group": 10, "cat_smooth": 1}, 20,
                      categorical=(0, 2), missing=False))
    # Random forest mode averages the trees.
    cases.append(case("random_forest", Xd, yd,
                      {"boosting": "rf", "objective": "regression", "num_leaves": 15,
                       "bagging_freq": 1, "bagging_fraction": 0.7}, 10))

    out = ["// Generated by tools/gen_lgb_fixtures.py with lightgbm " + lgb.__version__
           + ". DO NOT EDIT."]
    for c in cases:
        out.append("\n///|")
        out.append(f"let lgb_fixture_{c['name']} : LgbFixture = {{")
        out.append(f"  name: \"{c['name']}\",")
        out.append("  model: (")
        for line in c["model"].split("\n"):
            if line.startswith("feature_importances:"):
                break
            out.append("    #|" + line)
        out.append("  ),")
        for k in ("num_feature", "width", "num_trees"):
            out.append(f"  {k}: {c[k]},")
        for k in ("x", "raw", "pred", "leaf"):
            out.append(f"  {k}: {c[k]},")
        out.append("}")
    with open("lightgbm_fixtures_test.mbt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", len(cases), "cases")


if __name__ == "__main__":
    main()
