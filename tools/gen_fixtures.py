"""Generate differential-test fixtures from the reference XGBoost implementation.

For every case we train a small model with XGBoost, save it as JSON, and record
the exact float32 bit patterns of the inputs, the raw margins and the final
predictions. `moon test` then checks moonboost against these numbers.

Usage: python3 tools/gen_fixtures.py && moon fmt
       (needs xgboost, scikit-learn, numpy, pandas)

XGBoost training is not bit-identical across CPU architectures, so the
committed file is the one this script produces on x86-64 (the CI `fixtures`
job, which uploads it as an artifact when it differs). Predictions are
compared bit for bit whatever machine runs the tests.
"""
import json
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

ROWS = 40
rng = np.random.default_rng(42)


def bits(a):
    return [int(v) for v in np.asarray(a, dtype=np.float32).view(np.uint32).ravel()]


def with_missing(X, frac=0.1):
    X = X.astype(np.float32).copy()
    mask = rng.random(X.shape) < frac
    X[mask] = np.nan
    return X


def case(name, X, y, params, rounds, feature_types=None):
    X = with_missing(X)
    d = xgb.DMatrix(X, label=y, feature_types=feature_types,
                    enable_categorical=feature_types is not None)
    b = xgb.train(params | {"seed": 0, "nthread": 1}, d, rounds)
    model = b.save_raw("json").decode()
    rows = X[:ROWS]
    dm = xgb.DMatrix(rows, feature_types=feature_types,
                     enable_categorical=feature_types is not None)
    margin = b.predict(dm, output_margin=True)
    pred = b.predict(dm)
    ngroup = 1 if margin.ndim == 1 else margin.shape[1]
    return {
        "name": name,
        "model": model,
        "num_feature": X.shape[1],
        "num_group": ngroup,
        "x": bits(rows),
        "margin": bits(margin),
        "pred": bits(pred),
        "pred_width": 1 if pred.ndim == 1 else pred.shape[1],
    }


def wrap_json(text, width=100):
    """Break minified JSON after commas so every newline lands between tokens."""
    lines, cur = [], ""
    for part in text.split(","):
        piece = part + ","
        if cur and len(cur) + len(piece) > width:
            lines.append(cur)
            cur = ""
        cur += piece
    lines.append(cur[:-1])
    return lines


def main():
    cases = []
    Xd, yd = load_diabetes(return_X_y=True)
    cases.append(case("regression", Xd, yd,
                      {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.3}, 30))
    Xb, yb = load_breast_cancer(return_X_y=True)
    cases.append(case("binary_logistic", Xb, yb,
                      {"objective": "binary:logistic", "max_depth": 3}, 30))
    Xi, yi = load_iris(return_X_y=True)
    cases.append(case("multi_softprob", Xi, yi,
                      {"objective": "multi:softprob", "num_class": 3, "max_depth": 3}, 15))
    cases.append(case("multi_softmax", Xi, yi,
                      {"objective": "multi:softmax", "num_class": 3, "max_depth": 3}, 15))
    yp = rng.poisson(np.exp(Xd[:, 2] * 5 + 1))
    cases.append(case("poisson", Xd, yp,
                      {"objective": "count:poisson", "max_depth": 4}, 30))

    # Strictly positive targets for the gamma and tweedie log-link objectives.
    yg = rng.gamma(2.0, np.exp(Xd[:, 2] * 4 + 0.5))
    cases.append(case("gamma", Xd, yg,
                      {"objective": "reg:gamma", "max_depth": 4}, 30))
    yt = np.where(rng.random(len(Xd)) < 0.3, 0.0, yg)
    cases.append(case("tweedie", Xd, yt,
                      {"objective": "reg:tweedie", "tweedie_variance_power": 1.5,
                       "max_depth": 4}, 30))
    cases.append(case("binary_logitraw", Xb, yb,
                      {"objective": "binary:logitraw", "max_depth": 3}, 30))

    # Two categorical features (codes 0..9 and 0..4) plus two numeric ones.
    n = 3000
    c1 = rng.integers(0, 10, n)
    c2 = rng.integers(0, 5, n)
    z = rng.normal(size=(n, 2))
    yc = (np.isin(c1, [1, 4, 7]) * 1.5 + (c2 == 3) - z[:, 0] + 0.3 * z[:, 1]
          + rng.normal(scale=0.5, size=n) > 0.8).astype(int)
    Xc = np.column_stack([c1, z[:, 0], c2, z[:, 1]])
    cases.append(case("categorical", Xc, yc,
                      {"objective": "binary:logistic", "max_depth": 4,
                       "max_cat_to_onehot": 1}, 20,
                      feature_types=["c", "q", "c", "q"]))

    # Multiclass with unbalanced classes: 3.x stores one base score per class.
    Xu = rng.normal(size=(1500, 6))
    yu = rng.choice(3, len(Xu), p=[0.6, 0.3, 0.1])
    cases.append(case("multi_base_vector", Xu, yu,
                      {"objective": "multi:softprob", "num_class": 3, "max_depth": 3}, 10))

    # DART: trees carry weights. 3.x saves them in the gbtree model.
    cases.append(case("dart_binary", Xb, yb,
                      {"booster": "dart", "objective": "binary:logistic", "max_depth": 3,
                       "rate_drop": 0.3, "skip_drop": 0.0}, 20))
    cases.append(case("dart_multi", Xi, yi,
                      {"booster": "dart", "objective": "multi:softprob", "num_class": 3,
                       "max_depth": 3, "rate_drop": 0.2, "skip_drop": 0.0}, 10))
    # The same weights in the layout XGBoost 1.x/2.x wrote: a `dart` booster
    # wrapping the gbtree, with `weight_drop` beside it.
    legacy = dict(cases[-2], name="dart_legacy_layout")
    j = json.loads(legacy["model"])
    gbm = j["learner"]["gradient_booster"]["model"]
    weights = gbm.pop("weight_drop")
    j["learner"]["gradient_booster"] = {
        "name": "dart", "weight_drop": weights,
        "gbtree": {"name": "gbtree", "model": gbm}}
    legacy["model"] = json.dumps(j, separators=(",", ":"))
    cases.append(legacy)

    # gblinear: one weight per feature and class plus a bias.
    cases.append(case("gblinear_regression", Xd, yd,
                      {"booster": "gblinear", "objective": "reg:squarederror"}, 20))
    cases.append(case("gblinear_logistic", Xb, yb,
                      {"booster": "gblinear", "objective": "binary:logistic"}, 20))
    cases.append(case("gblinear_multi", Xu, yu,
                      {"booster": "gblinear", "objective": "multi:softprob", "num_class": 3}, 20))

    out = ["// Generated by tools/gen_fixtures.py with xgboost " + xgb.__version__ + ". DO NOT EDIT."]
    for c in cases:
        out.append("\n///|")
        out.append(f"let fixture_{c['name']} : Fixture = {{")
        out.append(f"  name: \"{c['name']}\",")
        out.append("  model: (")
        for line in wrap_json(c["model"]):
            out.append("    #|" + line)
        out.append("  ),")
        for k in ("num_feature", "num_group", "pred_width"):
            out.append(f"  {k}: {c[k]},")
        for k in ("x", "margin", "pred"):
            out.append(f"  {k}: {c[k]},")
        out.append("}")
    with open("xgboost_fixtures_test.mbt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", len(cases), "cases")


if __name__ == "__main__":
    main()
