"""One-time held-out evaluation. Run ONLY after thresholds are frozen.
Nothing is tuned from this output. Uses a bootstrap to show uncertainty,
because the test set has only ~68 failures."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.features import MODE_COLUMNS, TARGET, make_xy
from ml.train_final import metrics_at

ARTIFACTS = Path("artifacts")
N_BOOT = 1000
SEED = 42


def ci(values) -> list:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]


def bootstrap(y, scores, thresholds):
    rng = np.random.default_rng(SEED)
    out = {name: {"precision": [], "recall": []} for name in thresholds}
    out["pr_auc"] = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], scores[idx]
        if yb.sum() == 0:
            continue
        out["pr_auc"].append(average_precision_score(yb, sb))
        for name, t in thresholds.items():
            pred = sb >= t
            tp = int((pred & (yb == 1)).sum())
            fp = int((pred & (yb == 0)).sum())
            fn = int((~pred & (yb == 1)).sum())
            if tp + fp:
                out[name]["precision"].append(tp / (tp + fp))
            out[name]["recall"].append(tp / (tp + fn))
    return out


def main() -> None:
    art = joblib.load(ARTIFACTS / "model.joblib")
    pipe, meta = art["pipeline"], art["meta"]
    thresholds = meta["thresholds"]

    dev = pd.read_csv("data/dev.csv")
    test = pd.read_csv("data/test.csv")
    assert set(dev["row_id"]).isdisjoint(set(test["row_id"])), "dev/test overlap!"

    X, y_series = make_xy(test)
    y = y_series.to_numpy()
    scores = pipe.predict_proba(X)[:, 1]

    print(f"Model {meta['model_version']} | test rows={len(y)} failures={int(y.sum())}")
    print("Frozen thresholds:", {k: round(v, 4) for k, v in thresholds.items()})
    print(f"\nPR-AUC : {average_precision_score(y, scores):.4f}  (no-skill = {y.mean():.4f})")
    print(f"ROC-AUC: {roc_auc_score(y, scores):.4f}")
    print(f"Accuracy of predicting 'no failure' for every row: {1 - y.mean():.4f}")

    boot = bootstrap(y, scores, thresholds)
    print("PR-AUC 95% bootstrap interval:", ci(boot["pr_auc"]))

    result = {"pr_auc": float(average_precision_score(y, scores)),
              "roc_auc": float(roc_auc_score(y, scores)), "levels": {}}
    print("\nTest operating points vs dev out-of-fold (OOF):")
    for name, t in thresholds.items():
        m = metrics_at(y, scores, t)
        dev_m = meta["oof_operating_points"][name]
        p_ci = ci(boot[name]["precision"])
        r_ci = ci(boot[name]["recall"])
        print(f"\n[{name}] cutoff={t:.4f}")
        print(f"  test: precision={m['precision']} {p_ci} recall={m['recall']} {r_ci} "
              f"tp={m['tp']} fp={m['fp']} fn={m['fn']} flagged/1000={m['flagged_per_1000']}")
        print(f"  dev OOF: precision={dev_m['precision']} recall={dev_m['recall']} "
              f"flagged/1000={dev_m['flagged_per_1000']}")
        result["levels"][name] = {**m, "precision_ci95": p_ci, "recall_ci95": r_ci}

    pred = scores >= thresholds["elevated"]
    print("\nCaught at 'elevated' (real failures only, by label):")
    for mode in MODE_COLUMNS:
        mask = (test[mode].to_numpy() == 1) & (y == 1)
        print(f"  {mode}: {int(pred[mask].sum())}/{int(mask.sum())}")
    no_mode = (y == 1) & (test[MODE_COLUMNS].sum(axis=1).to_numpy() == 0)
    print(f"  no mode label: {int(pred[no_mode].sum())}/{int(no_mode.sum())}")

    with open(ARTIFACTS / "test_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved artifacts/test_metrics.json")


if __name__ == "__main__":
    main()