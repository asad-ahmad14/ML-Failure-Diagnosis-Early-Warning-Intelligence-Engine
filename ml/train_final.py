"""Train the final model on the full DEV set, choose alert thresholds from
out-of-fold (OOF) scores, run error analysis, and save one artifact.
The TEST set is not read here.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

from ml.compare_models import SEED, candidate_models
from ml.features import FEATURES, MODE_COLUMNS, TARGET, build_preprocessor, make_xy

MODEL_NAME = "random_forest"
MODEL_VERSION = "rf_v1"
DEV_PATH = Path("data/dev.csv")
ARTIFACTS = Path("artifacts")

THRESHOLD_RULE = (
    "watch: highest cutoff with OOF recall >= 0.90; "
    "elevated: highest cutoff with OOF recall >= 0.80; "
   "high: lowest cutoff with OOF precision >= 0.95 (never below elevated)"
)


def metrics_at(y: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    pred = scores >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {
        "threshold": round(float(thr), 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4),
        "tp": tp, "fp": fp, "fn": fn,
        "flagged_per_1000": round((tp + fp) / len(y) * 1000, 1),
    }


def pick_thresholds(y: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, thr = precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]  # align with thr
    watch = float(thr[r >= 0.90].max())
    elevated = float(thr[r >= 0.80].max())
    ok = thr[p >= 0.95]
    high = float(ok.min()) if len(ok) else elevated
    high = max(high, elevated)
    return {"watch": watch, "elevated": elevated, "high": high}


def main() -> None:
    dev = pd.read_csv(DEV_PATH)
    X, y_series = make_xy(dev)
    y = y_series.to_numpy()

    oof = pd.read_csv(ARTIFACTS / "oof_scores.csv")
    assert oof["row_id"].equals(dev["row_id"]), "OOF rows not aligned with dev rows"
    scores = oof[MODEL_NAME].to_numpy()

    thresholds = pick_thresholds(y, scores)
    op_points = {name: metrics_at(y, scores, t) for name, t in thresholds.items()}
    print("Operating points on OOF dev scores (n=%d, failures=%d):" % (len(y), y.sum()))
    print(pd.DataFrame(op_points).T.to_string())

    # Error analysis at the 'elevated' cutoff, by labelled failure mode
    pred = scores >= thresholds["elevated"]
    print("\nRecall by failure mode at the 'elevated' cutoff (OOF):")
    for mode in MODE_COLUMNS:
        mask = dev[mode].to_numpy() == 1
        caught = int(pred[mask].sum())
        print(f"  {mode}: {caught}/{int(mask.sum())} caught")
    no_mode = (y == 1) & (dev[MODE_COLUMNS].sum(axis=1).to_numpy() == 0)
    print(f"  failure with no mode label: {int(pred[no_mode].sum())}/{int(no_mode.sum())} caught")

    # Final fit on ALL dev rows. Preprocessing and model travel together.
    pipe = Pipeline([("prep", build_preprocessor()), ("model", candidate_models()[MODEL_NAME])])
    pipe.fit(X, y)

    meta = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "features": FEATURES,
        "n_train_rows": int(len(y)),
        "n_train_failures": int(y.sum()),
        "thresholds": thresholds,
        "threshold_rule": THRESHOLD_RULE,
        "oof_operating_points": op_points,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "meta": meta}, ARTIFACTS / "model.joblib")
    with open(ARTIFACTS / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Reload check: the saved artifact must reproduce the in-memory predictions
    loaded = joblib.load(ARTIFACTS / "model.joblib")
    same = np.allclose(
        loaded["pipeline"].predict_proba(X.head(200)), pipe.predict_proba(X.head(200))
    )
    print("\nSaved artifacts/model.joblib and artifacts/model_metadata.json")
    print("Reload gives identical predictions on 200 rows:", same)


if __name__ == "__main__":
    main()