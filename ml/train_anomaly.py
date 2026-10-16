"""Unsupervised anomaly detector (Isolation Forest). Failure labels are NOT used
for fitting. They are used afterwards only as a proxy sanity check on DEV data."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.features import NUMERIC, make_xy

SEED = 42
ARTIFACTS = Path("artifacts")
FLAG_PERCENTILE = 99.0
MODEL_VERSION = "iforest_v1"


def main() -> None:
    dev = pd.read_csv("data/dev.csv")
    X, y_series = make_xy(dev)
    y = y_series.to_numpy()
    Xn = X[NUMERIC]

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=-1)
    iso.fit(Xn)  # no labels used

    scores = -iso.score_samples(Xn)  # higher = more unusual
    threshold = float(np.percentile(scores, FLAG_PERCENTILE))
    flagged = scores >= threshold

    print(f"Anomaly score range on dev: {scores.min():.4f} to {scores.max():.4f}")
    print(f"Flag threshold (p{FLAG_PERCENTILE:g} of dev scores): {threshold:.4f}")
    print(f"Flagged rows: {int(flagged.sum())} of {len(y)}")
    print("\nPROXY check against failure labels (dev, in-sample; labels were not used to fit):")
    print(f"  ROC-AUC of anomaly score vs failure: {roc_auc_score(y, scores):.4f}")
    print(f"  PR-AUC  of anomaly score vs failure: {average_precision_score(y, scores):.4f}  (no-skill = {y.mean():.4f})")
    print(f"  Failure rate among flagged rows: {y[flagged].mean():.4f}")
    print(f"  Failure rate among unflagged rows: {y[~flagged].mean():.4f}")
    print(f"  Share of all failures that are flagged: {y[flagged].sum() / y.sum():.4f}")

    meta = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "features": NUMERIC,
        "flag_percentile": FLAG_PERCENTILE,
        "flag_threshold": threshold,
        "n_train_rows": int(len(y)),
        "rule": "unusual if score >= p99 of dev scores; score = -score_samples",
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"model": iso, "meta": meta}, ARTIFACTS / "anomaly.joblib")
    with open(ARTIFACTS / "anomaly_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    loaded = joblib.load(ARTIFACTS / "anomaly.joblib")["model"]
    same = np.allclose(loaded.score_samples(Xn.head(200)), iso.score_samples(Xn.head(200)))
    print("\nSaved artifacts/anomaly.joblib and artifacts/anomaly_metadata.json")
    print("Reload gives identical scores on 200 rows:", same)


if __name__ == "__main__":
    main()