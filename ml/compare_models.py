"""Compare candidate models with stratified 5-fold CV on the DEV set only.

The test set is never read here. Out-of-fold (OOF) scores are saved so the
next step can choose the alert threshold without touching test data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ml.features import build_preprocessor, make_xy

SEED = 42
DEV_PATH = Path("data/dev.csv")
ARTIFACTS = Path("artifacts")


def candidate_models() -> dict:
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED
        ),
    }


def main() -> None:
    dev = pd.read_csv(DEV_PATH)
    X, y = make_xy(dev)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_table = pd.DataFrame({"row_id": dev["row_id"], "y_true": y})
    rows = []

    for name, estimator in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        oof = np.zeros(len(X))
        fold_ap = []
        for train_idx, val_idx in skf.split(X, y):
            model = clone(pipe)  # fresh, unfitted copy for every fold
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores = model.predict_proba(X.iloc[val_idx])[:, 1]
            oof[val_idx] = scores
            fold_ap.append(average_precision_score(y.iloc[val_idx], scores))

        oof_table[name] = oof
        rows.append(
            {
                "model": name,
                "pr_auc_oof": average_precision_score(y, oof),
                "pr_auc_fold_mean": float(np.mean(fold_ap)),
                "pr_auc_fold_std": float(np.std(fold_ap)),
                "roc_auc_oof": roc_auc_score(y, oof),
            }
        )
        print(f"finished {name}")

    results = pd.DataFrame(rows).sort_values("pr_auc_oof", ascending=False)
    pd.set_option("display.width", 200)
    print("\nDev-set cross-validation results (5-fold, stratified, seed 42):")
    print(results.round(4).to_string(index=False))
    print(f"\nPositive rate in dev: {y.mean():.4f}  (a no-skill PR-AUC is about this)")

    ARTIFACTS.mkdir(exist_ok=True)
    oof_table.to_csv(ARTIFACTS / "oof_scores.csv", index=False)
    with open(ARTIFACTS / "cv_results.json", "w") as f:
        json.dump(results.to_dict(orient="records"), f, indent=2)
    print("Saved artifacts/oof_scores.csv and artifacts/cv_results.json")


if __name__ == "__main__":
    main()