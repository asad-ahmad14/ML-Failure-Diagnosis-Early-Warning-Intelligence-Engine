"""Local and global explanations for the Random Forest using SHAP TreeExplainer.

SHAP values describe how the MODEL's score changes with each feature. They are
model-derived evidence, not proof of a physical cause. The three one-hot Type
columns are summed back into a single 'Type' contribution.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from ml.features import FEATURES, NUMERIC, make_xy

ARTIFACTS = Path("artifacts")
N_SAMPLE = 400  # rows used for the additivity check and global importance


def _positive_class(values) -> np.ndarray:
    """Failure-class SHAP array, whichever layout this shap version returns."""
    if isinstance(values, list):  # older shap: [class0, class1]
        return np.asarray(values[1])
    arr = np.asarray(values)
    if arr.ndim == 3:  # newer shap: (rows, features, classes)
        return arr[:, :, 1]
    return arr


def _positive_base(expected) -> float:
    arr = np.atleast_1d(np.asarray(expected, dtype=float))
    return float(arr[1]) if arr.size > 1 else float(arr[0])


class Explainer:
    def __init__(self, pipeline):
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["model"]
        self.tree = shap.TreeExplainer(self.model)
        self.base_value = _positive_base(self.tree.expected_value)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to the failure score (in score units)."""
        Z = self.prep.transform(X)
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        sv = _positive_class(self.tree.shap_values(Z))
        n_num = len(NUMERIC)  # transformed layout: 7 numeric, then Type_L/M/H
        out = pd.DataFrame(sv[:, :n_num], columns=NUMERIC, index=X.index)
        out["Type"] = sv[:, n_num:].sum(axis=1)
        return out[FEATURES]

    def explain_row(self, X_row: pd.DataFrame, top_k: int = 3) -> dict:
        contrib = self.contributions(X_row).iloc[0]
        raw = X_row.iloc[0]
        ranked = contrib.abs().sort_values(ascending=False).index[:top_k]
        top = []
        for f in ranked:
            v = raw[f]
            v = v.item() if hasattr(v, "item") else v
            top.append(
                {
                    "feature": f,
                    "value": v,
                    "contribution": float(contrib[f]),
                    "direction": "raises risk score" if contrib[f] > 0 else "lowers risk score",
                }
            )
        return {
            "base_value": self.base_value,
            "model_score": float(self.base_value + contrib.sum()),
            "top_features": top,
        }


def build_reference_stats(X: pd.DataFrame) -> dict:
    ref = {
        "n_rows": int(len(X)),
        "type_share": X["Type"].value_counts(normalize=True).round(6).to_dict(),
        "numeric": {},
    }
    for col in NUMERIC:
        s = X[col]
        deciles = s.quantile(np.linspace(0, 1, 11)).to_numpy()
        # interior decile edges, de-duplicated (integer columns can have ties)
        bin_edges = np.unique(deciles[1:-1])
        idx = np.digitize(s.to_numpy(), bin_edges)
        expected = np.bincount(idx, minlength=len(bin_edges) + 1) / len(s)
        ref["numeric"][col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "decile_edges": [float(v) for v in deciles],
            "bin_edges": [float(v) for v in bin_edges],
            "bin_expected": [float(v) for v in expected],
        }
    return ref


def main() -> None:
    art = joblib.load(ARTIFACTS / "model.joblib")
    pipe = art["pipeline"]
    dev = pd.read_csv("data/dev.csv")
    X, y = make_xy(dev)
    ex = Explainer(pipe)

    sample = X.sample(N_SAMPLE, random_state=42)
    contrib = ex.contributions(sample)

    recon = ex.base_value + contrib.sum(axis=1).to_numpy()
    actual = pipe.predict_proba(sample)[:, 1]
    print(f"Base value (average model score): {ex.base_value:.4f}")
    print(f"Additivity check on {N_SAMPLE} rows, max |base + sum(contributions) - predict_proba|:",
          f"{float(np.abs(recon - actual).max()):.2e}")

    importance = contrib.abs().mean().sort_values(ascending=False)
    print(f"\nGlobal importance (mean |SHAP|, {N_SAMPLE} dev rows):")
    print(importance.round(4).to_string())

    # Demonstration rows from dev. Scores are in-sample, so they look more
    # confident than they would on unseen data.
    fail_i = int(np.where(y.to_numpy() == 1)[0][0])
    norm_i = int(np.where(y.to_numpy() == 0)[0][0])
    for label, i in [("a failure row", fail_i), ("a normal row", norm_i)]:
        e = ex.explain_row(X.iloc[[i]])
        print(f"\nExample: {label} (row_id {int(dev['row_id'].iloc[i])}, true label {int(y.iloc[i])})")
        print(f"  model score {e['model_score']:.3f} (average score {e['base_value']:.3f})")
        for t in e["top_features"]:
            print(f"  {t['feature']} = {t['value']}: {t['contribution']:+.3f} ({t['direction']})")

    with open(ARTIFACTS / "global_importance.json", "w") as f:
        json.dump({"method": "mean |SHAP| on dev sample", "n_rows": N_SAMPLE,
                   "importance": importance.round(6).to_dict()}, f, indent=2)
    with open(ARTIFACTS / "reference_stats.json", "w") as f:
        json.dump(build_reference_stats(X), f, indent=2)
    print("\nSaved artifacts/global_importance.json and artifacts/reference_stats.json")


if __name__ == "__main__":
    main()