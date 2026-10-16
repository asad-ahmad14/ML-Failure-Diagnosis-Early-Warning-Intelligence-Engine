"""Builds the /model response from the saved artifact files. No metric is typed in by hand."""
import json
from pathlib import Path

LIMITATIONS = [
    "Trained on the synthetic AI4I 2020 dataset; results say nothing about real machines.",
    "risk_score is a ranking score from a class-weighted model, not a probability of failure.",
    "Tool-wear failures are largely missed: UCI describes them as occurring at a randomly "
    "chosen wear time.",
    "Rows have no timestamps or machine ids, so this is a per-reading screen, not a forecast.",
    "The anomaly detector flags unusual readings; unusual does not mean failing.",
]


def _load(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def build_model_info(meta: dict, anomaly_meta: dict, artifacts_dir: Path) -> dict:
    return {
        "failure_risk_model": {
            "version": meta["model_version"],
            "algorithm": meta["model_name"],
            "trained_at": meta["trained_at"],
            "features": meta["features"],
            "n_train_rows": meta["n_train_rows"],
            "n_train_failures": meta["n_train_failures"],
            "thresholds": meta["thresholds"],
            "threshold_rule": meta["threshold_rule"],
            "dev_cross_validation": _load(artifacts_dir / "cv_results.json"),
            "held_out_test_metrics": _load(artifacts_dir / "test_metrics.json"),
            "global_importance": _load(artifacts_dir / "global_importance.json"),
        },
        "anomaly_model": {
            "version": anomaly_meta["model_version"],
            "algorithm": "IsolationForest",
            "features": anomaly_meta["features"],
            "flag_percentile": anomaly_meta["flag_percentile"],
            "flag_threshold": anomaly_meta["flag_threshold"],
            "rule": anomaly_meta["rule"],
        },
        "metric_sources": [
            "artifacts/cv_results.json",
            "artifacts/test_metrics.json",
            "artifacts/global_importance.json",
        ],
        "limitations": LIMITATIONS,
    }