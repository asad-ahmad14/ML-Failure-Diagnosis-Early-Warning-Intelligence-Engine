"""Input-drift and usage statistics. Pure functions over stored predictions."""
import numpy as np
import pandas as pd

from ml.features import NUMERIC, add_engineered_features

MIN_SAMPLES = 30
EPS = 1e-4


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate_shift"
    return "significant_shift"


def feature_drift(values, ref_col: dict) -> dict:
    values = np.asarray(values, dtype=float)
    edges = np.asarray(ref_col["bin_edges"], dtype=float)
    idx = np.digitize(values, edges)  # same binning as the reference stats
    actual = np.bincount(idx, minlength=len(edges) + 1) / len(values)
    value = psi(ref_col["bin_expected"], actual)
    return {
        "psi": round(value, 4),
        "status": psi_label(value),
        "reference_mean": round(ref_col["mean"], 4),
        "current_mean": round(float(values.mean()), 4),
    }


def drift_report(df: pd.DataFrame, reference: dict) -> dict:
    n = len(df)
    if n < MIN_SAMPLES:
        return {"status": "insufficient_data", "n": n, "min_samples": MIN_SAMPLES}
    enriched = add_engineered_features(df)
    features = {
        col: feature_drift(enriched[col].to_numpy(), reference["numeric"][col])
        for col in NUMERIC
    }
    current_share = df["Type"].value_counts(normalize=True).to_dict()
    type_share = {
        t: {
            "reference": round(ref_share, 4),
            "current": round(float(current_share.get(t, 0.0)), 4),
        }
        for t, ref_share in reference["type_share"].items()
    }
    worst = max(features.values(), key=lambda f: f["psi"])
    return {
        "status": "computed",
        "n": n,
        "features": features,
        "type_share": type_share,
        "max_psi": worst["psi"],
        "overall": worst["status"],
    }


def rows_to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Type": r.product_type,
                "Air temperature": r.air_temperature_k,
                "Process temperature": r.process_temperature_k,
                "Rotational speed": r.rotational_speed_rpm,
                "Torque": r.torque_nm,
                "Tool wear": r.tool_wear_min,
            }
            for r in rows
        ]
    )


def build_report(rows, reference: dict) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "n_predictions": 0,
            "warning_level_counts": {},
            "drift": {"status": "insufficient_data", "n": 0, "min_samples": MIN_SAMPLES},
        }
    scores = np.array([r.risk_score for r in rows])
    levels = pd.Series([r.warning_level for r in rows]).value_counts().to_dict()
    return {
        "n_predictions": n,
        "oldest_id_in_window": min(r.id for r in rows),
        "newest_id_in_window": max(r.id for r in rows),
        "warning_level_counts": {k: int(v) for k, v in levels.items()},
        "share_statistically_unusual": round(
            float(np.mean([r.is_statistically_unusual for r in rows])), 4
        ),
        "share_with_range_notes": round(
            float(np.mean([bool(r.data_quality_notes) for r in rows])), 4
        ),
        "risk_score": {
            "mean": round(float(scores.mean()), 4),
            "p50": round(float(np.percentile(scores, 50)), 4),
            "p90": round(float(np.percentile(scores, 90)), 4),
            "p99": round(float(np.percentile(scores, 99)), 4),
        },
        "drift": drift_report(rows_to_frame(rows), reference),
    }