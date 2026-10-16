"""Loads the saved artifacts once and turns readings into scored, explained results."""
import json
from pathlib import Path

import joblib
import pandas as pd

from app.schemas.predict import SensorReading
from app.services.warning_engine import evaluate_warning
from ml.explain import Explainer
from ml.features import NUMERIC, RAW_NUMERIC, make_xy

DISCLAIMER = (
    "Screening signal from a statistical model trained on a synthetic dataset. "
    "Not confirmation of mechanical failure. risk_score is a ranking score from a "
    "class-weighted model, not a probability of failure."
)


def readings_to_frame(readings: list[SensorReading]) -> pd.DataFrame:
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
            for r in readings
        ]
    )


class ModelService:
    def __init__(self, artifacts_dir: Path):
        needed = ["model.joblib", "anomaly.joblib", "reference_stats.json"]
        missing = [n for n in needed if not (artifacts_dir / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifacts in {artifacts_dir}: {missing}. "
                "Run the training scripts first."
            )
        art = joblib.load(artifacts_dir / "model.joblib")
        self.pipeline, self.meta = art["pipeline"], art["meta"]
        anomaly = joblib.load(artifacts_dir / "anomaly.joblib")
        self.anomaly_model, self.anomaly_meta = anomaly["model"], anomaly["meta"]
        with open(artifacts_dir / "reference_stats.json") as f:
            self.reference = json.load(f)
        self.explainer = Explainer(self.pipeline)
        self.thresholds = {k: float(v) for k, v in self.meta["thresholds"].items()}
        self.anomaly_threshold = float(self.anomaly_meta["flag_threshold"])

    def _range_notes(self, raw_row: pd.Series) -> list[str]:
        notes = []
        for col in RAW_NUMERIC:
            lo, hi = self.reference["numeric"][col]["min"], self.reference["numeric"][col]["max"]
            v = float(raw_row[col])
            if v < lo or v > hi:
                notes.append(
                    f"{col} = {v:g} is outside the training range [{lo:g}, {hi:g}]; "
                    "the model is extrapolating."
                )
        return notes

    def predict(self, readings: list[SensorReading]) -> list[dict]:
        df = readings_to_frame(readings)
        X, _ = make_xy(df)
        risk = self.pipeline.predict_proba(X)[:, 1]
        anomaly = -self.anomaly_model.score_samples(X[NUMERIC])
        results = []
        for i in range(len(X)):
            w = evaluate_warning(
                float(risk[i]), self.thresholds, float(anomaly[i]), self.anomaly_threshold
            )
            expl = self.explainer.explain_row(X.iloc[[i]])
            results.append(
                {
                    "risk_score": float(risk[i]),
                    "model_level": w["model_level"],
                    "anomaly_score": float(anomaly[i]),
                    "is_statistically_unusual": w["unusual"],
                    "warning_level": w["level"],
                    "warning_sources": w["sources"],
                    "message": w["message"],
                    "evidence": expl["top_features"],
                    "data_quality_notes": self._range_notes(df.iloc[i]),
                    "model_version": self.meta["model_version"],
                    "anomaly_model_version": self.anomaly_meta["model_version"],
                    "thresholds": self.thresholds,
                    "disclaimer": DISCLAIMER,
                }
            )
        return results