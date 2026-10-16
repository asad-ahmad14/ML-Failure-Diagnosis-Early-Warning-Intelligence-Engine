"""Writes predictions and alerts. Does not commit: the caller owns the transaction."""
import hashlib
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.alert import Alert
from app.models.prediction import Prediction
from app.schemas.predict import SensorReading

ACTIVE_STATUSES = ("open", "acknowledged")


def input_hash(reading: SensorReading) -> str:
    canonical = json.dumps(reading.model_dump(), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def dedup_key(ihash: str, result: dict) -> str:
    raw = f"{ihash}|{result['model_version']}|{result['anomaly_model_version']}"
    return hashlib.sha256(raw.encode()).hexdigest()


def save_result(db: Session, reading: SensorReading, result: dict):
    """Stage a prediction (and an alert when warranted). Returns (prediction, alert, alert_created)."""
    ihash = input_hash(reading)
    prediction = Prediction(
        product_type=reading.product_type,
        air_temperature_k=reading.air_temperature_k,
        process_temperature_k=reading.process_temperature_k,
        rotational_speed_rpm=reading.rotational_speed_rpm,
        torque_nm=reading.torque_nm,
        tool_wear_min=reading.tool_wear_min,
        input_hash=ihash,
        risk_score=result["risk_score"],
        anomaly_score=result["anomaly_score"],
        is_statistically_unusual=result["is_statistically_unusual"],
        model_level=result["model_level"],
        warning_level=result["warning_level"],
        warning_sources=result["warning_sources"],
        evidence=result["evidence"],
        data_quality_notes=result["data_quality_notes"],
        thresholds=result["thresholds"],
        model_version=result["model_version"],
        anomaly_model_version=result["anomaly_model_version"],
    )
    db.add(prediction)
    db.flush()  # assigns prediction.id

    alert, created = None, False
    if result["warning_level"] != "normal":
        key = dedup_key(ihash, result)
        alert = db.scalar(
            select(Alert).where(Alert.dedup_key == key, Alert.status.in_(ACTIVE_STATUSES))
        )
        if alert is None:
            alert = Alert(
                prediction_id=prediction.id,
                level=result["warning_level"],
                sources=result["warning_sources"],
                message=result["message"],
                dedup_key=key,
                model_version=result["model_version"],
            )
            db.add(alert)
            db.flush()
            created = True
    return prediction, alert, created