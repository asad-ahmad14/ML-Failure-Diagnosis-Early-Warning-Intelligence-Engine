"""Request and response schemas. Bounds below are generous sanity limits, not the
training range. Readings outside the training range are accepted but get a note."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SensorReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_type: Literal["L", "M", "H"] = Field(description="Product quality variant")
    air_temperature_k: float = Field(ge=250, le=350, allow_inf_nan=False)
    process_temperature_k: float = Field(ge=250, le=400, allow_inf_nan=False)
    rotational_speed_rpm: float = Field(gt=0, le=10000, allow_inf_nan=False)
    torque_nm: float = Field(ge=0, le=500, allow_inf_nan=False)
    tool_wear_min: float = Field(ge=0, le=1000, allow_inf_nan=False)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    readings: list[SensorReading] = Field(min_length=1)


class FeatureEvidence(BaseModel):
    feature: str
    value: float | str
    contribution: float
    direction: str


class PredictionResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    risk_score: float
    model_level: str
    anomaly_score: float
    is_statistically_unusual: bool
    warning_level: Literal["normal", "watch", "elevated", "high"]
    warning_sources: list[str]
    message: str
    evidence: list[FeatureEvidence]
    data_quality_notes: list[str]
    model_version: str
    anomaly_model_version: str
    thresholds: dict[str, float]
    disclaimer: str

    prediction_id: int
    alert_id: int | None
    alert_created: bool


class BatchResponse(BaseModel):
    count: int
    results: list[PredictionResult]