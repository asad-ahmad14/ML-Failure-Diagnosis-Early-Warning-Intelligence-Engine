from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    @field_validator(
        "created_at", "updated_at", "acknowledged_at", "resolved_at",
        mode="after", check_fields=False,
    )
    @classmethod
    def _as_utc(cls, v):
        # SQLite drops tzinfo on storage; values are always written as UTC.
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class PredictionOut(_Out):
    id: int
    created_at: datetime
    product_type: str
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    risk_score: float
    anomaly_score: float
    is_statistically_unusual: bool
    model_level: str
    warning_level: str
    warning_sources: list[str]
    evidence: list[dict]
    data_quality_notes: list[str]
    thresholds: dict[str, float]
    model_version: str
    anomaly_model_version: str


class AlertOut(_Out):
    id: int
    prediction_id: int
    created_at: datetime
    updated_at: datetime
    level: str
    status: str
    sources: list[str]
    message: str
    model_version: str
    acknowledged_at: datetime | None
    resolved_at: datetime | None


class PredictionPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PredictionOut]


class AlertPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AlertOut]


class StatusUpdate(BaseModel):
    status: Literal["acknowledged", "resolved"]