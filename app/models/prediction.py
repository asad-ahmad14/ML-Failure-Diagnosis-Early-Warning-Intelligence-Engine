from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:  # only for the editor/type checker; avoids a circular import at runtime
    from app.models.alert import Alert


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    # inputs
    product_type: Mapped[str] = mapped_column(String(1))
    air_temperature_k: Mapped[float] = mapped_column(Float)
    process_temperature_k: Mapped[float] = mapped_column(Float)
    rotational_speed_rpm: Mapped[float] = mapped_column(Float)
    torque_nm: Mapped[float] = mapped_column(Float)
    tool_wear_min: Mapped[float] = mapped_column(Float)
    input_hash: Mapped[str] = mapped_column(String(64), index=True)

    # outputs
    risk_score: Mapped[float] = mapped_column(Float)
    anomaly_score: Mapped[float] = mapped_column(Float)
    is_statistically_unusual: Mapped[bool] = mapped_column(Boolean)
    model_level: Mapped[str] = mapped_column(String(10))
    warning_level: Mapped[str] = mapped_column(String(10), index=True)
    warning_sources: Mapped[list] = mapped_column(JSON)
    evidence: Mapped[list] = mapped_column(JSON)
    data_quality_notes: Mapped[list] = mapped_column(JSON)
    thresholds: Mapped[dict] = mapped_column(JSON)

    # provenance
    model_version: Mapped[str] = mapped_column(String(50))
    anomaly_model_version: Mapped[str] = mapped_column(String(50))

    alerts: Mapped[list["Alert"]] = relationship(back_populates="prediction")