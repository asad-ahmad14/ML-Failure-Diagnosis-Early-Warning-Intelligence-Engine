from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.alert import Alert
from app.models.prediction import Prediction, utcnow
from app.schemas.records import (
    AlertOut, AlertPage, PredictionOut, PredictionPage, StatusUpdate,
)

router = APIRouter()

WarningLevel = Literal["normal", "watch", "elevated", "high"]
AlertLevel = Literal["watch", "elevated", "high"]
AlertStatus = Literal["open", "acknowledged", "resolved"]

ALLOWED_TRANSITIONS = {
    "open": {"acknowledged", "resolved"},
    "acknowledged": {"resolved"},
    "resolved": set(),
}


@router.get("/predictions", response_model=PredictionPage, tags=["history"])
def list_predictions(
    warning_level: WarningLevel | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(Prediction)
    count_stmt = select(func.count()).select_from(Prediction)
    if warning_level:
        stmt = stmt.where(Prediction.warning_level == warning_level)
        count_stmt = count_stmt.where(Prediction.warning_level == warning_level)
    total = db.scalar(count_stmt)
    rows = db.scalars(stmt.order_by(Prediction.id.desc()).limit(limit).offset(offset)).all()
    return PredictionPage(
        total=total, limit=limit, offset=offset,
        items=[PredictionOut.model_validate(r) for r in rows],
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionOut, tags=["history"])
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return PredictionOut.model_validate(row)


@router.get("/alerts", response_model=AlertPage, tags=["alerts"])
def list_alerts(
    status: AlertStatus | None = None,
    level: AlertLevel | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(Alert)
    count_stmt = select(func.count()).select_from(Alert)
    if status:
        stmt = stmt.where(Alert.status == status)
        count_stmt = count_stmt.where(Alert.status == status)
    if level:
        stmt = stmt.where(Alert.level == level)
        count_stmt = count_stmt.where(Alert.level == level)
    total = db.scalar(count_stmt)
    rows = db.scalars(stmt.order_by(Alert.id.desc()).limit(limit).offset(offset)).all()
    return AlertPage(
        total=total, limit=limit, offset=offset,
        items=[AlertOut.model_validate(r) for r in rows],
    )


@router.get("/alerts/{alert_id}", response_model=AlertOut, tags=["alerts"])
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    row = db.get(Alert, alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return AlertOut.model_validate(row)


@router.patch("/alerts/{alert_id}/status", response_model=AlertOut, tags=["alerts"])
def update_alert_status(alert_id: int, body: StatusUpdate, db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    if body.status not in ALLOWED_TRANSITIONS[alert.status]:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot change an alert from '{alert.status}' to '{body.status}'.",
        )
    now = utcnow()
    alert.status = body.status
    if body.status == "acknowledged":
        alert.acknowledged_at = now
    else:
        alert.resolved_at = now
    alert.updated_at = now
    db.commit()
    return AlertOut.model_validate(alert)