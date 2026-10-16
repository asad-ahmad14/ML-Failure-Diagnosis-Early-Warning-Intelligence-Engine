from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_model_service
from app.core.config import get_settings
from app.db.session import get_db
from app.models.alert import Alert
from app.models.prediction import Prediction
from app.services.model_info import build_model_info
from app.services.model_service import ModelService
from app.services.monitoring import build_report

router = APIRouter(tags=["monitoring"])

NOTES = [
    "Drift compares the inputs sent to this API with the training data. It says nothing "
    "about model accuracy: incoming readings carry no failure labels here.",
    "PSI bands (under 0.1 stable, 0.1 to 0.25 moderate, over 0.25 significant) are "
    "conventional rules of thumb, not statistical tests.",
]


@router.get("/model")
def model_info(service: ModelService = Depends(get_model_service)) -> dict:
    return build_model_info(service.meta, service.anomaly_meta, get_settings().artifacts_dir)


@router.get("/monitoring/report")
def monitoring_report(
    window: int = Query(1000, ge=1, le=10000, description="Most recent predictions to include"),
    service: ModelService = Depends(get_model_service),
    db: Session = Depends(get_db),
) -> dict:
    rows = db.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(window)).all()
    report = build_report(rows, service.reference)
    counts = dict(db.execute(select(Alert.status, func.count()).group_by(Alert.status)).all())
    report["alerts_by_status"] = {
        s: int(counts.get(s, 0)) for s in ("open", "acknowledged", "resolved")
    }
    report["window"] = window
    report["notes"] = NOTES
    return report