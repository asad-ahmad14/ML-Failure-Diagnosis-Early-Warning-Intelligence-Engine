from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_model_service
from app.db.session import get_db
from app.services.model_service import ModelService

router = APIRouter(tags=["health"])


@router.get("/health")
def health(service: ModelService = Depends(get_model_service), db: Session = Depends(get_db)) -> dict:
    try:
        db.execute(select(1))
        database = "ok"
    except Exception:  # never leak connection details to the caller
        database = "unavailable"
    return {
        "status": "ok" if database == "ok" else "degraded",
        "model_loaded": True,
        "database": database,
        "model_version": service.meta["model_version"],
        "anomaly_model_version": service.anomaly_meta["model_version"],
    }