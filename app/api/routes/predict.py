from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_model_service
from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.predict import BatchRequest, BatchResponse, PredictionResult, SensorReading
from app.services.model_service import ModelService
from app.services.persistence import save_result

router = APIRouter(tags=["prediction"])


def _with_ids(result: dict, prediction, alert, created: bool) -> dict:
    return result | {
        "prediction_id": prediction.id,
        "alert_id": alert.id if alert else None,
        "alert_created": created,
    }


@router.post("/predict", response_model=PredictionResult)
def predict(
    reading: SensorReading,
    service: ModelService = Depends(get_model_service),
    db: Session = Depends(get_db),
):
    result = service.predict([reading])[0]
    prediction, alert, created = save_result(db, reading, result)
    db.commit()
    return _with_ids(result, prediction, alert, created)


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(
    body: BatchRequest,
    service: ModelService = Depends(get_model_service),
    db: Session = Depends(get_db),
):
    limit = get_settings().max_batch_size
    if len(body.readings) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Batch of {len(body.readings)} exceeds the limit of {limit} readings.",
        )
    results = service.predict(body.readings)
    out = []
    for reading, result in zip(body.readings, results):
        prediction, alert, created = save_result(db, reading, result)
        out.append(_with_ids(result, prediction, alert, created))
    db.commit()  # one transaction: all rows of the batch, or none
    return {"count": len(out), "results": out}