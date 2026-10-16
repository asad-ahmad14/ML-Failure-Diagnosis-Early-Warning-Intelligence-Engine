from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health, monitoring, predict, records
from app.core.config import get_settings
from app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.model_service = ModelService(settings.artifacts_dir)  # fail fast if missing
    yield


app = FastAPI(
    title=get_settings().app_name,
    version="0.3.0",
    description="Anomaly detection and failure-risk screening for sensor readings. "
                "Trained on the synthetic AI4I 2020 dataset.",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(records.router)
app.include_router(monitoring.router)