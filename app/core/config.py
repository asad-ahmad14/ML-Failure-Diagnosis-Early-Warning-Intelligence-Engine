"""Application settings, overridable with MFE_* environment variables or a .env file."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MFE_", extra="ignore")

    app_name: str = "ML Failure Diagnosis & Early-Warning Engine"
    artifacts_dir: Path = Path("artifacts")
    max_batch_size: int = 100
    database_url: str = "sqlite:///./app.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()