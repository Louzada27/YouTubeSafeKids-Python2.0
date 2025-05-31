from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Dict, ClassVar

class Settings(BaseSettings):
    PROJECT_NAME: str = "YouTube Safe Kids"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    YOUTUBE_API_KEY: str
    PORT: int = 10000
    HOST: str = "0.0.0.0"
    RELOAD: bool = True
    HUGGINGFACE_HUB_TOKEN: str
    MODEL_PATH: str = "./ModeloToxidade"

    FILTER_WEIGHTS: ClassVar[Dict[str, float]] = {
        "duration": 1.0,
        "engagement": 1.0,
        "age_rating": 1.0,
        "interactivity": 1.0,
        "language": 1.0,
        "toxicity": 1.0,
        "sentiment": 1.0,
        "educational": 1.0,
        "diversity": 1.0,
        "sensitive": 1.0
    }

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()
