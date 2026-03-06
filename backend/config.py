from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/stockrag_db"
    db_echo: bool = False

    # Google AI
    google_api_key: str = ""

    # API
    cors_origins: List[str] = ["http://localhost:3000"]
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # ML hyperparameters
    lstm_epochs: int = 50
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_lookback: int = 60
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 5
    xgboost_learning_rate: float = 0.1
    prophet_changepoint_prior_scale: float = 0.05

    # Default symbols
    default_symbols: List[str] = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    # Embedding
    embedding_model: str = "models/gemini-embedding-001"
    generation_model: str = "gemini-2.5-flash"
    embedding_batch_size: int = 100

    # Cache
    cache_ttl_seconds: int = 300  # 5 minutes

    model_config = {
        "env_file": str(Path(__file__).resolve().parent / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
