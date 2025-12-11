from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = [] # List of {role: "user"|"model", parts: [{text: "..."}]}

class ChatResponse(BaseModel):
    text: str

class IngestResponse(BaseModel):
    message: str
    symbol: str

class StockDataPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class StockResponse(BaseModel):
    symbol: str
    data: List[StockDataPoint]

class RefreshAllResponse(BaseModel):
    message: str
    symbols: List[str]
    success_count: int
    failed_count: int

# Prediction-related schemas
class TrainModelResponse(BaseModel):
    message: str
    symbol: str
    models_trained: List[str]
    results: dict

class PredictionDataPoint(BaseModel):
    target_date: str
    predicted_price: float
    confidence_lower: float
    confidence_upper: float
    prediction_date: str

class PredictionResponse(BaseModel):
    symbol: str
    predictions: List[PredictionDataPoint]
    days_ahead: int

class ModelInfo(BaseModel):
    type: str
    version: str
    training_date: str
    metrics: dict
    training_samples: int
    is_active: bool

class ModelStatusResponse(BaseModel):
    symbol: str
    models: List[ModelInfo]
    status: str

