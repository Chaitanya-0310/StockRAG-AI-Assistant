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
