from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.services.db_service import init_db, get_db
from backend.services.etl_service import ingest_stock_data
from backend.services.rag_service import generate_rag_response
from backend.models.schemas import (
    ChatRequest, ChatResponse, IngestResponse, StockResponse, RefreshAllResponse,
    TrainModelResponse, PredictionResponse, ModelStatusResponse
)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ingest/{symbol}", response_model=IngestResponse)
async def ingest_endpoint(symbol: str, db: AsyncSession = Depends(get_db)):
    try:
        await ingest_stock_data(symbol.upper(), db)
        return {"message": "Ingestion successful", "symbol": symbol.upper()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    try:
        response_text = await generate_rag_response(request.query, db)
        return {"text": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock/{symbol}", response_model=StockResponse)
async def get_stock_endpoint(symbol: str, db: AsyncSession = Depends(get_db)):
    try:
        # Check if data exists in DB, if not fetch and ingest
        # For simplicity, let's just fetch fresh data using our service for now
        # In a real app, we would query the DB
        from backend.services.stock_service import fetch_stock_history
        
        df = fetch_stock_history(symbol.upper())
        
        data_points = []
        for _, row in df.iterrows():
             # Handle date conversion safely
            date_str = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
            
            data_points.append({
                "date": date_str,
                "open": row['Open'],
                "high": row['High'],
                "low": row['Low'],
                "close": row['Close'],
                "volume": int(row['Volume'])
            })
            
        return {"symbol": symbol.upper(), "data": data_points}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh-all", response_model=RefreshAllResponse)
async def refresh_all_endpoint(db: AsyncSession = Depends(get_db)):
    """
    Refresh data for all tracked symbols by fetching latest data and ingesting into DB
    """
    try:
        # Default symbols to refresh
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        success_count = 0
        failed_count = 0
        
        for symbol in symbols:
            try:
                await ingest_stock_data(symbol, db)
                success_count += 1
            except Exception as e:
                print(f"Failed to ingest {symbol}: {str(e)}")
                failed_count += 1
        
        return {
            "message": f"Refresh completed: {success_count} successful, {failed_count} failed",
            "symbols": symbols,
            "success_count": success_count,
            "failed_count": failed_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Prediction endpoints
@app.post("/train/{symbol}", response_model=TrainModelResponse)
async def train_model_endpoint(symbol: str, db: AsyncSession = Depends(get_db)):
    """Train ML models for stock price prediction"""
    try:
        from backend.services.ml_service import train_all_models
        
        results = await train_all_models(symbol.upper(), db)
        
        models_trained = [model_type for model_type in results.keys() if 'error' not in results[model_type]]
        
        return {
            "message": f"Training completed for {symbol.upper()}",
            "symbol": symbol.upper(),
            "models_trained": models_trained,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{symbol}", response_model=PredictionResponse)
async def get_predictions_endpoint(
    symbol: str, 
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """Get price predictions for a stock"""
    try:
        from backend.services.prediction_service import generate_predictions, get_predictions_for_symbol
        
        # Check if predictions exist and are recent
        existing_preds = await get_predictions_for_symbol(symbol.upper(), db, days)
        
        if not existing_preds:
            # Generate new predictions
            await generate_predictions(symbol.upper(), db, days)
            existing_preds = await get_predictions_for_symbol(symbol.upper(), db, days)
        
        return {
            "symbol": symbol.upper(),
            "predictions": existing_preds,
            "days_ahead": len(existing_preds)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{symbol}", response_model=ModelStatusResponse)
async def get_model_status_endpoint(symbol: str, db: AsyncSession = Depends(get_db)):
    """Get model training status and performance metrics"""
    try:
        from backend.services.prediction_service import get_model_status
        
        status = await get_model_status(symbol.upper(), db)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "StockRAG API is running"}

