import logging
import re
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from backend.services.db_service import init_db, get_db
from backend.services.etl_service import ingest_stock_data
from backend.services.rag_service import generate_rag_response
from backend.models.models import StockPrice, RagDocument, MLModel
from backend.models.schemas import (
    ChatRequest, ChatResponse, IngestResponse, StockResponse, RefreshAllResponse,
    TrainModelResponse, PredictionResponse, ModelStatusResponse
)
from backend.config import settings
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.logging_config import setup_logging
    setup_logging(json_format=False, level="INFO")
    await init_db()
    yield


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ticker format validation
TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}$')


def validate_symbol(symbol: str) -> str:
    """Validate and normalize ticker symbol."""
    symbol = symbol.upper().strip()
    if not TICKER_PATTERN.match(symbol):
        raise HTTPException(status_code=400, detail=f"Invalid ticker symbol format: {symbol}")
    return symbol


@app.post("/ingest/{symbol}", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest_endpoint(request: Request, symbol: str, db: AsyncSession = Depends(get_db)):
    symbol = validate_symbol(symbol)
    try:
        await ingest_stock_data(symbol, db)
        return {"message": "Ingestion successful", "symbol": symbol}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest, db: AsyncSession = Depends(get_db)):
    try:
        response_text = await generate_rag_response(chat_request.query, db)
        return {"text": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock/{symbol}", response_model=StockResponse)
@limiter.limit("60/minute")
async def get_stock_endpoint(request: Request, symbol: str, db: AsyncSession = Depends(get_db)):
    symbol = validate_symbol(symbol)
    try:
        from backend.services.stock_service import fetch_stock_history

        df = fetch_stock_history(symbol)

        data_points = []
        for _, row in df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
            data_points.append({
                "date": date_str,
                "open": row['Open'],
                "high": row['High'],
                "low": row['Low'],
                "close": row['Close'],
                "volume": int(row['Volume'])
            })

        return JSONResponse(
            content={"symbol": symbol, "data": data_points},
            headers={"Cache-Control": f"public, max-age={settings.cache_ttl_seconds}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refresh-all", response_model=RefreshAllResponse)
@limiter.limit("5/minute")
async def refresh_all_endpoint(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        symbols = settings.default_symbols
        success_count = 0
        failed_count = 0

        for symbol in symbols:
            try:
                await ingest_stock_data(symbol, db)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to ingest {symbol}: {str(e)}")
                failed_count += 1

        return {
            "message": f"Refresh completed: {success_count} successful, {failed_count} failed",
            "symbols": symbols,
            "success_count": success_count,
            "failed_count": failed_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/{symbol}", response_model=TrainModelResponse)
@limiter.limit("5/minute")
async def train_model_endpoint(request: Request, symbol: str, db: AsyncSession = Depends(get_db)):
    """Train ML models for stock price prediction"""
    symbol = validate_symbol(symbol)
    try:
        from backend.services.ml_service import train_all_models

        results = await train_all_models(symbol, db)
        models_trained = [model_type for model_type in results.keys() if 'error' not in results[model_type]]

        return {
            "message": f"Training completed for {symbol}",
            "symbol": symbol,
            "models_trained": models_trained,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{symbol}", response_model=PredictionResponse)
@limiter.limit("30/minute")
async def get_predictions_endpoint(
    request: Request,
    symbol: str,
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """Get price predictions for a stock"""
    symbol = validate_symbol(symbol)
    try:
        from backend.services.prediction_service import generate_predictions, get_predictions_for_symbol

        existing_preds = await get_predictions_for_symbol(symbol, db, days)

        if not existing_preds:
            await generate_predictions(symbol, db, days)
            existing_preds = await get_predictions_for_symbol(symbol, db, days)

        return {
            "symbol": symbol,
            "predictions": existing_preds,
            "days_ahead": len(existing_preds)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{symbol}", response_model=ModelStatusResponse)
@limiter.limit("60/minute")
async def get_model_status_endpoint(request: Request, symbol: str, db: AsyncSession = Depends(get_db)):
    """Get model training status and performance metrics"""
    symbol = validate_symbol(symbol)
    try:
        from backend.services.prediction_service import get_model_status

        status = await get_model_status(symbol, db)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{symbol}/feature-importance")
@limiter.limit("60/minute")
async def get_feature_importance_endpoint(request: Request, symbol: str, db: AsyncSession = Depends(get_db)):
    """Get XGBoost feature importance for a symbol"""
    symbol = validate_symbol(symbol)
    try:
        stmt = select(MLModel).where(
            MLModel.symbol == symbol,
            MLModel.model_type == 'xgboost',
            MLModel.is_active == True
        ).order_by(MLModel.training_date.desc()).limit(1)
        result = await db.execute(stmt)
        model = result.scalar_one_or_none()

        if not model or not model.metrics:
            raise HTTPException(status_code=404, detail=f"No XGBoost model found for {symbol}")

        importances = model.metrics.get('feature_importances', {})
        if not importances:
            raise HTTPException(status_code=404, detail=f"No feature importances stored for {symbol}")

        # Sort by importance descending
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        return {
            "symbol": symbol,
            "features": [{"name": name, "importance": score} for name, score in sorted_features]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/status")
@limiter.limit("60/minute")
async def get_pipeline_status(request: Request, db: AsyncSession = Depends(get_db)):
    """Get real pipeline status: per-symbol record counts, embedding counts, latest dates."""
    try:
        # Get all symbols with data
        symbol_stats = []

        stmt = select(
            StockPrice.symbol,
            func.count(StockPrice.id).label('record_count'),
            func.max(StockPrice.date).label('latest_date'),
            func.min(StockPrice.date).label('earliest_date')
        ).group_by(StockPrice.symbol)
        result = await db.execute(stmt)
        price_rows = result.fetchall()

        for row in price_rows:
            symbol = row[0]

            # Count RAG documents
            rag_stmt = select(func.count(RagDocument.id)).where(RagDocument.symbol == symbol)
            rag_result = await db.execute(rag_stmt)
            embedding_count = rag_result.scalar() or 0

            # Count models
            model_stmt = select(func.count(MLModel.id)).where(
                MLModel.symbol == symbol, MLModel.is_active == True
            )
            model_result = await db.execute(model_stmt)
            model_count = model_result.scalar() or 0

            symbol_stats.append({
                "symbol": symbol,
                "record_count": row[1],
                "embedding_count": embedding_count,
                "model_count": model_count,
                "latest_date": row[2].isoformat() if row[2] else None,
                "earliest_date": row[3].isoformat() if row[3] else None,
            })

        return {"symbols": symbol_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "StockRAG API is running"}
