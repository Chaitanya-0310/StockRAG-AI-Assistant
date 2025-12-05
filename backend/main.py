from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.services.db_service import init_db, get_db
from backend.services.etl_service import ingest_stock_data
from backend.services.rag_service import generate_rag_response
from backend.models.schemas import ChatRequest, ChatResponse, IngestResponse, StockResponse, RefreshAllResponse
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

@app.get("/")
async def root():
    return {"message": "StockRAG API is running"}
