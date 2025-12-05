from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from backend.models.models import StockPrice
from backend.services.stock_service import fetch_stock_history
from backend.services.db_service import get_db
import pandas as pd

async def ingest_stock_data(symbol: str, session: AsyncSession):
    """
    Fetches data for a symbol and loads it into the database.
    """
    print(f"Fetching data for {symbol}...")
    df = fetch_stock_history(symbol)
    
    # Check if data already exists to avoid duplicates (simple check for now)
    # In production, use upsert or check max date
    
    records = []
    for _, row in df.iterrows():
        # Convert timezone-aware datetime to naive or keep as is depending on DB
        date_val = row['Date'].date() if hasattr(row['Date'], 'date') else row['Date']
        
        stock_price = StockPrice(
            symbol=symbol,
            date=date_val,
            open=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            volume=row['Volume']
        )
        records.append(stock_price)
    
    # Batch insert
    session.add_all(records)
    await session.commit()
    print(f"Successfully loaded {len(records)} records for {symbol}.")
    
    # Trigger RAG ingestion
    from backend.services.rag_service import ingest_rag_documents
    await ingest_rag_documents(symbol, records, session)
