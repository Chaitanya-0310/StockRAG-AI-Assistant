import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, insert
from backend.models.models import StockPrice, RagDocument
from backend.services.stock_service import fetch_stock_history

async def ingest_stock_data(symbol: str, session: AsyncSession):
    """
    Fetches data for a symbol and loads it into the database.
    Idempotent: deletes existing data for symbol before inserting.
    """
    print(f"Fetching data for {symbol}...")
    df = await asyncio.to_thread(fetch_stock_history, symbol)
    if df.empty:
        print(f"No data returned for {symbol}. Skipping ingestion.")
        return
    
    # Delete existing stock prices for this symbol to ensure idempotency
    print(f"Deleting existing data for {symbol}...")
    await session.execute(
        delete(StockPrice).where(StockPrice.symbol == symbol)
    )
    
    # Delete existing RAG documents for this symbol
    await session.execute(
        delete(RagDocument).where(RagDocument.symbol == symbol)
    )
    
    stock_rows = []
    for row in df.itertuples(index=False):
        date_val = row.Date.date() if hasattr(row.Date, "date") else row.Date
        stock_rows.append(
            {
                "symbol": symbol,
                "date": date_val,
                "open": float(row.Open),
                "high": float(row.High),
                "low": float(row.Low),
                "close": float(row.Close),
                "volume": int(row.Volume),
            }
        )
    
    if stock_rows:
        await session.execute(insert(StockPrice), stock_rows)
    await session.commit()
    print(f"Successfully loaded {len(stock_rows)} records for {symbol}.")
    
    # Trigger RAG ingestion
    from backend.services.rag_service import ingest_rag_documents
    await ingest_rag_documents(symbol, stock_rows, session)


async def ingest_latest_stock_data(symbol: str, session: AsyncSession):
    """
    Fetches the latest data point for a symbol and loads it into the database.
    Idempotent: deletes existing data for the latest date before inserting.
    """
    print(f"Fetching latest data for {symbol}...")
    df = await asyncio.to_thread(fetch_stock_history, symbol, "5d")
    if df.empty:
        print(f"No data returned for {symbol}. Skipping latest ingestion.")
        return

    latest_row = df.tail(1)
    row = latest_row.itertuples(index=False).__next__()
    date_val = row.Date.date() if hasattr(row.Date, "date") else row.Date

    print(f"Deleting existing latest data for {symbol} on {date_val}...")
    await session.execute(
        delete(StockPrice).where(
            StockPrice.symbol == symbol,
            StockPrice.date == date_val,
        )
    )
    await session.execute(
        delete(RagDocument).where(
            RagDocument.symbol == symbol,
            RagDocument.date == date_val,
        )
    )

    stock_row = {
        "symbol": symbol,
        "date": date_val,
        "open": float(row.Open),
        "high": float(row.High),
        "low": float(row.Low),
        "close": float(row.Close),
        "volume": int(row.Volume),
    }

    await session.execute(insert(StockPrice), [stock_row])
    await session.commit()
    print(f"Successfully loaded latest record for {symbol} on {date_val}.")

    from backend.services.rag_service import ingest_rag_documents
    await ingest_rag_documents(symbol, [stock_row], session)
