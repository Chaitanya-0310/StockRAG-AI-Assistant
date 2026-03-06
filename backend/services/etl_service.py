import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from backend.models.models import StockPrice, RagDocument
from backend.services.stock_service import fetch_stock_history
import pandas as pd

logger = logging.getLogger(__name__)


async def ingest_stock_data(symbol: str, session: AsyncSession):
    """
    Incrementally fetches and ingests stock data for a symbol.
    Only fetches new data since the latest date already in the DB.
    """
    # Check latest date in DB for this symbol
    stmt = select(func.max(StockPrice.date)).where(StockPrice.symbol == symbol)
    result = await session.execute(stmt)
    latest_date = result.scalar()

    logger.info(f"Fetching data for {symbol} (latest in DB: {latest_date})...")
    df = fetch_stock_history(symbol)

    if df.empty:
        logger.warning(f"No data returned from yfinance for {symbol}")
        return

    # Filter to only new records
    new_records = []
    for _, row in df.iterrows():
        date_val = row['Date'].date() if hasattr(row['Date'], 'date') else row['Date']

        # Skip records we already have
        if latest_date and date_val <= latest_date:
            continue

        stock_price = StockPrice(
            symbol=symbol,
            date=date_val,
            open=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            volume=int(row['Volume'])
        )
        new_records.append(stock_price)

    if not new_records:
        logger.info(f"No new records for {symbol}. Checking if RAG documents need generating...")
        # RAG docs may be missing (e.g. table was recreated). Generate from existing data.
        from backend.services.rag_service import ingest_rag_documents, ingest_weekly_monthly_summaries
        from backend.models.models import RagDocument
        rag_count_stmt = select(func.count(RagDocument.id)).where(RagDocument.symbol == symbol)
        rag_count_result = await session.execute(rag_count_stmt)
        rag_count = rag_count_result.scalar() or 0

        if rag_count == 0:
            logger.info(f"No RAG documents for {symbol}. Generating from existing stock prices...")
            all_stmt = select(StockPrice).where(StockPrice.symbol == symbol).order_by(StockPrice.date)
            all_result = await session.execute(all_stmt)
            all_prices = all_result.scalars().all()
            await ingest_rag_documents(symbol, all_prices, session)
            await ingest_weekly_monthly_summaries(symbol, all_prices, session)
        return

    # Batch insert new records
    session.add_all(new_records)
    await session.commit()
    logger.info(f"Inserted {len(new_records)} new records for {symbol}.")

    # Generate RAG embeddings only for new records
    from backend.services.rag_service import ingest_rag_documents, ingest_weekly_monthly_summaries
    await ingest_rag_documents(symbol, new_records, session)

    # Also generate weekly/monthly summaries from ALL data for this symbol
    all_stmt = select(StockPrice).where(StockPrice.symbol == symbol).order_by(StockPrice.date)
    all_result = await session.execute(all_stmt)
    all_prices = all_result.scalars().all()

    # Delete old weekly/monthly summaries and regenerate
    await session.execute(
        delete(RagDocument).where(
            RagDocument.symbol == symbol,
            RagDocument.granularity.in_(['weekly', 'monthly'])
        )
    )
    await session.commit()
    await ingest_weekly_monthly_summaries(symbol, all_prices, session)


async def full_reingest_stock_data(symbol: str, session: AsyncSession):
    """
    Full re-ingestion: deletes all data and re-fetches everything.
    Use only when data needs to be corrected.
    """
    from sqlalchemy import delete as sql_delete

    logger.info(f"Full re-ingest for {symbol}...")

    await session.execute(sql_delete(StockPrice).where(StockPrice.symbol == symbol))
    await session.execute(sql_delete(RagDocument).where(RagDocument.symbol == symbol))
    await session.commit()

    await ingest_stock_data(symbol, session)
