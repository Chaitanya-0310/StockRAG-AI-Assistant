import logging
import google.generativeai as genai
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from backend.models.models import RagDocument, StockPrice
from backend.config import settings
from typing import List
import re

# Configure Gemini
genai.configure(api_key=settings.google_api_key)

EMBEDDING_MODEL = settings.embedding_model
GENERATION_MODEL = settings.generation_model
EMBEDDING_BATCH_SIZE = settings.embedding_batch_size

logger = logging.getLogger(__name__)


def _extract_embedding(raw) -> List[float]:
    """Extract a flat float list from an embedding value, handling both formats:
    - flat list: [0.1, 0.2, ...]  (text-embedding-004 style)
    - dict: {'values': [0.1, 0.2, ...]}  (gemini-embedding-001 style)
    """
    if isinstance(raw, dict):
        return raw.get('values', [])
    return raw  # already a flat list


async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text."""
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
        if isinstance(result, dict) and 'embedding' in result:
            return _extract_embedding(result['embedding'])
        raise ValueError(f"Unexpected embedding result format: {result}")
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts.

    Tries a single batched API call first (supported by some models). Falls back`
    to individual calls per text to ensure compatibility with gemini-embedding-001.
    """
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=texts)
        if isinstance(result, dict):
            # Batch response key (plural)
            if 'embeddings' in result:
                return [_extract_embedding(e) for e in result['embeddings']]
            # Single-key response (singular) — may be list-of-lists or list-of-dicts
            if 'embedding' in result:
                raw = result['embedding']
                if isinstance(raw, list) and raw:
                    first = raw[0]
                    if isinstance(first, (int, float)):
                        # Flat list → single embedding returned for a 1-item batch
                        return [raw]
                    # List of lists or list of dicts
                    return [_extract_embedding(e) for e in raw]
                if isinstance(raw, dict):
                    return [_extract_embedding(raw)]
        raise ValueError(f"Unexpected batch embedding result: {result}")
    except Exception as e:
        logger.warning(f"Batch embed failed ({e}), falling back to individual calls")
        # Fall back: embed each text individually
        embeddings = []
        for text in texts:
            embeddings.append(await generate_embedding(text))
        return embeddings


async def ingest_rag_documents(symbol: str, prices: List[StockPrice], session: AsyncSession):
    """Ingest RAG documents with batched embedding generation."""
    logger.info(f"Ingesting RAG documents for {symbol} ({len(prices)} records)...")

    # Build all texts first
    texts = []
    price_refs = []
    for p in prices:
        content = (
            f"On {p.date}, {symbol} opened at ${p.open:.2f}, reached a high of ${p.high:.2f}, "
            f"low of ${p.low:.2f}, and closed at ${p.close:.2f}. Volume was {p.volume:,}."
        )
        texts.append(content)
        price_refs.append(p)

    # Batch embed in groups of EMBEDDING_BATCH_SIZE
    rag_docs = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
        batch_prices = price_refs[i:i + EMBEDDING_BATCH_SIZE]

        try:
            embeddings = await generate_embeddings_batch(batch_texts)
            for text, embedding, price in zip(batch_texts, embeddings, batch_prices):
                doc = RagDocument(
                    symbol=symbol,
                    date=price.date,
                    content=text,
                    embedding=embedding,
                    granularity='daily'
                )
                rag_docs.append(doc)
        except Exception as e:
            logger.error(f"Skipping batch {i}-{i + len(batch_texts)} due to error: {e}")
            continue

    if rag_docs:
        session.add_all(rag_docs)
        await session.commit()
        logger.info(f"Stored {len(rag_docs)} vector documents for {symbol}.")


async def ingest_weekly_monthly_summaries(symbol: str, prices: List[StockPrice], session: AsyncSession):
    """Generate weekly and monthly summary RAG documents for better trend queries."""
    from collections import defaultdict

    weekly_data = defaultdict(list)
    monthly_data = defaultdict(list)

    for p in prices:
        # ISO week key
        week_key = p.date.isocalendar()[:2]  # (year, week)
        weekly_data[week_key].append(p)
        # Month key
        month_key = (p.date.year, p.date.month)
        monthly_data[month_key].append(p)

    texts = []
    granularities = []
    dates = []

    # Weekly summaries
    for week_key, week_prices in sorted(weekly_data.items()):
        if len(week_prices) < 2:
            continue
        week_prices.sort(key=lambda x: x.date)
        open_price = week_prices[0].open
        close_price = week_prices[-1].close
        high_price = max(p.high for p in week_prices)
        low_price = min(p.low for p in week_prices)
        total_volume = sum(p.volume for p in week_prices)
        change_pct = ((close_price - open_price) / open_price * 100) if open_price else 0

        text = (
            f"Week of {week_prices[0].date} to {week_prices[-1].date}: {symbol} "
            f"opened at ${open_price:.2f}, closed at ${close_price:.2f} "
            f"({change_pct:+.2f}%). Weekly high ${high_price:.2f}, low ${low_price:.2f}. "
            f"Total volume: {total_volume:,}."
        )
        texts.append(text)
        granularities.append('weekly')
        dates.append(week_prices[-1].date)

    # Monthly summaries
    for month_key, month_prices in sorted(monthly_data.items()):
        if len(month_prices) < 5:
            continue
        month_prices.sort(key=lambda x: x.date)
        open_price = month_prices[0].open
        close_price = month_prices[-1].close
        high_price = max(p.high for p in month_prices)
        low_price = min(p.low for p in month_prices)
        total_volume = sum(p.volume for p in month_prices)
        change_pct = ((close_price - open_price) / open_price * 100) if open_price else 0
        month_name = month_prices[0].date.strftime('%B %Y')

        text = (
            f"{month_name} summary for {symbol}: "
            f"Opened at ${open_price:.2f}, closed at ${close_price:.2f} "
            f"({change_pct:+.2f}%). Monthly high ${high_price:.2f}, low ${low_price:.2f}. "
            f"Total volume: {total_volume:,}. Trading days: {len(month_prices)}."
        )
        texts.append(text)
        granularities.append('monthly')
        dates.append(month_prices[-1].date)

    if not texts:
        return

    # Batch embed
    rag_docs = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
        batch_grans = granularities[i:i + EMBEDDING_BATCH_SIZE]
        batch_dates = dates[i:i + EMBEDDING_BATCH_SIZE]

        try:
            embeddings = await generate_embeddings_batch(batch_texts)
            for text, embedding, gran, dt in zip(batch_texts, embeddings, batch_grans, batch_dates):
                doc = RagDocument(
                    symbol=symbol,
                    date=dt,
                    content=text,
                    embedding=embedding,
                    granularity=gran
                )
                rag_docs.append(doc)
        except Exception as e:
            logger.error(f"Skipping summary batch due to error: {e}")
            continue

    if rag_docs:
        session.add_all(rag_docs)
        await session.commit()
        logger.info(f"Stored {len(rag_docs)} summary documents for {symbol}.")


async def retrieve_context(query: str, session: AsyncSession, limit: int = 5) -> List[str]:
    try:
        query_embedding = await generate_embedding(query)

        stmt = (
            select(RagDocument)
            .order_by(RagDocument.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        result = await session.execute(stmt)
        docs = result.scalars().all()

        return [d.content for d in docs]
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []


def is_prediction_query(query: str) -> bool:
    """Detect if query is asking about future predictions"""
    prediction_keywords = [
        'predict', 'forecast', 'future', 'will', 'next', 'upcoming',
        'tomorrow', 'week', 'month', 'should i buy', 'should i sell',
        'going to', 'expected', 'outlook', 'projection'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in prediction_keywords)


async def get_known_symbols(session: AsyncSession) -> List[str]:
    """Get all symbols that have data in the database."""
    stmt = select(func.distinct(StockPrice.symbol))
    result = await session.execute(stmt)
    return [row[0] for row in result.fetchall()]


def extract_symbol_from_query(query: str, known_symbols: List[str] = None) -> str:
    """Extract stock symbol from query using known symbols from DB."""
    query_upper = query.upper()

    # Check against known symbols first
    if known_symbols:
        for symbol in known_symbols:
            if symbol in query_upper:
                return symbol

    # Fallback: common symbols
    common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    for symbol in common_symbols:
        if symbol in query_upper:
            return symbol

    # Try to find ticker pattern (2-5 uppercase letters)
    match = re.search(r'\b[A-Z]{2,5}\b', query)
    if match:
        return match.group(0)

    return None


async def get_prediction_context(symbol: str, session: AsyncSession) -> str:
    """Retrieve prediction data for context"""
    try:
        from backend.services.prediction_service import get_predictions_for_symbol

        predictions = await get_predictions_for_symbol(symbol, session, days=30)

        if not predictions:
            return f"No predictions available for {symbol}. Models may need to be trained first."

        context = f"\n\n**Price Predictions for {symbol}:**\n\n"

        for days in [7, 14, 30]:
            if days <= len(predictions):
                pred = predictions[days - 1]
                context += f"**{days}-day forecast:**\n"
                context += f"- Predicted Price: ${pred['predicted_price']:.2f}\n"
                context += f"- Confidence Range: ${pred['confidence_lower']:.2f} - ${pred['confidence_upper']:.2f}\n\n"

        return context
    except Exception as e:
        logger.error(f"Error getting prediction context: {e}")
        return ""


async def generate_rag_response(query: str, session: AsyncSession) -> str:
    try:
        is_pred_query = is_prediction_query(query)
        known_symbols = await get_known_symbols(session)
        context_str = ""

        if is_pred_query:
            symbol = extract_symbol_from_query(query, known_symbols)

            if symbol:
                pred_context = await get_prediction_context(symbol, session)
                context_str += pred_context

            historical_contexts = await retrieve_context(query, session, limit=3)
            context_str += "\n\n**Historical Data:**\n" + "\n".join(historical_contexts)
        else:
            contexts = await retrieve_context(query, session)
            context_str = "\n".join(contexts)

        system_instruction = f"""
        You are a financial analyst AI assistant. Use the following context to answer the user's question.
        Answer in a readable format like a human, using **bold** and *italic* to make it more readable and add emojis to make it more engaging.

        If the question is about future predictions:
        - Provide the prediction data clearly
        - Include confidence intervals
        - Explain the trend (bullish/bearish)
        - Add appropriate disclaimers about prediction uncertainty
        - Always end with: "This is not financial advice. Predictions are based on historical patterns and should not be the sole basis for investment decisions."

        If the answer is not in the context, say you don't have that data.

        Context:
        {context_str}
        """

        model = genai.GenerativeModel(
            model_name=GENERATION_MODEL,
            system_instruction=system_instruction
        )

        response = await model.generate_content_async(query)
        return response.text
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return f"I encountered an error processing your request. Error: {str(e)}"
