import asyncio
import logging
import os
import re
from typing import Iterable, List, Optional

import google.generativeai as genai
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.models import RagDocument

from dotenv import load_dotenv
from pathlib import Path

# Load env vars
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash"

async def _embed_content_async(content: str, task_type: str) -> List[float]:
    result = await asyncio.to_thread(
        genai.embed_content,
        model=EMBEDDING_MODEL,
        content=content,
        task_type=task_type,
    )
    return result["embedding"]

async def generate_embedding(text: str) -> List[float]:
    try:
        return await _embed_content_async(text, "retrieval_document")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

async def generate_query_embedding(text: str) -> List[float]:
    try:
        return await _embed_content_async(text, "retrieval_query")
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        raise

def _get_row_value(row: object, key: str):
    return row[key] if isinstance(row, dict) else getattr(row, key)

def _format_currency(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)

async def _embed_texts(texts: List[str], concurrency: int = 5) -> List[Optional[List[float]]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _embed(text: str) -> Optional[List[float]]:
        async with semaphore:
            try:
                return await generate_embedding(text)
            except Exception as exc:
                print(f"Skipping document due to embedding error: {exc}")
                return None

    return await asyncio.gather(*[_embed(text) for text in texts])

async def ingest_rag_documents(symbol: str, prices: Iterable[object], session: AsyncSession):
    print(f"Ingesting RAG documents for {symbol}...")
    rag_rows = []
    texts = []

    for p in prices:
        date_value = _get_row_value(p, "date")
        content = (
            f"On {date_value}, {symbol} opened at ${_format_currency(_get_row_value(p, 'open'))}, "
            f"reached a high of ${_format_currency(_get_row_value(p, 'high'))}, "
            f"low of ${_format_currency(_get_row_value(p, 'low'))}, "
            f"and closed at ${_format_currency(_get_row_value(p, 'close'))}. "
            f"Volume was {_get_row_value(p, 'volume')}."
        )
        texts.append(content)
        rag_rows.append({"symbol": symbol, "date": date_value, "content": content})

    if not rag_rows:
        print(f"No documents to ingest for {symbol}.")
        return

    embeddings = await _embed_texts(texts)

    docs_to_insert = []
    for row, embedding in zip(rag_rows, embeddings):
        if embedding:
            docs_to_insert.append({**row, "embedding": embedding})

    if docs_to_insert:
        await session.execute(insert(RagDocument), docs_to_insert)
        await session.commit()
        print(f"Stored {len(docs_to_insert)} vector documents for {symbol}.")
    else:
        print(f"No embeddings available for {symbol}. Skipping insert.")

async def retrieve_context(query: str, session: AsyncSession, limit: int = 5) -> List[str]:
    try:
        query_embedding = await generate_query_embedding(query)
        stmt = select(RagDocument).order_by(
            RagDocument.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        symbol = extract_symbol_from_query(query)
        if symbol:
            stmt = stmt.where(RagDocument.symbol == symbol)
        result = await session.execute(stmt)
        docs = result.scalars().all()
        
        return [d.content for d in docs]
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

# Configure logging
logging.basicConfig(filename='backend/error.log', level=logging.ERROR)

def is_prediction_query(query: str) -> bool:
    """Detect if query is asking about future predictions"""
    prediction_keywords = [
        'predict', 'forecast', 'future', 'will', 'next', 'upcoming',
        'tomorrow', 'week', 'month', 'should i buy', 'should i sell',
        'going to', 'expected', 'outlook', 'projection'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in prediction_keywords)

COMPANY_SYMBOL_MAP = {
    "APPLE": "AAPL",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "TESLA": "TSLA",
    "META": "META",
    "NVIDIA": "NVDA",
}

def extract_symbol_from_query(query: str) -> Optional[str]:
    """Extract stock symbol from query"""
    # Common stock symbols
    common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    
    query_upper = query.upper()
    dollar_match = re.search(r'\$([A-Z]{1,5})\b', query_upper)
    if dollar_match:
        return dollar_match.group(1)

    for symbol in common_symbols:
        if symbol in query_upper:
            return symbol

    for company, symbol in COMPANY_SYMBOL_MAP.items():
        if company in query_upper:
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
        
        # Format predictions for context
        context = f"\n\n📊 **Price Predictions for {symbol}:**\n\n"
        
        # Show 7-day, 14-day, and 30-day predictions
        for i, days in enumerate([7, 14, 30]):
            if days <= len(predictions):
                pred = predictions[days - 1]
                context += f"**{days}-day forecast:**\n"
                context += f"- Predicted Price: ${pred['predicted_price']:.2f}\n"
                context += f"- Confidence Range: ${pred['confidence_lower']:.2f} - ${pred['confidence_upper']:.2f}\n\n"
        
        return context
    except Exception as e:
        print(f"Error getting prediction context: {e}")
        return ""

async def generate_rag_response(query: str, session: AsyncSession) -> str:
    try:
        # Check if this is a prediction query
        is_pred_query = is_prediction_query(query)
        
        context_str = ""
        
        if is_pred_query:
            # Extract symbol
            symbol = extract_symbol_from_query(query)
            
            if symbol:
                # Get prediction context
                pred_context = await get_prediction_context(symbol, session)
                context_str += pred_context
            
            # Also get historical context
            historical_contexts = await retrieve_context(query, session, limit=3)
            context_str += "\n\n**Historical Data:**\n" + "\n".join(historical_contexts)
        else:
            # Regular historical query
            contexts = await retrieve_context(query, session)
            context_str = "\n".join(contexts)
        
        if not context_str.strip():
            return (
                "I couldn't find enough stock data to answer that yet. "
                "Try asking about a specific ticker (e.g., AAPL, MSFT) or ingest data first."
            )

        # 2. Augment
        system_instruction = f"""
        You are a financial analyst AI assistant. Use the following context to answer the user's question.
        Answer in a readable format like a human, using **bold** and *italic* to make it more readable and add emojis to make it more engaging.
        
        If the question is about future predictions:
        - Provide the prediction data clearly
        - Include confidence intervals
        - Explain the trend (bullish/bearish)
        - Add appropriate disclaimers about prediction uncertainty
        - Always end with: "⚠️ This is not financial advice. Predictions are based on historical patterns and should not be the sole basis for investment decisions."
        
        If the answer is not in the context, say you don't have that data.
        
        Context:
        {context_str}
        """
        
        model = genai.GenerativeModel(
            model_name=GENERATION_MODEL,
            system_instruction=system_instruction
        )
        
        # 3. Generate (Async)
        response = await model.generate_content_async(query)
        return response.text
    except Exception as e:
        logging.error(f"Error generating RAG response: {e}")
        print(f"Error generating RAG response: {e}")
        return f"I encountered an error processing your request. Error: {str(e)}"
