import os
import google.generativeai as genai
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from backend.models.models import RagDocument
from backend.models.models import StockPrice
from typing import List
import re

from dotenv import load_dotenv
from pathlib import Path

# Load env vars
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash"

async def generate_embedding(text: str) -> List[float]:
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

async def ingest_rag_documents(symbol: str, prices: List[StockPrice], session: AsyncSession):
    print(f"Ingesting RAG documents for {symbol}...")
    rag_docs = []
    
    for p in prices:
        content = f"On {p.date}, {symbol} opened at ${p.open}, reached a high of ${p.high}, low of ${p.low}, and closed at ${p.close}. Volume was {p.volume}."
        
        try:
            embedding = await generate_embedding(content)
            
            doc = RagDocument(
                symbol=symbol,
                date=p.date,
                content=content,
                embedding=embedding
            )
            rag_docs.append(doc)
        except Exception as e:
            print(f"Skipping document due to embedding error: {e}")
            continue
    
    if rag_docs:
        session.add_all(rag_docs)
        await session.commit()
        print(f"Stored {len(rag_docs)} vector documents for {symbol}.")

async def retrieve_context(query: str, session: AsyncSession, limit: int = 5) -> List[str]:
    try:
        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        stmt = select(RagDocument).order_by(RagDocument.embedding.cosine_distance(query_embedding)).limit(limit)
        result = await session.execute(stmt)
        docs = result.scalars().all()
        
        return [d.content for d in docs]
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

import logging

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

def extract_symbol_from_query(query: str) -> str:
    """Extract stock symbol from query"""
    # Common stock symbols
    common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    
    query_upper = query.upper()
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

