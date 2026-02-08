import os
import google.generativeai as genai
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from backend.models.models import RagDocument
from backend.models.models import StockPrice
from typing import List
import re
import numpy as np

from dotenv import load_dotenv
from pathlib import Path

# Load env vars
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash"

# Check database type
DATABASE_URL = os.getenv("DATABASE_URL", "")
IS_POSTGRES = DATABASE_URL.startswith("postgresql")

async def generate_embedding(text: str) -> List[float]:
    """Generate text embedding using Google's text-embedding-004 model"""
    if not GOOGLE_API_KEY:
        # Fallback: return random embedding for development without API key
        import random
        return [random.uniform(-1, 1) for _ in range(768)]
    
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Fallback: return zeros
        return [0.0] * 768

async def ingest_rag_documents(symbol: str, prices: List[StockPrice], session: AsyncSession):
    """Generate embeddings and store RAG documents for stock prices"""
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
                embedding=embedding  # Will be Vector for Postgres, JSON for SQLite
            )
            rag_docs.append(doc)
        except Exception as e:
            print(f"Skipping document due to embedding error: {e}")
            continue
    
    if rag_docs:
        session.add_all(rag_docs)
        await session.commit()
        print(f"Stored {len(rag_docs)} vector documents for {symbol}.")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def retrieve_context(query: str, session: AsyncSession, limit: int = 5) -> List[str]:
    """Retrieve relevant context for a query using vector similarity"""
    try:
        if not GOOGLE_API_KEY:
            # Fallback without API key - return recent stock data
            stmt = select(RagDocument).order_by(RagDocument.date.desc()).limit(limit)
            result = await session.execute(stmt)
            docs = result.scalars().all()
            return [d.content for d in docs]
        
        # Generate query embedding
        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        if IS_POSTGRES:
            # Use pgvector's cosine distance for PostgreSQL
            from pgvector.sqlalchemy import Vector
            stmt = select(RagDocument).order_by(
                RagDocument.embedding.cosine_distance(query_embedding)
            ).limit(limit)
            result = await session.execute(stmt)
            docs = result.scalars().all()
        else:
            # For SQLite, fetch all and compute similarity in Python
            stmt = select(RagDocument)
            result = await session.execute(stmt)
            all_docs = result.scalars().all()
            
            # Calculate cosine similarity for each document
            similarities = []
            for doc in all_docs:
                if doc.embedding:
                    sim = cosine_similarity(query_embedding, doc.embedding)
                    similarities.append((sim, doc))
            
            # Sort by similarity and take top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            docs = [doc for _, doc in similarities[:limit]]
        
        return [d.content for d in docs]
    except Exception as e:
        print(f"Error retrieving context: {e}")
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

def extract_symbol_from_query(query: str) -> str:
    """Extract stock symbol from query"""
    # Common stock symbols
    common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    
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
    """Generate RAG-based response to user query"""
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
        
        # If no API key, return a mock response
        if not GOOGLE_API_KEY:
            return generate_mock_response(query, context_str, is_pred_query)
        
        # 2. Augment with system instruction
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
        
        # 3. Generate response
        response = await model.generate_content_async(query)
        return response.text
    except Exception as e:
        print(f"Error generating RAG response: {e}")
        return f"I encountered an error processing your request. Error: {str(e)}"

def generate_mock_response(query: str, context: str, is_prediction: bool) -> str:
    """Generate a mock response when API key is not available"""
    symbol = extract_symbol_from_query(query)
    
    if is_prediction and symbol:
        return f"""## 📊 Prediction for {symbol}

Based on the available data, here are the key insights:

**Note:** AI predictions are currently unavailable because no Google API key is configured.

To get full AI-powered predictions, please:
1. Get a Google API key from https://makersuite.google.com/app/apikey
2. Add it to your `.env` file: `GOOGLE_API_KEY=your-key-here`
3. Restart the server

⚠️ This is not financial advice. Predictions are based on historical patterns and should not be the sole basis for investment decisions."""
    
    if symbol and context:
        return f"""## 📈 {symbol} Stock Analysis

{context[:500]}...

**Note:** For AI-generated insights, please configure your Google API key in the `.env` file.

You can ask about:
- Historical price data
- 30-day price predictions
- Stock comparisons
- Market trends

⚠️ This is not financial advice."""
    
    return """## AI Financial Assistant

I can help you with:
- 📊 Stock price predictions (30-day forecasts)
- 📈 Historical data analysis
- 🔍 Technical indicators
- 💡 Investment insights

**To get started:**
1. Enter a stock symbol like "AAPL" or "TSLA"
2. Ask a question like "What will AAPL price be next week?"
3. Configure your Google API key for AI-powered responses

⚠️ This is not financial advice. Always do your own research before making investment decisions."""
