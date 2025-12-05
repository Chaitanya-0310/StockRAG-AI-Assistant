import os
import google.generativeai as genai
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from backend.models.models import RagDocument
from backend.models.models import StockPrice
from typing import List

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

async def generate_rag_response(query: str, session: AsyncSession) -> str:
    try:
        # 1. Retrieve
        contexts = await retrieve_context(query, session)
        context_str = "\n".join(contexts)
        
        # 2. Augment
        system_instruction = f"""
        You are a financial analyst AI. Use the following context to answer the user's question.
        And answer in readable format like a human adding bold and italic to make it more readable and also add emojis to make it more engaging.
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
