import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from dotenv import load_dotenv
from backend.models.models import Base
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Default to SQLite for easy development
    DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/stockrag.db"
    print(f"DATABASE_URL not set, using SQLite: {DATABASE_URL}")

# Check if using SQLite or PostgreSQL
IS_SQLITE = DATABASE_URL.startswith("sqlite")

if IS_SQLITE:
    # SQLite doesn't support vector extension, so we'll use a different approach for embeddings
    engine = create_async_engine(DATABASE_URL, echo=False)
else:
    engine = create_async_engine(DATABASE_URL, echo=True)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def init_db():
    async with engine.begin() as conn:
        if not IS_SQLITE:
            # PostgreSQL with pgvector
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception as e:
                print(f"Warning: Could not create pgvector extension: {e}")
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("Database initialized successfully")

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
