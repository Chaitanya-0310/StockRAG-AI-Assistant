import pytest
import asyncio
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import date, timedelta


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_stock_df():
    """Generate a realistic mock stock DataFrame."""
    dates = pd.date_range(end=date.today(), periods=252, freq='B')
    np.random.seed(42)
    prices = 150 + np.cumsum(np.random.randn(252) * 2)

    df = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(252) * 0.5,
        'High': prices + abs(np.random.randn(252)) * 2,
        'Low': prices - abs(np.random.randn(252)) * 2,
        'Close': prices,
        'Volume': np.random.randint(1_000_000, 50_000_000, 252)
    })
    return df


@pytest.fixture
def mock_yfinance(mock_stock_df):
    """Mock yfinance to avoid real API calls during tests."""
    with patch('backend.services.stock_service.yf') as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_stock_df.set_index('Date')
        mock_yf.Ticker.return_value = mock_ticker

        # Reset the cache so mock takes effect
        from backend.services.stock_service import _stock_cache
        _stock_cache.clear()

        # Return the mock_stock_df with reset index for fetch_stock_history
        mock_ticker.history.return_value = mock_stock_df.set_index('Date').reset_index()

        yield mock_yf


@pytest.fixture
def mock_gemini():
    """Mock Gemini API to avoid real API calls."""
    with patch('backend.services.rag_service.genai') as mock_genai:
        # Mock embed_content to return fake embeddings
        mock_genai.embed_content.return_value = {
            'embedding': [0.1] * 768
        }

        # Mock GenerativeModel
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a mock AI response."
        mock_model.generate_content_async = asyncio.coroutine(lambda *args, **kwargs: mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        yield mock_genai
