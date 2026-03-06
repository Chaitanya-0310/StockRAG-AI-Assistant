import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import date


class TestStockService:
    def test_fetch_stock_history_returns_dataframe(self, mock_yfinance):
        from backend.services.stock_service import fetch_stock_history, _stock_cache
        _stock_cache.clear()

        df = fetch_stock_history("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert 'Date' in df.columns or 'date' in df.columns.str.lower()

    def test_fetch_stock_history_caches_result(self, mock_yfinance):
        from backend.services.stock_service import fetch_stock_history, _stock_cache
        _stock_cache.clear()

        df1 = fetch_stock_history("MSFT")
        df2 = fetch_stock_history("MSFT")

        # yfinance should only be called once due to caching
        assert mock_yfinance.Ticker.call_count == 1

    def test_cache_separates_symbols(self, mock_yfinance):
        from backend.services.stock_service import fetch_stock_history, _stock_cache
        _stock_cache.clear()

        fetch_stock_history("AAPL")
        fetch_stock_history("GOOGL")

        assert mock_yfinance.Ticker.call_count == 2


class TestRSICalculation:
    def test_rsi_basic(self):
        from backend.services.prediction_service import _calculate_rsi

        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        rsi = _calculate_rsi(prices)

        # RSI should be between 0 and 100 (ignoring NaN)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
