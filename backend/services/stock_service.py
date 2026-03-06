import logging
import yfinance as yf
import pandas as pd
from cachetools import TTLCache
from backend.config import settings

logger = logging.getLogger(__name__)

# Cache: key=symbol, value=DataFrame, TTL from config
_stock_cache = TTLCache(maxsize=100, ttl=settings.cache_ttl_seconds)


def fetch_stock_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetches historical stock data for a given symbol.
    Results are cached with a TTL to avoid hammering the API.
    """
    cache_key = f"{symbol}_{period}"
    if cache_key in _stock_cache:
        logger.info(f"Cache hit for {symbol}")
        return _stock_cache[cache_key]

    logger.info(f"Fetching {symbol} from yfinance...")
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    hist.reset_index(inplace=True)

    _stock_cache[cache_key] = hist
    return hist


def get_stock_info(symbol: str):
    """Fetches basic info for a stock."""
    ticker = yf.Ticker(symbol)
    return ticker.info
