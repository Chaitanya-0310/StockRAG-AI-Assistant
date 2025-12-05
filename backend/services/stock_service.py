import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetches historical stock data for a given symbol.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    
    # Reset index to make Date a column
    hist.reset_index(inplace=True)
    
    # Ensure Date is just date object (if needed) or keep as datetime
    # yfinance returns datetime with timezone usually
    
    return hist

def get_stock_info(symbol: str):
    """
    Fetches basic info for a stock.
    """
    ticker = yf.Ticker(symbol)
    return ticker.info
