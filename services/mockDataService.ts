import { StockDataPoint, StockTicker } from '../types';

// Helper to generate random walk data mimicking stocks
const generateStockHistory = (symbol: string, startPrice: number, volatility: number, days: number): StockDataPoint[] => {
  const data: StockDataPoint[] = [];
  let currentPrice = startPrice;
  const now = new Date();

  for (let i = days; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    const dateStr = date.toISOString().split('T')[0];

    // Random daily change
    const changePercent = (Math.random() - 0.5) * volatility;
    const changeAmount = currentPrice * changePercent;
    
    const open = currentPrice;
    const close = currentPrice + changeAmount;
    const high = Math.max(open, close) + (Math.random() * (currentPrice * 0.01));
    const low = Math.min(open, close) - (Math.random() * (currentPrice * 0.01));
    const volume = Math.floor(Math.random() * 1000000) + 500000;

    data.push({
      date: dateStr,
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      close: parseFloat(close.toFixed(2)),
      volume,
      symbol
    });

    currentPrice = close;
  }
  return data;
};

// Expanded dataset with diversifed sectors and ETFs
export const INITIAL_TICKERS: StockTicker[] = [
  // Tech Giants
  { symbol: 'AAPL', name: 'Apple Inc.', data: generateStockHistory('AAPL', 175, 0.02, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'NVDA', name: 'Nvidia Corp.', data: generateStockHistory('NVDA', 450, 0.04, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', data: generateStockHistory('GOOGL', 135, 0.025, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', data: generateStockHistory('AMZN', 145, 0.03, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'NFLX', name: 'Netflix Inc.', data: generateStockHistory('NFLX', 400, 0.035, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'MSFT', name: 'Microsoft Corp.', data: generateStockHistory('MSFT', 330, 0.02, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'META', name: 'Meta Platforms', data: generateStockHistory('META', 300, 0.03, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'TSLA', name: 'Tesla Inc.', data: generateStockHistory('TSLA', 240, 0.045, 180), lastUpdated: new Date().toISOString() },

  // Financials
  { symbol: 'JPM', name: 'JPMorgan Chase', data: generateStockHistory('JPM', 145, 0.015, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'BAC', name: 'Bank of America', data: generateStockHistory('BAC', 28, 0.015, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'V', name: 'Visa Inc.', data: generateStockHistory('V', 240, 0.015, 180), lastUpdated: new Date().toISOString() },

  // Healthcare
  { symbol: 'JNJ', name: 'Johnson & Johnson', data: generateStockHistory('JNJ', 160, 0.01, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'PFE', name: 'Pfizer Inc.', data: generateStockHistory('PFE', 33, 0.015, 180), lastUpdated: new Date().toISOString() },
  
  // Consumer
  { symbol: 'KO', name: 'Coca-Cola Co.', data: generateStockHistory('KO', 58, 0.01, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'PG', name: 'Procter & Gamble', data: generateStockHistory('PG', 150, 0.01, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'WMT', name: 'Walmart Inc.', data: generateStockHistory('WMT', 160, 0.015, 180), lastUpdated: new Date().toISOString() },

  // Energy
  { symbol: 'XOM', name: 'Exxon Mobil', data: generateStockHistory('XOM', 105, 0.02, 180), lastUpdated: new Date().toISOString() },

  // ETFs
  { symbol: 'SPY', name: 'SPDR S&P 500 ETF', data: generateStockHistory('SPY', 440, 0.01, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'QQQ', name: 'Invesco QQQ Trust', data: generateStockHistory('QQQ', 370, 0.015, 180), lastUpdated: new Date().toISOString() },
  { symbol: 'IWM', name: 'iShares Russell 2000', data: generateStockHistory('IWM', 180, 0.02, 180), lastUpdated: new Date().toISOString() },
];

export const getTickerData = (symbol: string): StockTicker | undefined => {
  return INITIAL_TICKERS.find(t => t.symbol === symbol);
};

export const getAllTickers = (): StockTicker[] => {
  return INITIAL_TICKERS;
};