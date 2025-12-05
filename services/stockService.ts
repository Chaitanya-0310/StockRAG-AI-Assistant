import { api } from './api';
import { StockTicker } from '../types';

export const getTickerData = async (symbol: string): Promise<StockTicker> => {
    try {
        const response = await api.get(`/stock/${symbol}`);
        return {
            symbol: response.symbol,
            name: response.symbol, // Backend currently doesn't return name, using symbol as fallback
            data: response.data,
            lastUpdated: new Date().toISOString()
        };
    } catch (error) {
        console.error(`Failed to fetch data for ${symbol}:`, error);
        throw error;
    }
};

// For the dashboard list, we might want a specific endpoint or just fetch a few default tickers
export const getAllTickers = async (): Promise<StockTicker[]> => {
    const defaultSymbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'];
    const promises = defaultSymbols.map(sym => getTickerData(sym));
    return Promise.all(promises);
};
