import { StockDataPoint, RagContext, StockTicker } from '../types';
import { getAllTickers } from './mockDataService';

/**
 * In a real backend, this would involve embedding user queries and searching a vector DB.
 * Here, we implement a "Keyword + Time" based retrieval strategy to simulate RAG logic client-side.
 */

// 1. Ingestion: Convert raw JSON rows into "Documents" (Text representations)
const createDocumentFromRow = (point: StockDataPoint): string => {
  return `On ${point.date}, ${point.symbol} opened at $${point.open}, reached a high of $${point.high}, a low of $${point.low}, and closed at $${point.close}. Volume was ${point.volume}.`;
};

// 2. Retrieval: Find relevant data based on query
export const retrieveContext = (query: string): RagContext[] => {
  const allTickers = getAllTickers();
  const lowerQuery = query.toLowerCase();
  
  const relevantContexts: RagContext[] = [];

  // Keyword extraction (Simple Heuristic)
  const mentionedSymbols = allTickers.filter(t => 
    lowerQuery.includes(t.symbol.toLowerCase()) || 
    lowerQuery.includes(t.name.toLowerCase().split(' ')[0].toLowerCase())
  );

  // Default to all symbols if none specifically mentioned (broad market query) or limit context size
  const targetTickers = mentionedSymbols.length > 0 ? mentionedSymbols : allTickers;

  // Time extraction (Simple Heuristic)
  let daysLookback = 7; // Default to last week
  if (lowerQuery.includes('month') || lowerQuery.includes('30 days')) daysLookback = 30;
  if (lowerQuery.includes('year') || lowerQuery.includes('long term')) daysLookback = 90;
  if (lowerQuery.includes('yesterday')) daysLookback = 1;

  targetTickers.forEach(ticker => {
    // Get last N days of data
    const recentData = ticker.data.slice(-daysLookback);

    recentData.forEach(point => {
      // Calculate a pseudo "relevance" score based on recency
      // In a real vector DB, this would be cosine similarity
      const dateRecency = new Date(point.date).getTime();
      const now = new Date().getTime();
      const recencyScore = 1 - ((now - dateRecency) / (1000 * 60 * 60 * 24 * 365)); // Higher is better

      relevantContexts.push({
        symbol: ticker.symbol,
        date: point.date,
        summary: createDocumentFromRow(point),
        relevanceScore: recencyScore
      });
    });
  });

  // Sort by score (recency in this heuristic) and take top K chunks to fit in context
  // Gemini 2.5 Flash has a large context window, so we can be generous, but let's simulate efficient RAG.
  return relevantContexts.sort((a, b) => b.relevanceScore - a.relevanceScore).slice(0, 100); 
};


export const formatContextForPrompt = (contexts: RagContext[]): string => {
  if (contexts.length === 0) return "No specific stock data found in the knowledge base.";
  return contexts.map(c => `[Context ID: ${c.symbol}-${c.date}] ${c.summary}`).join('\n');
};