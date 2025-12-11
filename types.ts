export interface StockDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  symbol: string;
}

export interface StockTicker {
  symbol: string;
  name: string;
  data: StockDataPoint[];
  lastUpdated: string;
}

export enum AppTab {
  CHAT = 'CHAT',
  DASHBOARD = 'DASHBOARD',
  PIPELINE = 'PIPELINE',
  PREDICTIONS = 'PREDICTIONS'
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model' | 'system';
  content: string;
  timestamp: number;
  ragContext?: RagContext[]; // Debug info showing what was retrieved
  isThinking?: boolean;
}

export interface RagContext {
  symbol: string;
  date: string;
  summary: string;
  relevanceScore: number;
}

export interface PipelineStep {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  logs: string[];
}