import React, { useState, useEffect } from 'react';
import {
  CheckCircle, Loader2, Database, Server, RefreshCw,
  Play, AlertCircle, Clock, Hash, FileText
} from 'lucide-react';
import { API_BASE_URL } from '../services/api';

interface SymbolStatus {
  symbol: string;
  record_count: number;
  embedding_count: number;
  model_count: number;
  latest_date: string | null;
  earliest_date: string | null;
}

const PipelineVisualizer: React.FC = () => {
  const [symbolStatuses, setSymbolStatuses] = useState<SymbolStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState<Record<string, boolean>>({});
  const [newSymbol, setNewSymbol] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPipelineStatus();
  }, []);

  const fetchPipelineStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/pipeline/status`);
      if (!response.ok) throw new Error('Failed to fetch pipeline status');
      const data = await response.json();
      setSymbolStatuses(data.symbols || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const triggerIngest = async (symbol: string) => {
    setIngesting(prev => ({ ...prev, [symbol]: true }));
    try {
      const response = await fetch(`${API_BASE_URL}/ingest/${symbol}`, { method: 'POST' });
      if (!response.ok) throw new Error(`Ingest failed for ${symbol}`);
      await fetchPipelineStatus();
    } catch (err) {
      console.error(`Error ingesting ${symbol}:`, err);
    } finally {
      setIngesting(prev => ({ ...prev, [symbol]: false }));
    }
  };

  const handleAddSymbol = async () => {
    const symbol = newSymbol.trim().toUpperCase();
    if (!symbol || !/^[A-Z]{1,5}$/.test(symbol)) return;
    setNewSymbol('');
    await triggerIngest(symbol);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500">
        <Loader2 className="w-8 h-8 animate-spin mr-2" />
        <span>Loading pipeline status...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <AlertCircle className="mx-auto mb-4 text-red-400" size={48} />
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={fetchPipelineStatus}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const totalRecords = symbolStatuses.reduce((sum, s) => sum + s.record_count, 0);
  const totalEmbeddings = symbolStatuses.reduce((sum, s) => sum + s.embedding_count, 0);
  const totalModels = symbolStatuses.reduce((sum, s) => sum + s.model_count, 0);

  return (
    <div className="p-6 max-w-6xl mx-auto flex flex-col gap-6">
      {/* Header */}
      <div className="flex justify-between items-center bg-slate-900/50 p-6 rounded-xl border border-slate-800">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">ETL Pipeline Status</h2>
          <p className="text-slate-400 text-sm">Real-time data ingestion and embedding status</p>
        </div>
        <button
          onClick={fetchPipelineStatus}
          className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg border border-slate-700 flex items-center gap-2"
        >
          <RefreshCw size={16} /> Refresh
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
            <Database size={16} /> Symbols Tracked
          </div>
          <div className="text-3xl font-bold text-white">{symbolStatuses.length}</div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
            <Hash size={16} /> Total Records
          </div>
          <div className="text-3xl font-bold text-emerald-400">{totalRecords.toLocaleString()}</div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
            <FileText size={16} /> Embeddings
          </div>
          <div className="text-3xl font-bold text-indigo-400">{totalEmbeddings.toLocaleString()}</div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
            <Server size={16} /> ML Models
          </div>
          <div className="text-3xl font-bold text-amber-400">{totalModels}</div>
        </div>
      </div>

      {/* Add New Symbol */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
        <input
          type="text"
          value={newSymbol}
          onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
          onKeyDown={(e) => e.key === 'Enter' && handleAddSymbol()}
          placeholder="Enter ticker (e.g. JPM)"
          maxLength={5}
          className="bg-slate-800 border border-slate-700 text-white text-sm rounded-lg px-4 py-2 w-48 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
        />
        <button
          onClick={handleAddSymbol}
          disabled={!newSymbol.trim()}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg font-medium flex items-center gap-2"
        >
          <Play size={16} /> Ingest New Symbol
        </button>
      </div>

      {/* Symbol Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-800 text-left text-xs text-slate-500 uppercase tracking-wider">
              <th className="p-4">Symbol</th>
              <th className="p-4">Records</th>
              <th className="p-4">Embeddings</th>
              <th className="p-4">Models</th>
              <th className="p-4">Date Range</th>
              <th className="p-4">Status</th>
              <th className="p-4">Actions</th>
            </tr>
          </thead>
          <tbody>
            {symbolStatuses.map(status => (
              <tr key={status.symbol} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                <td className="p-4">
                  <span className="font-bold text-white text-lg">{status.symbol}</span>
                </td>
                <td className="p-4 font-mono text-slate-300">{status.record_count.toLocaleString()}</td>
                <td className="p-4 font-mono text-indigo-400">{status.embedding_count.toLocaleString()}</td>
                <td className="p-4 font-mono text-amber-400">{status.model_count}</td>
                <td className="p-4 text-sm text-slate-400">
                  {status.earliest_date && status.latest_date ? (
                    <span className="flex items-center gap-1">
                      <Clock size={14} />
                      {status.earliest_date} to {status.latest_date}
                    </span>
                  ) : '—'}
                </td>
                <td className="p-4">
                  {status.embedding_count > 0 ? (
                    <span className="flex items-center gap-1 text-emerald-400 text-sm">
                      <CheckCircle size={14} /> Ready
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-amber-400 text-sm">
                      <AlertCircle size={14} /> No embeddings
                    </span>
                  )}
                </td>
                <td className="p-4">
                  <button
                    onClick={() => triggerIngest(status.symbol)}
                    disabled={ingesting[status.symbol]}
                    className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded-lg flex items-center gap-1"
                  >
                    {ingesting[status.symbol] ? (
                      <><Loader2 size={14} className="animate-spin" /> Ingesting...</>
                    ) : (
                      <><RefreshCw size={14} /> Update</>
                    )}
                  </button>
                </td>
              </tr>
            ))}
            {symbolStatuses.length === 0 && (
              <tr>
                <td colSpan={7} className="p-8 text-center text-slate-500">
                  No data ingested yet. Use the form above to add a stock symbol.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PipelineVisualizer;
