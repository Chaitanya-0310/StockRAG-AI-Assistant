import React, { useState, useEffect } from 'react';
import { getAllTickers } from '../services/stockService';
import { refreshAllData } from '../services/api';
import { StockTicker } from '../types';
import StockChart from './StockChart';
import { TrendingUp, TrendingDown, Activity, Search, X, BarChart2, LineChart, SlidersHorizontal, Loader2, RefreshCw } from 'lucide-react';

const StockDashboard: React.FC = () => {
  const [allTickers, setAllTickers] = useState<StockTicker[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTicker, setSelectedTicker] = useState<StockTicker | null>(null);

  // Chart Controls State
  const [chartType, setChartType] = useState<'area' | 'candle'>('area');
  const [showSMA, setShowSMA] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const tickers = await getAllTickers();
        setAllTickers(tickers);
      } catch (error) {
        console.error("Failed to fetch tickers:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const filteredTickers = allTickers.filter(t =>
    t.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    t.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleOpenDetail = (ticker: StockTicker) => {
    setSelectedTicker(ticker);
    setChartType('area'); // Reset to default
    setShowSMA(false);
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await refreshAllData();
      // Refetch tickers after refresh
      const tickers = await getAllTickers();
      setAllTickers(tickers);
    } catch (error) {
      console.error("Failed to refresh data:", error);
    } finally {
      setRefreshing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500">
        <Loader2 className="w-8 h-8 animate-spin mb-2" />
        <span className="ml-2">Loading market data...</span>
      </div>
    );
  }

  return (
    <div className="p-6 overflow-y-auto h-[calc(100vh-100px)]">
      <div className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Market Overview</h2>
          <p className="text-slate-400">
            Real-time mock data currently indexed in the RAG Knowledge Base.
            <br />
            <span className="text-indigo-400 text-sm font-semibold">{allTickers.length} Assets Tracked</span>
          </p>
        </div>

        <div className="flex gap-3 items-end">
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/40"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            {refreshing ? 'Refreshing...' : 'Refresh Data'}
          </button>

          <div className="relative w-full md:w-72">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              placeholder="Search tickers (e.g., SPY, AAPL)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-sm rounded-lg pl-10 pr-4 py-2.5 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition-all"
            />
          </div>
        </div>
      </div>

      {filteredTickers.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-slate-500">
          <Activity className="w-12 h-12 mb-4 opacity-20" />
          <p>No tickers found matching "{searchTerm}"</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {filteredTickers.map((ticker) => {
            const latest = ticker.data[ticker.data.length - 1];
            const prev = ticker.data[ticker.data.length - 2];
            const change = latest.close - prev.close;
            const changePercent = (change / prev.close) * 100;
            const isPositive = change >= 0;

            return (
              <div
                key={ticker.symbol}
                onClick={() => handleOpenDetail(ticker)}
                className="bg-slate-900 rounded-xl border border-slate-800 p-5 hover:border-indigo-500/50 hover:bg-slate-800/50 transition-all shadow-lg group cursor-pointer"
              >
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-bold text-white flex items-center gap-2 group-hover:text-indigo-300 transition-colors">
                      {ticker.symbol}
                      <span className="text-xs font-normal text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full truncate max-w-[120px]">{ticker.name}</span>
                    </h3>
                    <div className="flex items-baseline gap-2 mt-1">
                      <span className="text-2xl font-mono text-slate-200">${latest.close.toFixed(2)}</span>
                      <span className={`flex items-center text-sm font-semibold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {isPositive ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
                        {Math.abs(changePercent).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                  <div className={`p-2 rounded-lg ${isPositive ? 'bg-emerald-950/50' : 'bg-rose-950/50'}`}>
                    <Activity className={`w-6 h-6 ${isPositive ? 'text-emerald-500' : 'text-rose-500'}`} />
                  </div>
                </div>

                {/* Mini Chart */}
                <div className="bg-slate-950/50 rounded-lg p-2 border border-slate-800/50 pointer-events-none">
                  <StockChart
                    data={ticker.data.slice(-30)}
                    color={isPositive ? '#10b981' : '#f43f5e'}
                    className="h-[250px]"
                  />
                </div>

                <div className="grid grid-cols-3 gap-2 mt-4 text-xs text-slate-500 border-t border-slate-800 pt-3">
                  <div className="flex flex-col">
                    <span>Open</span>
                    <span className="text-slate-300">${latest.open.toFixed(2)}</span>
                  </div>
                  <div className="flex flex-col">
                    <span>High</span>
                    <span className="text-slate-300">${latest.high.toFixed(2)}</span>
                  </div>
                  <div className="flex flex-col text-right">
                    <span>Vol</span>
                    <span className="text-slate-300">{(latest.volume / 1000).toFixed(1)}k</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Detail Modal */}
      {selectedTicker && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-slate-900 w-full max-w-5xl rounded-2xl border border-slate-700 shadow-2xl flex flex-col overflow-hidden max-h-[95vh]">

            {/* Modal Header */}
            <div className="p-6 border-b border-slate-800 flex justify-between items-start bg-slate-900">
              <div>
                <h2 className="text-3xl font-bold text-white flex items-center gap-3">
                  {selectedTicker.symbol}
                  <span className="text-lg font-normal text-slate-400">{selectedTicker.name}</span>
                </h2>
                <div className="flex items-center gap-4 mt-2">
                  <span className="text-4xl font-mono text-white">
                    ${selectedTicker.data[selectedTicker.data.length - 1].close.toFixed(2)}
                  </span>
                  {(() => {
                    const latest = selectedTicker.data[selectedTicker.data.length - 1];
                    const prev = selectedTicker.data[selectedTicker.data.length - 2];
                    const change = latest.close - prev.close;
                    const changePercent = (change / prev.close) * 100;
                    const isUp = change >= 0;
                    return (
                      <span className={`px-3 py-1 rounded-full text-sm font-bold flex items-center ${isUp ? 'bg-emerald-950 text-emerald-400 border border-emerald-900' : 'bg-rose-950 text-rose-400 border border-rose-900'}`}>
                        {isUp ? <TrendingUp className="w-4 h-4 mr-2" /> : <TrendingDown className="w-4 h-4 mr-2" />}
                        {Math.abs(changePercent).toFixed(2)}%
                      </span>
                    )
                  })()}
                </div>
              </div>
              <button
                onClick={() => setSelectedTicker(null)}
                className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Toolbar */}
            <div className="px-6 py-3 bg-slate-950 border-b border-slate-800 flex flex-wrap gap-4 items-center">
              <div className="flex items-center bg-slate-900 rounded-lg p-1 border border-slate-800">
                <button
                  onClick={() => setChartType('area')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${chartType === 'area' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  <LineChart className="w-4 h-4" /> Area
                </button>
                <button
                  onClick={() => setChartType('candle')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${chartType === 'candle' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  <BarChart2 className="w-4 h-4" /> Candles
                </button>
              </div>

              <div className="h-6 w-px bg-slate-800 mx-2" />

              <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300 hover:text-white select-none">
                <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${showSMA ? 'bg-amber-500 border-amber-500' : 'border-slate-600 bg-slate-800'}`}>
                  {showSMA && <Activity className="w-3 h-3 text-black" />}
                </div>
                <input
                  type="checkbox"
                  checked={showSMA}
                  onChange={(e) => setShowSMA(e.target.checked)}
                  className="hidden"
                />
                Show SMA (20)
              </label>

              <div className="ml-auto text-xs text-slate-500 flex items-center gap-1">
                <SlidersHorizontal className="w-3 h-3" />
                <span>6 Month History</span>
              </div>
            </div>

            {/* Main Chart Area */}
            <div className="flex-1 bg-slate-900 p-6 min-h-[500px]">
              <StockChart
                data={selectedTicker.data}
                type={chartType}
                showSMA={showSMA}
                className="h-full w-full"
                color={selectedTicker.data[selectedTicker.data.length - 1].close >= selectedTicker.data[selectedTicker.data.length - 2].close ? '#10b981' : '#f43f5e'}
              />
            </div>

            {/* Footer Stats */}
            <div className="grid grid-cols-4 divide-x divide-slate-800 bg-slate-950 border-t border-slate-800">
              {[
                { label: 'Open', val: selectedTicker.data[selectedTicker.data.length - 1].open },
                { label: 'High', val: selectedTicker.data[selectedTicker.data.length - 1].high },
                { label: 'Low', val: selectedTicker.data[selectedTicker.data.length - 1].low },
                { label: 'Volume', val: selectedTicker.data[selectedTicker.data.length - 1].volume.toLocaleString() }
              ].map((stat, i) => (
                <div key={i} className="p-4 text-center">
                  <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">{stat.label}</div>
                  <div className="text-lg font-mono text-slate-200 font-medium">
                    {typeof stat.val === 'number' ? `$${stat.val.toFixed(2)}` : stat.val}
                  </div>
                </div>
              ))}
            </div>

          </div>
        </div>
      )}
    </div>
  );
};

export default StockDashboard;