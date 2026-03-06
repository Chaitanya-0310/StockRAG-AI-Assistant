import React, { useState, useEffect, useMemo } from 'react';
import { getAllTickers } from '../services/stockService';
import { refreshAllData } from '../services/api';
import { StockTicker } from '../types';
import StockChart from './StockChart';
import {
  AreaChart as CompAreaChart, Area as CompArea, XAxis as CompXAxis, YAxis as CompYAxis,
  CartesianGrid as CompGrid, Tooltip as CompTooltip, Legend as CompLegend, ResponsiveContainer as CompRC
} from 'recharts';
import { TrendingUp, TrendingDown, Search, Loader2, RefreshCw, LineChart, BarChart2, Activity, Download } from 'lucide-react';

const StockDashboard: React.FC = () => {
  const [allTickers, setAllTickers] = useState<StockTicker[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTicker, setSelectedTicker] = useState<StockTicker | null>(null);

  const [chartType, setChartType] = useState<'area' | 'candle'>('area');
  const [showSMA, setShowSMA] = useState(false);

  // Comparison mode
  const [compareSymbols, setCompareSymbols] = useState<string[]>([]);
  const [showCompare, setShowCompare] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const tickers = await getAllTickers();
        setAllTickers(tickers);
        if (tickers.length > 0) setSelectedTicker(tickers[0]);
      } catch (error) {
        console.error("Failed to fetch tickers:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await refreshAllData();
      const tickers = await getAllTickers();
      setAllTickers(tickers);
    } catch (error) {
      console.error("Failed to refresh data:", error);
    } finally {
      setRefreshing(false);
    }
  };

  const exportCSV = () => {
    if (!selectedTicker) return;
    const headers = 'Date,Open,High,Low,Close,Volume\n';
    const rows = selectedTicker.data.map(d =>
      `${d.date},${d.open},${d.high},${d.low},${d.close},${d.volume}`
    ).join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedTicker.symbol}_data.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleCompare = (symbol: string) => {
    setCompareSymbols(prev => {
      if (prev.includes(symbol)) return prev.filter(s => s !== symbol);
      if (prev.length >= 4) return prev;
      return [...prev, symbol];
    });
  };

  // Comparison chart data: normalize to % change
  const comparisonData = useMemo(() => {
    if (compareSymbols.length < 2) return [];
    const tickerMap = new Map(allTickers.map(t => [t.symbol, t]));
    const series = compareSymbols.map(sym => tickerMap.get(sym)).filter(Boolean) as StockTicker[];
    if (series.length < 2) return [];

    // Use shortest series length
    const minLen = Math.min(...series.map(s => s.data.length));
    const result: any[] = [];

    for (let i = 0; i < minLen; i++) {
      const point: any = { date: series[0].data[i].date };
      for (const ticker of series) {
        const base = ticker.data[0].close;
        const change = ((ticker.data[i].close - base) / base) * 100;
        point[ticker.symbol] = parseFloat(change.toFixed(2));
      }
      result.push(point);
    }
    return result;
  }, [compareSymbols, allTickers]);

  const filteredTickers = allTickers.filter(t =>
    t.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    t.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const MainChartHeader = () => {
    if (!selectedTicker) return null;
    const latest = selectedTicker.data[selectedTicker.data.length - 1];
    const prev = selectedTicker.data[selectedTicker.data.length - 2];
    const change = latest.close - prev.close;
    const changePercent = (change / prev.close) * 100;
    const isPositive = change >= 0;

    return (
      <div className="p-6 border-b border-slate-800">
        <div className="flex justify-between items-center mb-2">
          <div>
            <h2 className="text-3xl font-bold text-white flex items-center gap-3">
              {selectedTicker.symbol}
              <span className="text-lg font-normal text-slate-400">{selectedTicker.name}</span>
            </h2>
          </div>
          <div className="flex items-center gap-4">
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
            <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300 hover:text-white select-none">
                <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${showSMA ? 'bg-amber-500 border-amber-500' : 'border-slate-600 bg-slate-800'}`}>
                  {showSMA && <Activity className="w-3 h-3 text-black" />}
                </div>
                <input type="checkbox" checked={showSMA} onChange={(e) => setShowSMA(e.target.checked)} className="hidden" />
                Show SMA (20)
            </label>
            <button onClick={exportCSV} className="flex items-center gap-1 px-3 py-1.5 rounded-md text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition-all">
              <Download className="w-4 h-4" /> CSV
            </button>
            <button
              onClick={() => setShowCompare(!showCompare)}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${showCompare ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Compare
            </button>
          </div>
        </div>
        <div className="flex items-baseline gap-4">
          <span className="text-4xl font-mono text-white">${latest.close.toFixed(2)}</span>
          <span className={`flex items-center text-xl font-semibold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
            {isPositive ? <TrendingUp className="w-5 h-5 mr-1" /> : <TrendingDown className="w-5 h-5 mr-1" />}
            {change.toFixed(2)} ({changePercent.toFixed(2)}%)
          </span>
        </div>

        {/* Compare selector */}
        {showCompare && (
          <div className="mt-4 flex flex-wrap gap-2">
            <span className="text-xs text-slate-500 self-center">Select up to 4:</span>
            {allTickers.map(t => (
              <button
                key={t.symbol}
                onClick={() => toggleCompare(t.symbol)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
                  compareSymbols.includes(t.symbol)
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                } ${compareSymbols.length >= 4 && !compareSymbols.includes(t.symbol) ? 'opacity-50 cursor-not-allowed' : ''}`}
                disabled={compareSymbols.length >= 4 && !compareSymbols.includes(t.symbol)}
              >
                {t.symbol}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  };

  const ComparisonChart = () => {
    if (comparisonData.length === 0) return null;
    const colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444'];

    return (
      <div className="p-6 border-t border-slate-800">
        <h3 className="text-lg font-bold text-white mb-4">Comparison (% Change)</h3>
        <CompRC width="100%" height={300}>
          <CompAreaChart data={comparisonData}>
            <CompGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <CompXAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} tickFormatter={(s: string) => { const d = new Date(s); return `${d.getMonth()+1}/${d.getDate()}`; }} minTickGap={30} />
            <CompYAxis stroke="#64748b" tick={{ fontSize: 11 }} tickFormatter={(v: number) => `${v.toFixed(0)}%`} />
            <CompTooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} formatter={(v: number) => `${v.toFixed(2)}%`} />
            <CompLegend />
            {compareSymbols.map((sym, i) => (
              <CompArea key={sym} type="monotone" dataKey={sym} stroke={colors[i % colors.length]} fill="none" strokeWidth={2} dot={false} />
            ))}
          </CompAreaChart>
        </CompRC>
      </div>
    );
  };

  const StatsBar = () => {
    if(!selectedTicker) return null;
    const latest = selectedTicker.data[selectedTicker.data.length - 1];
    const stats = [
        { label: 'Open', value: latest.open.toFixed(2) },
        { label: 'High', value: latest.high.toFixed(2) },
        { label: 'Low', value: latest.low.toFixed(2) },
        { label: 'Volume', value: (latest.volume / 1_000_000).toFixed(2) + 'M' },
    ];
    return (
        <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-slate-800 bg-slate-950 border-t border-slate-800">
        {stats.map(stat => (
          <div key={stat.label} className="p-4 text-center">
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">{stat.label}</div>
            <div className="text-lg font-mono text-slate-200 font-medium">
              {stat.label.match(/Open|High|Low/) ? `$${stat.value}`: stat.value}
            </div>
          </div>
        ))}
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500">
        <Loader2 className="w-8 h-8 animate-spin mb-2" />
        <span className="ml-2">Loading market data...</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col md:flex-row h-[calc(100vh-80px)] bg-slate-950">
      <main className="flex-1 flex flex-col min-h-0">
        <MainChartHeader />
        {showCompare && compareSymbols.length >= 2 && <ComparisonChart />}
        <div className="flex-1 p-1 sm:p-2 md:p-4 lg:p-6 bg-slate-900 min-h-0">
         {selectedTicker && <StockChart
            data={selectedTicker.data}
            type={chartType}
            showSMA={showSMA}
            className="h-full w-full"
            color={selectedTicker.data[selectedTicker.data.length - 1].close >= selectedTicker.data[selectedTicker.data.length - 2].close ? '#10b981' : '#f43f5e'}
         />}
        </div>
        <StatsBar />
      </main>

      <aside className="w-full md:w-1/3 lg:w-1/4 border-t md:border-t-0 md:border-l border-slate-800 flex flex-col">
        <div className="p-4 border-b border-slate-800 flex items-center gap-2">
        <div className="relative w-full">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-sm rounded-lg pl-10 pr-4 py-2 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition-all"
            />
          </div>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="p-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {filteredTickers.map(ticker => {
            const latest = ticker.data[ticker.data.length - 1];
            const prev = ticker.data[ticker.data.length - 2];
            const change = latest.close - prev.close;
            const changePercent = (change / prev.close) * 100;
            const isPositive = change >= 0;

            return (
              <div
                key={ticker.symbol}
                onClick={() => setSelectedTicker(ticker)}
                className={`p-4 border-b border-slate-800 cursor-pointer flex justify-between items-center transition-colors ${selectedTicker?.symbol === ticker.symbol ? 'bg-indigo-600/10' : 'hover:bg-slate-800/50'}`}
              >
                <div className='w-1/3'>
                  <h3 className={`font-bold text-white ${selectedTicker?.symbol === ticker.symbol ? 'text-indigo-300': ''}`}>{ticker.symbol}</h3>
                  <p className="text-xs text-slate-400 truncate">{ticker.name}</p>
                </div>
                <div className="w-1/3 h-10">
                    <StockChart
                        data={ticker.data.slice(-30)}
                        color={isPositive ? '#10b981' : '#f43f5e'}
                        className="h-full w-full"
                        detailed={false}
                    />
                </div>
                <div className="w-1/3 text-right">
                  <p className="font-mono text-white">${latest.close.toFixed(2)}</p>
                  <p className={`text-sm font-semibold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {changePercent.toFixed(2)}%
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </aside>
    </div>
  );
};

export default StockDashboard;
