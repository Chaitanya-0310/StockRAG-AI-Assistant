import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { TrendingUp, TrendingDown, AlertCircle, Calendar, BarChart3, Layers } from 'lucide-react';

interface PredictionData {
    target_date: string;
    predicted_price: number;
    confidence_lower: number;
    confidence_upper: number;
    prediction_date: string;
}

interface PredictionChartProps {
    symbol: string;
    days?: number;
}

const PredictionChart: React.FC<PredictionChartProps> = ({ symbol, days = 30 }) => {
    const [predictions, setPredictions] = useState<PredictionData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [historicalData, setHistoricalData] = useState<any[]>([]);
    const [selectedRange, setSelectedRange] = useState<7 | 14 | 30>(30);

    useEffect(() => {
        fetchPredictions();
        fetchHistoricalData();
    }, [symbol, days]);

    const fetchPredictions = async () => {
        try {
            setLoading(true);
            const response = await fetch(`http://localhost:8000/predict/${symbol}?days=${days}`);

            if (!response.ok) {
                throw new Error('Failed to fetch predictions');
            }

            const data = await response.json();
            setPredictions(data.predictions);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    };

    const fetchHistoricalData = async () => {
        try {
            const response = await fetch(`http://localhost:8000/stock/${symbol}`);
            if (response.ok) {
                const data = await response.json();
                // Get last 30 days
                const recent = data.data.slice(-30);
                setHistoricalData(recent);
            }
        } catch (err) {
            console.error('Error fetching historical data:', err);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96 bg-slate-900 rounded-xl border border-slate-800">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto mb-4"></div>
                    <p className="text-slate-400">Generating ensemble predictions...</p>
                    <p className="text-xs text-slate-600 mt-1">Combining LSTM, Transformer, Prophet & XGBoost</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-slate-900 rounded-xl border border-red-500/30 p-6">
                <div className="flex items-center gap-3 text-red-400">
                    <AlertCircle size={24} />
                    <div>
                        <h3 className="font-semibold">Error Loading Predictions</h3>
                        <p className="text-sm text-slate-400">{error}</p>
                        <p className="text-xs text-slate-500 mt-2">Make sure all 4 models are trained for {symbol}</p>
                    </div>
                </div>
            </div>
        );
    }

    // Filter predictions by selected range
    const filteredPredictions = predictions.slice(0, selectedRange);

    // Combine historical and prediction data for chart
    const chartData = [
        ...historicalData.map(d => ({
            date: d.date,
            actual: d.close,
            type: 'historical'
        })),
        ...filteredPredictions.map(p => ({
            date: p.target_date,
            predicted: p.predicted_price,
            lower: p.confidence_lower,
            upper: p.confidence_upper,
            type: 'prediction'
        }))
    ];

    // Calculate trend
    const firstPred = predictions[0]?.predicted_price || 0;
    const lastPred = filteredPredictions[filteredPredictions.length - 1]?.predicted_price || 0;
    const lastHistorical = historicalData.length > 0 ? historicalData[historicalData.length - 1].close : firstPred;
    const trend = lastPred > lastHistorical ? 'up' : 'down';
    const trendPercent = lastHistorical > 0 ? ((lastPred - lastHistorical) / lastHistorical * 100).toFixed(2) : '0.00';

    // Calculate confidence spread for last prediction
    const lastPredData = filteredPredictions[filteredPredictions.length - 1];
    const confidenceSpread = lastPredData
        ? ((lastPredData.confidence_upper - lastPredData.confidence_lower) / lastPredData.predicted_price * 100).toFixed(1)
        : '0';

    return (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-xl font-bold text-white flex items-center gap-2">
                        <Layers className="text-indigo-400" size={24} />
                        {symbol} Ensemble Forecast
                        {trend === 'up' ? (
                            <TrendingUp className="text-emerald-400" size={24} />
                        ) : (
                            <TrendingDown className="text-red-400" size={24} />
                        )}
                    </h3>
                    <p className="text-sm text-slate-400 mt-1">
                        4-model weighted ensemble with dynamic confidence intervals
                    </p>
                </div>

                <div className="flex items-center gap-4">
                    {/* Range Selector */}
                    <div className="flex items-center gap-1 bg-slate-800 rounded-lg p-1">
                        {([7, 14, 30] as const).map(range => (
                            <button
                                key={range}
                                onClick={() => setSelectedRange(range)}
                                className={`px-3 py-1 text-xs rounded-md font-medium transition-all ${
                                    selectedRange === range
                                        ? 'bg-indigo-600 text-white'
                                        : 'text-slate-400 hover:text-white'
                                }`}
                            >
                                {range}D
                            </button>
                        ))}
                    </div>

                    <div className="text-right">
                        <div className={`text-2xl font-bold ${trend === 'up' ? 'text-emerald-400' : 'text-red-400'}`}>
                            {trend === 'up' ? '+' : ''}{trendPercent}%
                        </div>
                        <p className="text-xs text-slate-500">{selectedRange}-day forecast</p>
                    </div>
                </div>
            </div>

            {/* Chart */}
            <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={chartData}>
                    <defs>
                        <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0.05} />
                        </linearGradient>
                        <linearGradient id="historicalGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.15} />
                            <stop offset="95%" stopColor="#10b981" stopOpacity={0.02} />
                        </linearGradient>
                    </defs>

                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />

                    <XAxis
                        dataKey="date"
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                        tickFormatter={(value) => {
                            const date = new Date(value);
                            return `${date.getMonth() + 1}/${date.getDate()}`;
                        }}
                    />

                    <YAxis
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                        tickFormatter={(value) => `$${value.toFixed(0)}`}
                        domain={['auto', 'auto']}
                    />

                    <Tooltip
                        contentStyle={{
                            backgroundColor: '#1e293b',
                            border: '1px solid #334155',
                            borderRadius: '8px',
                            color: '#e2e8f0',
                            fontSize: '12px',
                        }}
                        formatter={(value: any, name: string) => {
                            const labels: Record<string, string> = {
                                actual: 'Historical Price',
                                predicted: 'Ensemble Prediction',
                                upper: 'Confidence Upper',
                                lower: 'Confidence Lower',
                            };
                            return [`$${Number(value).toFixed(2)}`, labels[name] || name];
                        }}
                        labelFormatter={(label) => {
                            const d = new Date(label);
                            return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
                        }}
                    />

                    <Legend
                        wrapperStyle={{ paddingTop: '20px', fontSize: '12px' }}
                        iconType="line"
                    />

                    {/* Confidence interval area */}
                    <Area
                        type="monotone"
                        dataKey="upper"
                        stroke="none"
                        fill="url(#confidenceGradient)"
                        fillOpacity={1}
                        name="Confidence Range"
                    />

                    {/* Historical actual prices */}
                    <Area
                        type="monotone"
                        dataKey="actual"
                        stroke="#10b981"
                        strokeWidth={2}
                        fill="url(#historicalGradient)"
                        name="Historical Price"
                        dot={false}
                    />

                    {/* Predicted prices */}
                    <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#6366f1"
                        strokeWidth={3}
                        strokeDasharray="5 5"
                        dot={{ fill: '#6366f1', r: 3, strokeWidth: 0 }}
                        activeDot={{ r: 6, fill: '#6366f1', stroke: '#818cf8', strokeWidth: 2 }}
                        name="Ensemble Prediction"
                    />

                    {/* Lower bound */}
                    <Line
                        type="monotone"
                        dataKey="lower"
                        stroke="#6366f1"
                        strokeWidth={1}
                        strokeOpacity={0.3}
                        strokeDasharray="2 4"
                        dot={false}
                        name="Confidence Lower"
                    />
                </AreaChart>
            </ResponsiveContainer>

            {/* Key Predictions Cards */}
            <div className="grid grid-cols-4 gap-4 mt-6">
                {[7, 14, 30].map(day => {
                    const pred = predictions[day - 1];
                    if (!pred) return null;

                    const changeFromCurrent = lastHistorical > 0
                        ? ((pred.predicted_price - lastHistorical) / lastHistorical * 100).toFixed(2)
                        : '0.00';
                    const isPositive = Number(changeFromCurrent) >= 0;

                    return (
                        <div key={day} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                            <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                                <Calendar size={14} />
                                {day}-Day
                            </div>
                            <div className="text-xl font-bold text-white">
                                ${pred.predicted_price.toFixed(2)}
                            </div>
                            <div className={`text-sm font-medium ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                                {isPositive ? '+' : ''}{changeFromCurrent}%
                            </div>
                            <div className="text-[10px] text-slate-500 mt-1">
                                ${pred.confidence_lower.toFixed(2)} — ${pred.confidence_upper.toFixed(2)}
                            </div>
                        </div>
                    );
                })}

                {/* Confidence Spread Card */}
                <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                    <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                        <BarChart3 size={14} />
                        Uncertainty
                    </div>
                    <div className="text-xl font-bold text-amber-400">
                        {confidenceSpread}%
                    </div>
                    <div className="text-[10px] text-slate-500 mt-1">
                        Confidence spread at {selectedRange}D horizon
                    </div>
                    <div className="text-[10px] text-slate-600 mt-0.5">
                        Widens with prediction distance
                    </div>
                </div>
            </div>

            {/* Disclaimer */}
            <div className="mt-6 p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                <p className="text-xs text-amber-200 flex items-center gap-2">
                    <AlertCircle size={16} className="shrink-0" />
                    <span>
                        Predictions are generated by a 4-model ensemble (LSTM+Attention, Transformer, Prophet, XGBoost) 
                        using adaptive softmax weighting. Confidence intervals widen with prediction horizon. 
                        This is not financial advice. Always do your own research.
                    </span>
                </p>
            </div>
        </div>
    );
};

export default PredictionChart;
