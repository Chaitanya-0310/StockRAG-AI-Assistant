import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { TrendingUp, TrendingDown, AlertCircle, Calendar } from 'lucide-react';

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
                    <p className="text-slate-400">Loading predictions...</p>
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
                        <p className="text-xs text-slate-500 mt-2">Make sure models are trained for {symbol}</p>
                    </div>
                </div>
            </div>
        );
    }

    // Combine historical and prediction data for chart
    const chartData = [
        ...historicalData.map(d => ({
            date: d.date,
            actual: d.close,
            type: 'historical'
        })),
        ...predictions.map(p => ({
            date: p.target_date,
            predicted: p.predicted_price,
            lower: p.confidence_lower,
            upper: p.confidence_upper,
            type: 'prediction'
        }))
    ];

    // Calculate trend
    const firstPred = predictions[0]?.predicted_price || 0;
    const lastPred = predictions[predictions.length - 1]?.predicted_price || 0;
    const trend = lastPred > firstPred ? 'up' : 'down';
    const trendPercent = ((lastPred - firstPred) / firstPred * 100).toFixed(2);

    return (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-xl font-bold text-white flex items-center gap-2">
                        {symbol} Price Forecast
                        {trend === 'up' ? (
                            <TrendingUp className="text-emerald-400" size={24} />
                        ) : (
                            <TrendingDown className="text-red-400" size={24} />
                        )}
                    </h3>
                    <p className="text-sm text-slate-400 mt-1">
                        {days}-day prediction with confidence intervals
                    </p>
                </div>

                <div className="text-right">
                    <div className={`text-2xl font-bold ${trend === 'up' ? 'text-emerald-400' : 'text-red-400'}`}>
                        {trend === 'up' ? '+' : ''}{trendPercent}%
                    </div>
                    <p className="text-xs text-slate-500">Expected change</p>
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
                    </defs>

                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />

                    <XAxis
                        dataKey="date"
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 12 }}
                        tickFormatter={(value) => {
                            const date = new Date(value);
                            return `${date.getMonth() + 1}/${date.getDate()}`;
                        }}
                    />

                    <YAxis
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 12 }}
                        tickFormatter={(value) => `$${value.toFixed(0)}`}
                    />

                    <Tooltip
                        contentStyle={{
                            backgroundColor: '#1e293b',
                            border: '1px solid #334155',
                            borderRadius: '8px',
                            color: '#e2e8f0'
                        }}
                        formatter={(value: any) => [`$${value.toFixed(2)}`, '']}
                    />

                    <Legend
                        wrapperStyle={{ paddingTop: '20px' }}
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
                    <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                        name="Historical Price"
                    />

                    {/* Predicted prices */}
                    <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#6366f1"
                        strokeWidth={3}
                        strokeDasharray="5 5"
                        dot={{ fill: '#6366f1', r: 4 }}
                        name="Predicted Price"
                    />

                    {/* Lower bound */}
                    <Line
                        type="monotone"
                        dataKey="lower"
                        stroke="#6366f1"
                        strokeWidth={1}
                        strokeOpacity={0.3}
                        dot={false}
                        name="Lower Bound"
                    />
                </AreaChart>
            </ResponsiveContainer>

            {/* Key Predictions */}
            <div className="grid grid-cols-3 gap-4 mt-6">
                {[7, 14, 30].map(day => {
                    const pred = predictions[day - 1];
                    if (!pred) return null;

                    return (
                        <div key={day} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                            <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                                <Calendar size={16} />
                                {day}-Day Forecast
                            </div>
                            <div className="text-2xl font-bold text-white">
                                ${pred.predicted_price.toFixed(2)}
                            </div>
                            <div className="text-xs text-slate-500 mt-1">
                                Range: ${pred.confidence_lower.toFixed(2)} - ${pred.confidence_upper.toFixed(2)}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Disclaimer */}
            <div className="mt-6 p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                <p className="text-xs text-amber-200 flex items-center gap-2">
                    <AlertCircle size={16} />
                    <span>
                        ⚠️ These predictions are based on historical patterns and ML models.
                        Not financial advice. Always do your own research before making investment decisions.
                    </span>
                </p>
            </div>
        </div>
    );
};

export default PredictionChart;
