import React, { useEffect, useState } from 'react';
import { Brain, CheckCircle, XCircle, Clock, TrendingUp, AlertTriangle, RefreshCw, BarChart3, Target, Zap } from 'lucide-react';

interface ModelMetrics {
    rmse: number;
    mae: number;
    r2?: number;
    mape?: number;
    directional_accuracy: number;
}

interface ModelInfo {
    type: string;
    version: string;
    training_date: string;
    metrics: ModelMetrics;
    training_samples: number;
    is_active: boolean;
}

interface ModelStatus {
    symbol: string;
    models: ModelInfo[];
    status: string;
}

interface ModelDashboardProps {
    symbols?: string[];
}

const MODEL_TYPE_META: Record<string, { icon: string; label: string; color: string; description: string }> = {
    lstm: {
        icon: '🧠',
        label: 'LSTM + Attention',
        color: 'text-purple-400',
        description: 'Deep recurrent network with multi-head self-attention and residual connections',
    },
    transformer: {
        icon: '⚡',
        label: 'Transformer',
        color: 'text-cyan-400',
        description: 'Pure encoder-based Transformer with positional encoding for temporal patterns',
    },
    prophet: {
        icon: '📈',
        label: 'Prophet',
        color: 'text-blue-400',
        description: 'Bayesian time-series model with automatic seasonality and trend detection',
    },
    xgboost: {
        icon: '🌲',
        label: 'XGBoost',
        color: 'text-orange-400',
        description: 'Gradient boosted trees with cross-validated hyperparameter tuning',
    },
};

const ModelDashboard: React.FC<ModelDashboardProps> = ({
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
}) => {
    const [modelStatuses, setModelStatuses] = useState<Record<string, ModelStatus>>({});
    const [loading, setLoading] = useState(true);
    const [training, setTraining] = useState<Record<string, boolean>>({});

    useEffect(() => {
        fetchAllModelStatuses();
    }, []);

    const fetchAllModelStatuses = async () => {
        setLoading(true);
        const statuses: Record<string, ModelStatus> = {};

        for (const symbol of symbols) {
            try {
                const response = await fetch(`http://localhost:8000/models/${symbol}`);
                if (response.ok) {
                    const data = await response.json();
                    statuses[symbol] = data;
                }
            } catch (err) {
                console.error(`Error fetching status for ${symbol}:`, err);
            }
        }

        setModelStatuses(statuses);
        setLoading(false);
    };

    const trainModel = async (symbol: string) => {
        setTraining(prev => ({ ...prev, [symbol]: true }));

        try {
            const response = await fetch(`http://localhost:8000/train/${symbol}`, {
                method: 'POST'
            });

            if (response.ok) {
                setTimeout(() => {
                    fetchAllModelStatuses();
                    setTraining(prev => ({ ...prev, [symbol]: false }));
                }, 2000);
            } else {
                setTraining(prev => ({ ...prev, [symbol]: false }));
                alert(`Failed to train models for ${symbol}`);
            }
        } catch (err) {
            setTraining(prev => ({ ...prev, [symbol]: false }));
            alert(`Error training models for ${symbol}`);
        }
    };

    const getAccuracyColor = (accuracy: number) => {
        if (accuracy >= 70) return 'text-emerald-400';
        if (accuracy >= 60) return 'text-yellow-400';
        return 'text-red-400';
    };

    const getR2Color = (r2: number) => {
        if (r2 >= 0.8) return 'text-emerald-400';
        if (r2 >= 0.5) return 'text-yellow-400';
        return 'text-red-400';
    };

    const getMapeColor = (mape: number) => {
        if (mape <= 5) return 'text-emerald-400';
        if (mape <= 15) return 'text-yellow-400';
        return 'text-red-400';
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto mb-4"></div>
                    <p className="text-slate-400">Loading model statuses...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                        <Brain className="text-indigo-400" size={28} />
                        ML Model Dashboard
                    </h2>
                    <p className="text-slate-400 mt-1">
                        4-model ensemble: LSTM + Attention, Transformer, Prophet, XGBoost
                    </p>
                </div>

                <button
                    onClick={fetchAllModelStatuses}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg border border-slate-700 transition-colors flex items-center gap-2"
                >
                    <RefreshCw size={16} />
                    Refresh
                </button>
            </div>

            {/* Model Cards Grid */}
            <div className="flex-1 overflow-y-auto grid grid-cols-1 xl:grid-cols-2 gap-6">
                {symbols.map(symbol => {
                    const status = modelStatuses[symbol];
                    const isTrained = status?.status === 'trained';
                    const isTraining = training[symbol];
                    const modelList = isTrained ? status.models : [];

                    // Best model by directional accuracy
                    const bestModel = modelList.length > 0
                        ? modelList.reduce((prev, cur) =>
                            (prev.metrics.directional_accuracy > cur.metrics.directional_accuracy) ? prev : cur
                        )
                        : null;

                    return (
                        <div
                            key={symbol}
                            className="bg-slate-900 rounded-xl border border-slate-800 p-6 hover:border-slate-700 transition-colors flex flex-col"
                        >
                            {/* Symbol Header */}
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <div className="w-12 h-12 bg-indigo-600/20 rounded-lg flex items-center justify-center">
                                        <span className="text-xl font-bold text-indigo-400">{symbol}</span>
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold text-white">{symbol}</h3>
                                        <div className="flex items-center gap-2 mt-1">
                                            {isTrained ? (
                                                <span className="flex items-center gap-1 text-xs text-emerald-400">
                                                    <CheckCircle size={14} />
                                                    {modelList.length} Models Trained
                                                </span>
                                            ) : (
                                                <span className="flex items-center gap-1 text-xs text-slate-500">
                                                    <XCircle size={14} />
                                                    Not Trained
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <button
                                    onClick={() => trainModel(symbol)}
                                    disabled={isTraining}
                                    className={`px-4 py-2 rounded-lg font-medium transition-all ${isTraining
                                        ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                                        : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                                    }`}
                                >
                                    {isTraining ? (
                                        <span className="flex items-center gap-2">
                                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                                            Training...
                                        </span>
                                    ) : (
                                        'Train Models'
                                    )}
                                </button>
                            </div>

                            {/* Best Model Highlight */}
                            {bestModel && (
                                <div className="bg-gradient-to-r from-indigo-500/10 to-purple-500/10 rounded-lg p-4 mb-4 border border-indigo-500/20">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <p className="text-xs text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                                <Target size={12} />
                                                Best Model — Directional Accuracy
                                            </p>
                                            <p className={`text-3xl font-bold ${getAccuracyColor(bestModel.metrics.directional_accuracy)}`}>
                                                {bestModel.metrics.directional_accuracy.toFixed(1)}%
                                            </p>
                                        </div>
                                        <div className="text-right">
                                            <p className="text-xs text-slate-500">
                                                Type: <span className={`font-medium ${MODEL_TYPE_META[bestModel.type]?.color || 'text-white'}`}>
                                                    {MODEL_TYPE_META[bestModel.type]?.label || bestModel.type}
                                                </span>
                                            </p>
                                            <p className="text-xs text-slate-500">Version: <span className="text-white">{bestModel.version}</span></p>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Individual Model Cards */}
                            {modelList.length > 0 && (
                                <div className="space-y-3">
                                    {modelList.map((model, idx) => {
                                        const meta = MODEL_TYPE_META[model.type] || {
                                            icon: '🤖', label: model.type, color: 'text-white', description: ''
                                        };

                                        return (
                                            <div
                                                key={`${model.type}-${idx}`}
                                                className="bg-slate-800/50 rounded-lg p-4 border border-slate-700 hover:border-slate-600 transition-colors"
                                            >
                                                {/* Model Header */}
                                                <div className="flex items-center justify-between mb-3">
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-xl">{meta.icon}</span>
                                                        <div>
                                                            <h4 className={`font-semibold ${meta.color}`}>
                                                                {meta.label}
                                                            </h4>
                                                            <p className="text-[10px] text-slate-600 max-w-[260px] truncate">{meta.description}</p>
                                                        </div>
                                                    </div>
                                                    <div className={`text-right ${getAccuracyColor(model.metrics.directional_accuracy)}`}>
                                                        <div className="text-xl font-bold">
                                                            {model.metrics.directional_accuracy.toFixed(1)}%
                                                        </div>
                                                        <p className="text-[10px] text-slate-500">Dir. Accuracy</p>
                                                    </div>
                                                </div>

                                                {/* Metrics Grid */}
                                                <div className="grid grid-cols-5 gap-2 text-sm">
                                                    <div>
                                                        <p className="text-slate-500 text-[10px] uppercase">RMSE</p>
                                                        <p className="text-white font-medium text-xs">{model.metrics.rmse.toFixed(3)}</p>
                                                    </div>
                                                    <div>
                                                        <p className="text-slate-500 text-[10px] uppercase">MAE</p>
                                                        <p className="text-white font-medium text-xs">{model.metrics.mae.toFixed(3)}</p>
                                                    </div>
                                                    <div>
                                                        <p className="text-slate-500 text-[10px] uppercase">R&sup2;</p>
                                                        <p className={`font-medium text-xs ${model.metrics.r2 !== undefined ? getR2Color(model.metrics.r2) : 'text-slate-400'}`}>
                                                            {model.metrics.r2 !== undefined ? model.metrics.r2.toFixed(3) : 'N/A'}
                                                        </p>
                                                    </div>
                                                    <div>
                                                        <p className="text-slate-500 text-[10px] uppercase">MAPE</p>
                                                        <p className={`font-medium text-xs ${model.metrics.mape !== undefined ? getMapeColor(model.metrics.mape) : 'text-slate-400'}`}>
                                                            {model.metrics.mape !== undefined ? `${model.metrics.mape.toFixed(1)}%` : 'N/A'}
                                                        </p>
                                                    </div>
                                                    <div>
                                                        <p className="text-slate-500 text-[10px] uppercase">Samples</p>
                                                        <p className="text-white font-medium text-xs">{model.training_samples.toLocaleString()}</p>
                                                    </div>
                                                </div>

                                                {/* Footer */}
                                                <div className="mt-3 pt-2 border-t border-slate-700 flex items-center justify-between text-[10px] text-slate-500">
                                                    <span className="flex items-center gap-1">
                                                        <Clock size={10} />
                                                        {new Date(model.training_date).toLocaleDateString()}
                                                    </span>
                                                    <span>{model.version}</span>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}

                            {!isTrained && (
                                <div className="text-center py-8 text-slate-500">
                                    <AlertTriangle className="mx-auto mb-2" size={32} />
                                    <p className="text-sm">No models trained yet</p>
                                    <p className="text-xs mt-1">Click "Train Models" to start 4-model ensemble training</p>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Info Box */}
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mt-6">
                <p className="text-sm text-blue-200">
                    <strong>💡 Enhanced Prediction System:</strong> The ensemble combines{' '}
                    <span className="text-purple-300">LSTM + Attention</span>,{' '}
                    <span className="text-cyan-300">Transformer</span>,{' '}
                    <span className="text-blue-300">Prophet</span>, and{' '}
                    <span className="text-orange-300">XGBoost</span> using softmax-weighted averaging.
                    Models are weighted by directional accuracy, inverse RMSE, and R&sup2; score for optimal predictions.
                </p>
            </div>
        </div>
    );
};

export default ModelDashboard;
