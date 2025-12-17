import React, { useEffect, useState } from 'react';
import { Brain, CheckCircle, XCircle, Clock, TrendingUp, AlertTriangle, RefreshCw } from 'lucide-react';

interface ModelMetrics {
    rmse: number;
    mae: number;
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
                // Refresh status after training
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

    const getModelTypeIcon = (type: string) => {
        switch (type) {
            case 'lstm':
                return '🧠';
            case 'prophet':
                return '📈';
            case 'xgboost':
                return '🌲';
            default:
                return '🤖';
        }
    };

    const getAccuracyColor = (accuracy: number) => {
        if (accuracy >= 70) return 'text-emerald-400';
        if (accuracy >= 60) return 'text-yellow-400';
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
                    <p className="text-slate-400 mt-1">Training status and performance metrics</p>
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

                    // Find the model with the highest directional accuracy to display as the "latest" percentage
                    const latestModel = isTrained
                        ? status.models.reduce((prev, current) => 
                            (prev.metrics.directional_accuracy > current.metrics.directional_accuracy) ? prev : current
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
                                                    Trained
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

                            {/* Latest Accuracy Percentage */}
                            {latestModel && (
                                <div className="bg-slate-800/50 rounded-lg p-4 mb-4 border border-slate-700 flex items-center justify-between">
                                    <div>
                                        <p className="text-xs text-slate-500 uppercase tracking-wider">Latest Directional Accuracy</p>
                                        <p className={`text-3xl font-bold ${getAccuracyColor(latestModel.metrics.directional_accuracy)}`}>
                                            {latestModel.metrics.directional_accuracy.toFixed(1)}%
                                        </p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-xs text-slate-500">Model Type: <span className="text-white capitalize">{latestModel.type}</span></p>
                                        <p className="text-xs text-slate-500">Version: <span className="text-white">{latestModel.version}</span></p>
                                    </div>
                                </div>
                            )}

            {/* Models List */}
            {latestModel && (
                <div
                    className="bg-slate-800/50 rounded-lg p-4 border border-slate-700 mt-4"
                >
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                            <span className="text-2xl">{getModelTypeIcon(latestModel.type)}</span>
                            <div>
                                <h4 className="font-semibold text-white capitalize">
                                    {latestModel.type}
                                </h4>
                                <p className="text-xs text-slate-500">{latestModel.version}</p>
                            </div>
                        </div>

                        <div className={`text-right ${getAccuracyColor(latestModel.metrics.directional_accuracy)}`}>
                            <div className="text-2xl font-bold">
                                {latestModel.metrics.directional_accuracy.toFixed(1)}%
                            </div>
                            <p className="text-xs text-slate-500">Accuracy</p>
                        </div>
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-3 gap-3 text-sm">
                        <div>
                            <p className="text-slate-500 text-xs">RMSE</p>
                            <p className="text-white font-medium">{latestModel.metrics.rmse.toFixed(2)}</p>
                        </div>
                        <div>
                            <p className="text-slate-500 text-xs">MAE</p>
                            <p className="text-white font-medium">{latestModel.metrics.mae.toFixed(2)}</p>
                        </div>
                        <div>
                            <p className="text-slate-500 text-xs">Samples</p>
                            <p className="text-white font-medium">{latestModel.training_samples}</p>
                        </div>
                    </div>

                    {/* Training Date */}
                    <div className="mt-3 pt-3 border-t border-slate-700 flex items-center gap-2 text-xs text-slate-400">
                        <Clock size={12} />
                        Trained: {new Date(latestModel.training_date).toLocaleDateString()}
                    </div>
                </div>
            )}
            {!isTrained && (
                <div className="text-center py-8 text-slate-500">
                    <AlertTriangle className="mx-auto mb-2" size={32} />
                    <p className="text-sm">No models trained yet</p>
                    <p className="text-xs mt-1">Click "Train Models" to get started</p>
                </div>
            )}
                        </div>
                    );
                })}
            </div>

            {/* Info Box */}
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mt-6">
                <p className="text-sm text-blue-200">
                    <strong>💡 Tip:</strong> Models should be retrained weekly to capture new market patterns.
                    Higher directional accuracy ({">"} 60%) indicates better prediction reliability.
                </p>
            </div>
        </div>
    );
};

export default ModelDashboard;
