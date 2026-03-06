import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from prophet import Prophet
import joblib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from backend.models.models import StockPrice, MLModel, StockPrediction
from backend.config import settings
from backend.services.db_service import AsyncSessionLocal
import pandas_ta as ta
from pathlib import Path
import mlflow
import mlflow.pytorch
import mlflow.prophet
import mlflow.xgboost
import mlflow.sklearn

logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

# Model storage directory
MODEL_DIR = Path(__file__).parent.parent / "ml_models"
MODEL_DIR.mkdir(exist_ok=True)


class LSTMModel(nn.Module):
    """LSTM Neural Network for time series prediction"""

    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


class StockDataset(Dataset):
    """PyTorch Dataset for stock price sequences"""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


async def prepare_training_data(symbol: str, session: AsyncSession, lookback: int = 60) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Fetch and prepare training data with technical indicators."""
    stmt = select(StockPrice).where(StockPrice.symbol == symbol).order_by(StockPrice.date)
    result = await session.execute(stmt)
    prices = result.scalars().all()

    if len(prices) < lookback + 30:
        raise ValueError(f"Insufficient data for {symbol}. Need at least {lookback + 30} days.")

    df = pd.DataFrame([{
        'date': p.date,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume
    } for p in prices])

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Add technical indicators
    df['SMA_10'] = ta.sma(df['close'], length=10)
    df['SMA_30'] = ta.sma(df['close'], length=30)
    df['RSI'] = ta.rsi(df['close'], length=14)

    macd = ta.macd(df['close'])
    if macd is not None and not macd.empty:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']

    bbands = ta.bbands(df['close'], length=20)
    if bbands is not None and not bbands.empty:
        df['BB_upper'] = bbands.iloc[:, 2]
        df['BB_lower'] = bbands.iloc[:, 0]

    df['volume_sma'] = ta.sma(df['volume'], length=20)
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()

    df.dropna(inplace=True)

    return df, MinMaxScaler()


def create_sequences(data: np.ndarray, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # Predict close price (first column)
    return np.array(X), np.array(y)


def _train_lstm_sync(symbol: str, df: pd.DataFrame, scaler: MinMaxScaler, epochs: int) -> Dict:
    """Synchronous LSTM training (runs in thread)."""
    logger.info(f"Training LSTM model for {symbol}...")

    feature_cols = ['close', 'volume', 'SMA_10', 'SMA_30', 'RSI']
    features = df[feature_cols].values

    scaler.fit(features)
    normalized_data = scaler.transform(features)

    lookback = settings.lstm_lookback
    X, y = create_sequences(normalized_data, lookback=lookback)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=settings.lstm_hidden_size,
        num_layers=settings.lstm_num_layers
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            logger.info(f"LSTM Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions.extend(outputs.squeeze().numpy())
            actuals.extend(batch_y.numpy())

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    actual_direction = np.diff(actuals) > 0
    pred_direction = np.diff(predictions) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    model_path = MODEL_DIR / f"{symbol}_lstm.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'lookback': lookback
    }, model_path)

    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'directional_accuracy': float(directional_accuracy)
    }

    logger.info(f"LSTM trained. RMSE: {rmse:.4f}, Dir Acc: {directional_accuracy:.2f}%")

    try:
        with mlflow.start_run(run_name=f"LSTM_{symbol}"):
            mlflow.log_params({
                "model_type": "lstm", "symbol": symbol, "epochs": epochs,
                "hidden_size": settings.lstm_hidden_size,
                "num_layers": settings.lstm_num_layers,
                "lookback": lookback
            })
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, "lstm_model")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    return {
        'model_path': str(model_path),
        'metrics': metrics,
        'training_samples': len(X_train)
    }


def _train_prophet_sync(symbol: str, prices_data: list) -> Dict:
    """Synchronous Prophet training (runs in thread)."""
    logger.info(f"Training Prophet model for {symbol}...")

    df = pd.DataFrame(prices_data)

    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=settings.prophet_changepoint_prior_scale
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)

    test_predictions = forecast.tail(len(test_df))['yhat'].values
    test_actuals = test_df['y'].values

    rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
    mae = mean_absolute_error(test_actuals, test_predictions)

    actual_direction = np.diff(test_actuals) > 0
    pred_direction = np.diff(test_predictions) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    model_path = MODEL_DIR / f"{symbol}_prophet.pkl"
    joblib.dump(model, model_path)

    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'directional_accuracy': float(directional_accuracy)
    }

    logger.info(f"Prophet trained. RMSE: {rmse:.4f}, Dir Acc: {directional_accuracy:.2f}%")

    try:
        with mlflow.start_run(run_name=f"Prophet_{symbol}"):
            mlflow.log_params({
                "model_type": "prophet", "symbol": symbol,
                "changepoint_prior_scale": settings.prophet_changepoint_prior_scale
            })
            mlflow.log_metrics(metrics)
            mlflow.prophet.log_model(model, "prophet_model")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    return {
        'model_path': str(model_path),
        'metrics': metrics,
        'training_samples': len(train_df)
    }


def _train_xgboost_sync(symbol: str, df: pd.DataFrame, scaler: MinMaxScaler) -> Dict:
    """Synchronous XGBoost training (runs in thread)."""
    logger.info(f"Training XGBoost model for {symbol}...")

    # Create lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

    df.dropna(inplace=True)

    feature_cols = [col for col in df.columns if col not in ['close', 'open', 'high', 'low']]
    X = df[feature_cols].values
    y = df['close'].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=settings.xgboost_n_estimators,
        max_depth=settings.xgboost_max_depth,
        learning_rate=settings.xgboost_learning_rate,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    actual_direction = np.diff(y_test) > 0
    pred_direction = np.diff(predictions) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    # Store feature importances
    feature_importances = dict(zip(feature_cols, model.feature_importances_.tolist()))

    model_path = MODEL_DIR / f"{symbol}_xgboost.pkl"
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'scaler': scaler
    }, model_path)

    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'directional_accuracy': float(directional_accuracy),
        'feature_importances': feature_importances
    }

    logger.info(f"XGBoost trained. RMSE: {rmse:.4f}, Dir Acc: {directional_accuracy:.2f}%")

    try:
        with mlflow.start_run(run_name=f"XGBoost_{symbol}"):
            mlflow.log_params({
                "model_type": "xgboost", "symbol": symbol,
                "n_estimators": settings.xgboost_n_estimators,
                "max_depth": settings.xgboost_max_depth,
                "learning_rate": settings.xgboost_learning_rate
            })
            mlflow.log_metrics({
                "rmse": float(rmse),
                "mae": float(mae),
                "directional_accuracy": float(directional_accuracy)
            })
            mlflow.sklearn.log_model(model, "xgboost_model")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    return {
        'model_path': str(model_path),
        'metrics': metrics,
        'training_samples': len(X_train)
    }


async def train_lstm_model(symbol: str, session: AsyncSession, epochs: int = None) -> Dict:
    """Train LSTM model for stock price prediction."""
    if epochs is None:
        epochs = settings.lstm_epochs
    df, scaler = await prepare_training_data(symbol, session)
    return await asyncio.to_thread(_train_lstm_sync, symbol, df.copy(), scaler, epochs)


async def train_prophet_model(symbol: str, session: AsyncSession) -> Dict:
    """Train Prophet model for stock price prediction."""
    stmt = select(StockPrice).where(StockPrice.symbol == symbol).order_by(StockPrice.date)
    result = await session.execute(stmt)
    prices = result.scalars().all()

    prices_data = [{'ds': p.date, 'y': p.close} for p in prices]
    return await asyncio.to_thread(_train_prophet_sync, symbol, prices_data)


async def train_xgboost_model(symbol: str, session: AsyncSession) -> Dict:
    """Train XGBoost model for stock price prediction."""
    df, scaler = await prepare_training_data(symbol, session)
    return await asyncio.to_thread(_train_xgboost_sync, symbol, df.copy(), scaler)


async def train_all_models(symbol: str, session: AsyncSession) -> Dict[str, Dict]:
    """
    Train all prediction models (LSTM, Prophet, XGBoost) in parallel.
    Uses asyncio.gather with separate DB sessions per task for thread safety.
    """
    async def _train_and_save(train_fn, model_type, hyperparams):
        """Train a model using its own DB session, return result."""
        async with AsyncSessionLocal() as task_session:
            try:
                result = await train_fn(symbol, task_session)
                return model_type, result, hyperparams
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                return model_type, {'error': str(e)}, hyperparams

    # Launch all three training tasks in parallel
    tasks = [
        _train_and_save(
            train_lstm_model, 'lstm',
            {'epochs': settings.lstm_epochs, 'hidden_size': settings.lstm_hidden_size, 'num_layers': settings.lstm_num_layers}
        ),
        _train_and_save(
            train_prophet_model, 'prophet',
            {'changepoint_prior_scale': settings.prophet_changepoint_prior_scale}
        ),
        _train_and_save(
            train_xgboost_model, 'xgboost',
            {'n_estimators': settings.xgboost_n_estimators, 'max_depth': settings.xgboost_max_depth, 'learning_rate': settings.xgboost_learning_rate}
        ),
    ]

    completed = await asyncio.gather(*tasks)

    results = {}
    for model_type, result, hyperparams in completed:
        results[model_type] = result
        if 'error' not in result:
            model_record = MLModel(
                symbol=symbol,
                model_type=model_type,
                version='v1.0',
                file_path=result['model_path'],
                metrics=result['metrics'],
                training_samples=result['training_samples'],
                is_active=True,
                hyperparameters=hyperparams
            )
            session.add(model_record)

    await session.commit()
    return results
