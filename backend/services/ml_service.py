import os
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
import pandas_ta as ta
from pathlib import Path

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
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
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
    """
    Fetch and prepare training data with technical indicators
    
    Args:
        symbol: Stock symbol
        session: Database session
        lookback: Number of days to look back for sequence creation
        
    Returns:
        DataFrame with features and fitted scaler
    """
    # Fetch historical data from database
    stmt = select(StockPrice).where(StockPrice.symbol == symbol).order_by(StockPrice.date)
    result = await session.execute(stmt)
    prices = result.scalars().all()
    
    if len(prices) < lookback + 30:
        raise ValueError(f"Insufficient data for {symbol}. Need at least {lookback + 30} days.")
    
    # Convert to DataFrame
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
    
    # MACD
    macd = ta.macd(df['close'])
    if macd is not None and not macd.empty:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
    
    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20)
    if bbands is not None and not bbands.empty:
        df['BB_upper'] = bbands['BBU_20_2.0']
        df['BB_lower'] = bbands['BBL_20_2.0']
    
    # Volume indicators
    df['volume_sma'] = ta.sma(df['volume'], length=20)
    
    # Price changes
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Drop NaN values created by indicators
    df.dropna(inplace=True)
    
    return df, MinMaxScaler()


def create_sequences(data: np.ndarray, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training
    
    Args:
        data: Normalized feature array
        lookback: Number of time steps to look back
        
    Returns:
        X (sequences) and y (targets)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # Predict close price (first column)
    
    return np.array(X), np.array(y)


async def train_lstm_model(symbol: str, session: AsyncSession, epochs: int = 50) -> Dict:
    """
    Train LSTM model for stock price prediction
    
    Args:
        symbol: Stock symbol
        session: Database session
        epochs: Number of training epochs
        
    Returns:
        Dictionary with model metrics
    """
    print(f"Training LSTM model for {symbol}...")
    
    # Prepare data
    df, scaler = await prepare_training_data(symbol, session)
    
    # Select features for LSTM
    feature_cols = ['close', 'volume', 'SMA_10', 'SMA_30', 'RSI']
    features = df[feature_cols].values
    
    # Normalize
    scaler.fit(features)
    normalized_data = scaler.transform(features)
    
    # Create sequences
    X, y = create_sequences(normalized_data, lookback=60)
    
    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create datasets
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = LSTMModel(input_size=len(feature_cols), hidden_size=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
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
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    # Evaluation
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions.extend(outputs.squeeze().numpy())
            actuals.extend(batch_y.numpy())
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    # Directional accuracy
    actual_direction = np.diff(actuals) > 0
    pred_direction = np.diff(predictions) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Save model
    model_path = MODEL_DIR / f"{symbol}_lstm.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'lookback': 60
    }, model_path)
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'directional_accuracy': float(directional_accuracy)
    }
    
    print(f"LSTM Model trained. RMSE: {rmse:.4f}, MAE: {mae:.4f}, Dir Acc: {directional_accuracy:.2f}%")
    
    return {
        'model_path': str(model_path),
        'metrics': metrics,
        'training_samples': len(X_train)
    }


async def train_prophet_model(symbol: str, session: AsyncSession) -> Dict:
    """
    Train Prophet model for stock price prediction
    
    Args:
        symbol: Stock symbol
        session: Database session
        
    Returns:
        Dictionary with model metrics
    """
    print(f"Training Prophet model for {symbol}...")
    
    # Fetch data
    stmt = select(StockPrice).where(StockPrice.symbol == symbol).order_by(StockPrice.date)
    result = await session.execute(stmt)
    prices = result.scalars().all()
    
    # Prepare Prophet format (ds, y)
    df = pd.DataFrame([{
        'ds': p.date,
        'y': p.close
    } for p in prices])
    
    # Train/test split
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    # Initialize and train Prophet
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    model.fit(train_df)
    
    # Make predictions on test set
    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)
    
    # Evaluate
    test_predictions = forecast.tail(len(test_df))['yhat'].values
    test_actuals = test_df['y'].values
    
    rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
    mae = mean_absolute_error(test_actuals, test_predictions)
    
    # Directional accuracy
    actual_direction = np.diff(test_actuals) > 0
    pred_direction = np.diff(test_predictions) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Save model
    model_path = MODEL_DIR / f"{symbol}_prophet.pkl"
    joblib.dump(model, model_path)
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'directional_accuracy': float(directional_accuracy)
    }
    
    print(f"Prophet Model trained. RMSE: {rmse:.4f}, MAE: {mae:.4f}, Dir Acc: {directional_accuracy:.2f}%")
    
    return {
        'model_path': str(model_path),
        'metrics': metrics,
        'training_samples': len(train_df)
    }


async def train_xgboost_model(symbol: str, session: AsyncSession) -> Dict:
    """
    Train XGBoost model for stock price prediction
    
    Args:
        symbol: Stock symbol
        session: Database session
        
    Returns:
        Dictionary with model metrics
    """
    print(f"Training XGBoost model for {symbol}...")
    
    # Prepare data with features
    df, scaler = await prepare_training_data(symbol, session)
    
    # Create lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    df.dropna(inplace=True)
    
    # Select features
    feature_cols = [col for col in df.columns if col not in ['close', 'open', 'high', 'low']]
    X = df[feature_cols].values
    y = df['close'].values
    
    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    # Directional accuracy
    actual_direction = np.diff(y_test) > 0
    pred_direction = np.diff(predictions) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Save model
    model_path = MODEL_DIR / f"{symbol}_xgboost.pkl"
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'scaler': scaler
    }, model_path)
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'directional_accuracy': float(directional_accuracy)
    }
    
    print(f"XGBoost Model trained. RMSE: {rmse:.4f}, MAE: {mae:.4f}, Dir Acc: {directional_accuracy:.2f}%")
    
    return {
        'model_path': str(model_path),
        'metrics': metrics,
        'training_samples': len(X_train)
    }


async def train_all_models(symbol: str, session: AsyncSession) -> Dict[str, Dict]:
    """
    Train all prediction models (LSTM, Prophet, XGBoost) for a symbol
    
    Args:
        symbol: Stock symbol
        session: Database session
        
    Returns:
        Dictionary with results for each model type
    """
    results = {}
    
    try:
        # Train LSTM
        lstm_result = await train_lstm_model(symbol, session)
        results['lstm'] = lstm_result
        
        # Save to database
        model_record = MLModel(
            symbol=symbol,
            model_type='lstm',
            version='v1.0',
            file_path=lstm_result['model_path'],
            metrics=lstm_result['metrics'],
            training_samples=lstm_result['training_samples'],
            is_active=True,
            hyperparameters={'epochs': 50, 'hidden_size': 64, 'num_layers': 2}
        )
        session.add(model_record)
        
    except Exception as e:
        print(f"Error training LSTM: {e}")
        results['lstm'] = {'error': str(e)}
    
    try:
        # Train Prophet
        prophet_result = await train_prophet_model(symbol, session)
        results['prophet'] = prophet_result
        
        # Save to database
        model_record = MLModel(
            symbol=symbol,
            model_type='prophet',
            version='v1.0',
            file_path=prophet_result['model_path'],
            metrics=prophet_result['metrics'],
            training_samples=prophet_result['training_samples'],
            is_active=True,
            hyperparameters={'changepoint_prior_scale': 0.05}
        )
        session.add(model_record)
        
    except Exception as e:
        print(f"Error training Prophet: {e}")
        results['prophet'] = {'error': str(e)}
    
    try:
        # Train XGBoost
        xgb_result = await train_xgboost_model(symbol, session)
        results['xgboost'] = xgb_result
        
        # Save to database
        model_record = MLModel(
            symbol=symbol,
            model_type='xgboost',
            version='v1.0',
            file_path=xgb_result['model_path'],
            metrics=xgb_result['metrics'],
            training_samples=xgb_result['training_samples'],
            is_active=True,
            hyperparameters={'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
        )
        session.add(model_record)
        
    except Exception as e:
        print(f"Error training XGBoost: {e}")
        results['xgboost'] = {'error': str(e)}
    
    await session.commit()
    
    return results
