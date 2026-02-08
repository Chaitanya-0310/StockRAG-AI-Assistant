import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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
import warnings
warnings.filterwarnings('ignore')

# Model storage directory
MODEL_DIR = Path(__file__).parent.parent / "ml_models"
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Enhanced Technical Feature Engineering
# ---------------------------------------------------------------------------

def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a rich set of technical indicators and derived features.
    Expects columns: open, high, low, close, volume (all lowercase).
    Returns the DataFrame with new feature columns appended.
    """
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']

    # --- Moving Averages ---
    df['SMA_10'] = ta.sma(c, length=10)
    df['SMA_30'] = ta.sma(c, length=30)
    df['SMA_50'] = ta.sma(c, length=50)
    df['EMA_12'] = ta.ema(c, length=12)
    df['EMA_26'] = ta.ema(c, length=26)

    # --- Momentum Indicators ---
    df['RSI'] = ta.rsi(c, length=14)
    df['RSI_fast'] = ta.rsi(c, length=7)

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_signal'] = macd.iloc[:, 1]
        df['MACD_hist'] = macd.iloc[:, 2]

    # Stochastic Oscillator
    stoch = ta.stoch(h, l, c, k=14, d=3)
    if stoch is not None and not stoch.empty:
        df['STOCH_K'] = stoch.iloc[:, 0]
        df['STOCH_D'] = stoch.iloc[:, 1]

    # Williams %R
    df['WILLR'] = ta.willr(h, l, c, length=14)

    # Rate of Change
    df['ROC_10'] = ta.roc(c, length=10)

    # --- Volatility Indicators ---
    bbands = ta.bbands(c, length=20, std=2)
    if bbands is not None and not bbands.empty:
        df['BB_upper'] = bbands.iloc[:, 2]  # BBU
        df['BB_middle'] = bbands.iloc[:, 1]  # BBM
        df['BB_lower'] = bbands.iloc[:, 0]  # BBL
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_pct'] = (c - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    atr = ta.atr(h, l, c, length=14)
    if atr is not None:
        df['ATR'] = atr
        df['ATR_pct'] = atr / c  # Normalised volatility

    # Historical Volatility (20-day)
    df['HV_20'] = c.pct_change().rolling(20).std() * np.sqrt(252)

    # --- Volume Indicators ---
    df['volume_sma_20'] = ta.sma(v, length=20)
    df['volume_ratio'] = v / df['volume_sma_20']

    obv = ta.obv(c, v)
    if obv is not None:
        df['OBV'] = obv

    # --- Price-Derived Features ---
    df['price_change'] = c.pct_change()
    df['price_change_5d'] = c.pct_change(5)
    df['price_change_10d'] = c.pct_change(10)
    df['volume_change'] = v.pct_change()
    df['high_low_range'] = (h - l) / c
    df['close_open_range'] = (c - df['open']) / df['open']

    # Price relative to moving averages
    df['price_to_sma10'] = c / df['SMA_10']
    df['price_to_sma30'] = c / df['SMA_30']
    df['price_to_sma50'] = c / df['SMA_50']
    df['sma10_to_sma30'] = df['SMA_10'] / df['SMA_30']

    # --- Lag Features ---
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = c.pct_change(lag)
        df[f'volume_lag_{lag}'] = v.shift(lag)
        df[f'close_lag_{lag}'] = c.shift(lag)

    # --- Calendar Features ---
    if hasattr(df.index, 'dayofweek'):
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
    else:
        df['day_of_week'] = 0
        df['month'] = 1
        df['quarter'] = 1

    return df


async def prepare_training_data(
    symbol: str,
    session: AsyncSession,
    lookback: int = 60,
    use_robust_scaler: bool = False,
) -> Tuple[pd.DataFrame, object]:
    """
    Fetch and prepare training data with comprehensive technical indicators.
    """
    stmt = select(StockPrice).where(StockPrice.symbol == symbol).order_by(StockPrice.date)
    result = await session.execute(stmt)
    prices = result.scalars().all()

    min_needed = lookback + 60  # buffer for indicators + sequences
    if len(prices) < min_needed:
        raise ValueError(f"Insufficient data for {symbol}. Need at least {min_needed} days, got {len(prices)}.")

    df = pd.DataFrame([{
        'date': p.date,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume,
    } for p in prices])

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Compute all indicators
    df = compute_advanced_features(df)

    # Drop NaN values created by indicators
    df.dropna(inplace=True)

    scaler = RobustScaler() if use_robust_scaler else MinMaxScaler()
    return df, scaler


# ---------------------------------------------------------------------------
# Sequence Helpers
# ---------------------------------------------------------------------------

def create_sequences(data: np.ndarray, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time-series models (LSTM / Transformer)."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])  # Predict close price (first column after scaling)
    return np.array(X), np.array(y)


class StockDataset(Dataset):
    """PyTorch Dataset for stock price sequences."""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Enhanced LSTM with multi-head attention and residual connections."""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_size: int = 1,
        dropout: float = 0.3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Multi-head attention over LSTM outputs
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        # Self-attention over time steps
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm1(lstm_out + attn_out)  # residual
        last_output = attn_out[:, -1, :]
        return self.head(last_output)


class TransformerPredictor(nn.Module):
    """
    Pure Transformer encoder for stock time-series prediction.
    Uses positional encoding, multi-head self-attention, and feed-forward layers.
    """

    def __init__(
        self,
        input_size: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 120,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm = nn.LayerNorm(d_model)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: (B, T, input_size)
        B, T, _ = x.shape
        x = self.input_proj(x) * math.sqrt(self.d_model)  # (B, T, d_model)
        x = x + self.pos_embedding[:, :T, :]
        x = self.encoder(x)
        x = self.norm(x)
        # Use last token representation for prediction
        return self.head(x[:, -1, :])


# ---------------------------------------------------------------------------
# Metrics Helpers
# ---------------------------------------------------------------------------

def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    mae = float(mean_absolute_error(actuals, predictions))
    r2 = float(r2_score(actuals, predictions))

    # MAPE (avoid division by zero)
    mask = actuals != 0
    mape = float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100) if mask.any() else 0.0

    # Directional accuracy
    if len(actuals) > 1:
        actual_dir = np.diff(actuals) > 0
        pred_dir = np.diff(predictions) > 0
        directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        directional_accuracy = 50.0

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
    }


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------

async def train_lstm_model(symbol: str, session: AsyncSession, epochs: int = 80) -> Dict:
    """Train enhanced LSTM model with attention, LR scheduling, early stopping."""
    print(f"Training Enhanced LSTM model for {symbol}...")

    df, scaler = await prepare_training_data(symbol, session, use_robust_scaler=False)

    # Feature selection for LSTM
    feature_cols = [
        'close', 'volume', 'SMA_10', 'SMA_30', 'EMA_12', 'RSI',
        'MACD', 'ATR_pct', 'BB_pct', 'price_change', 'volume_ratio',
        'high_low_range', 'HV_20',
    ]
    # Filter to columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    features = df[feature_cols].values
    scaler.fit(features)
    normalized = scaler.transform(features)

    lookback = 60
    X, y = create_sequences(normalized, lookback=lookback)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_ds = StockDataset(X_train, y_train)
    test_ds = StockDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        num_heads=4,
    )

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Early stopping
    best_val_loss = float('inf')
    patience, patience_counter = 15, 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in test_loader:
                vl = criterion(model(bx).squeeze(), by)
                val_loss += vl.item()
        val_loss /= max(len(test_loader), 1)
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(train_loader)
            print(f"  LSTM Epoch [{epoch+1}/{epochs}] train_loss={avg:.6f} val_loss={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")

        if patience_counter >= patience:
            print(f"  LSTM early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds.extend(model(bx).squeeze().numpy())
            acts.extend(by.numpy())

    metrics = compute_metrics(np.array(acts), np.array(preds))

    model_path = MODEL_DIR / f"{symbol}_lstm.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'lookback': lookback,
        'input_size': len(feature_cols),
    }, model_path)

    print(f"  LSTM trained. RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} "
          f"MAPE={metrics['mape']:.2f}% DirAcc={metrics['directional_accuracy']:.1f}%")

    return {'model_path': str(model_path), 'metrics': metrics, 'training_samples': len(X_train)}


async def train_transformer_model(symbol: str, session: AsyncSession, epochs: int = 80) -> Dict:
    """Train Transformer encoder model for stock prediction."""
    print(f"Training Transformer model for {symbol}...")

    df, scaler = await prepare_training_data(symbol, session, use_robust_scaler=False)

    feature_cols = [
        'close', 'volume', 'SMA_10', 'SMA_30', 'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_hist', 'ATR_pct', 'BB_pct', 'BB_width',
        'price_change', 'volume_ratio', 'high_low_range', 'HV_20',
        'ROC_10', 'STOCH_K',
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    features = df[feature_cols].values
    scaler.fit(features)
    normalized = scaler.transform(features)

    lookback = 60
    X, y = create_sequences(normalized, lookback=lookback)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_ds = StockDataset(X_train, y_train)
    test_ds = StockDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = TransformerPredictor(
        input_size=len(feature_cols),
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        max_seq_len=lookback,
    )

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    patience, patience_counter = 15, 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in test_loader:
                vl = criterion(model(bx).squeeze(), by)
                val_loss += vl.item()
        val_loss /= max(len(test_loader), 1)
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(train_loader)
            print(f"  Transformer Epoch [{epoch+1}/{epochs}] train_loss={avg:.6f} val_loss={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Transformer early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds.extend(model(bx).squeeze().numpy())
            acts.extend(by.numpy())

    metrics = compute_metrics(np.array(acts), np.array(preds))

    model_path = MODEL_DIR / f"{symbol}_transformer.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'lookback': lookback,
        'input_size': len(feature_cols),
    }, model_path)

    print(f"  Transformer trained. RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} "
          f"MAPE={metrics['mape']:.2f}% DirAcc={metrics['directional_accuracy']:.1f}%")

    return {'model_path': str(model_path), 'metrics': metrics, 'training_samples': len(X_train)}


async def train_prophet_model(symbol: str, session: AsyncSession) -> Dict:
    """Train Prophet model with tuned hyperparameters."""
    print(f"Training Prophet model for {symbol}...")

    stmt = select(StockPrice).where(StockPrice.symbol == symbol).order_by(StockPrice.date)
    result = await session.execute(stmt)
    prices = result.scalars().all()

    df = pd.DataFrame([{'ds': p.date, 'y': p.close} for p in prices])

    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
        changepoint_range=0.9,
        n_changepoints=30,
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)

    test_predictions = forecast.tail(len(test_df))['yhat'].values
    test_actuals = test_df['y'].values

    metrics = compute_metrics(test_actuals, test_predictions)

    model_path = MODEL_DIR / f"{symbol}_prophet.pkl"
    joblib.dump(model, model_path)

    print(f"  Prophet trained. RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} "
          f"DirAcc={metrics['directional_accuracy']:.1f}%")

    return {'model_path': str(model_path), 'metrics': metrics, 'training_samples': len(train_df)}


async def train_xgboost_model(symbol: str, session: AsyncSession) -> Dict:
    """Train XGBoost with time-series cross-validation and tuned hyperparameters."""
    print(f"Training XGBoost model for {symbol}...")

    df, scaler = await prepare_training_data(symbol, session, use_robust_scaler=True)

    # Use all non-OHLC columns as features
    exclude = {'close', 'open', 'high', 'low'}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df['close'].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Time-series cross-validation for best params
    tscv = TimeSeriesSplit(n_splits=3)
    best_rmse = float('inf')
    best_params = {}

    param_grid = [
        {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.85, 'colsample_bytree': 0.85},
        {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9},
    ]

    for params in param_grid:
        cv_rmses = []
        for train_idx, val_idx in tscv.split(X_train):
            xtr, xval = X_train[train_idx], X_train[val_idx]
            ytr, yval = y_train[train_idx], y_train[val_idx]

            m = xgb.XGBRegressor(
                objective='reg:squarederror',
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                **params,
            )
            m.fit(xtr, ytr, eval_set=[(xval, yval)], verbose=False)
            p = m.predict(xval)
            cv_rmses.append(np.sqrt(mean_squared_error(yval, p)))

        avg_rmse = np.mean(cv_rmses)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params

    # Train final model with best params
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        **best_params,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)

    model_path = MODEL_DIR / f"{symbol}_xgboost.pkl"
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'best_params': best_params,
    }, model_path)

    print(f"  XGBoost trained. RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} "
          f"DirAcc={metrics['directional_accuracy']:.1f}%")

    return {'model_path': str(model_path), 'metrics': metrics, 'training_samples': len(X_train)}


# ---------------------------------------------------------------------------
# Train All Models
# ---------------------------------------------------------------------------

async def train_all_models(symbol: str, session: AsyncSession) -> Dict[str, Dict]:
    """Train all four prediction models (LSTM, Transformer, Prophet, XGBoost)."""
    results = {}

    model_trainers = [
        ('lstm', train_lstm_model, {'epochs': 80, 'hidden_size': 128, 'num_layers': 3, 'attention': True}),
        ('transformer', train_transformer_model, {'epochs': 80, 'd_model': 128, 'nhead': 4, 'num_layers': 3}),
        ('prophet', train_prophet_model, {'changepoint_prior_scale': 0.1, 'n_changepoints': 30}),
        ('xgboost', train_xgboost_model, {'cv_folds': 3, 'tuned': True}),
    ]

    for model_type, trainer, hyperparams in model_trainers:
        try:
            if model_type in ('lstm', 'transformer'):
                result = await trainer(symbol, session, epochs=80)
            else:
                result = await trainer(symbol, session)

            results[model_type] = result

            model_record = MLModel(
                symbol=symbol,
                model_type=model_type,
                version='v2.0',
                file_path=result['model_path'],
                metrics=result['metrics'],
                training_samples=result['training_samples'],
                is_active=True,
                hyperparameters=hyperparams,
            )
            session.add(model_record)

        except Exception as e:
            print(f"Error training {model_type}: {e}")
            import traceback; traceback.print_exc()
            results[model_type] = {'error': str(e)}

    await session.commit()
    return results
