import logging
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, delete
from backend.models.models import StockPrice, MLModel, StockPrediction
from backend.services.ml_service import MODEL_DIR, LSTMModel
from pathlib import Path

logger = logging.getLogger(__name__)


async def generate_predictions(
    symbol: str,
    session: AsyncSession,
    days_ahead: int = 30
) -> List[Dict]:
    """Generate ensemble predictions combining LSTM, Prophet, and XGBoost."""
    logger.info(f"Generating {days_ahead}-day predictions for {symbol}...")

    stmt = select(MLModel).where(
        MLModel.symbol == symbol,
        MLModel.is_active == True
    )
    result = await session.execute(stmt)
    models = result.scalars().all()

    if not models:
        raise ValueError(f"No trained models found for {symbol}. Please train models first.")

    predictions_by_model = {}

    for model_record in models:
        try:
            if model_record.model_type == 'lstm':
                preds = await _predict_lstm(symbol, session, days_ahead, model_record)
                predictions_by_model['lstm'] = preds
            elif model_record.model_type == 'prophet':
                preds = await _predict_prophet(symbol, session, days_ahead, model_record)
                predictions_by_model['prophet'] = preds
            elif model_record.model_type == 'xgboost':
                preds = await _predict_xgboost(symbol, session, days_ahead, model_record)
                predictions_by_model['xgboost'] = preds
        except Exception as e:
            logger.error(f"Error predicting with {model_record.model_type}: {e}")
            continue

    if not predictions_by_model:
        raise ValueError(f"All models failed to generate predictions for {symbol}")

    ensemble_predictions = _ensemble_predictions(predictions_by_model, models)
    await _store_predictions(symbol, ensemble_predictions, session)

    return ensemble_predictions


async def _predict_lstm(
    symbol: str,
    session: AsyncSession,
    days_ahead: int,
    model_record: MLModel
) -> List[Dict]:
    """Generate predictions using LSTM model with proper feature recomputation."""
    checkpoint = torch.load(model_record.file_path, weights_only=False)
    model = LSTMModel(input_size=len(checkpoint['feature_cols']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = checkpoint['scaler']
    feature_cols = checkpoint['feature_cols']
    lookback = checkpoint['lookback']

    # Get enough recent data for indicators + lookback
    stmt = select(StockPrice).where(
        StockPrice.symbol == symbol
    ).order_by(desc(StockPrice.date)).limit(lookback + 50)
    result = await session.execute(stmt)
    prices = list(reversed(result.scalars().all()))

    # Build a DataFrame with raw values for recomputing indicators
    raw_closes = [p.close for p in prices]
    raw_volumes = [float(p.volume) for p in prices]

    predictions = []
    current_closes = list(raw_closes)
    current_volumes = list(raw_volumes)

    with torch.no_grad():
        for i in range(days_ahead):
            # Recompute all features from current data
            close_series = pd.Series(current_closes)
            volume_series = pd.Series(current_volumes)

            sma_10 = close_series.rolling(10).mean()
            sma_30 = close_series.rolling(30).mean()
            rsi = _calculate_rsi(close_series)

            # Build feature row from most recent values
            feature_row = {
                'close': current_closes[-1],
                'volume': current_volumes[-1],
                'SMA_10': sma_10.iloc[-1] if not pd.isna(sma_10.iloc[-1]) else current_closes[-1],
                'SMA_30': sma_30.iloc[-1] if not pd.isna(sma_30.iloc[-1]) else current_closes[-1],
                'RSI': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0,
            }

            # Build the full lookback sequence
            features_df = pd.DataFrame({
                'close': current_closes,
                'volume': current_volumes,
                'SMA_10': sma_10.values,
                'SMA_30': sma_30.values,
                'RSI': rsi.values,
            })
            features_df.fillna(method='bfill', inplace=True)
            features_df.fillna(method='ffill', inplace=True)

            sequence = features_df[feature_cols].tail(lookback).values
            if len(sequence) < lookback:
                break

            normalized = scaler.transform(sequence)
            input_tensor = torch.FloatTensor(normalized).unsqueeze(0)

            pred = model(input_tensor).item()

            # Denormalize
            dummy = np.zeros((1, len(feature_cols)))
            dummy[0, 0] = pred
            pred_price = scaler.inverse_transform(dummy)[0, 0]

            predictions.append({
                'day': i + 1,
                'price': float(pred_price)
            })

            # Update rolling data with prediction
            current_closes.append(float(pred_price))
            current_volumes.append(current_volumes[-1])  # Carry forward volume

    return predictions


async def _predict_prophet(
    symbol: str,
    session: AsyncSession,
    days_ahead: int,
    model_record: MLModel
) -> List[Dict]:
    """Generate predictions using Prophet model."""
    model = joblib.load(model_record.file_path)

    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)

    predictions = []
    for i in range(1, days_ahead + 1):
        pred_row = forecast.iloc[-days_ahead + i - 1]
        predictions.append({
            'day': i,
            'price': float(pred_row['yhat']),
            'lower': float(pred_row['yhat_lower']),
            'upper': float(pred_row['yhat_upper'])
        })

    return predictions


async def _predict_xgboost(
    symbol: str,
    session: AsyncSession,
    days_ahead: int,
    model_record: MLModel
) -> List[Dict]:
    """Generate predictions using XGBoost model with proper feature recomputation."""
    model_data = joblib.load(model_record.file_path)
    model = model_data['model']
    feature_cols = model_data['feature_cols']

    # Get recent data
    stmt = select(StockPrice).where(
        StockPrice.symbol == symbol
    ).order_by(desc(StockPrice.date)).limit(100)
    result = await session.execute(stmt)
    prices = list(reversed(result.scalars().all()))

    # Build DataFrame with all needed columns
    df = pd.DataFrame([{
        'close': p.close,
        'volume': float(p.volume),
        'high': p.high,
        'low': p.low,
        'open': p.open,
    } for p in prices])

    predictions = []

    for i in range(days_ahead):
        # Recompute technical indicators
        df['SMA_10'] = df['close'].rolling(10).mean()
        df['SMA_30'] = df['close'].rolling(30).mean()
        df['RSI'] = _calculate_rsi(df['close'])

        # Recompute lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

        # Add other features that might be in feature_cols
        if 'volume_sma' in feature_cols:
            df['volume_sma'] = df['volume'].rolling(20).mean()
        if 'price_change' in feature_cols:
            df['price_change'] = df['close'].pct_change()
        if 'volume_change' in feature_cols:
            df['volume_change'] = df['volume'].pct_change()
        if 'MACD' in feature_cols:
            import pandas_ta as ta
            macd = ta.macd(df['close'])
            if macd is not None and not macd.empty:
                df['MACD'] = macd['MACD_12_26_9']
                df['MACD_signal'] = macd['MACDs_12_26_9']
        if 'BB_upper' in feature_cols:
            import pandas_ta as ta
            bbands = ta.bbands(df['close'], length=20)
            if bbands is not None and not bbands.empty:
                df['BB_upper'] = bbands.iloc[:, 2]
                df['BB_lower'] = bbands.iloc[:, 0]

        clean_df = df.dropna()
        if clean_df.empty:
            break

        # Get available feature columns (some may not exist)
        available_cols = [c for c in feature_cols if c in clean_df.columns]
        if len(available_cols) != len(feature_cols):
            # Fill missing columns with 0
            for c in feature_cols:
                if c not in clean_df.columns:
                    clean_df[c] = 0.0

        last_features = clean_df[feature_cols].iloc[-1:].values
        pred_price = model.predict(last_features)[0]

        predictions.append({
            'day': i + 1,
            'price': float(pred_price)
        })

        # Append prediction as new row for next iteration
        new_row = df.iloc[-1].copy()
        new_row['close'] = float(pred_price)
        new_row['open'] = df.iloc[-1]['close']
        new_row['high'] = max(float(pred_price), df.iloc[-1]['close'])
        new_row['low'] = min(float(pred_price), df.iloc[-1]['close'])
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return predictions


def _ensemble_predictions(
    predictions_by_model: Dict[str, List[Dict]],
    models: List[MLModel]
) -> List[Dict]:
    """Combine predictions from multiple models using weighted average."""
    weights = {}
    total_weight = 0

    for model in models:
        if model.model_type in predictions_by_model:
            acc = model.metrics.get('directional_accuracy', 50)
            weight = acc / 100.0
            weights[model.model_type] = weight
            total_weight += weight

    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        weights = {k: 1.0 / len(predictions_by_model) for k in predictions_by_model.keys()}

    ensemble = []
    days_ahead = len(list(predictions_by_model.values())[0])

    for day in range(days_ahead):
        weighted_price = 0
        lower_bound = float('inf')
        upper_bound = float('-inf')

        for model_type, preds in predictions_by_model.items():
            if day >= len(preds):
                continue
            weight = weights[model_type]
            weighted_price += preds[day]['price'] * weight

            if 'lower' in preds[day]:
                lower_bound = min(lower_bound, preds[day]['lower'])
                upper_bound = max(upper_bound, preds[day]['upper'])

        if lower_bound == float('inf'):
            lower_bound = weighted_price * 0.9
            upper_bound = weighted_price * 1.1

        ensemble.append({
            'day': day + 1,
            'predicted_price': float(weighted_price),
            'confidence_lower': float(lower_bound),
            'confidence_upper': float(upper_bound)
        })

    return ensemble


async def _store_predictions(
    symbol: str,
    predictions: List[Dict],
    session: AsyncSession
):
    """Store predictions in database."""
    await session.execute(
        delete(StockPrediction).where(StockPrediction.symbol == symbol)
    )

    prediction_date = date.today()

    for pred in predictions:
        target_date = prediction_date + timedelta(days=pred['day'])
        db_prediction = StockPrediction(
            symbol=symbol,
            prediction_date=prediction_date,
            target_date=target_date,
            predicted_price=pred['predicted_price'],
            confidence_lower=pred['confidence_lower'],
            confidence_upper=pred['confidence_upper'],
            model_version='v1.0',
            model_type='ensemble'
        )
        session.add(db_prediction)

    await session.commit()
    logger.info(f"Stored {len(predictions)} predictions for {symbol}")


async def get_predictions_for_symbol(
    symbol: str,
    session: AsyncSession,
    days: Optional[int] = None
) -> List[Dict]:
    """Retrieve stored predictions for a symbol."""
    stmt = select(StockPrediction).where(
        StockPrediction.symbol == symbol,
        StockPrediction.target_date >= date.today()
    ).order_by(StockPrediction.target_date)

    if days:
        max_date = date.today() + timedelta(days=days)
        stmt = stmt.where(StockPrediction.target_date <= max_date)

    result = await session.execute(stmt)
    predictions = result.scalars().all()

    return [{
        'target_date': p.target_date.isoformat(),
        'predicted_price': p.predicted_price,
        'confidence_lower': p.confidence_lower,
        'confidence_upper': p.confidence_upper,
        'prediction_date': p.prediction_date.isoformat()
    } for p in predictions]


async def get_model_status(symbol: str, session: AsyncSession) -> Dict:
    """Get training status and metrics for all models of a symbol."""
    stmt = select(MLModel).where(MLModel.symbol == symbol)
    result = await session.execute(stmt)
    models = result.scalars().all()

    if not models:
        return {'symbol': symbol, 'models': [], 'status': 'not_trained'}

    model_info = []
    for model in models:
        model_info.append({
            'type': model.model_type,
            'version': model.version,
            'training_date': model.training_date.isoformat(),
            'metrics': model.metrics,
            'training_samples': model.training_samples,
            'is_active': model.is_active
        })

    return {
        'symbol': symbol,
        'models': model_info,
        'status': 'trained'
    }


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
