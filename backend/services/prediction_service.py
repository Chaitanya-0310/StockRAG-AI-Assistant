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


async def generate_predictions(
    symbol: str, 
    session: AsyncSession, 
    days_ahead: int = 30
) -> List[Dict]:
    """
    Generate ensemble predictions combining LSTM, Prophet, and XGBoost
    
    Args:
        symbol: Stock symbol
        session: Database session
        days_ahead: Number of days to predict ahead
        
    Returns:
        List of prediction dictionaries
    """
    print(f"Generating {days_ahead}-day predictions for {symbol}...")
    
    # Get active models
    stmt = select(MLModel).where(
        MLModel.symbol == symbol,
        MLModel.is_active == True
    )
    result = await session.execute(stmt)
    models = result.scalars().all()
    
    if not models:
        raise ValueError(f"No trained models found for {symbol}. Please train models first.")
    
    predictions_by_model = {}
    
    # Generate predictions from each model
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
            print(f"Error predicting with {model_record.model_type}: {e}")
            continue
    
    if not predictions_by_model:
        raise ValueError(f"All models failed to generate predictions for {symbol}")
    
    # Ensemble: Average predictions with weights based on model performance
    ensemble_predictions = _ensemble_predictions(predictions_by_model, models)
    
    # Store predictions in database
    await _store_predictions(symbol, ensemble_predictions, session)
    
    return ensemble_predictions


async def _predict_lstm(
    symbol: str, 
    session: AsyncSession, 
    days_ahead: int,
    model_record: MLModel
) -> List[Dict]:
    """Generate predictions using LSTM model"""
    
    # Load model
    checkpoint = torch.load(model_record.file_path)
    model = LSTMModel(input_size=len(checkpoint['feature_cols']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    feature_cols = checkpoint['feature_cols']
    lookback = checkpoint['lookback']
    
    # Get recent data
    stmt = select(StockPrice).where(
        StockPrice.symbol == symbol
    ).order_by(desc(StockPrice.date)).limit(lookback + 50)
    result = await session.execute(stmt)
    prices = list(reversed(result.scalars().all()))
    
    # Prepare features
    df = pd.DataFrame([{
        'close': p.close,
        'volume': p.volume,
        'high': p.high,
        'low': p.low
    } for p in prices])
    
    # Add technical indicators (simplified)
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['SMA_30'] = df['close'].rolling(30).mean()
    df['RSI'] = _calculate_rsi(df['close'])
    
    df = df[feature_cols].tail(lookback).values
    
    # Normalize
    normalized = scaler.transform(df)
    
    # Predict iteratively
    predictions = []
    current_sequence = normalized.copy()
    
    with torch.no_grad():
        for i in range(days_ahead):
            # Prepare input
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
            
            # Predict
            pred = model(input_tensor).item()
            
            # Denormalize
            dummy = np.zeros((1, len(feature_cols)))
            dummy[0, 0] = pred
            pred_price = scaler.inverse_transform(dummy)[0, 0]
            
            predictions.append({
                'day': i + 1,
                'price': float(pred_price)
            })
            
            # Update sequence (simplified - just shift and add prediction)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred
    
    return predictions


async def _predict_prophet(
    symbol: str,
    session: AsyncSession,
    days_ahead: int,
    model_record: MLModel
) -> List[Dict]:
    """Generate predictions using Prophet model"""
    
    # Load model
    model = joblib.load(model_record.file_path)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)
    
    # Extract predictions
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
    """Generate predictions using XGBoost model"""
    
    # Load model
    model_data = joblib.load(model_record.file_path)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    # Get recent data
    stmt = select(StockPrice).where(
        StockPrice.symbol == symbol
    ).order_by(desc(StockPrice.date)).limit(100)
    result = await session.execute(stmt)
    prices = list(reversed(result.scalars().all()))
    
    # Prepare features (simplified)
    df = pd.DataFrame([{
        'close': p.close,
        'volume': p.volume
    } for p in prices])
    
    # Add basic features
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['SMA_30'] = df['close'].rolling(30).mean()
    df['RSI'] = _calculate_rsi(df['close'])
    
    # Create lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    df.dropna(inplace=True)
    
    # Predict iteratively
    predictions = []
    last_features = df[feature_cols].iloc[-1:].values
    feature_state = {col: float(last_features[0][idx]) for idx, col in enumerate(feature_cols)}

    for i in range(days_ahead):
        pred_price = model.predict(last_features)[0]

        predictions.append({
            "day": i + 1,
            "price": float(pred_price),
        })

        if "close" in feature_state:
            feature_state["close"] = float(pred_price)

        for lag in [10, 5, 3, 2, 1]:
            close_key = f"close_lag_{lag}"
            if close_key in feature_state:
                if lag == 1:
                    feature_state[close_key] = float(pred_price)
                else:
                    prev_key = f"close_lag_{lag - 1}"
                    feature_state[close_key] = feature_state.get(prev_key, feature_state[close_key])

            volume_key = f"volume_lag_{lag}"
            if volume_key in feature_state:
                if lag == 1 and "volume" in feature_state:
                    feature_state[volume_key] = feature_state["volume"]
                else:
                    prev_key = f"volume_lag_{lag - 1}"
                    feature_state[volume_key] = feature_state.get(prev_key, feature_state[volume_key])

        last_features = np.array([[feature_state[col] for col in feature_cols]])

    return predictions


def _calculate_model_weights(
    predictions_by_model: Dict[str, List[Dict]],
    models: List[MLModel]
) -> Dict[str, float]:
    weights = {}
    total_weight = 0.0

    for model in models:
        if model.model_type not in predictions_by_model:
            continue
        metrics = model.metrics or {}
        rmse = metrics.get("rmse")
        mae = metrics.get("mae")
        directional_accuracy = metrics.get("directional_accuracy")

        weight = 0.0
        if rmse and rmse > 0:
            weight = 1.0 / rmse
        elif mae and mae > 0:
            weight = 1.0 / mae
        elif directional_accuracy:
            weight = directional_accuracy / 100.0

        if weight <= 0:
            weight = 1.0

        weights[model.model_type] = weight
        total_weight += weight

    if total_weight == 0:
        return {k: 1.0 / len(predictions_by_model) for k in predictions_by_model.keys()}

    return {k: v / total_weight for k, v in weights.items()}


def _estimate_interval(
    weighted_price: float,
    model_prices: List[float],
    prophet_bounds: Optional[Tuple[float, float]]
) -> Tuple[float, float]:
    if len(model_prices) > 1:
        spread = float(np.std(model_prices))
        lower = weighted_price - (1.5 * spread)
        upper = weighted_price + (1.5 * spread)
    else:
        lower = weighted_price * 0.9
        upper = weighted_price * 1.1

    if prophet_bounds:
        lower = min(lower, prophet_bounds[0])
        upper = max(upper, prophet_bounds[1])

    return lower, upper


def _ensemble_predictions(
    predictions_by_model: Dict[str, List[Dict]],
    models: List[MLModel]
) -> List[Dict]:
    """
    Combine predictions from multiple models using weighted average.
    """
    weights = _calculate_model_weights(predictions_by_model, models)

    ensemble = []
    days_ahead = min(len(preds) for preds in predictions_by_model.values())

    for day in range(days_ahead):
        weighted_price = 0.0
        prophet_bounds = None
        model_prices = []

        for model_type, preds in predictions_by_model.items():
            weight = weights.get(model_type, 0.0)
            price = preds[day]["price"]
            model_prices.append(price)
            weighted_price += price * weight

            if "lower" in preds[day] and "upper" in preds[day]:
                prophet_bounds = (preds[day]["lower"], preds[day]["upper"])

        lower_bound, upper_bound = _estimate_interval(weighted_price, model_prices, prophet_bounds)

        ensemble.append({
            "day": day + 1,
            "predicted_price": float(weighted_price),
            "confidence_lower": float(lower_bound),
            "confidence_upper": float(upper_bound),
        })

    return ensemble


async def _store_predictions(
    symbol: str,
    predictions: List[Dict],
    session: AsyncSession
):
    """Store predictions in database"""
    
    # Delete old predictions for this symbol
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
    print(f"Stored {len(predictions)} predictions for {symbol}")


async def get_predictions_for_symbol(
    symbol: str,
    session: AsyncSession,
    days: Optional[int] = None
) -> List[Dict]:
    """
    Retrieve stored predictions for a symbol
    
    Args:
        symbol: Stock symbol
        session: Database session
        days: Optional filter for number of days ahead
        
    Returns:
        List of predictions
    """
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
    """Get training status and metrics for all models of a symbol"""
    
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
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
