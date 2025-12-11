from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, Boolean, JSON
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
from datetime import datetime

Base = declarative_base()

class StockPrice(Base):
    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

class RagDocument(Base):
    __tablename__ = "rag_documents"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(Date)
    content = Column(Text)  # The text chunk
    embedding = Column(Vector(768))  # Google text-embedding-004 dimension is 768

class StockPrediction(Base):
    __tablename__ = "stock_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    prediction_date = Column(Date, index=True)  # When prediction was made
    target_date = Column(Date, index=True)  # Date being predicted
    predicted_price = Column(Float)
    confidence_lower = Column(Float)  # Lower bound of confidence interval
    confidence_upper = Column(Float)  # Upper bound of confidence interval
    model_version = Column(String)
    model_type = Column(String)  # 'lstm', 'prophet', 'xgboost', 'ensemble'
    created_at = Column(DateTime, default=datetime.utcnow)

class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    model_type = Column(String)  # 'lstm', 'prophet', 'xgboost', 'ensemble'
    version = Column(String)
    file_path = Column(String)  # Path to saved model file
    training_date = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)  # Store RMSE, MAE, directional accuracy, etc.
    training_samples = Column(Integer)  # Number of data points used for training
    is_active = Column(Boolean, default=True)
    hyperparameters = Column(JSON)  # Store model configuration
