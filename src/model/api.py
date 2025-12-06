"""
Water Anomaly Detection Model API
==================================
FastAPI server for serving KNN and LSTM predictions
- Single prediction endpoint
- Batch prediction endpoint
- Model health check
- Ensemble predictions (KNN + LSTM)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import tensorflow as tf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

CONFIG_PATH = Path(__file__).parent.parent / "config" / "paths_config.json"

def load_config():
    """Load configuration from JSON file"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

CONFIG = load_config()
MODEL_CONFIG = CONFIG.get("models", {})
API_CONFIG = CONFIG.get("api", {})

# Model paths
KNN_MODEL_FILE = MODEL_CONFIG.get("knn_model_file", "src/model/weights/knn_model.pkl")
LSTM_MODEL_FILE = MODEL_CONFIG.get("lstm_model_file", "src/model/weights/lstm_model.h5")
SCALER_FILE = MODEL_CONFIG.get("scaler_file", "src/model/weights/scaler.pkl")
LABEL_ENCODERS_FILE = MODEL_CONFIG.get("label_encoders_file", "src/model/weights/label_encoders.pkl")
TARGET_ENCODER_FILE = MODEL_CONFIG.get("target_encoder_file", "src/model/weights/target_encoder.pkl")

# API settings
MODEL_API_PORT = API_CONFIG.get("model_port", 8002)
MODEL_API_HOST = API_CONFIG.get("model_host", "0.0.0.0")

# ============================================================================
# Request/Response Models
# ============================================================================

class SensorData(BaseModel):
    """Single sensor reading for prediction"""
    pressure: float = Field(..., description="Water pressure in PSI")
    temperature: float = Field(..., description="Temperature in Celsius")
    ph_level: float = Field(..., description="pH level (0-14)")
    dissolved_oxygen: float = Field(..., description="Dissolved oxygen in mg/L")
    turbidity: float = Field(..., description="Turbidity in NTU")
    flow_rate: float = Field(..., description="Flow rate in L/min")
    location: Optional[str] = Field(None, description="Sensor location (e.g., valve_a)")
    sensor_type: Optional[str] = Field(None, description="Sensor type (e.g., digital)")

class BatchPredictRequest(BaseModel):
    """Batch prediction request"""
    samples: List[SensorData] = Field(..., description="List of sensor readings")

class PredictionResponse(BaseModel):
    """Single prediction result"""
    knn_prediction: str
    knn_confidence: float
    lstm_prediction: str
    lstm_confidence: float
    ensemble_prediction: str
    ensemble_confidence: float
    anomaly_detected: bool
    risk_level: Optional[str] = None

class SinglePredictResponse(BaseModel):
    """Response for single prediction"""
    success: bool
    prediction: PredictionResponse
    analysis: Dict[str, Any]
    execution_time_ms: float
    model_version: str

class BatchPredictResponse(BaseModel):
    """Response for batch prediction"""
    success: bool
    predictions: List[Dict[str, Any]]
    total_processed: int
    anomalies_found: int
    execution_time_ms: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models: Dict[str, str]
    version: str
    timestamp: str

# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages loading and using prediction models"""
    
    def __init__(self):
        """Initialize and load models"""
        self.knn_model = None
        self.lstm_model = None
        self.scaler = None
        self.label_encoders = None
        self.target_encoder = None
        self.numeric_features = [
            'pressure', 'temperature', 'ph_level', 'dissolved_oxygen',
            'turbidity', 'flow_rate'
        ]
        self.categorical_features = ['location', 'sensor_type']
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            if Path(KNN_MODEL_FILE).exists():
                self.knn_model = joblib.load(KNN_MODEL_FILE)
                logger.info(f"[OK] Loaded KNN model from {KNN_MODEL_FILE}")
            else:
                logger.warning(f"[WARNING] KNN model not found at {KNN_MODEL_FILE}")
            
            if Path(LSTM_MODEL_FILE).exists():
                self.lstm_model = tf.keras.models.load_model(LSTM_MODEL_FILE)
                logger.info(f"[OK] Loaded LSTM model from {LSTM_MODEL_FILE}")
            else:
                logger.warning(f"[WARNING] LSTM model not found at {LSTM_MODEL_FILE}")
            
            if Path(SCALER_FILE).exists():
                self.scaler = joblib.load(SCALER_FILE)
                logger.info(f"[OK] Loaded scaler from {SCALER_FILE}")
            
            if Path(LABEL_ENCODERS_FILE).exists():
                self.label_encoders = joblib.load(LABEL_ENCODERS_FILE)
                logger.info(f"[OK] Loaded label encoders from {LABEL_ENCODERS_FILE}")
            
            if Path(TARGET_ENCODER_FILE).exists():
                self.target_encoder = joblib.load(TARGET_ENCODER_FILE)
                logger.info(f"[OK] Loaded target encoder from {TARGET_ENCODER_FILE}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading models: {e}")
            raise
    
    def preprocess_sample(self, sample_dict: Dict[str, Any]) -> np.ndarray:
        """Preprocess a single sample for prediction"""
        df = pd.DataFrame([sample_dict])
        
        # Handle missing categorical features
        for col in self.categorical_features:
            if col not in df.columns or pd.isna(df[col]).any():
                df[col] = 'unknown'
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in self.label_encoders:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except:
                    # Use 0 for unknown categories
                    df[col] = 0
        
        # Select and scale numeric features
        X = df[self.numeric_features].values
        if self.scaler:
            X = self.scaler.transform(X)
        
        return X, df[self.categorical_features].values
    
    def predict_single(self, sample: SensorData) -> Dict[str, Any]:
        """Predict anomaly for single sample"""
        sample_dict = sample.dict()
        
        try:
            X_numeric, X_categorical = self.preprocess_sample(sample_dict)
            X = np.hstack([X_numeric, X_categorical])
            
            predictions = {
                'knn_prediction': 'unknown',
                'knn_confidence': 0.0,
                'lstm_prediction': 'unknown',
                'lstm_confidence': 0.0,
                'ensemble_prediction': 'unknown',
                'ensemble_confidence': 0.0,
                'anomaly_detected': False
            }
            
            # KNN prediction
            if self.knn_model:
                knn_pred = self.knn_model.predict(X)
                knn_proba = self.knn_model.predict_proba(X)
                predictions['knn_prediction'] = self.target_encoder.inverse_transform(knn_pred)[0]
                predictions['knn_confidence'] = float(np.max(knn_proba))
            
            # LSTM prediction
            if self.lstm_model:
                X_lstm = X_numeric.reshape((1, 1, X_numeric.shape[1]))
                lstm_proba = self.lstm_model.predict(X_lstm, verbose=0)
                lstm_pred_idx = np.argmax(lstm_proba)
                lstm_pred = self.target_encoder.inverse_transform([lstm_pred_idx])[0]
                predictions['lstm_prediction'] = lstm_pred
                predictions['lstm_confidence'] = float(np.max(lstm_proba))
            
            # Ensemble prediction (average confidence)
            if predictions['knn_confidence'] > 0 and predictions['lstm_confidence'] > 0:
                avg_confidence = (predictions['knn_confidence'] + predictions['lstm_confidence']) / 2
                ensemble_pred = predictions['knn_prediction']  # Use KNN as primary
                predictions['ensemble_prediction'] = ensemble_pred
                predictions['ensemble_confidence'] = avg_confidence
                
                # Detect anomaly if either model predicts anomaly with high confidence
                if (predictions['knn_prediction'] == 'anomaly' or 
                    predictions['lstm_prediction'] == 'anomaly'):
                    predictions['anomaly_detected'] = True
            
            return predictions
        
        except Exception as e:
            logger.error(f"[ERROR] Prediction error: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if models are loaded and ready"""
        return self.knn_model is not None and self.lstm_model is not None


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Water Anomaly Detection Model API",
    description="ML prediction service for water system anomalies (KNN + LSTM)",
    version="1.0.0"
)

# Initialize model manager
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global model_manager
    logger.info("[START] Starting Water Anomaly Detection Model API...")
    logger.info("=" * 70)
    logger.info(f"[INFO] Model Directory: {Path('src/model/weights').absolute()}")
    logger.info(f"[INFO] KNN Model: {KNN_MODEL_FILE}")
    logger.info(f"[INFO] LSTM Model: {LSTM_MODEL_FILE}")
    logger.info("=" * 70)
    
    try:
        model_manager = ModelManager()
        logger.info("[OK] Model Manager initialized successfully")
        logger.info(f"[INFO] Server running on http://{MODEL_API_HOST}:{MODEL_API_PORT}")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize model manager: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("[STOP] Shutting down Model API...")

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return HealthResponse(
        status="operational" if model_manager.is_ready() else "degraded",
        models={
            "knn": "loaded" if model_manager.knn_model else "not_loaded",
            "lstm": "loaded" if model_manager.lstm_model else "not_loaded"
        },
        version="1.0",
        timestamp=datetime.now().isoformat()
    )

# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict", response_model=SinglePredictResponse)
async def predict_single(sample: SensorData):
    """Predict anomaly for single sensor reading"""
    if not model_manager or not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Models not ready")
    
    start_time = time.time()
    
    try:
        prediction = model_manager.predict_single(sample)
        execution_time = (time.time() - start_time) * 1000
        
        # Determine risk level
        risk_level = "low"
        if prediction['anomaly_detected']:
            avg_confidence = prediction['ensemble_confidence']
            if avg_confidence > 0.85:
                risk_level = "critical"
            elif avg_confidence > 0.75:
                risk_level = "high"
            else:
                risk_level = "medium"
        
        return SinglePredictResponse(
            success=True,
            prediction=PredictionResponse(**prediction),
            analysis={
                "anomaly_detected": prediction['anomaly_detected'],
                "risk_level": risk_level,
                "features_used": model_manager.numeric_features
            },
            execution_time_ms=execution_time,
            model_version="v1.0"
        )
    
    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Predict anomalies for multiple sensor readings"""
    if not model_manager or not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Models not ready")
    
    start_time = time.time()
    predictions = []
    anomalies_count = 0
    
    try:
        for sample in request.samples:
            prediction = model_manager.predict_single(sample)
            
            # Add risk level
            risk_level = "low"
            if prediction['anomaly_detected']:
                anomalies_count += 1
                avg_confidence = prediction['ensemble_confidence']
                if avg_confidence > 0.85:
                    risk_level = "critical"
                elif avg_confidence > 0.75:
                    risk_level = "high"
                else:
                    risk_level = "medium"
            
            prediction['risk_level'] = risk_level
            predictions.append(prediction)
        
        execution_time = (time.time() - start_time) * 1000
        
        return BatchPredictResponse(
            success=True,
            predictions=predictions,
            total_processed=len(request.samples),
            anomalies_found=anomalies_count,
            execution_time_ms=execution_time
        )
    
    except Exception as e:
        logger.error(f"[ERROR] Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Root endpoint
# ============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Water Anomaly Detection Model API",
        "version": "1.0.0",
        "description": "ML prediction service using KNN and LSTM models",
        "endpoints": {
            "health": "GET /health",
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "docs": "GET /docs",
            "redoc": "GET /redoc"
        },
        "documentation": "http://localhost:8002/docs"
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"[START] Starting Model API server...")
    logger.info(f"[INFO] Host: {MODEL_API_HOST}")
    logger.info(f"[INFO] Port: {MODEL_API_PORT}")
    logger.info(f"[INFO] API Documentation: http://localhost:{MODEL_API_PORT}/docs")
    
    uvicorn.run(
        app,
        host=MODEL_API_HOST,
        port=MODEL_API_PORT,
        log_level="info"
    )
