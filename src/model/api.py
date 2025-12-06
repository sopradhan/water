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

# Zone-specific model paths
ZONE_MODELS_BASE = "src/model/model_weights"
AVAILABLE_ZONES = ['Zone0', 'Zone1', 'Zone2']

# API settings
MODEL_API_PORT = API_CONFIG.get("model_port", 8002)
MODEL_API_HOST = API_CONFIG.get("model_host", "0.0.0.0")

# ============================================================================
# Request/Response Models
# ============================================================================

class SensorData(BaseModel):
    """Single sensor reading for prediction"""
    zone_name: Optional[str] = Field(None, description="Zone (Zone0, Zone1, Zone2)", alias="ZoneName")
    pressure_psi: float = Field(..., description="Water pressure in PSI", alias="Pressure_PSI")
    master_flow_lpm: float = Field(..., description="Master flow rate in L/min", alias="Master_Flow_LPM")
    temperature_c: float = Field(..., description="Temperature in Celsius", alias="Temperature_C")
    vibration: float = Field(..., description="Vibration in mm/s", alias="Vibration")
    rpm: float = Field(..., description="RPM - revolutions per minute", alias="RPM")
    operation_hours: float = Field(..., description="Operation hours - cumulative", alias="OperationHours")
    acoustic_level: float = Field(..., description="Acoustic level in dB", alias="AcousticLevel")
    ultrasonic_signal: float = Field(..., description="Ultrasonic signal in V", alias="UltrasonicSignal")
    pipe_age: float = Field(..., description="Pipe age in years", alias="PipeAge")
    soil_type: Optional[str] = Field(None, description="Soil type (e.g., clay, sand)", alias="SoilType")
    material: Optional[str] = Field(None, description="Pipe material (e.g., PVC, cast_iron)", alias="Material")
    
    class Config:
        populate_by_name = True

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
    available_zones: List[str]
    current_zone: str
    version: str
    timestamp: str

# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages loading and using prediction models - supports zone-specific models"""
    
    def __init__(self):
        """Initialize and load models"""
        self.models = {}  # Store models for each zone
        self.current_zone = 'Zone0'  # Default zone
        self.knn_model = None
        self.lstm_model = None
        self.scaler = None
        self.label_encoders = None
        self.target_encoder = None
        # Must match training data features exactly
        self.numeric_features = [
            'Pressure_PSI',
            'Master_Flow_LPM',
            'Temperature_C',
            'Vibration',
            'RPM',
            'OperationHours',
            'AcousticLevel',
            'UltrasonicSignal',
            'PipeAge'
        ]
        self.categorical_features = ['SoilType', 'Material']
        self.load_models()
    
    def load_models_for_zone(self, zone_name: str = 'Zone0'):
        """Load zone-specific models or fallback to main models"""
        try:
            zone_dir = Path(ZONE_MODELS_BASE) / f"{zone_name}_models"
            
            # Try zone-specific directory first
            if zone_dir.exists():
                logger.info(f"Loading {zone_name}-specific models from {zone_dir}")
                knn_path = zone_dir / "knn_model.pkl"
                lstm_path = zone_dir / "lstm_model.h5"
                scaler_path = zone_dir / "scaler.pkl"
                encoders_path = zone_dir / "label_encoders.pkl"
                target_path = zone_dir / "target_encoder.pkl"
            else:
                # Fallback to main model directory
                logger.info(f"Zone directory not found, using main models")
                knn_path = Path(ZONE_MODELS_BASE) / "knn_lazy_model.pkl"
                lstm_path = Path(ZONE_MODELS_BASE) / "lstm_model.h5"
                scaler_path = Path(ZONE_MODELS_BASE) / "scaler.pkl"
                encoders_path = Path(ZONE_MODELS_BASE) / "label_encoders.pkl"
                target_path = Path(ZONE_MODELS_BASE) / "target_encoder.pkl"
            
            # Load models
            knn_model = joblib.load(knn_path) if knn_path.exists() else None
            lstm_model = tf.keras.models.load_model(lstm_path) if lstm_path.exists() else None
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            label_encoders = joblib.load(encoders_path) if encoders_path.exists() else None
            target_encoder = joblib.load(target_path) if target_path.exists() else None
            
            if knn_model and lstm_model and scaler:
                logger.info(f"[OK] Loaded all models for {zone_name}")
                return {
                    'knn': knn_model,
                    'lstm': lstm_model,
                    'scaler': scaler,
                    'label_encoders': label_encoders,
                    'target_encoder': target_encoder
                }
            else:
                logger.error(f"[ERROR] Failed to load models for {zone_name}")
                return None
        
        except Exception as e:
            logger.error(f"[ERROR] Error loading models for {zone_name}: {e}")
            return None
    
    def load_models(self):
        """Load all zone-specific models"""
        try:
            # Load models for each zone
            for zone in AVAILABLE_ZONES:
                models = self.load_models_for_zone(zone)
                if models:
                    self.models[zone] = models
                    logger.info(f"[OK] Loaded models for {zone}")
            
            # Set default to Zone0
            if 'Zone0' in self.models:
                zone_models = self.models['Zone0']
                self.knn_model = zone_models['knn']
                self.lstm_model = zone_models['lstm']
                self.scaler = zone_models['scaler']
                self.label_encoders = zone_models['label_encoders']
                self.target_encoder = zone_models['target_encoder']
                self.current_zone = 'Zone0'
                logger.info("[OK] Initialized with Zone0 models")
            else:
                logger.warning("[WARNING] Could not load any zone models!")
        
        except Exception as e:
            logger.error(f"[ERROR] Error loading models: {e}")
            raise
    
    def set_zone(self, zone_name: str = 'Zone0'):
        """Switch to zone-specific models"""
        if zone_name not in AVAILABLE_ZONES:
            logger.warning(f"Zone {zone_name} not available, using Zone0")
            zone_name = 'Zone0'
        
        if zone_name in self.models:
            zone_models = self.models[zone_name]
            self.knn_model = zone_models['knn']
            self.lstm_model = zone_models['lstm']
            self.scaler = zone_models['scaler']
            self.label_encoders = zone_models['label_encoders']
            self.target_encoder = zone_models['target_encoder']
            self.current_zone = zone_name
            logger.info(f"[OK] Switched to {zone_name} models")
        else:
            logger.error(f"Models for {zone_name} not found")
    
    def preprocess_sample(self, sample_dict: Dict[str, Any]) -> tuple:
        """Preprocess a single sample for prediction"""
        # Map API field names to training data field names
        field_mapping = {
            'pressure_psi': 'Pressure_PSI',
            'master_flow_lpm': 'Master_Flow_LPM',
            'temperature_c': 'Temperature_C',
            'vibration': 'Vibration',
            'rpm': 'RPM',
            'operation_hours': 'OperationHours',
            'acoustic_level': 'AcousticLevel',
            'ultrasonic_signal': 'UltrasonicSignal',
            'pipe_age': 'PipeAge',
            'soil_type': 'SoilType',
            'material': 'Material'
        }
        
        # Rename fields to match training data
        mapped_dict = {}
        for api_key, value in sample_dict.items():
            if api_key in field_mapping:
                mapped_dict[field_mapping[api_key]] = value
            else:
                mapped_dict[api_key] = value
        
        df = pd.DataFrame([mapped_dict])
        
        # Handle missing categorical features
        for col in self.categorical_features:
            if col not in df.columns or pd.isna(df[col]).any():
                df[col] = 'unknown'
        
        # Encode categorical features
        categorical_encoded = []
        for col in self.categorical_features:
            if col in self.label_encoders:
                try:
                    encoded_val = self.label_encoders[col].transform([df[col].iloc[0]])[0]
                    categorical_encoded.append(float(encoded_val))
                except:
                    # Use 0 for unknown categories
                    categorical_encoded.append(0.0)
            else:
                categorical_encoded.append(0.0)
        
        # Select numeric features (don't scale yet)
        X_numeric = df[self.numeric_features].values  # Shape: (1, 9)
        X_categorical = np.array([categorical_encoded])  # Shape: (1, 2)
        
        # Combine numeric and categorical BEFORE scaling (scaler expects 11 features)
        X_combined = np.hstack([X_numeric, X_categorical])  # Shape: (1, 11)
        
        # Now scale all 11 features
        if self.scaler:
            X_combined = self.scaler.transform(X_combined)
        
        return X_combined
    
    def predict_single(self, sample: SensorData) -> Dict[str, Any]:
        """Predict anomaly for single sample"""
        sample_dict = sample.dict()
        
        try:
            # Switch zone models if zone_name is provided
            zone_name = sample_dict.pop('zone_name', None)
            if zone_name:
                self.set_zone(zone_name)
            
            # X_combined shape: (1, 11) - already preprocessed and scaled
            X_combined = self.preprocess_sample(sample_dict)
            
            predictions = {
                'knn_prediction': 'unknown',
                'knn_confidence': 0.0,
                'knn_all_confidences': {},
                'lstm_prediction': 'unknown',
                'lstm_confidence': 0.0,
                'lstm_all_confidences': {},
                'ensemble_prediction': 'unknown',
                'ensemble_confidence': 0.0,
                'anomaly_detected': False,
                'agreement': 'none'
            }
            
            # KNN prediction - uses all 11 features
            knn_pred_idx = None
            knn_max_confidence = 0.0
            if self.knn_model:
                knn_pred_idx = self.knn_model.predict(X_combined)[0]
                knn_proba = self.knn_model.predict_proba(X_combined)[0]
                knn_pred = self.target_encoder.inverse_transform([knn_pred_idx])[0]
                knn_max_confidence = float(np.max(knn_proba))
                
                predictions['knn_prediction'] = knn_pred
                predictions['knn_confidence'] = knn_max_confidence
                # Store all class confidences
                for i, class_name in enumerate(self.target_encoder.classes_):
                    predictions['knn_all_confidences'][class_name] = float(knn_proba[i])
            
            # LSTM prediction - uses all 11 features with time dimension
            lstm_pred_idx = None
            lstm_max_confidence = 0.0
            if self.lstm_model:
                # Reshape to (batch_size=1, time_steps=1, features=11)
                X_lstm = X_combined.reshape((1, 1, X_combined.shape[1]))
                lstm_proba = self.lstm_model.predict(X_lstm, verbose=0)[0]
                lstm_pred_idx = np.argmax(lstm_proba)
                lstm_pred = self.target_encoder.inverse_transform([lstm_pred_idx])[0]
                lstm_max_confidence = float(np.max(lstm_proba))
                
                predictions['lstm_prediction'] = lstm_pred
                predictions['lstm_confidence'] = lstm_max_confidence
                # Store all class confidences
                for i, class_name in enumerate(self.target_encoder.classes_):
                    predictions['lstm_all_confidences'][class_name] = float(lstm_proba[i])
            
            # Hybrid decision: if both agree, use that; else choose higher confidence
            if knn_pred_idx is not None and lstm_pred_idx is not None:
                if knn_pred_idx == lstm_pred_idx:
                    # Both models agree
                    final_pred_idx = knn_pred_idx
                    predictions['agreement'] = 'both_agree'
                    avg_confidence = (knn_max_confidence + lstm_max_confidence) / 2
                else:
                    # Models disagree - choose the one with higher confidence
                    if knn_max_confidence > lstm_max_confidence:
                        final_pred_idx = knn_pred_idx
                        predictions['agreement'] = 'knn_higher_confidence'
                        avg_confidence = knn_max_confidence
                    else:
                        final_pred_idx = lstm_pred_idx
                        predictions['agreement'] = 'lstm_higher_confidence'
                        avg_confidence = lstm_max_confidence
                
                ensemble_pred = self.target_encoder.inverse_transform([final_pred_idx])[0]
                predictions['ensemble_prediction'] = ensemble_pred
                predictions['ensemble_confidence'] = avg_confidence
                
                # Detect anomaly if prediction is not "Normal"
                if ensemble_pred != 'Normal':
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
    logger.info(f"[INFO] Zone Models Base: {ZONE_MODELS_BASE}")
    logger.info(f"[INFO] Available Zones: {AVAILABLE_ZONES}")
    logger.info("=" * 70)
    
    try:
        model_manager = ModelManager()
        logger.info("[OK] Model Manager initialized successfully")
        logger.info(f"[OK] Available zones: {list(model_manager.models.keys())}")
        logger.info(f"[OK] Current zone: {model_manager.current_zone}")
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
        status="operational" if model_manager.knn_model and model_manager.lstm_model else "degraded",
        models={
            "knn": "loaded" if model_manager.knn_model else "not_loaded",
            "lstm": "loaded" if model_manager.lstm_model else "not_loaded"
        },
        available_zones=list(model_manager.models.keys()),
        current_zone=model_manager.current_zone,
        version="2.0 (Zone-Specific)",
        timestamp=datetime.now().isoformat()
    )

# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict", response_model=SinglePredictResponse)
async def predict_single(sample: SensorData):
    """Predict anomaly for single sensor reading (supports zone-specific models)"""
    if not model_manager or not model_manager.knn_model or not model_manager.lstm_model:
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
                "zone": model_manager.current_zone,
                "features_used": model_manager.numeric_features,
                "categorical_features": model_manager.categorical_features
            },
            execution_time_ms=execution_time,
            model_version=f"v2.0-{model_manager.current_zone}"
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
