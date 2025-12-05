"""
Hybrid Water Leakage Anomaly Detector
-------------------------------------
- Lazy Learning: KNN (instant anomaly detection)
- Temporal Model: LSTM (sequential patterns)
- Real-time hybrid prediction for a single row
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model

from src.config import config


# --------- Load preprocessing and models ---------
def load_models():
    """Load all required models and preprocessing objects"""
    scaler = joblib.load(config.SCALER_FILE)
    label_encoders = joblib.load(config.LABEL_ENCODERS_FILE)
    target_encoder = joblib.load(config.TARGET_ENCODER_FILE)
    knn = joblib.load(config.KNN_MODEL_FILE)

    # Load LSTM for inference only to suppress warnings
    lstm_model = load_model(config.LSTM_MODEL_FILE, compile=False)

    return {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "knn": knn,
        "lstm": lstm_model,
    }


# --------- Prediction Function ---------
def preprocess_sample(sample: dict, scaler, label_encoders):
    """Encode and scale a single sample"""
    numeric_features = config.NUMERIC_FEATURES
    categorical_features = config.CATEGORICAL_FEATURES

    df = pd.DataFrame([sample])

    # Encode categorical features
    for col in categorical_features:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = le.transform(df[col].astype(str))

    # Scale numeric features
    df[numeric_features] = scaler.transform(df[numeric_features])

    return df


def hybrid_predict(sample: dict, models: dict = None):
    """Hybrid KNN + LSTM prediction"""
    if models is None:
        models = load_models()

    numeric_features = config.NUMERIC_FEATURES
    categorical_features = config.CATEGORICAL_FEATURES

    scaler = models["scaler"]
    label_encoders = models["label_encoders"]
    target_encoder = models["target_encoder"]
    knn = models["knn"]
    lstm_model = models["lstm"]

    df = preprocess_sample(sample, scaler, label_encoders)
    X = df[numeric_features + categorical_features].values
    X_rnn = X.reshape((X.shape[0], 1, X.shape[1]))

    # KNN prediction (lazy learning)
    knn_pred = knn.predict(X)[0]
    knn_proba = knn.predict_proba(X)[0]

    # LSTM prediction
    lstm_pred = lstm_model.predict(X_rnn, verbose=0)
    lstm_class = np.argmax(lstm_pred, axis=1)[0]
    lstm_confidence = lstm_pred[0].tolist()

    # Hybrid decision: if both agree, use that; else choose higher confidence
    if knn_pred == lstm_class:
        final_class = knn_pred
    else:
        final_class = (
            knn_pred
            if knn_proba[knn_pred] > lstm_confidence[lstm_class]
            else lstm_class
        )

    # Decode class labels
    class_label = target_encoder.inverse_transform([final_class])[0]

    return {
        "final_class": class_label,
        "knn_class": target_encoder.inverse_transform([knn_pred])[0],
        "knn_confidence": {
            target_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(knn_proba)
        },
        "lstm_class": target_encoder.inverse_transform([lstm_class])[0],
        "lstm_confidence": {
            target_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(lstm_confidence)
        },
    }


# --------- Example Usage ---------
if __name__ == "__main__":
    sample = {
        "Pressure": 35,
        "FlowRate": 700,
        "Temperature": 28,
        "Vibration": 5,
        "RPM": 1500,
        "OperationHours": 20000,
        "AcousticLevel": 88,
        "UltrasonicSignal": 0.8,
        "PipeAge": 25,
        "SoilType": "Sandy",
        "Material": "PVC",
    }

    result = hybrid_predict(sample)
    print(json.dumps(result, indent=2))
