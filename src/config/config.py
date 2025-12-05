"""
Configuration management for Water Network Anomaly Detection
Loads environment variables and provides configuration constants
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # If .env doesn't exist, use .env.example as fallback
    env_example_path = Path(__file__).parent.parent.parent / ".env.example"
    if env_example_path.exists():
        load_dotenv(env_example_path)


class Config:
    """Base configuration class"""

    # Project root
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # Data paths
    TRAINING_DATA_DIR = Path(
        os.getenv("TRAINING_DATA_DIR", str(PROJECT_ROOT / "src" / "data" / "training_dataset"))
    )
    TRAINING_DATA_OUTPUT_DIR = Path(
        os.getenv("TRAINING_DATA_OUTPUT_DIR", str(PROJECT_ROOT / "src" / "data" / "training_dataset"))
    )
    PROD_DATA_DIR = Path(
        os.getenv("PROD_DATA_DIR", str(PROJECT_ROOT / "src" / "data" / "prod_data"))
    )

    # Model paths
    MODEL_WEIGHTS_DIR = Path(os.getenv("MODEL_WEIGHTS_DIR", str(PROJECT_ROOT / "src" / "model" / "model_weights")))
    SCALER_FILE = Path(os.getenv("SCALER_FILE", str(MODEL_WEIGHTS_DIR / "scaler.pkl")))
    KNN_MODEL_FILE = Path(os.getenv("KNN_MODEL_FILE", str(MODEL_WEIGHTS_DIR / "knn_lazy_model.pkl")))
    LSTM_MODEL_FILE = Path(os.getenv("LSTM_MODEL_FILE", str(MODEL_WEIGHTS_DIR / "lstm_model.h5")))
    LABEL_ENCODERS_FILE = Path(
        os.getenv("LABEL_ENCODERS_FILE", str(MODEL_WEIGHTS_DIR / "label_encoders.pkl"))
    )
    TARGET_ENCODER_FILE = Path(
        os.getenv("TARGET_ENCODER_FILE", str(MODEL_WEIGHTS_DIR / "target_encoder.pkl"))
    )

    # Model parameters
    KNN_NEIGHBORS = int(os.getenv("KNN_NEIGHBORS", "5"))
    LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "50"))
    LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", "64"))
    LSTM_VALIDATION_SPLIT = float(os.getenv("LSTM_VALIDATION_SPLIT", "0.1"))
    TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", "0.2"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))

    # Feature columns
    NUMERIC_FEATURES = [
        "Pressure_PSI",
        "Master_Flow_LPM",
        "Temperature_C",
        "Vibration",
        "RPM",
        "OperationHours",
        "AcousticLevel",
        "UltrasonicSignal",
        "PipeAge",
    ]

    CATEGORICAL_FEATURES = ["SoilType", "Material"]

    TARGET_COLUMN = "class_label"

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist"""
        cls.TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.TRAINING_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


# Create directories on import
Config.ensure_directories()

# Export config instance
config = Config()
