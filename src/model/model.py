"""
Hybrid RNN + Lazy Learning Water Network Classifier
---------------------------------------------------
- Uses synthetic water dataset
- Lazy learning (KNN) for instant anomaly detection
- LSTM for temporal pattern learning
- Saves models and evaluation metrics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.config import config


def load_dataset():
    """Load all water sensor data batches and extract values from unit structures"""
    data_dir = config.TRAINING_DATA_DIR
    all_files = list(data_dir.glob("water_batch_*.json"))

    if not all_files:
        raise FileNotFoundError(f"No water_batch_*.json files found in {data_dir}")

    dfs = []
    for f in all_files:
        with open(f) as jf:
            dfs.append(pd.DataFrame(json.load(jf)))

    df = pd.concat(dfs, ignore_index=True)
    
    # Extract values from {'value': X, 'unit': Y} structure for numeric features
    numeric_features = config.NUMERIC_FEATURES
    for col in numeric_features:
        if col in df.columns:
            # Check if column contains dict structure
            if isinstance(df[col].iloc[0], dict):
                df[col] = df[col].apply(lambda x: x.get('value', x) if isinstance(x, dict) else x)
    
    print(f"Total samples: {len(df)}")
    return df


def train_models():
    """Train KNN and LSTM models"""

    # Load dataset
    df = load_dataset()

    # -------- Preprocessing --------
    numeric_features = config.NUMERIC_FEATURES
    categorical_features = config.CATEGORICAL_FEATURES
    target_col = config.TARGET_COLUMN

    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Encode target
    target_le = LabelEncoder()
    df[target_col] = target_le.fit_transform(df[target_col].astype(str))

    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # -------- Train/Test Split --------
    X = df[numeric_features + categorical_features].values
    y = df[target_col].values

    # For RNN we need sequences (1-step sequences for simplicity)
    X_rnn = X.reshape((X.shape[0], 1, X.shape[1]))

    X_train, X_test, X_rnn_train, X_rnn_test, y_train, y_test = train_test_split(
        X,
        X_rnn,
        y,
        test_size=config.TRAIN_TEST_SPLIT,
        random_state=42,
        stratify=y,
    )

    # -------- Lazy Learning: KNN --------
    print("\n" + "=" * 50)
    print("Training KNN classifier...")
    print("=" * 50)
    knn = KNeighborsClassifier(n_neighbors=config.KNN_NEIGHBORS)
    knn.fit(X_train, y_train)
    y_knn_pred = knn.predict(X_test)
    print("KNN Classification Report:\n", classification_report(y_test, y_knn_pred))

    config.MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, config.KNN_MODEL_FILE)
    print(f"✓ Saved KNN model to {config.KNN_MODEL_FILE}")

    # -------- RNN Model (LSTM) --------
    print("\n" + "=" * 50)
    print("Training LSTM model...")
    print("=" * 50)
    num_classes = len(np.unique(y))
    y_train_rnn = pd.get_dummies(y_train).values
    y_test_rnn = pd.get_dummies(y_test).values

    lstm_model = Sequential(
        [
            LSTM(
                64,
                activation="relu",
                input_shape=(X_rnn_train.shape[1], X_rnn_train.shape[2]),
                return_sequences=True,
            ),
            Dropout(0.3),
            LSTM(32, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    lstm_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = lstm_model.fit(
        X_rnn_train,
        y_train_rnn,
        validation_split=config.LSTM_VALIDATION_SPLIT,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_rnn_pred = lstm_model.predict(X_rnn_test)
    y_rnn_classes = np.argmax(y_rnn_pred, axis=1)
    print("LSTM Classification Report:\n", classification_report(y_test, y_rnn_classes))

    lstm_model.save(config.LSTM_MODEL_FILE)
    print(f"✓ Saved LSTM model to {config.LSTM_MODEL_FILE}")

    # -------- Confusion Matrices --------
    print("\n" + "=" * 50)
    print("Confusion Matrices")
    print("=" * 50)
    print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_knn_pred))
    print("\nLSTM Confusion Matrix:\n", confusion_matrix(y_test, y_rnn_classes))

    # -------- Save preprocessing objects --------
    joblib.dump(scaler, config.SCALER_FILE)
    joblib.dump(label_encoders, config.LABEL_ENCODERS_FILE)
    joblib.dump(target_le, config.TARGET_ENCODER_FILE)
    print(f"\n✓ Saved preprocessing objects")
    print(f"  - Scaler: {config.SCALER_FILE}")
    print(f"  - Label Encoders: {config.LABEL_ENCODERS_FILE}")
    print(f"  - Target Encoder: {config.TARGET_ENCODER_FILE}")


if __name__ == "__main__":
    train_models()
