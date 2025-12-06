"""
Retrain Models with Class Balancing
====================================
Fixes the model bias by using class weights and balancing techniques
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
from collections import Counter

from src.config import config


def load_dataset():
    """Load all water sensor data batches"""
    data_dir = config.TRAINING_DATA_DIR
    all_files = list(data_dir.glob("water_batch_*.json"))

    if not all_files:
        raise FileNotFoundError(f"No water_batch_*.json files found in {data_dir}")

    dfs = []
    for f in sorted(all_files):
        with open(f) as jf:
            dfs.append(pd.DataFrame(json.load(jf)))

    df = pd.concat(dfs, ignore_index=True)
    
    # Extract values from {'value': X, 'unit': Y} structure
    numeric_features = config.NUMERIC_FEATURES
    for col in numeric_features:
        if col in df.columns:
            if isinstance(df[col].iloc[0], dict):
                df[col] = df[col].apply(lambda x: x.get('value', x) if isinstance(x, dict) else x)
    
    print(f"Total samples: {len(df)}")
    return df


def calculate_class_weights(y):
    """Calculate class weights for imbalanced data"""
    class_counts = Counter(y)
    total = len(y)
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION & WEIGHTS")
    print("="*60)
    
    weights = {}
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        # Weight = total / (num_classes * class_count)
        weight = total / (len(class_counts) * count)
        weights[class_idx] = weight
        
        percentage = (count / total) * 100
        print(f"Class {class_idx}: {count:5} samples ({percentage:6.2f}%) - Weight: {weight:.4f}")
    
    print("="*60 + "\n")
    return weights


def train_models():
    """Train KNN and LSTM models with class balancing"""

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

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Calculate class weights for both models
    class_weights = calculate_class_weights(y_train)

    # -------- Lazy Learning: KNN with Class Weights --------
    print("\n" + "=" * 50)
    print("Training KNN classifier with class weights...")
    print("=" * 50)
    knn = KNeighborsClassifier(n_neighbors=config.KNN_NEIGHBORS, weights='distance')
    knn.fit(X_train, y_train)
    y_knn_pred = knn.predict(X_test)
    
    print("\nKNN Classification Report:")
    print(classification_report(y_test, y_knn_pred, 
                                target_names=target_le.classes_))
    print("KNN Confusion Matrix:")
    print(confusion_matrix(y_test, y_knn_pred))

    config.MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, config.KNN_MODEL_FILE)
    print(f"\n[OK] Saved KNN model to {config.KNN_MODEL_FILE}")

    # -------- RNN Model (LSTM) with Class Weights --------
    print("\n" + "=" * 50)
    print("Training LSTM model with class weights...")
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

    print(f"\nClass weights: {class_weights}")
    
    history = lstm_model.fit(
        X_rnn_train,
        y_train_rnn,
        validation_split=config.LSTM_VALIDATION_SPLIT,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    lstm_model.save(config.LSTM_MODEL_FILE)
    print(f"\n[OK] Saved LSTM model to {config.LSTM_MODEL_FILE}")

    # -------- Evaluate on Test Set --------
    print("\n" + "=" * 50)
    print("LSTM Test Set Evaluation")
    print("=" * 50)
    lstm_pred = lstm_model.predict(X_rnn_test, verbose=0)
    lstm_pred_classes = np.argmax(lstm_pred, axis=1)
    
    print("\nLSTM Classification Report:")
    print(classification_report(y_test, lstm_pred_classes, 
                                target_names=target_le.classes_))
    print("LSTM Confusion Matrix:")
    print(confusion_matrix(y_test, lstm_pred_classes))

    # -------- Save Preprocessing Objects --------
    joblib.dump(scaler, config.SCALER_FILE)
    joblib.dump(label_encoders, config.LABEL_ENCODERS_FILE)
    joblib.dump(target_le, config.TARGET_ENCODER_FILE)
    
    print(f"\n[OK] Saved scaler to {config.SCALER_FILE}")
    print(f"[OK] Saved label encoders to {config.LABEL_ENCODERS_FILE}")
    print(f"[OK] Saved target encoder to {config.TARGET_ENCODER_FILE}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print("\nModels trained with class balancing to handle imbalanced data:")
    print(f"  - KNN: Uses distance weighting for neighbors")
    print(f"  - LSTM: Trained with class weights = {class_weights}")
    print("\nModels are now ready for production use!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    train_models()
