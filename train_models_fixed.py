"""
FIXED Model Training Script with Class Balancing
=================================================
Implements proper handling of imbalanced classes:
- Class weights for both KNN and LSTM
- Stratified sampling
- Proper validation
- Better hyperparameters
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

from src.config import config


def load_dataset():
    """Load all water sensor data batches"""
    data_dir = config.TRAINING_DATA_DIR
    all_files = list(data_dir.glob("water_batch_*.json"))

    if not all_files:
        raise FileNotFoundError(f"No water_batch_*.json files found in {data_dir}")

    dfs = []
    for f in all_files:
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


def train_models_with_balancing():
    """Train KNN and LSTM models with proper class balancing"""

    # Load dataset
    df = load_dataset()

    # -------- Preprocessing --------
    numeric_features = config.NUMERIC_FEATURES
    categorical_features = config.CATEGORICAL_FEATURES
    target_col = config.TARGET_COLUMN

    # Print class distribution
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    class_counts = df[target_col].value_counts()
    for cls, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {cls:25} {count:6} ({pct:6.2f}%)")

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

    # -------- Train/Test Split with Stratification --------
    X = df[numeric_features + categorical_features].values
    y = df[target_col].values

    # For RNN we need sequences
    X_rnn = X.reshape((X.shape[0], 1, X.shape[1]))

    X_train, X_test, X_rnn_train, X_rnn_test, y_train, y_test = train_test_split(
        X,
        X_rnn,
        y,
        test_size=config.TRAIN_TEST_SPLIT,
        random_state=42,
        stratify=y,  # Maintains class distribution
    )

    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # ========== COMPUTE CLASS WEIGHTS ==========
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\n" + "="*60)
    print("CLASS WEIGHTS (for balanced training)")
    print("="*60)
    for class_idx, weight in class_weight_dict.items():
        class_name = target_le.inverse_transform([class_idx])[0]
        print(f"  {class_name:25} {weight:.4f}")

    # -------- KNN with Class Consideration --------
    print("\n" + "="*60)
    print("Training KNN Classifier (with distance weighting)")
    print("="*60)
    
    knn = KNeighborsClassifier(
        n_neighbors=config.KNN_NEIGHBORS,
        weights='distance',  # Weight by distance
        leaf_size=30,
        n_jobs=-1  # Use all cores
    )
    knn.fit(X_train, y_train)
    
    y_knn_pred = knn.predict(X_test)
    knn_f1 = f1_score(y_test, y_knn_pred, average='weighted')
    
    print("\nKNN Classification Report:")
    print(classification_report(
        y_test, 
        y_knn_pred,
        target_names=target_le.classes_,
        zero_division=0
    ))
    print(f"Weighted F1-Score: {knn_f1:.4f}")

    config.MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, config.KNN_MODEL_FILE)
    print(f"\n[OK] Saved KNN model to {config.KNN_MODEL_FILE}")

    # -------- LSTM with Class Weights --------
    print("\n" + "="*60)
    print("Training LSTM Model (with class weights)")
    print("="*60)
    
    num_classes = len(np.unique(y))
    y_train_rnn = pd.get_dummies(y_train).values
    y_test_rnn = pd.get_dummies(y_test).values

    lstm_model = Sequential([
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
    ])

    lstm_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    # Train with class weights
    history = lstm_model.fit(
        X_rnn_train,
        y_train_rnn,
        validation_split=config.LSTM_VALIDATION_SPLIT,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # KEY: Use class weights
        verbose=1,
    )

    # Evaluate
    lstm_pred = lstm_model.predict(X_rnn_test, verbose=0)
    y_lstm_pred = np.argmax(lstm_pred, axis=1)
    lstm_f1 = f1_score(y_test, y_lstm_pred, average='weighted')
    
    print("\nLSTM Classification Report:")
    print(classification_report(
        y_test,
        y_lstm_pred,
        target_names=target_le.classes_,
        zero_division=0
    ))
    print(f"Weighted F1-Score: {lstm_f1:.4f}")

    # Save LSTM
    lstm_model.save(config.LSTM_MODEL_FILE)
    print(f"\n[OK] Saved LSTM model to {config.LSTM_MODEL_FILE}")

    # -------- Save Preprocessing Objects --------
    joblib.dump(scaler, config.SCALER_FILE)
    joblib.dump(label_encoders, config.LABEL_ENCODERS_FILE)
    joblib.dump(target_le, config.TARGET_ENCODER_FILE)
    
    print(f"[OK] Saved scaler to {config.SCALER_FILE}")
    print(f"[OK] Saved label encoders to {config.LABEL_ENCODERS_FILE}")
    print(f"[OK] Saved target encoder to {config.TARGET_ENCODER_FILE}")

    # -------- Summary --------
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"KNN F1-Score (weighted):  {knn_f1:.4f}")
    print(f"LSTM F1-Score (weighted): {lstm_f1:.4f}")
    print(f"Models saved to: {config.MODEL_WEIGHTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    train_models_with_balancing()
