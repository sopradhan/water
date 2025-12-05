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

# -----------------------------
# Load Dataset
# -----------------------------
data_dir = Path(r"C:\Users\GENAIKOLGPUSR36\Desktop\water\water_sensor_data")
model_dir= Path(r"C:\Users\GENAIKOLGPUSR36\Desktop\water\model\binary")
all_files = list(data_dir.glob("water_batch_*.json"))

dfs = []
for f in all_files:
    with open(f) as jf:
        dfs.append(pd.DataFrame(json.load(jf)))

df = pd.concat(dfs, ignore_index=True)
print(f"Total samples: {len(df)}")

# -----------------------------
# Preprocessing
# -----------------------------
numeric_features = ["Pressure", "FlowRate", "Temperature", "Vibration",
                    "RPM", "OperationHours", "AcousticLevel", "UltrasonicSignal", "PipeAge"]

categorical_features = ["SoilType", "Material"]
target_col = "class_label"

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

# -----------------------------
# Train/Test Split
# -----------------------------
X = df[numeric_features + categorical_features].values
y = df[target_col].values

# For RNN we need sequences (we'll use 1-step sequences for simplicity)
X_rnn = X.reshape((X.shape[0], 1, X.shape[1]))

X_train, X_test, X_rnn_train, X_rnn_test, y_train, y_test = train_test_split(
    X, X_rnn, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Lazy Learning: KNN
# -----------------------------
print("Training KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_knn_pred = knn.predict(X_test)
print("KNN Classification Report:\n", classification_report(y_test, y_knn_pred))

joblib.dump(knn, model_dir / "knn_lazy_model.pkl")
print("✓ Saved KNN model")

# -----------------------------
# RNN Model (LSTM)
# -----------------------------
print("Training LSTM model...")
num_classes = len(np.unique(y))
y_train_rnn = pd.get_dummies(y_train).values
y_test_rnn = pd.get_dummies(y_test).values

lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_rnn_train.shape[1], X_rnn_train.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

lstm_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

history = lstm_model.fit(
    X_rnn_train, y_train_rnn,
    validation_split=0.1,
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
y_rnn_pred = lstm_model.predict(X_rnn_test)
y_rnn_classes = np.argmax(y_rnn_pred, axis=1)
print("LSTM Classification Report:\n", classification_report(y_test, y_rnn_classes))

lstm_model.save(model_dir / "lstm_model.h5")
print("✓ Saved LSTM model")

# -----------------------------
# Confusion Matrices
# -----------------------------
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_knn_pred))
print("LSTM Confusion Matrix:\n", confusion_matrix(y_test, y_rnn_classes))

# -----------------------------
# Save preprocessing objects
# -----------------------------
joblib.dump(scaler, model_dir / "scaler.pkl")
joblib.dump(label_encoders, model_dir / "label_encoders.pkl")
joblib.dump(target_le, model_dir / "target_encoder.pkl")
print("✓ Saved scaler and encoders")
