#!/usr/bin/env python
"""Debug preprocessing pipeline"""

import sys
sys.path.insert(0, 'c:/Users/PRADHAN/OneDrive/Desktop/water')

from src.model.api import ModelManager, SensorData
import numpy as np

# Initialize model manager
mm = ModelManager()

# Create test data
test_data = SensorData(
    zone_name="Zone0",
    pressure_psi=117.29,
    master_flow_lpm=3946.92,
    temperature_c=32.96,
    vibration=1.39,
    rpm=292.63,
    operation_hours=14663.72,
    acoustic_level=19.19,
    ultrasonic_signal=0.4,
    pipe_age=10,
    soil_type="Rocky",
    material="PVC"
)

print("Test data created:")
print(f"  Zone: {test_data.zone_name}")
print(f"  Pressure: {test_data.pressure_psi}")

# Get dict
sample_dict = test_data.dict()
print(f"\nSample dict keys: {sample_dict.keys()}")

# Pop zone_name
zone_name = sample_dict.pop('zone_name', None)
print(f"Zone name: {zone_name}")
print(f"Remaining keys: {sample_dict.keys()}")

# Preprocess
try:
    print("\nCalling preprocess_sample...")
    X_combined = mm.preprocess_sample(sample_dict)
    print(f"X_combined shape: {X_combined.shape}")
    print(f"X_combined dtype: {X_combined.dtype}")
    print(f"X_combined:\n{X_combined}")
    
    # Try prediction
    print("\nTrying KNN prediction...")
    knn_pred = mm.knn_model.predict(X_combined)
    print(f"KNN prediction: {knn_pred}")
    
    print("\nTrying LSTM prediction...")
    X_lstm = X_combined.reshape((1, 1, X_combined.shape[1]))
    print(f"X_lstm shape: {X_lstm.shape}")
    lstm_proba = mm.lstm_model.predict(X_lstm, verbose=0)
    print(f"LSTM output shape: {lstm_proba.shape}")
    print(f"LSTM output: {lstm_proba}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
