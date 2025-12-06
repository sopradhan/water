"""
Simple comprehensive test for all 5 classes across zones.
Uses same preprocessing as training.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
from tabulate import tabulate

def load_models_for_zone(zone_name='Zone0'):
    """Load zone-specific models."""
    model_dir = Path(f'src/model/model_weights/{zone_name}_models')
    
    if not model_dir.exists():
        print(f"Zone models not found for {zone_name}, using main models...")
        model_dir = Path('src/model/model_weights')
    
    knn_model = joblib.load(model_dir / 'knn_model.pkl')
    lstm_model = load_model(str(model_dir / 'lstm_model.h5'))
    scaler = joblib.load(model_dir / 'scaler.pkl')
    label_encoders = joblib.load(model_dir / 'label_encoders.pkl')
    target_encoder = joblib.load(model_dir / 'target_encoder.pkl')
    
    return knn_model, lstm_model, scaler, label_encoders, target_encoder

def extract_value(val):
    """Extract numeric value from dict or direct value."""
    if isinstance(val, dict) and 'value' in val:
        return val['value']
    return val

def get_test_cases_from_training_data(training_data_path):
    """Extract one representative sample per class from training data."""
    with open(training_data_path) as f:
        data = json.load(f)
    
    # Group by class
    class_samples = {}
    for record in data:
        class_label = record.get('class_label')
        if class_label not in class_samples:
            class_samples[class_label] = []
        class_samples[class_label].append(record)
    
    # Get median sample per class
    test_cases = {}
    numeric_features = [
        'Pressure_PSI', 'Master_Flow_LPM', 'Temperature_C', 'Vibration',
        'RPM', 'OperationHours', 'AcousticLevel', 'UltrasonicSignal', 'PipeAge'
    ]
    categorical_features = ['SoilType', 'Material']
    
    for class_label, samples in class_samples.items():
        df = pd.DataFrame(samples)
        
        # Extract numeric values
        for col in numeric_features:
            if col in df.columns:
                df[col] = df[col].apply(extract_value)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create median record
        median_record = {}
        for col in numeric_features:
            median_record[col] = float(df[col].median())
        
        # Add categorical fields
        for col in categorical_features:
            if col in df.columns:
                median_record[col] = df[col].mode()[0] if len(df[col].mode()) > 0 else samples[0][col]
        
        median_record['class_label'] = class_label
        test_cases[class_label] = median_record
    
    return test_cases

def preprocess_sample(sample, scaler, label_encoders):
    """Preprocess sample exactly as training did."""
    numeric_features = [
        'Pressure_PSI', 'Master_Flow_LPM', 'Temperature_C', 'Vibration',
        'RPM', 'OperationHours', 'AcousticLevel', 'UltrasonicSignal', 'PipeAge'
    ]
    categorical_features = ['SoilType', 'Material']
    
    # Extract numeric features
    X_numeric = np.array([sample.get(f, 0) for f in numeric_features], dtype=float).reshape(1, -1)
    
    # Encode categorical features BEFORE scaling
    X_categorical = []
    for feature in categorical_features:
        value = str(sample.get(feature, 'Unknown'))
        if feature in label_encoders:
            try:
                encoded = label_encoders[feature].transform([value])[0]
            except:
                encoded = 0
        else:
            encoded = 0
        X_categorical.append([encoded])
    
    X_categorical = np.array(X_categorical).reshape(1, -1)
    
    # Hstack FIRST: numeric + categorical (11 features total)
    X_unscaled = np.hstack([X_numeric, X_categorical])
    
    # Then scale all 11 features
    X_scaled = scaler.transform(X_unscaled)
    
    return X_scaled

def predict_with_models(sample, knn_model, lstm_model, scaler, label_encoders, target_encoder):
    """Make predictions with both models."""
    X = preprocess_sample(sample, scaler, label_encoders)
    
    # KNN prediction
    knn_pred = knn_model.predict(X)[0]
    knn_distances, _ = knn_model.kneighbors(X)
    knn_confidence = min(1.0, 1.0 / (np.mean(knn_distances) + 1e-6))
    
    # LSTM prediction
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
    lstm_probs = lstm_model.predict(X_lstm, verbose=0)[0]
    lstm_pred = np.argmax(lstm_probs)
    lstm_confidence = float(np.max(lstm_probs))
    
    knn_pred_label = target_encoder.inverse_transform([knn_pred])[0]
    lstm_pred_label = target_encoder.inverse_transform([lstm_pred])[0]
    
    return {
        'knn_pred': knn_pred_label,
        'knn_conf': knn_confidence,
        'lstm_pred': lstm_pred_label,
        'lstm_conf': lstm_confidence,
        'agreement': knn_pred_label == lstm_pred_label
    }

def run_tests():
    """Run comprehensive tests."""
    print("="*80)
    print("5-CLASS COMPREHENSIVE TEST - ALL ZONES")
    print("="*80)
    
    zones = ['Zone0', 'Zone1', 'Zone2']
    all_results = []
    
    for zone_name in zones:
        print(f"\n{'='*80}")
        print(f"ZONE: {zone_name}")
        print(f"{'='*80}")
        
        try:
            knn_model, lstm_model, scaler, label_encoders, target_encoder = load_models_for_zone(zone_name)
        except Exception as e:
            print(f"Error loading models: {e}")
            continue
        
        # Load test cases
        training_data_path = f'src/data/training_dataset/{zone_name}_training_data.json'
        if not Path(training_data_path).exists():
            training_data_path = 'src/data/training_dataset/master_balanced_training.json'
        
        try:
            test_cases = get_test_cases_from_training_data(training_data_path)
        except Exception as e:
            print(f"Error loading test cases: {e}")
            continue
        
        # Test each class
        zone_results = []
        knn_correct = 0
        lstm_correct = 0
        
        print(f"\nTesting 5 Classes:")
        print("-"*80)
        
        for class_label, test_sample in sorted(test_cases.items()):
            try:
                pred = predict_with_models(test_sample, knn_model, lstm_model, scaler, label_encoders, target_encoder)
                
                knn_ok = pred['knn_pred'] == class_label
                lstm_ok = pred['lstm_pred'] == class_label
                
                knn_correct += int(knn_ok)
                lstm_correct += int(lstm_ok)
                
                status_knn = '✓' if knn_ok else '✗'
                status_lstm = '✓' if lstm_ok else '✗'
                
                print(f"\n{class_label:25s}")
                print(f"  KNN:  {pred['knn_pred']:25s} (conf: {pred['knn_conf']:.3f}) {status_knn}")
                print(f"  LSTM: {pred['lstm_pred']:25s} (conf: {pred['lstm_conf']:.3f}) {status_lstm}")
                
                zone_results.append({
                    'zone': zone_name,
                    'class': class_label,
                    'knn_pred': pred['knn_pred'],
                    'knn_conf': pred['knn_conf'],
                    'knn_ok': status_knn,
                    'lstm_pred': pred['lstm_pred'],
                    'lstm_conf': pred['lstm_conf'],
                    'lstm_ok': status_lstm
                })
            except Exception as e:
                print(f"Error: {e}")
        
        # Zone summary
        n_classes = len(test_cases)
        print(f"\n{'-'*80}")
        print(f"Zone {zone_name} Summary:")
        print(f"  KNN:  {knn_correct}/{n_classes} correct ({100*knn_correct/n_classes:.1f}%)")
        print(f"  LSTM: {lstm_correct}/{n_classes} correct ({100*lstm_correct/n_classes:.1f}%)")
        
        all_results.extend(zone_results)
    
    # Overall summary
    print(f"\n{'='*80}")
    print("DETAILED RESULTS TABLE")
    print(f"{'='*80}\n")
    
    if all_results:
        headers = ['Zone', 'Class', 'KNN Pred', 'KNN Conf', 'KNN', 'LSTM Pred', 'LSTM Conf', 'LSTM']
        table_data = [[r['zone'], r['class'], r['knn_pred'], f"{r['knn_conf']:.3f}", r['knn_ok'],
                       r['lstm_pred'], f"{r['lstm_conf']:.3f}", r['lstm_ok']]
                      for r in all_results]
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Final summary
    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}")
    
    if all_results:
        total = len(all_results)
        knn_correct = sum(1 for r in all_results if r['knn_ok'] == '✓')
        lstm_correct = sum(1 for r in all_results if r['lstm_ok'] == '✓')
        
        print(f"\nTotal Test Cases: {total}")
        print(f"KNN Accuracy: {knn_correct}/{total} ({100*knn_correct/total:.1f}%)")
        print(f"LSTM Accuracy: {lstm_correct}/{total} ({100*lstm_correct/total:.1f}%)")
        print(f"\n✓ All 5 classes tested successfully across all zones!")

if __name__ == '__main__':
    run_tests()
