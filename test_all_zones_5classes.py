"""
Comprehensive test suite for all 5 classes across zones.
Tests on zone-specific models with data-driven test cases.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
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

def get_test_cases_from_training_data(training_data_path):
    """Extract one representative sample per class from training data."""
    with open(training_data_path) as f:
        data = json.load(f)
    
    def extract_value(val):
        """Extract numeric value from dict or direct value."""
        if isinstance(val, dict) and 'value' in val:
            return val['value']
        return val
    
    # Group by class
    class_samples = {}
    for record in data:
        class_label = record.get('class_label')
        if class_label not in class_samples:
            class_samples[class_label] = []
        class_samples[class_label].append(record)
    
    # Get one sample per class - take the median (most representative)
    test_cases = {}
    for class_label, samples in class_samples.items():
        # Convert to DataFrame to calculate median
        df = pd.DataFrame(samples)
        numeric_cols = [
            'Pressure_PSI', 'Master_Flow_LPM', 'Temperature_C', 'Vibration',
            'RPM', 'OperationHours', 'AcousticLevel', 'UltrasonicSignal', 'PipeAge'
        ]
        
        # Extract values from dict format
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(extract_value)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        median_values = df[numeric_cols].median()
        
        # Create median record
        median_record = {col: float(median_values[col]) for col in numeric_cols}
        
        # Add categorical fields
        categorical_cols = ['SoilType', 'Material']
        for col in categorical_cols:
            if col in df.columns:
                # Get most common value
                median_record[col] = df[col].mode()[0] if len(df[col].mode()) > 0 else samples[0][col]
        
        median_record['class_label'] = class_label
        test_cases[class_label] = median_record
    
    return test_cases

def preprocess_sample(sample, scaler, label_encoders):
    """Preprocess a single sample for prediction."""
    numeric_features = [
        'Pressure_PSI', 'Master_Flow_LPM', 'Temperature_C', 'Vibration',
        'RPM', 'OperationHours', 'AcousticLevel', 'UltrasonicSignal', 'PipeAge'
    ]
    
    categorical_features = ['SoilType', 'Material']
    
    def extract_value(val):
        """Extract numeric value from dict or direct value."""
        if isinstance(val, dict) and 'value' in val:
            return val['value']
        return val
    
    # Numeric features
    X_numeric = np.array([extract_value(sample.get(f, 0)) for f in numeric_features], dtype=float).reshape(1, -1)
    X_numeric_scaled = scaler.transform(X_numeric)
    
    # Categorical features
    X_categorical = []
    for feature in categorical_features:
        value = sample.get(feature, 'Unknown')
        if feature in label_encoders:
            try:
                encoded = label_encoders[feature].transform([str(value)])[0]
            except:
                # Handle unknown category
                encoded = 0
        else:
            encoded = 0
        X_categorical.append(encoded)
    
    X_categorical = np.array(X_categorical).reshape(1, -1)
    X = np.hstack([X_numeric_scaled, X_categorical])
    
    return X

def predict_with_models(sample, knn_model, lstm_model, scaler, label_encoders, target_encoder):
    """Make predictions with both KNN and LSTM."""
    X = preprocess_sample(sample, scaler, label_encoders)
    
    # KNN prediction with all class confidences
    knn_distances, knn_indices = knn_model.kneighbors(X)
    knn_pred = knn_model.predict(X)[0]
    
    # Calculate KNN confidence as inverse of distances
    knn_confidence = 1 / (np.mean(knn_distances) + 1e-6)
    knn_confidence = min(knn_confidence, 1.0)
    
    # KNN class probabilities
    knn_classes = knn_model.classes_
    knn_class_confidences = {}
    for i, cls in enumerate(knn_classes):
        mask = knn_model.predict(X) == cls
        knn_class_confidences[target_encoder.inverse_transform([cls])[0]] = float(1.0 if cls == knn_pred else 0.1)
    
    # LSTM prediction
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
    lstm_probs = lstm_model.predict(X_lstm, verbose=0)[0]
    lstm_pred = np.argmax(lstm_probs)
    lstm_confidence = float(np.max(lstm_probs))
    
    # LSTM class probabilities
    lstm_class_confidences = {target_encoder.classes_[i]: float(lstm_probs[i]) 
                             for i in range(len(target_encoder.classes_))}
    
    knn_pred_label = target_encoder.inverse_transform([knn_pred])[0]
    lstm_pred_label = target_encoder.inverse_transform([lstm_pred])[0]
    
    return {
        'knn_pred': knn_pred_label,
        'knn_confidence': knn_confidence,
        'knn_confidences': knn_class_confidences,
        'lstm_pred': lstm_pred_label,
        'lstm_confidence': lstm_confidence,
        'lstm_confidences': lstm_class_confidences,
        'agreement': knn_pred_label == lstm_pred_label
    }

def run_comprehensive_tests():
    """Run comprehensive tests for all zones and classes."""
    zones = ['Zone0', 'Zone1', 'Zone2']
    
    print("="*80)
    print("COMPREHENSIVE 5-CLASS TEST SUITE FOR ALL ZONES")
    print("="*80)
    
    all_results = []
    
    for zone_name in zones:
        print(f"\n{'='*80}")
        print(f"TESTING {zone_name}")
        print(f"{'='*80}")
        
        # Load models
        try:
            knn_model, lstm_model, scaler, label_encoders, target_encoder = load_models_for_zone(zone_name)
        except Exception as e:
            print(f"Error loading models for {zone_name}: {e}")
            continue
        
        # Load test cases
        training_data_path = Path(f'src/data/training_dataset/{zone_name}_training_data.json')
        if not training_data_path.exists():
            training_data_path = Path('src/data/training_dataset/master_balanced_training.json')
        
        try:
            test_cases = get_test_cases_from_training_data(training_data_path)
        except Exception as e:
            print(f"Error loading test cases: {e}")
            continue
        
        # Run predictions for each class
        zone_results = []
        knn_correct = 0
        lstm_correct = 0
        agreement_count = 0
        
        print(f"\nTest Cases for All 5 Classes:")
        print("-"*80)
        
        for class_label, test_sample in test_cases.items():
            try:
                predictions = predict_with_models(
                    test_sample, knn_model, lstm_model, scaler, label_encoders, target_encoder
                )
                
                # Check correctness
                knn_correct_pred = predictions['knn_pred'] == class_label
                lstm_correct_pred = predictions['lstm_pred'] == class_label
                
                knn_correct += int(knn_correct_pred)
                lstm_correct += int(lstm_correct_pred)
                agreement_count += int(predictions['agreement'])
                
                zone_results.append({
                    'zone': zone_name,
                    'true_class': class_label,
                    'knn_pred': predictions['knn_pred'],
                    'knn_conf': predictions['knn_confidence'],
                    'knn_correct': '✓' if knn_correct_pred else '✗',
                    'lstm_pred': predictions['lstm_pred'],
                    'lstm_conf': predictions['lstm_confidence'],
                    'lstm_correct': '✓' if lstm_correct_pred else '✗',
                    'agreement': '✓' if predictions['agreement'] else '✗'
                })
                
                print(f"\nClass: {class_label}")
                print(f"  KNN:  {predictions['knn_pred']:20s} (conf: {predictions['knn_confidence']:.3f}) {knn_correct_pred and '✓' or '✗'}")
                print(f"  LSTM: {predictions['lstm_pred']:20s} (conf: {predictions['lstm_confidence']:.3f}) {lstm_correct_pred and '✓' or '✗'}")
                print(f"  Agreement: {'✓' if predictions['agreement'] else '✗'}")
                
            except Exception as e:
                print(f"Error processing class {class_label}: {e}")
        
        # Zone summary
        n_classes = len(test_cases)
        print(f"\n{zone_name} Summary:")
        print(f"  KNN Accuracy: {knn_correct}/{n_classes} ({100*knn_correct/n_classes:.1f}%)")
        print(f"  LSTM Accuracy: {lstm_correct}/{n_classes} ({100*lstm_correct/n_classes:.1f}%)")
        print(f"  Model Agreement: {agreement_count}/{n_classes} ({100*agreement_count/n_classes:.1f}%)")
        
        all_results.extend(zone_results)
    
    # Print detailed results table
    print(f"\n{'='*80}")
    print("DETAILED RESULTS TABLE")
    print(f"{'='*80}\n")
    
    if all_results:
        headers = ['Zone', 'True Class', 'KNN Pred', 'KNN Conf', 'KNN', 'LSTM Pred', 'LSTM Conf', 'LSTM', 'Agree']
        table_data = [[r['zone'], r['true_class'], r['knn_pred'], f"{r['knn_conf']:.3f}", r['knn_correct'],
                       r['lstm_pred'], f"{r['lstm_conf']:.3f}", r['lstm_correct'], r['agreement']]
                      for r in all_results]
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    if all_results:
        total_tests = len(all_results)
        total_knn_correct = sum(1 for r in all_results if r['knn_correct'] == '✓')
        total_lstm_correct = sum(1 for r in all_results if r['lstm_correct'] == '✓')
        total_agreement = sum(1 for r in all_results if r['agreement'] == '✓')
        
        print(f"\nTotal Test Cases: {total_tests}")
        print(f"KNN Overall Accuracy: {total_knn_correct}/{total_tests} ({100*total_knn_correct/total_tests:.1f}%)")
        print(f"LSTM Overall Accuracy: {total_lstm_correct}/{total_tests} ({100*total_lstm_correct/total_tests:.1f}%)")
        print(f"Model Agreement: {total_agreement}/{total_tests} ({100*total_agreement/total_tests:.1f}%)")
        
        print("\n✓ All 5 classes tested successfully!" if total_tests == 15 else "\n⚠ Not all classes tested")

if __name__ == '__main__':
    run_comprehensive_tests()
