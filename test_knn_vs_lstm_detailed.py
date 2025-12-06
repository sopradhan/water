"""
Detailed KNN vs LSTM Comparison
================================
Shows detailed comparison of KNN and LSTM predictions with all confidence scores
"""

import json
import requests
from typing import Dict, List, Any

MODEL_API_URL = "http://localhost:8002"

# Test cases
TEST_CASES = {
    "Normal": {
        "pressure_psi": 65.0,
        "master_flow_lpm": 150.0,
        "temperature_c": 22.0,
        "vibration": 0.5,
        "acoustic_level": 18.0,
        "rpm": 300.0,
        "operation_hours": 1000.0,
        "ultrasonic_signal": 0.15,
        "pipe_age": 5.0,
        "soil_type": "clay",
        "material": "PVC"
    },
    "Leak": {
        "pressure_psi": 45.0,
        "master_flow_lpm": 80.0,
        "temperature_c": 28.0,
        "vibration": 5.5,
        "acoustic_level": 35.0,
        "rpm": 280.0,
        "operation_hours": 3000.0,
        "ultrasonic_signal": 0.85,
        "pipe_age": 15.0,
        "soil_type": "clay",
        "material": "cast_iron"
    },
    "Defect": {
        "pressure_psi": 55.0,
        "master_flow_lpm": 120.0,
        "temperature_c": 30.0,
        "vibration": 8.5,
        "acoustic_level": 45.0,
        "rpm": 320.0,
        "operation_hours": 5000.0,
        "ultrasonic_signal": 0.95,
        "pipe_age": 25.0,
        "soil_type": "silty",
        "material": "cast_iron"
    },
    "MaintenanceRequired": {
        "pressure_psi": 85.0,
        "master_flow_lpm": 180.0,
        "temperature_c": 28.0,
        "vibration": 3.5,
        "acoustic_level": 28.0,
        "rpm": 370.0,
        "operation_hours": 7000.0,
        "ultrasonic_signal": 0.55,
        "pipe_age": 18.0,
        "soil_type": "sandy",
        "material": "PVC"
    },
    "IllegalConnection": {
        "pressure_psi": 55.0,
        "master_flow_lpm": 220.0,
        "temperature_c": 20.0,
        "vibration": 2.2,
        "acoustic_level": 22.0,
        "rpm": 380.0,
        "operation_hours": 2500.0,
        "ultrasonic_signal": 0.38,
        "pipe_age": 10.0,
        "soil_type": "clay",
        "material": "HDPE"
    }
}


def check_api_health():
    """Check API health"""
    try:
        response = requests.get(f"{MODEL_API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction"""
    try:
        response = requests.post(
            f"{MODEL_API_URL}/predict",
            json=sensor_data,
            timeout=30
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None


def print_comparison(class_name: str, pred_result: Dict[str, Any]):
    """Print detailed KNN vs LSTM comparison"""
    if not pred_result or 'prediction' not in pred_result:
        print(f"[ERROR] No prediction result")
        return
    
    pred = pred_result['prediction']
    
    print(f"\n{'='*80}")
    print(f"EXPECTED CLASS: {class_name}")
    print(f"{'='*80}")
    
    # KNN Details
    print(f"\nKNN MODEL:")
    print(f"  Prediction: {pred['knn_prediction']}")
    print(f"  Confidence: {pred['knn_confidence']*100:.2f}%")
    if pred.get('knn_all_confidences'):
        print(f"  All Scores:")
        for class_name_score, score in sorted(pred['knn_all_confidences'].items(), 
                                               key=lambda x: x[1], reverse=True):
            bar = '█' * int(score * 20)
            print(f"    {class_name_score:25} {score*100:6.2f}% {bar}")
    
    # LSTM Details
    print(f"\nLSTM MODEL:")
    print(f"  Prediction: {pred['lstm_prediction']}")
    print(f"  Confidence: {pred['lstm_confidence']*100:.2f}%")
    if pred.get('lstm_all_confidences'):
        print(f"  All Scores:")
        for class_name_score, score in sorted(pred['lstm_all_confidences'].items(), 
                                               key=lambda x: x[1], reverse=True):
            bar = '█' * int(score * 20)
            print(f"    {class_name_score:25} {score*100:6.2f}% {bar}")
    
    # Ensemble Decision
    print(f"\nENSEMBLE DECISION:")
    print(f"  Agreement Status: {pred.get('agreement', 'N/A')}")
    print(f"  Final Prediction: {pred['ensemble_prediction']}")
    print(f"  Ensemble Confidence: {pred['ensemble_confidence']*100:.2f}%")
    print(f"  Anomaly Detected: {pred['anomaly_detected']}")
    
    # Correctness
    is_correct = pred['ensemble_prediction'] == class_name
    status = "[OK]" if is_correct else "[X]"
    print(f"\n  {status} Result: {'CORRECT' if is_correct else 'INCORRECT'}")


def main():
    print("\n" + "="*80)
    print("DETAILED KNN vs LSTM PREDICTION ANALYSIS")
    print("="*80)
    
    if not check_api_health():
        print("[ERROR] API not running")
        return
    
    print("\nTesting all 5 classes with detailed model comparison...\n")
    
    correct_count = 0
    total_count = len(TEST_CASES)
    
    for expected_class, sensor_data in TEST_CASES.items():
        result = predict(sensor_data)
        if result:
            print_comparison(expected_class, result)
            if result['prediction']['ensemble_prediction'] == expected_class:
                correct_count += 1
        else:
            print(f"[ERROR] Failed to get prediction for {expected_class}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {total_count}")
    print(f"Correct: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"Incorrect: {total_count - correct_count}/{total_count}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
