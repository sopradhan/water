"""
Test with Data-Driven Test Cases
=================================
Test cases designed based on actual training data distributions
"""

import requests
import time
from typing import Dict, Any

MODEL_API_URL = "http://localhost:8002"

# Test cases based on actual feature distributions from training data
TEST_CASES = {
    "Normal": [
        {
            "name": "Normal - Typical Operation",
            "data": {
                "pressure_psi": 117.53,      # Normal mean
                "master_flow_lpm": 4053.61,
                "temperature_c": 29.39,
                "vibration": 1.88,           # Low vibration (Normal mean)
                "acoustic_level": 21.70,     # Low acoustic
                "rpm": 295.99,
                "operation_hours": 25487.55,
                "ultrasonic_signal": 0.39,
                "pipe_age": 21.94,           # Mid-age pipes (Normal mean)
                "soil_type": "clay",
                "material": "PVC"
            }
        }
    ],
    "Leak": [
        {
            "name": "Leak - Low Pressure High Vibration",
            "data": {
                "pressure_psi": 82.22,       # Low pressure (Leak mean)
                "master_flow_lpm": 4031.65,
                "temperature_c": 29.46,
                "vibration": 3.90,           # Medium vibration (Leak mean)
                "acoustic_level": 21.76,
                "rpm": 295.98,
                "operation_hours": 25575.13,
                "ultrasonic_signal": 0.39,
                "pipe_age": 22.13,
                "soil_type": "clay",
                "material": "PVC"
            }
        }
    ],
    "Defect": [
        {
            "name": "Defect - High Pressure High Vibration",
            "data": {
                "pressure_psi": 127.25,      # Very high pressure (Defect mean)
                "master_flow_lpm": 4085.70,
                "temperature_c": 29.28,
                "vibration": 7.92,           # Very high vibration (Defect mean)
                "acoustic_level": 36.90,     # High acoustic (Defect mean)
                "rpm": 295.47,
                "operation_hours": 25835.69,
                "ultrasonic_signal": 0.40,
                "pipe_age": 22.11,
                "soil_type": "clay",
                "material": "cast_iron"
            }
        }
    ],
    "MaintenanceRequired": [
        {
            "name": "MaintenanceRequired - Old Pipes High Vibration",
            "data": {
                "pressure_psi": 107.25,      # Medium pressure
                "master_flow_lpm": 4095.65,
                "temperature_c": 29.35,
                "vibration": 9.42,           # Very high vibration (MaintenanceRequired mean)
                "acoustic_level": 21.95,
                "rpm": 295.47,
                "operation_hours": 25227.31,
                "ultrasonic_signal": 0.40,
                "pipe_age": 42.44,           # Very old pipes (MaintenanceRequired mean)
                "soil_type": "silty",
                "material": "asbestos_cement"
            }
        }
    ],
    "IllegalConnection": [
        {
            "name": "IllegalConnection - High Pressure Low Vibration",
            "data": {
                "pressure_psi": 122.60,      # High pressure (IllegalConnection mean)
                "master_flow_lpm": 4047.19,
                "temperature_c": 29.38,
                "vibration": 1.88,           # Very low vibration (IllegalConnection mean)
                "acoustic_level": 21.73,     # Very low acoustic (IllegalConnection mean)
                "rpm": 296.04,
                "operation_hours": 25724.02,
                "ultrasonic_signal": 0.39,
                "pipe_age": 21.96,
                "soil_type": "clay",
                "material": "PVC"
            }
        }
    ]
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


def print_prediction(class_name: str, pred_result: Dict[str, Any]):
    """Print prediction result"""
    if not pred_result or 'prediction' not in pred_result:
        print(f"[ERROR] No prediction")
        return False
    
    pred = pred_result['prediction']
    is_correct = pred['ensemble_prediction'] == class_name
    status = "[OK]" if is_correct else "[X]"
    
    print(f"\n{status} Expected: {class_name:25} | Got: {pred['ensemble_prediction']:25}")
    print(f"   KNN: {pred['knn_prediction']:15} ({pred['knn_confidence']*100:6.2f}%)")
    print(f"   LSTM: {pred['lstm_prediction']:15} ({pred['lstm_confidence']*100:6.2f}%)")
    print(f"   Agreement: {pred.get('agreement', 'N/A')}")
    print(f"   Confidence: {pred['ensemble_confidence']*100:6.2f}%")
    
    return is_correct


def main():
    print("\n" + "="*80)
    print("DATA-DRIVEN TEST CASES (Based on Actual Training Data Distributions)")
    print("="*80)
    
    if not check_api_health():
        print("[ERROR] API not running")
        return
    
    correct = 0
    total = 0
    
    for class_name in sorted(TEST_CASES.keys()):
        print(f"\n{'='*80}")
        print(f"CLASS: {class_name}")
        print(f"{'='*80}")
        
        for test_case in TEST_CASES[class_name]:
            total += 1
            print(f"\n{test_case['name']}")
            
            result = predict(test_case['data'])
            if result and print_prediction(class_name, result):
                correct += 1
            
            time.sleep(0.3)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
