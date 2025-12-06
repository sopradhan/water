"""
Test Model API with Different Classes
======================================
Tests single predictions for all 5 anomaly classes with various test cases
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Any

# Configuration
MODEL_API_URL = "http://localhost:8002"
PROD_DATA_PATH = "src/data/prod_data/prod_zone0_master.json"

# Test cases for each class - different sensor characteristics
TEST_CASES = {
    "Normal": [
        {
            "name": "Normal Operation - Good Conditions",
            "description": "All parameters within normal range, stable operation",
            "data": {
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
            }
        },
        {
            "name": "Normal Operation - Moderate Load",
            "description": "Operating at moderate capacity with normal parameters",
            "data": {
                "pressure_psi": 75.0,
                "master_flow_lpm": 200.0,
                "temperature_c": 24.0,
                "vibration": 0.8,
                "acoustic_level": 20.0,
                "rpm": 350.0,
                "operation_hours": 2000.0,
                "ultrasonic_signal": 0.20,
                "pipe_age": 8.0,
                "soil_type": "sandy",
                "material": "HDPE"
            }
        }
    ],
    "Leak": [
        {
            "name": "Leak - High Vibration",
            "description": "Elevated vibration and acoustic signals indicating potential leak",
            "data": {
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
            }
        },
        {
            "name": "Leak - Low Pressure Drop",
            "description": "Pressure drop with erratic flow indicates leak",
            "data": {
                "pressure_psi": 38.0,
                "master_flow_lpm": 60.0,
                "temperature_c": 26.0,
                "vibration": 4.2,
                "acoustic_level": 32.0,
                "rpm": 250.0,
                "operation_hours": 4000.0,
                "ultrasonic_signal": 0.75,
                "pipe_age": 20.0,
                "soil_type": "sandy",
                "material": "asbestos_cement"
            }
        }
    ],
    "Defect": [
        {
            "name": "Defect - High Vibration Anomaly",
            "description": "Extreme vibration indicating mechanical defect or structural issue",
            "data": {
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
            }
        },
        {
            "name": "Defect - Multiple Anomalies",
            "description": "Combination of high pressure, vibration, and acoustic anomalies",
            "data": {
                "pressure_psi": 120.0,
                "master_flow_lpm": 250.0,
                "temperature_c": 35.0,
                "vibration": 9.2,
                "acoustic_level": 50.0,
                "rpm": 400.0,
                "operation_hours": 6000.0,
                "ultrasonic_signal": 1.0,
                "pipe_age": 30.0,
                "soil_type": "clay",
                "material": "ductile_iron"
            }
        }
    ],
    "MaintenanceRequired": [
        {
            "name": "Maintenance - Moderate Wear",
            "description": "Elevated parameters suggesting need for maintenance",
            "data": {
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
            }
        },
        {
            "name": "Maintenance - Aging System",
            "description": "Older system with moderate anomaly indicators",
            "data": {
                "pressure_psi": 68.0,
                "master_flow_lpm": 140.0,
                "temperature_c": 27.0,
                "vibration": 2.8,
                "acoustic_level": 25.0,
                "rpm": 310.0,
                "operation_hours": 8000.0,
                "ultrasonic_signal": 0.48,
                "pipe_age": 35.0,
                "soil_type": "silty",
                "material": "asbestos_cement"
            }
        }
    ],
    "IllegalConnection": [
        {
            "name": "Illegal Connection - Flow Anomaly",
            "description": "Unusual flow pattern consistent with illegal connection",
            "data": {
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
        },
        {
            "name": "Illegal Connection - Irregular Pattern",
            "description": "Flow inconsistency with low vibration pattern",
            "data": {
                "pressure_psi": 62.0,
                "master_flow_lpm": 280.0,
                "temperature_c": 21.0,
                "vibration": 1.5,
                "acoustic_level": 19.0,
                "rpm": 420.0,
                "operation_hours": 3000.0,
                "ultrasonic_signal": 0.32,
                "pipe_age": 12.0,
                "soil_type": "sandy",
                "material": "PVC"
            }
        }
    ]
}


def check_api_health():
    """Check if Model API is running"""
    print(f"[CHECK] Checking Model API health at {MODEL_API_URL}...\n")
    
    try:
        response = requests.get(f"{MODEL_API_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"[OK] API is running")
            print(f"     Status: {health['status']}")
            print(f"     KNN Model: {health['models']['knn']}")
            print(f"     LSTM Model: {health['models']['lstm']}\n")
            return True
        else:
            print(f"[ERROR] API returned status {response.status_code}\n")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to API at {MODEL_API_URL}")
        print("[INFO] Make sure to start the Model API first:")
        print("       python -m src.model.api\n")
        return False
    except Exception as e:
        print(f"[ERROR] {str(e)}\n")
        return False


def predict_single(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a single prediction"""
    try:
        response = requests.post(
            f"{MODEL_API_URL}/predict",
            json=sensor_data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[ERROR] API returned status {response.status_code}")
            print(f"[ERROR] Response: {response.text}")
            return None
    
    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timeout")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to API")
        return None
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None


def print_prediction_result(pred_result: Dict[str, Any], test_case_name: str):
    """Print formatted prediction result"""
    if not pred_result or 'prediction' not in pred_result:
        print(f"[ERROR] Invalid prediction result")
        return
    
    pred = pred_result['prediction']
    
    print(f"\n  Test Case: {test_case_name}")
    print(f"  " + "-" * 66)
    print(f"  KNN Prediction:        {pred['knn_prediction']:20} ({pred['knn_confidence']*100:6.2f}%)")
    print(f"  LSTM Prediction:       {pred['lstm_prediction']:20} ({pred['lstm_confidence']*100:6.2f}%)")
    print(f"  Model Agreement:       {pred.get('agreement', 'N/A'):20}")
    print(f"  Ensemble Prediction:   {pred['ensemble_prediction']:20} ({pred['ensemble_confidence']*100:6.2f}%)")
    print(f"  Anomaly Detected:      {str(pred['anomaly_detected']):20}")
    print(f"  Risk Level:            {str(pred.get('risk_level', 'N/A')):20}")
    print(f"  Execution Time:        {pred_result['execution_time_ms']:.2f}ms")


def run_all_class_tests():
    """Run tests for all classes"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODEL PREDICTION TESTING - ALL CLASSES")
    print("=" * 70 + "\n")
    
    # Check API health
    if not check_api_health():
        print("[ERROR] API is not running. Exiting.")
        return False
    
    results_summary = {}
    total_tests = 0
    passed_tests = 0
    
    # Test each class
    for class_name in sorted(TEST_CASES.keys()):
        test_cases = TEST_CASES[class_name]
        results_summary[class_name] = []
        
        print(f"\n{'='*70}")
        print(f"CLASS: {class_name}")
        print(f"{'='*70}")
        
        for test_case in test_cases:
            total_tests += 1
            test_name = test_case['name']
            description = test_case['description']
            sensor_data = test_case['data']
            
            print(f"\n  Description: {description}")
            
            # Make prediction
            prediction = predict_single(sensor_data)
            
            if prediction:
                passed_tests += 1
                print_prediction_result(prediction, test_name)
                
                # Check if prediction matches expected class
                pred_class = prediction['prediction']['ensemble_prediction']
                matched = pred_class == class_name
                results_summary[class_name].append({
                    'name': test_name,
                    'predicted': pred_class,
                    'expected': class_name,
                    'matched': matched
                })
            else:
                print(f"  [ERROR] Prediction failed")
                results_summary[class_name].append({
                    'name': test_name,
                    'predicted': 'ERROR',
                    'expected': class_name,
                    'matched': False
                })
            
            time.sleep(0.5)  # Small delay between requests
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal Tests Run: {total_tests}")
    print(f"Successful: {passed_tests}/{total_tests}")
    
    print(f"\n{'='*70}")
    print("RESULTS BY CLASS")
    print(f"{'='*70}\n")
    
    for class_name in sorted(results_summary.keys()):
        results = results_summary[class_name]
        print(f"\n{class_name}:")
        for result in results:
            status = "[OK]" if result['matched'] else "[X]"
            print(f"  {status} {result['name']}")
            print(f"      Expected: {result['expected']} | Got: {result['predicted']}")
    
    return True


if __name__ == "__main__":
    run_all_class_tests()
    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70 + "\n")
