"""
Test script for updated zone-specific API
"""
import requests
import json
import time

BASE_URL = "http://localhost:8002"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_zone0():
    """Test prediction for Zone0"""
    print("\n" + "="*70)
    print("TEST 2: Predict for Zone0 - Normal Operation")
    print("="*70)
    
    payload = {
        "ZoneName": "Zone0",
        "Pressure_PSI": 117.29,
        "Master_Flow_LPM": 3946.92,
        "Temperature_C": 32.96,
        "Vibration": 1.39,
        "RPM": 292.63,
        "OperationHours": 14663.72,
        "AcousticLevel": 19.19,
        "UltrasonicSignal": 0.4,
        "PipeAge": 10,
        "SoilType": "Rocky",
        "Material": "PVC"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  KNN: {result['prediction']['knn_prediction']} ({result['prediction']['knn_confidence']:.3f})")
        print(f"  LSTM: {result['prediction']['lstm_prediction']} ({result['prediction']['lstm_confidence']:.3f})")
        print(f"  Ensemble: {result['prediction']['ensemble_prediction']} ({result['prediction']['ensemble_confidence']:.3f})")
        print(f"  Anomaly Detected: {result['prediction']['anomaly_detected']}")
        print(f"  Zone Used: {result['analysis']['zone']}")
        print(f"  Model Version: {result['model_version']}")
        print(f"  Execution Time: {result['execution_time_ms']:.2f}ms")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_zone1():
    """Test prediction for Zone1 - Leak case"""
    print("\n" + "="*70)
    print("TEST 3: Predict for Zone1 - Leak Detection")
    print("="*70)
    
    payload = {
        "ZoneName": "Zone1",
        "Pressure_PSI": 82.22,  # Low pressure (Leak signature)
        "Master_Flow_LPM": 3949.0,
        "Temperature_C": 30.0,
        "Vibration": 3.90,  # Medium vibration (Leak signature)
        "RPM": 290.0,
        "OperationHours": 12000.0,
        "AcousticLevel": 21.76,
        "UltrasonicSignal": 0.35,
        "PipeAge": 15,
        "SoilType": "Clay",
        "Material": "DI"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  KNN: {result['prediction']['knn_prediction']} ({result['prediction']['knn_confidence']:.3f})")
        print(f"  LSTM: {result['prediction']['lstm_prediction']} ({result['prediction']['lstm_confidence']:.3f})")
        print(f"  Ensemble: {result['prediction']['ensemble_prediction']} ({result['prediction']['ensemble_confidence']:.3f})")
        print(f"  Anomaly Detected: {result['prediction']['anomaly_detected']}")
        print(f"  Risk Level: {result['analysis']['risk_level']}")
        print(f"  Zone Used: {result['analysis']['zone']}")
        print(f"  Model Version: {result['model_version']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_zone2():
    """Test prediction for Zone2 - MaintenanceRequired"""
    print("\n" + "="*70)
    print("TEST 4: Predict for Zone2 - Maintenance Detection")
    print("="*70)
    
    payload = {
        "ZoneName": "Zone2",
        "Pressure_PSI": 107.25,
        "Master_Flow_LPM": 3200.0,
        "Temperature_C": 31.0,
        "Vibration": 9.42,  # Very high vibration (Maintenance signature)
        "RPM": 295.0,
        "OperationHours": 32000.0,
        "AcousticLevel": 25.0,
        "UltrasonicSignal": 0.45,
        "PipeAge": 42,  # Very old pipe (Maintenance signature)
        "SoilType": "Sandy",
        "Material": "HDPE"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  KNN: {result['prediction']['knn_prediction']} ({result['prediction']['knn_confidence']:.3f})")
        print(f"  LSTM: {result['prediction']['lstm_prediction']} ({result['prediction']['lstm_confidence']:.3f})")
        print(f"  Ensemble: {result['prediction']['ensemble_prediction']} ({result['prediction']['ensemble_confidence']:.3f})")
        print(f"  Anomaly Detected: {result['prediction']['anomaly_detected']}")
        print(f"  Risk Level: {result['analysis']['risk_level']}")
        print(f"  Zone Used: {result['analysis']['zone']}")
        print(f"  Model Version: {result['model_version']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("ZONE-SPECIFIC API TESTING")
    print("="*70)
    print(f"API URL: {BASE_URL}")
    print("Waiting 5 seconds for API to be ready...")
    time.sleep(5)
    
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Zone0 Prediction", test_predict_zone0()))
    results.append(("Zone1 Prediction", test_predict_zone1()))
    results.append(("Zone2 Prediction", test_predict_zone2()))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_pass = sum(1 for _, p in results if p)
    total_tests = len(results)
    print(f"\nTotal: {total_pass}/{total_tests} tests passed")

if __name__ == '__main__':
    main()
