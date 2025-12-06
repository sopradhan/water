"""
Test Model API with Production Data
====================================
Extracts a single row from prod_zone0_master.json and makes predictions
"""

import json
import time
import requests
from pathlib import Path

# Configuration
PROD_DATA_PATH = "src/data/prod_data/prod_zone0_master.json"
MODEL_API_URL = "http://localhost:8002"

def load_production_data():
    """Load production data from JSON file"""
    print("[INFO] Loading production data...")
    
    path = Path(PROD_DATA_PATH)
    if not path.exists():
        print(f"[ERROR] File not found: {PROD_DATA_PATH}")
        return None
    
    with open(path) as f:
        data = json.load(f)
    
    print(f"[OK] Loaded {len(data) if isinstance(data, list) else 1} records")
    return data

def extract_sensor_reading(record):
    """Extract sensor reading from production record"""
    print("[INFO] Extracting sensor reading...")
    
    # Handle if data is a list or dict
    if isinstance(record, list):
        record = record[0] if record else {}
    
    # Helper function to extract value from nested dict or direct value
    def get_value(data, key, default=0):
        if key in data:
            val = data[key]
            if isinstance(val, dict) and 'value' in val:
                return float(val['value'])
            return float(val)
        return default
    
    # Map production data fields to API fields (matching training data exactly)
    # Production data has all these fields with 'value' in nested dict
    sensor_data = {
        "pressure_psi": get_value(record, "Pressure_PSI", 65),
        "master_flow_lpm": get_value(record, "Master_Flow_LPM", 150),
        "temperature_c": get_value(record, "Temperature_C", 22),
        "vibration": get_value(record, "Vibration", 1.75),
        "acoustic_level": get_value(record, "AcousticLevel", 20),
        "rpm": get_value(record, "RPM", 300),
        "operation_hours": get_value(record, "OperationHours", 1000),
        "ultrasonic_signal": get_value(record, "UltrasonicSignal", 0.27),
        "pipe_age": get_value(record, "PipeAge", 5),
        "soil_type": record.get("SoilType", "clay"),
        "material": record.get("Material", "cast_iron")
    }
    
    print("[OK] Sensor data extracted:")
    for key, value in sensor_data.items():
        if key not in ["soil_type", "material"]:
            print(f"    {key}: {value}")
    print(f"    soil_type: {sensor_data['soil_type']}")
    print(f"    material: {sensor_data['material']}")
    
    return sensor_data

def check_api_health():
    """Check if Model API is running"""
    print(f"\n[CHECK] Checking Model API health at {MODEL_API_URL}...")
    
    try:
        response = requests.get(f"{MODEL_API_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"[OK] API is running: {health}")
            return True
        else:
            print(f"[ERROR] API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to API at {MODEL_API_URL}")
        print("[INFO] Make sure to start the Model API first:")
        print("       python -m src.model.api")
        return False
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False

def predict_single(sensor_data):
    """Make a single prediction"""
    print(f"\n[PREDICT] Making single prediction...")
    
    try:
        response = requests.post(
            f"{MODEL_API_URL}/predict",
            json=sensor_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("[OK] Prediction received!")
            
            # Display prediction results
            prediction = result.get("prediction", {})
            print("\n" + "="*70)
            print("PREDICTION RESULTS")
            print("="*70)
            print(f"KNN Prediction:        {prediction.get('knn_prediction')}")
            print(f"KNN Confidence:        {prediction.get('knn_confidence'):.2%}")
            print(f"LSTM Prediction:       {prediction.get('lstm_prediction')}")
            print(f"LSTM Confidence:       {prediction.get('lstm_confidence'):.2%}")
            print(f"Ensemble Prediction:   {prediction.get('ensemble_prediction')}")
            print(f"Ensemble Confidence:   {prediction.get('ensemble_confidence'):.2%}")
            print(f"Anomaly Detected:      {prediction.get('anomaly_detected')}")
            print(f"Risk Level:            {prediction.get('risk_level')}")
            print("="*70)
            
            # Display analysis
            analysis = result.get("analysis", {})
            if analysis:
                print("\nANALYSIS:")
                print(f"  Risk Score:          {analysis.get('risk_score')}")
                print(f"  Recommendation:      {analysis.get('recommendation')}")
            
            print(f"\nExecution Time:        {result.get('execution_time_ms'):.2f}ms")
            print(f"Model Version:         {result.get('model_version')}")
            
            return result
        else:
            print(f"[ERROR] API returned status {response.status_code}")
            print(f"[ERROR] Response: {response.text}")
            return None
    
    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to API")
        return None
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None

def predict_batch(sensor_readings):
    """Make batch predictions"""
    print(f"\n[PREDICT] Making batch prediction for {len(sensor_readings)} readings...")
    
    try:
        response = requests.post(
            f"{MODEL_API_URL}/predict/batch",
            json={"samples": sensor_readings},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("[OK] Batch prediction received!")
            
            print("\n" + "="*70)
            print("BATCH PREDICTION RESULTS")
            print("="*70)
            print(f"Total Processed:       {result.get('total_processed')}")
            print(f"Anomalies Found:       {result.get('anomalies_found')}")
            print(f"Success Rate:          {result.get('success_rate'):.2%}")
            print("="*70)
            
            return result
        else:
            print(f"[ERROR] API returned status {response.status_code}")
            return None
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None

def main():
    """Main test function"""
    print("\n" + "="*70)
    print("MODEL API TESTING - PRODUCTION DATA")
    print("="*70)
    
    # Step 1: Check API health
    if not check_api_health():
        print("\n[ERROR] API is not running. Exiting...")
        return
    
    # Step 2: Load production data
    prod_data = load_production_data()
    if not prod_data:
        print("\n[ERROR] Failed to load production data. Exiting...")
        return
    
    # Step 3: Extract sensor reading
    sensor_data = extract_sensor_reading(prod_data)
    
    # Step 4: Make single prediction
    print("\n[TEST 1] SINGLE PREDICTION")
    print("-" * 70)
    result = predict_single(sensor_data)
    
    if result:
        time.sleep(1)
        
        # Step 5: Make batch prediction with multiple readings
        print("\n\n[TEST 2] BATCH PREDICTION")
        print("-" * 70)
        
        # Create 3 sample readings with correct features
        batch_data = [
            sensor_data,
            {
                "pressure_psi": 120,
                "temperature_c": 28,
                "master_flow_lpm": 250,
                "vibration": 2.5,
                "acoustic_level": 25,
                "rpm": 350,
                "operation_hours": 2000,
                "ultrasonic_signal": 0.35,
                "pipe_age": 8,
                "soil_type": "silty",
                "material": "PVC"
            },
            {
                "pressure_psi": 45,
                "temperature_c": 18,
                "master_flow_lpm": 80,
                "vibration": 1.2,
                "acoustic_level": 18,
                "rpm": 250,
                "operation_hours": 500,
                "ultrasonic_signal": 0.15,
                "pipe_age": 3,
                "soil_type": "clay",
                "material": "cast_iron"
            }
        ]
        
        batch_result = predict_batch(batch_data)
        
        print("\n" + "="*70)
        print("TESTING COMPLETE")
        print("="*70)
        print(f"[OK] Single Prediction: SUCCESS")
        print(f"[OK] Batch Prediction: {'SUCCESS' if batch_result else 'FAILED'}")
        print("="*70)
    else:
        print("\n[ERROR] Single prediction failed. Not running batch test.")

if __name__ == "__main__":
    main()
