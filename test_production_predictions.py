"""
Test Model with Actual Production Data
=======================================
Uses real production data from prod_zone0_master.json to test model predictions
This helps determine if bias is in test cases or in the model itself
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Any

MODEL_API_URL = "http://localhost:8002"
PROD_DATA_PATH = "src/data/prod_data/prod_zone0_master.json"


def load_production_data() -> List[Dict[str, Any]]:
    """Load production data"""
    with open(PROD_DATA_PATH) as f:
        return json.load(f)


def extract_sensor_data(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract sensor data from production record"""
    def get_value(data, key, default=0):
        if key in data:
            val = data[key]
            if isinstance(val, dict) and 'value' in val:
                return float(val['value'])
            return float(val)
        return default
    
    return {
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


def predict_single(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make single prediction"""
    try:
        response = requests.post(
            f"{MODEL_API_URL}/predict",
            json=sensor_data,
            timeout=30
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None


def check_api_health():
    """Check API health"""
    try:
        response = requests.get(f"{MODEL_API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    print("\n" + "=" * 70)
    print("PRODUCTION DATA PREDICTION TEST")
    print("=" * 70 + "\n")
    
    if not check_api_health():
        print("[ERROR] API not running")
        return
    
    # Load production data
    prod_data = load_production_data()
    print(f"[OK] Loaded {len(prod_data)} production records\n")
    
    # Sample every 10th record for testing
    sample_indices = list(range(0, len(prod_data), 10))
    sample_records = [prod_data[i] for i in sample_indices]
    
    print(f"Testing {len(sample_records)} sampled records (every 10th record)\n")
    
    predictions_by_class = {}
    
    for idx, record in enumerate(sample_records):
        zone = record.get('ZoneName', 'Unknown')
        sensor_data = extract_sensor_data(record)
        
        prediction = predict_single(sensor_data)
        
        if prediction and 'prediction' in prediction:
            pred = prediction['prediction']
            predicted_class = pred['ensemble_prediction']
            
            if predicted_class not in predictions_by_class:
                predictions_by_class[predicted_class] = 0
            predictions_by_class[predicted_class] += 1
            
            if idx % 5 == 0:  # Print every 5th prediction
                print(f"[{idx+1:2}] {zone:10} => {predicted_class:20} (KNN: {pred['knn_prediction']:20}) "
                      f"(Confidence: {pred['ensemble_confidence']*100:6.2f}%)")
        
        time.sleep(0.2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY - ACTUAL PRODUCTION DATA")
    print("=" * 70 + "\n")
    
    total = len(sample_records)
    print(f"Total predictions: {total}\n")
    print("Distribution:")
    for class_name in sorted(predictions_by_class.keys()):
        count = predictions_by_class[class_name]
        percentage = (count / total) * 100
        print(f"  {class_name:25} {count:3} ({percentage:6.2f}%)")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    if len(predictions_by_class) == 1:
        print(f"\nWARNING: Model predicts ONLY '{list(predictions_by_class.keys())[0]}' class!")
        print("This indicates severe model bias or feature distribution mismatch.")
    else:
        print(f"\nModel predicts {len(predictions_by_class)} different classes")
        dominant_class = max(predictions_by_class, key=predictions_by_class.get)
        dominant_pct = (predictions_by_class[dominant_class] / total) * 100
        if dominant_pct > 80:
            print(f"WARNING: Model heavily biased towards '{dominant_class}' ({dominant_pct:.1f}%)")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
