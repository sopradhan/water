"""
Test script to predict anomaly from a single row in prod_zone0_master.json
"""

import json
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests
import pandas as pd
from typing import Dict, Any

def load_single_row_from_json(json_file: str, row_index: int = 0) -> Dict[str, Any]:
    """Load a single row from the production data JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if row_index >= len(data):
        raise ValueError(f"Row index {row_index} out of range. File has {len(data)} rows")
    
    return data[row_index]

def extract_numeric_features(row: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric features from a production data row"""
    features = {}
    
    # Define numeric fields from the JSON structure
    numeric_fields = {
        "Master_Elevation_m": "Master_Elevation_m",
        "Zone_Elevation_m": "Zone_Elevation_m",
        "Branch_Diameter_mm": "Branch_Diameter_mm",
        "Branch_Length_m": "Branch_Length_m",
        "Pump_Setpoint_PSI": "Pump_Setpoint_PSI",
        "Master_Flow_LPM": "Master_Flow_LPM",
        "Zone_Demand_LPM": "Zone_Demand_LPM",
        "Actual_Flow_LPM": "Actual_Flow_LPM",
        "Pressure_PSI": "Pressure_PSI",
        "Friction_Loss_PSI": "Friction_Loss_PSI",
        "Temperature_C": "Temperature_C",
        "Vibration": "Vibration",
        "RPM": "RPM",
        "OperationHours": "OperationHours",
        "AcousticLevel": "AcousticLevel",
        "UltrasonicSignal": "UltrasonicSignal",
        "PipeAge": "PipeAge",
        "Hour": "Hour",
        "Month": "Month",
    }
    
    for feature_name, json_key in numeric_fields.items():
        if json_key in row:
            value = row[json_key]
            # Handle nested dict structure with 'value' key
            if isinstance(value, dict) and 'value' in value:
                features[feature_name] = float(value['value'])
            else:
                features[feature_name] = float(value)
    
    # Add categorical features
    features['ZoneType'] = row.get('ZoneType', 'distribution_hub')
    features['SoilType'] = row.get('SoilType', 'Sandy')
    features['Material'] = row.get('Material', 'HDPE')
    
    return features

def format_prediction_payload(features: Dict[str, float]) -> Dict[str, Any]:
    """Format features into Model API payload"""
    return {
        "pressure": features.get("Pressure_PSI", 119.87),
        "temperature": features.get("Temperature_C", 19.01),
        "ph_level": 7.0,  # Default if not in data
        "dissolved_oxygen": 8.0,  # Default if not in data
        "turbidity": features.get("AcousticLevel", 20.9) / 100.0,  # Normalize
        "flow_rate": features.get("Actual_Flow_LPM", 484.98),
        "location": features.get("ZoneType", "distribution_hub"),
        "sensor_type": "water_sensor"
    }

def test_single_row_prediction(
    json_file: str = "src/data/prod_data/prod_zone0_master.json",
    row_index: int = 0,
    api_url: str = "http://localhost:8002",
    verbose: bool = True
) -> Dict[str, Any]:
    """Test prediction on a single row from production data"""
    
    print("\n" + "="*80)
    print("[TEST] Single Row Prediction from Production Data")
    print("="*80)
    
    # Step 1: Load row
    print(f"\n[STEP 1] Loading row {row_index} from {json_file}")
    try:
        row = load_single_row_from_json(json_file, row_index)
        print(f"[OK] Row loaded successfully")
        
        if verbose:
            print(f"\nRow Summary:")
            print(f"  Zone: {row.get('ZoneName')}")
            print(f"  Type: {row.get('ZoneType')}")
            print(f"  Timestamp: {row.get('Timestamp')}")
    except Exception as e:
        print(f"[ERROR] Failed to load row: {e}")
        return {"success": False, "error": str(e)}
    
    # Step 2: Extract features
    print(f"\n[STEP 2] Extracting features from row")
    try:
        features = extract_numeric_features(row)
        print(f"[OK] Extracted {len(features)} features")
        
        if verbose:
            print(f"\nExtracted Features:")
            numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
            for feature, value in list(numeric_features.items())[:10]:
                print(f"  {feature}: {value}")
    except Exception as e:
        print(f"[ERROR] Failed to extract features: {e}")
        return {"success": False, "error": str(e)}
    
    # Step 3: Format payload
    print(f"\n[STEP 3] Formatting prediction payload")
    try:
        payload = format_prediction_payload(features)
        print(f"[OK] Payload formatted")
        
        if verbose:
            print(f"\nPrediction Payload:")
            for key, value in payload.items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"[ERROR] Failed to format payload: {e}")
        return {"success": False, "error": str(e)}
    
    # Step 4: Make prediction
    print(f"\n[STEP 4] Making prediction via Model API ({api_url}/predict)")
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        print(f"[OK] Prediction successful (HTTP {response.status_code})")
        
        if verbose:
            print(f"\n[RESULTS] Prediction Output:")
            print(f"  Ensemble Prediction: {result['prediction']['ensemble_prediction']}")
            print(f"  Confidence: {result['prediction']['ensemble_confidence']:.2%}")
            print(f"  Anomaly Detected: {result['prediction']['anomaly_detected']}")
            print(f"  Risk Level: {result['analysis']['risk_level']}")
            print(f"  Execution Time: {result['execution_time_ms']:.2f}ms")
            
            print(f"\n[MODEL DETAILS]:")
            print(f"  KNN Prediction: {result['prediction']['knn_prediction']}")
            print(f"  LSTM Prediction: {result['prediction']['lstm_prediction']}")
            
            print(f"\n[ANALYSIS]:")
            print(f"  Summary: {result['analysis']['summary']}")
        
        return {
            "success": True,
            "row_index": row_index,
            "original_row": row,
            "features": features,
            "payload": payload,
            "prediction": result
        }
    
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to Model API at {api_url}")
        print(f"[HINT] Make sure to start the Model API:")
        print(f"       python -m src.model.api")
        return {"success": False, "error": "Connection refused"}
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return {"success": False, "error": str(e)}

def test_multiple_rows(
    json_file: str = "src/data/prod_data/prod_zone0_master.json",
    num_rows: int = 5,
    api_url: str = "http://localhost:8002",
    verbose: bool = False
) -> list:
    """Test predictions on multiple rows"""
    
    print("\n" + "="*80)
    print(f"[TEST] Multiple Row Predictions ({num_rows} rows)")
    print("="*80)
    
    results = []
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    num_rows = min(num_rows, len(data))
    
    for i in range(num_rows):
        print(f"\n{'='*80}")
        print(f"[ROW {i+1}/{num_rows}]")
        result = test_single_row_prediction(json_file, i, api_url, verbose=False)
        results.append(result)
        
        if result['success']:
            pred = result['prediction']['prediction']
            analysis = result['prediction']['analysis']
            print(f"  Prediction: {pred['ensemble_prediction']}")
            print(f"  Confidence: {pred['ensemble_confidence']:.2%}")
            print(f"  Anomaly: {pred['anomaly_detected']}")
            print(f"  Risk: {analysis['risk_level']}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"[SUMMARY] Processed {len([r for r in results if r['success']])} rows successfully")
    
    successful = [r for r in results if r['success']]
    if successful:
        anomalies = sum(1 for r in successful if r['prediction']['prediction']['anomaly_detected'])
        print(f"[ANOMALIES] {anomalies}/{len(successful)} rows detected anomalies")
    
    return results

if __name__ == "__main__":
    # Test single row
    result = test_single_row_prediction(row_index=0, verbose=True)
    
    if result['success']:
        print(f"\n{'='*80}")
        print("[SUCCESS] Single row prediction test passed!")
        print("="*80)
    else:
        print(f"\n[FAILED] Test failed: {result.get('error')}")
        sys.exit(1)
