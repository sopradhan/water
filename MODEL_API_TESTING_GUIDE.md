# Model API Testing Guide

## Quick Start

### Step 1: Start the Model API

```bash
# Terminal 1: Start the Model API
cd c:\Users\PRADHAN\OneDrive\Desktop\water
python -m src.model.api
```

You should see output like:
```
[INFO] Starting Model API server...
[INFO] Host: 0.0.0.0
[INFO] Port: 8002
[INFO] API Documentation: http://localhost:8002/docs
```

### Step 2: Run the Test Suite

```bash
# Terminal 2: Run all tests
cd c:\Users\PRADHAN\OneDrive\Desktop\water
python test_model_api.py
```

## Test Suite Overview

The test suite runs 7 comprehensive tests:

### Test 1: Health Check ✅
- **Endpoint:** `GET /health`
- **Purpose:** Verify API is running and models are loaded
- **Expected:** `{"status": "healthy", ...}`

### Test 2: Single Prediction - Normal Reading ✅
- **Endpoint:** `POST /predict`
- **Data:** Normal water sensor readings
- **Expected:** `anomaly_detected = False`, `risk_level = "low"`

### Test 3: Single Prediction - High Turbidity Anomaly ✅
- **Endpoint:** `POST /predict`
- **Data:** High turbidity (indicates suspended particles)
- **Expected:** `anomaly_detected = True`, `risk_level = "high"`

### Test 4: Single Prediction - Low Oxygen Anomaly ✅
- **Endpoint:** `POST /predict`
- **Data:** Low dissolved oxygen (indicates contamination)
- **Expected:** `anomaly_detected = True`, `risk_level = "high"`

### Test 5: Batch Prediction ✅
- **Endpoint:** `POST /predict/batch`
- **Data:** 4 samples (normal + 3 anomalies)
- **Expected:** Processes all and returns aggregated results

### Test 6: Error Handling - Missing Field ✅
- **Endpoint:** `POST /predict`
- **Data:** Missing required field
- **Expected:** HTTP 400 error with validation message

### Test 7: Performance ✅
- **Endpoint:** `POST /predict`
- **Purpose:** Measure prediction speed
- **Expected:** Average < 1000ms per prediction

## Running Individual Tests

You can also run the tests programmatically:

```python
from test_model_api import ModelAPITester

# Create tester
tester = ModelAPITester("http://localhost:8002")

# Run specific test
tester.test_health_check()
tester.test_single_prediction_normal()
tester.test_batch_prediction()

# Print summary
tester.print_summary()
```

## Example Output

```
================================================================================
  TEST 1: Health Check
================================================================================

Endpoint: GET http://localhost:8002/health
Status Code: 200

Response:
{
  "status": "healthy",
  "services": {
    "knn": "loaded",
    "lstm": "loaded",
    "scaler": "loaded",
    "encoders": "loaded"
  },
  "timestamp": "2025-12-06T10:30:45.123456"
}

✅ Health Check
   └─ Status: healthy

================================================================================
  TEST 2: Single Prediction - Normal Reading
================================================================================

Endpoint: POST http://localhost:8002/predict
Status Code: 200

Response:
{
  "prediction": {
    "ensemble_prediction": 0,
    "ensemble_confidence": 0.92,
    "anomaly_detected": false,
    "knn_prediction": 0,
    "lstm_prediction": 0,
    "execution_time_ms": 45.23
  },
  "analysis": {
    "risk_level": "low",
    "confidence_level": "high",
    "recommendations": [...]
  }
}

✅ Single Prediction (Normal)
   └─ Anomaly: False, Risk: low
```

## Response Structure

### Health Check Response
```json
{
  "status": "healthy",
  "services": {
    "knn": "loaded",
    "lstm": "loaded",
    "scaler": "loaded",
    "encoders": "loaded"
  },
  "timestamp": "2025-12-06T10:30:45.123456"
}
```

### Single Prediction Response
```json
{
  "prediction": {
    "ensemble_prediction": 0,      // 0=normal, 1=anomaly
    "ensemble_confidence": 0.95,    // Model confidence (0-1)
    "anomaly_detected": false,      // Boolean flag
    "knn_prediction": 0,            // KNN model output
    "knn_confidence": 0.88,         // KNN confidence
    "lstm_prediction": 0,           // LSTM model output
    "lstm_confidence": 0.92,        // LSTM confidence
    "execution_time_ms": 45.23      // Prediction time
  },
  "analysis": {
    "risk_level": "low",            // low/medium/high
    "confidence_level": "high",     // low/medium/high
    "knn_risk": "low",
    "lstm_risk": "low",
    "ensemble_risk": "low",
    "recommendations": [
      "Water quality is normal",
      "Continue regular monitoring"
    ]
  }
}
```

### Batch Prediction Response
```json
{
  "total_processed": 4,
  "anomalies_found": 2,
  "normal_count": 2,
  "total_time_ms": 182.45,
  "predictions": [
    {
      "ensemble_prediction": 0,
      "anomaly_detected": false,
      "risk_level": "low"
    },
    {
      "ensemble_prediction": 1,
      "anomaly_detected": true,
      "risk_level": "high"
    },
    ...
  ]
}
```

## API Endpoints Explained

### 1. Health Check
```bash
GET /health
```
**Purpose:** Verify API is running and all models are loaded
**Response:** Health status and model loading info

### 2. Single Prediction
```bash
POST /predict

# Request body:
{
  "pressure": 65.0,
  "temperature": 22.0,
  "ph_level": 7.2,
  "dissolved_oxygen": 8.5,
  "turbidity": 0.3,
  "flow_rate": 150.0,
  "location": "Plant A",
  "sensor_type": "Type1"
}
```
**Purpose:** Get anomaly prediction for a single sensor reading
**Returns:** Ensemble prediction with confidence scores

### 3. Batch Prediction
```bash
POST /predict/batch

# Request body:
{
  "samples": [
    {sensor reading 1},
    {sensor reading 2},
    ...
  ]
}
```
**Purpose:** Get predictions for multiple readings at once
**Returns:** Aggregated results for all samples

## Understanding the Models

### Ensemble Voting
The API uses an ensemble approach combining two models:

1. **KNN (K-Nearest Neighbors)**
   - Fast, memory-based model
   - Good for local pattern detection
   - Confidence reflects neighbor similarity

2. **LSTM (Long Short-Term Memory)**
   - Deep learning model
   - Captures temporal patterns
   - Better for sequential anomalies

**Ensemble Decision:**
- If both models agree → High confidence
- If models disagree → Medium confidence
- Final prediction = Majority vote

### Risk Level Classification

| Risk Level | Criteria | Action |
|-----------|----------|--------|
| **Low** | Normal readings, all parameters within bounds | Continue monitoring |
| **Medium** | Slight deviations, potential warning signs | Investigate further |
| **High** | Clear anomalies, critical parameters out of range | Immediate action needed |

## Troubleshooting

### Issue: Connection Refused
```
Error: Cannot connect to http://localhost:8002
```

**Solution:** Start the Model API
```bash
python -m src.model.api
```

### Issue: Models Not Found
```
Error: Model files not found
```

**Solution:** Verify model files exist
```bash
Test-Path "src/model/weights/knn_model.pkl"
Test-Path "src/model/weights/lstm_model.h5"
```

### Issue: Slow Predictions
If predictions take > 1000ms:
- Check CPU usage
- Verify no other processes running
- Restart the API

### Issue: Inconsistent Predictions
Different results for same input?
- Check model ensemble voting
- Verify scaler is loaded correctly
- Check for floating point precision issues

## Advanced Usage

### Using with Python Client

```python
from src.client import WaterAnomalyClient

# Initialize client
client = WaterAnomalyClient()

# Single prediction
result = client.predict_anomaly(
    pressure=65,
    temperature=22,
    ph_level=7.2,
    dissolved_oxygen=8.5,
    turbidity=0.3,
    flow_rate=150
)

print(f"Anomaly: {result.anomaly_detected}")
print(f"Confidence: {result.ensemble_confidence:.2%}")

# Batch prediction
samples = [
    {"pressure": 65, "temperature": 22, ...},
    {"pressure": 70, "temperature": 25, ...}
]

batch_results = client.predict_batch(samples)
print(f"Anomalies found: {batch_results['anomalies_found']}")
```

### Using with CURL

```bash
# Health check
curl http://localhost:8002/health

# Single prediction
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pressure": 65,
    "temperature": 22,
    "ph_level": 7.2,
    "dissolved_oxygen": 8.5,
    "turbidity": 0.3,
    "flow_rate": 150
  }'

# Batch prediction
curl -X POST http://localhost:8002/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"pressure": 65, ...},
      {"pressure": 70, ...}
    ]
  }'
```

### Using with httpie

```bash
# Single prediction
http POST localhost:8002/predict \
  pressure:=65 \
  temperature:=22 \
  ph_level:=7.2 \
  dissolved_oxygen:=8.5 \
  turbidity:=0.3 \
  flow_rate:=150
```

## Performance Benchmarks

Expected performance on standard hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| Health Check | 1-2ms | Just returns status |
| Single Prediction | 40-80ms | Both KNN + LSTM inference |
| Batch (10 samples) | 400-800ms | Roughly linear with batch size |
| Batch (100 samples) | 4-8s | Batch processing optimization |

## Next Steps

1. ✅ Start the Model API
2. ✅ Run the test suite
3. ✅ Verify all 7 tests pass
4. ✅ Try custom predictions
5. ✅ Integrate with RAG API for complete workflow

## See Also

- `MODEL_API_GUIDE.md` - Complete API documentation
- `src/model/api.py` - Model API source code
- `src/client.py` - Python client library
- `SETUP_GUIDE.md` - System setup instructions
