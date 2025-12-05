# Water Network Anomaly Detection

A machine learning system for detecting anomalies in water distribution networks using hybrid approaches combining lazy learning (KNN) and temporal modeling (LSTM).

## Project Structure

```
water/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py          # Configuration management
│   ├── data_generator/
│   │   ├── __init__.py
│   │   └── advanced_data_generator.py  # Synthetic data generation
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model.py           # Model training script
│   │   └── hybrid_water_leakage_anomaly.py  # Prediction inference
│   └── __init__.py
├── data/                      # Data storage directory
├── models/                    # Trained models storage
├── logs/                      # Logs directory
├── tests/                     # Unit tests
├── .env                       # Environment configuration (create from .env.example)
├── .env.example               # Example environment variables
├── .gitignore                 # Git ignore rules
├── pyproject.toml             # Project metadata and configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd water
```

2. **Create a virtual environment** (recommended)
```bash
# Using venv
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual paths
# On Windows:
notepad .env
# On macOS/Linux:
nano .env
```

## Configuration

Environment variables are managed through the `.env` file. Key variables:

- `DATA_DIR` - Directory containing input water sensor data
- `DATA_OUTPUT_DIR` - Directory for generated synthetic data
- `MODEL_DIR` - Directory for saving trained models
- `KNN_NEIGHBORS` - Number of neighbors for KNN classifier
- `LSTM_EPOCHS` - Training epochs for LSTM
- `LSTM_BATCH_SIZE` - Batch size for LSTM training

See `.env.example` for all available options.

## Usage

### 1. Generate Synthetic Data

```bash
python -m src.data_generator.advanced_data_generator
```

This generates three synthetic water sensor datasets:
- `water_batch_01.json` (10,000 samples)
- `water_batch_02.json` (12,000 samples)
- `water_batch_03.json` (15,000 samples)

### 2. Train Models

```bash
python -m src.model.model
```

This trains:
- KNN classifier for instant anomaly detection
- LSTM for temporal pattern learning
- Saves preprocessors (scaler, encoders)

### 3. Make Predictions

```bash
python -m src.model.hybrid_water_leakage_anomaly
```

Or use it as a module:

```python
from src.model.hybrid_water_leakage_anomaly import hybrid_predict

sample = {
    "Pressure": 35,
    "FlowRate": 700,
    "Temperature": 28,
    "Vibration": 5,
    "RPM": 1500,
    "OperationHours": 20000,
    "AcousticLevel": 88,
    "UltrasonicSignal": 0.8,
    "PipeAge": 25,
    "SoilType": "Sandy",
    "Material": "PVC"
}

result = hybrid_predict(sample)
print(result)
```

## Features

### Data Generation
- **Correlated synthetic data** - Realistic relationships between pressure, flow, vibration, etc.
- **5 anomaly classes**:
  - Normal operation
  - Leak detection
  - Illegal connections
  - Pipe defects
  - Maintenance required
- **Seasonal variations** - Temperature changes throughout the year

### Model Architecture
- **KNN (Lazy Learning)**: Fast, interpretable anomaly detection
- **LSTM (RNN)**: Captures temporal patterns in sequential data
- **Hybrid Decision**: Combines both models for robust predictions

### Output
Predictions include:
- Final class (normal/leak/illegal/defect/maintenance)
- KNN confidence scores
- LSTM confidence scores
- Class-wise probabilities

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
```bash
# Format with black
black src/ tests/

# Check with flake8
flake8 src/ tests/
```

## Dependencies

Core:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning (KNN, preprocessing)
- `tensorflow` - Deep learning (LSTM)
- `joblib` - Model serialization
- `python-dotenv` - Environment management

Development:
- `pytest` - Testing framework
- `black` - Code formatter
- `flake8` - Style checker

## License

MIT License

## Authors

Water Analytics Team

## Contributing

1. Create a feature branch (`git checkout -b feature/AmazingFeature`)
2. Commit changes (`git commit -m 'Add AmazingFeature'`)
3. Push to branch (`git push origin feature/AmazingFeature`)
4. Open a Pull Request

## Support

For issues, questions, or suggestions, please open an issue in the repository.
