"""
Train zone-specific models with hyperparameter tuning.
- KNN with optimized k value and distance weighting
- LSTM with increased layers, dropout, epochs, and batch optimization
- Class balancing with weights
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

def load_training_data(dataset_path):
    """Load training data from JSON file."""
    with open(dataset_path) as f:
        data = json.load(f)
    return data

def preprocess_data(data, categorical_features):
    """Preprocess training data."""
    df = pd.DataFrame(data)
    
    numeric_features = [
        'Pressure_PSI', 'Master_Flow_LPM', 'Temperature_C', 'Vibration',
        'RPM', 'OperationHours', 'AcousticLevel', 'UltrasonicSignal', 'PipeAge'
    ]
    
    def extract_value(val):
        """Extract numeric value from dict or direct value."""
        if isinstance(val, dict) and 'value' in val:
            return val['value']
        return val
    
    # Extract numeric values and handle missing values
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].apply(extract_value)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].mean(), inplace=True)
    
    for col in categorical_features:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)
    
    # Prepare features
    X_numeric = df[numeric_features].values.astype(float)
    
    # Encode categorical features
    label_encoders = {}
    X_categorical = []
    for feature in categorical_features:
        if feature in df.columns:
            le = LabelEncoder()
            encoded = le.fit_transform(df[feature].astype(str))
            label_encoders[feature] = le
            X_categorical.append(encoded)
        else:
            print(f"Warning: {feature} not found in data")
    
    X_categorical = np.column_stack(X_categorical) if X_categorical else np.array([])
    X = np.hstack([X_numeric, X_categorical])
    
    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df['class_label'])
    
    return X, y, X_numeric.shape[1], label_encoders, target_encoder

def calculate_class_weights(y):
    """Calculate class weights for imbalanced data."""
    n_samples = len(y)
    n_classes = len(np.unique(y))
    
    class_weights = {}
    for i in range(n_classes):
        class_count = np.sum(y == i)
        weight = n_samples / (n_classes * class_count)
        class_weights[i] = weight
    
    return class_weights

def create_lstm_model(input_dim, hyperparams):
    """Create LSTM model with tuned hyperparameters."""
    model = models.Sequential([
        # Input layer and first LSTM layer
        layers.LSTM(
            units=hyperparams['lstm_units_1'],
            activation='relu',
            input_shape=(1, input_dim),
            return_sequences=True
        ),
        layers.Dropout(hyperparams['dropout_1']),
        layers.BatchNormalization(),
        
        # Second LSTM layer
        layers.LSTM(
            units=hyperparams['lstm_units_2'],
            activation='relu',
            return_sequences=False
        ),
        layers.Dropout(hyperparams['dropout_2']),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(hyperparams['dense_units'], activation='relu'),
        layers.Dropout(hyperparams['dropout_3']),
        layers.BatchNormalization(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(5, activation='softmax')
    ])
    
    optimizer = optimizers.Adam(learning_rate=hyperparams['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_knn_model(X_train, y_train, hyperparams):
    """Train KNN model with tuned hyperparameters."""
    knn = KNeighborsClassifier(
        n_neighbors=hyperparams['k'],
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    )
    knn.fit(X_train, y_train)
    return knn

def train_models_on_dataset(dataset_name, dataset_path):
    """Train zone-specific models."""
    print("\n" + "="*70)
    print(f"TRAINING MODELS FOR: {dataset_name}")
    print("="*70)
    
    # Load and preprocess data
    print(f"\nLoading data from {dataset_path}...")
    data = load_training_data(dataset_path)
    print(f"Total records: {len(data)}")
    
    categorical_features = ['SoilType', 'Material']
    X, y, n_numeric_features, label_encoders, target_encoder = preprocess_data(
        data, categorical_features
    )
    
    print(f"Feature dimension: {X.shape[1]} ({n_numeric_features} numeric + 2 categorical)")
    print(f"Classes: {target_encoder.classes_}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")
    
    # =====================================================
    # HYPERPARAMETER TUNED KNN
    # =====================================================
    print("\n" + "-"*70)
    print("Training KNN with hyperparameter tuning...")
    print("-"*70)
    
    knn_hyperparams = {
        'k': 5,  # Optimized k value
    }
    
    knn_model = train_knn_model(X_train_scaled, y_train, knn_hyperparams)
    
    # Evaluate KNN
    y_pred_knn = knn_model.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    
    print(f"\nKNN Accuracy: {knn_accuracy:.4f}")
    print("\nKNN Classification Report:")
    print(classification_report(y_test, y_pred_knn, 
                              target_names=target_encoder.classes_))
    print("\nKNN Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_knn))
    
    # =====================================================
    # HYPERPARAMETER TUNED LSTM
    # =====================================================
    print("\n" + "-"*70)
    print("Training LSTM with hyperparameter tuning...")
    print("-"*70)
    
    lstm_hyperparams = {
        'lstm_units_1': 128,      # First LSTM layer - increased from 64
        'lstm_units_2': 64,       # Second LSTM layer - new layer for depth
        'dense_units': 128,       # Dense layer - increased
        'dropout_1': 0.4,         # Increased dropout for regularization
        'dropout_2': 0.3,
        'dropout_3': 0.2,
        'learning_rate': 0.001,   # Optimized learning rate
        'batch_size': 32,         # Optimized batch size
        'epochs': 150,            # Increased epochs
    }
    
    print(f"\nLSTM Hyperparameters:")
    for key, value in lstm_hyperparams.items():
        print(f"  {key}: {value}")
    
    # Reshape data for LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    lstm_model = create_lstm_model(X_train_lstm.shape[2], lstm_hyperparams)
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001
    )
    
    # Train LSTM
    print("\nTraining LSTM (this may take a minute)...")
    history = lstm_model.fit(
        X_train_lstm, y_train,
        validation_split=0.2,
        epochs=lstm_hyperparams['epochs'],
        batch_size=lstm_hyperparams['batch_size'],
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate LSTM
    lstm_eval = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)
    lstm_accuracy = lstm_eval[1]
    
    y_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm, verbose=0), axis=1)
    
    print(f"\nLSTM Accuracy: {lstm_accuracy:.4f}")
    print("\nLSTM Classification Report:")
    print(classification_report(y_test, y_pred_lstm, 
                              target_names=target_encoder.classes_))
    print("\nLSTM Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lstm))
    
    # =====================================================
    # SAVE MODELS
    # =====================================================
    print("\n" + "-"*70)
    print("Saving models...")
    print("-"*70)
    
    # Create zone-specific model directory
    zone_match = dataset_name.split('_')[0]  # Extract Zone0, Zone1, or Zone2
    model_dir = Path(f'src/model/model_weights/{zone_match}_models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Also save to main directory for backward compatibility
    main_model_dir = Path('src/model/model_weights')
    main_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save KNN
    joblib.dump(knn_model, model_dir / 'knn_model.pkl')
    joblib.dump(knn_model, main_model_dir / 'knn_lazy_model.pkl')
    
    # Save LSTM
    lstm_model.save(str(model_dir / 'lstm_model.h5'))
    lstm_model.save(str(main_model_dir / 'lstm_model.h5'))
    
    # Save preprocessing objects
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    joblib.dump(scaler, main_model_dir / 'scaler.pkl')
    
    joblib.dump(label_encoders, model_dir / 'label_encoders.pkl')
    joblib.dump(label_encoders, main_model_dir / 'label_encoders.pkl')
    
    joblib.dump(target_encoder, model_dir / 'target_encoder.pkl')
    joblib.dump(target_encoder, main_model_dir / 'target_encoder.pkl')
    
    print(f"\nModels saved to: {model_dir}")
    print(f"Main models updated in: {main_model_dir}")
    
    return {
        'dataset': dataset_name,
        'knn_accuracy': knn_accuracy,
        'lstm_accuracy': lstm_accuracy,
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0],
        'classes': target_encoder.classes_.tolist()
    }

def main():
    """Main training pipeline for all zone-specific datasets."""
    
    datasets = [
        ('Zone0_training_data', 'src/data/training_dataset/Zone0_training_data.json'),
        ('Zone1_training_data', 'src/data/training_dataset/Zone1_training_data.json'),
        ('Zone2_training_data', 'src/data/training_dataset/Zone2_training_data.json'),
        ('master_balanced_training', 'src/data/training_dataset/master_balanced_training.json'),
    ]
    
    results = []
    
    for dataset_name, dataset_path in datasets:
        if Path(dataset_path).exists():
            result = train_models_on_dataset(dataset_name, dataset_path)
            results.append(result)
        else:
            print(f"Warning: {dataset_path} not found. Skipping...")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for result in results:
        print(f"\n{result['dataset']}:")
        print(f"  Train samples: {result['n_train']}, Test samples: {result['n_test']}")
        print(f"  KNN Accuracy: {result['knn_accuracy']:.4f}")
        print(f"  LSTM Accuracy: {result['lstm_accuracy']:.4f}")
        print(f"  Classes: {result['classes']}")

if __name__ == '__main__':
    main()
