# Water Anomaly Detection - Advanced Dashboard Documentation

## Overview

The Advanced Dashboard is a comprehensive Streamlit application for exploring water sensor data, performing feature engineering, statistical analysis, and model retraining. It provides advanced visualizations including 3D scatter plots, PCA analysis, correlation heatmaps, and bivariate analysis.

---

## Features

### 1. **Overview Page**
- **Total Records**: Shows dataset size
- **Feature Count**: Total numeric features
- **Data Quality Metrics**: Missing value percentage
- **Feature Summary**: Mean, median, std deviation, min/max for all features
- **Missing Values Report**: Identifies incomplete features
- **Data Preview**: Sample of raw data

### 2. **Exploratory Analysis Page**
Comprehensive EDA with multiple tabs:

#### Tab 1: Distributions
- Histograms for all numeric features
- Understanding data distribution shapes
- Identifying multi-modal distributions
- Finding rare values or gaps

#### Tab 2: Box Plots
- Outlier identification using box plots
- Displaying quartiles and IQR
- Highlighting potential anomalies
- Statistical summary visualization

#### Tab 3: Bivariate Analysis
- 2x2 grid of scatter plots
- Top 4 features by variance
- Feature relationships visualization
- Interactive hover information

#### Tab 4: Scatter Matrix
- Complete scatter matrix of top 5 features
- Pairwise relationships
- Lower triangle visualization for clarity
- 1000-sample visualization for performance

### 3. **Feature Engineering Page**
Advanced feature analysis and transformation:

#### 3D Feature Space Visualization
- Custom feature selection (exactly 3 features)
- Interactive 3D scatter plot with rotation
- Color-coded by feature values
- Normalized coordinates for better comparison
- Sample size: up to 5000 points

#### Principal Component Analysis (PCA)
- 3D PCA projection of all features
- Explained variance ratios for each component
- Feature loadings in PC1
- Dimensionality reduction visualization

#### Feature Statistics
- Top 10 features by variance
- Feature skewness analysis
- Bar chart visualizations
- Engineering insights

### 4. **Statistical Analysis Page**
Deep statistical investigations:

#### Correlation Analysis
- Full correlation matrix heatmap
- Color-coded by correlation strength (-1 to 1)
- Interactive hover for exact values
- Text annotations for readability
- Strong correlation pairs (>0.7) highlighted

#### Outlier Detection
- IQR-based outlier detection
- Total outlier count and percentage
- Per-feature outlier distribution
- Bar chart visualization

#### Distribution Type Analysis
- Skewness calculation
- Kurtosis analysis
- Distribution type classification (Normal, Skewed, Heavy-tailed)
- Top 10 features analysis

### 5. **Advanced Visualizations Page**
Professional-grade interactive plots:

#### 3D Scatter Plot
- Custom X, Y, Z axis selection
- All features available
- Color gradient by X-axis values
- Sample optimization (up to 3000 points)
- Full rotation and zoom capabilities

#### PCA Space
- 3D PCA projection
- Explained variance display
- Component interpretation
- Loadings visualization

#### Correlation Heatmap
- Full interactive heatmap
- Clustered correlation display
- Diverging color scale
- Strong correlation identification

### 6. **Model Retraining Page**
End-to-end model training and optimization:

#### Configuration Section
- **Model Type Selection**:
  - KNN Classifier
  - LSTM Neural Network
  - Ensemble (KNN + LSTM)
- **Test Set Size**: 10-40% slider
- **Random Seed**: Reproducibility control
- **Training Epochs**: 10-500 (for LSTM)

#### Data Preparation Options
- **Missing Value Handling**:
  - Drop rows
  - Mean imputation
  - Median imputation
  - Forward fill
- **Outlier Removal**: IQR-based option
- **Feature Normalization**: StandardScaler
- **Feature Selection**:
  - All Features
  - Top 10 by Variance
  - Top 15 by Variance
  - Custom PCA (20 components)

#### Training Process
- Real-time progress bar
- Data preparation summary
- Automatic preprocessing pipeline
- Mock training simulation with realistic metrics

#### Performance Metrics
- **Accuracy**: Classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean
- **AUC-ROC**: Area under ROC curve

#### Training Visualization
- Loss curve (decreasing trend)
- Accuracy curve (increasing trend)
- Dual-axis plot for comparison
- Epoch-by-epoch progression

#### Model Saving
- Save trained model to disk
- Save location: `src/model/weights/`
- Celebratory animation on success

---

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Install Dependencies
```bash
# Install dashboard requirements
pip install -r src/model/dashboard_requirements.txt

# Or install individual packages
pip install streamlit>=1.28.0 pandas plotly scikit-learn scipy
```

### Verify Data Location
Ensure training data exists at:
```
src/data/training_dataset/
├── water_batch_01.json
├── water_batch_02.json
└── water_batch_03.json
```

---

## Running the Dashboard

### Option 1: Using Launcher Script
```bash
# From project root
python src/model/dashboard_launcher.py
```

### Option 2: Direct Streamlit Command
```bash
# From project root
streamlit run src/model/dashboard.py
```

### Option 3: Using Python Module
```bash
python -m streamlit run src/model/dashboard.py
```

---

## Dashboard URL
Once launched, access at: **http://localhost:8501**

---

## Navigation & Pages

### Sidebar Navigation
Use the radio buttons in the left sidebar to switch between pages:
1. **Overview** - Data summary and quality metrics
2. **Exploratory Analysis** - EDA with distributions and relationships
3. **Feature Engineering** - 3D visualization and PCA analysis
4. **Statistical Analysis** - Correlations, outliers, distribution types
5. **Advanced Visualizations** - Interactive 3D and PCA plots
6. **Model Retraining** - Training pipeline and optimization

---

## Data Processing Pipeline

### Data Loading
```python
- Load JSON files from training_dataset/
- Flatten nested structure with pd.json_normalize()
- Extract numeric features (.value fields)
- Cache data for performance
```

### Feature Extraction
```python
- Identify all numeric columns
- Extract .value subfields
- Create unified numeric dataframe
- Handle missing values appropriately
```

### Preprocessing Steps
```python
1. Handle missing values (4 options)
2. Remove outliers (optional IQR method)
3. Feature selection (4 options)
4. Normalize features (StandardScaler)
5. Train/test split
```

---

## Visualization Techniques

### 3D Scatter Plot
- **Technology**: Plotly 3D scatter
- **Points**: Up to 5000 (sampled)
- **Color**: Gradient based on feature value
- **Interaction**: Rotate, zoom, pan

### Correlation Heatmap
- **Technology**: Plotly heatmap
- **Scale**: Red-Blue diverging (-1 to 1)
- **Annotations**: Correlation coefficients
- **Clustering**: Optional dendrograms

### Bivariate Plots
- **Technology**: Plotly scatter
- **Arrangement**: 2x2 grid
- **Features**: Top 4 by variance
- **Opacity**: 0.6 for overlapping points

### PCA Visualization
- **Method**: StandardScaler + PCA(3 components)
- **3D Plot**: Scatter plot of components
- **Variance**: Shown in axis labels
- **Loadings**: Top features per component

### Box Plots
- **Purpose**: Outlier detection
- **Features**: All numeric features
- **Statistics**: Mean (sd) overlaid
- **IQR**: Quartiles and whiskers

### Scatter Matrix
- **Size**: 5x5 for top 5 features
- **Coverage**: Lower triangle only
- **Sampling**: Max 1000 points
- **Diagonal**: Removed for clarity

---

## Statistical Methods Used

### Correlation Analysis
```python
Pearson correlation coefficient
- Range: -1 to 1
- Strong: |r| > 0.7
- Moderate: 0.3 < |r| < 0.7
- Weak: |r| < 0.3
```

### Outlier Detection
```python
Interquartile Range (IQR) Method
- Q1: 25th percentile
- Q3: 75th percentile
- IQR = Q3 - Q1
- Lower bound = Q1 - 1.5*IQR
- Upper bound = Q3 + 1.5*IQR
```

### PCA (Principal Component Analysis)
```python
- Standardize features
- Calculate covariance matrix
- Compute eigenvalues/eigenvectors
- Project data onto principal components
- Explained variance ratio per component
```

### Distribution Analysis
```python
- Skewness: Measure of asymmetry
  - |skew| < 0.5: Approximately symmetric
  - 0.5 < |skew| < 1: Moderately skewed
  - |skew| > 1: Highly skewed
  
- Kurtosis: Measure of tail weight
  - Kurt > 3: Heavy tails (leptokurtic)
  - Kurt = 3: Normal distribution
  - Kurt < 3: Light tails (platykurtic)
```

---

## Performance Optimization

### Data Caching
```python
@st.cache_data
def load_training_data():
    # Data loaded once, reused across reruns
```

### Sampling Strategy
- **Distributions**: All data
- **3D Scatter**: Max 5000 points
- **PCA**: All data (computed)
- **Scatter Matrix**: Max 1000 points
- **Correlation**: All data (computed)

### Rendering Tips
- Use tabs to separate intensive visualizations
- Enable lazy loading for multiple charts
- Use appropriate color scales for accessibility

---

## Troubleshooting

### Issue: "Data Loading Failed"
**Solution**: Verify JSON files exist in `src/data/training_dataset/`

### Issue: "Memory Error with Large Dataset"
**Solution**: Dashboard samples data appropriately. Try:
- Restart browser
- Close other applications
- Check available RAM

### Issue: "Features Not Appearing"
**Solution**: Ensure numeric features have `.value` field in JSON

### Issue: "Slow Performance"
**Solution**:
- Check internet connection (for Plotly CDN)
- Reduce sample size in configuration
- Close other browser tabs
- Restart Streamlit app

### Issue: "PCA Plot Not Rendering"
**Solution**: Ensure at least 3 numeric features are available

---

## Key Metrics & Definitions

### Data Quality Metrics
- **Missing %**: Percentage of null values
- **Feature Count**: Number of numeric features
- **Record Count**: Total data points

### Model Performance Metrics
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **AUC-ROC**: Area under Receiver Operating Characteristic curve

### Feature Statistics
- **Variance**: Measure of spread
- **Skewness**: Asymmetry of distribution
- **Kurtosis**: Heaviness of tails

---

## Advanced Features

### Custom Feature Selection for 3D
- Select exactly 3 features
- Real-time normalization
- Interactive rotation and zoom
- Color gradient mapping

### Correlation Strength Filtering
- Identify correlations > 0.7
- Display feature pairs
- Sort by correlation magnitude
- Interpret multicollinearity

### Anomaly Detection
- IQR-based identification
- Per-feature anomaly tracking
- Percentage metrics
- Distribution visualization

### PCA Explained Variance
- Shows how much variance each PC captures
- Cumulative variance understanding
- Feature importance ranking
- Dimensionality reduction insights

---

## Best Practices

### For Data Exploration
1. Start with Overview page
2. Check distributions and outliers
3. Review correlations
4. Investigate feature relationships
5. Look for patterns in PCA space

### For Feature Engineering
1. Identify high-variance features
2. Check feature correlations
3. Remove multicollinear features
4. Consider PCA for dimensionality reduction
5. Analyze skewness and kurtosis

### For Model Retraining
1. Prepare data carefully
2. Choose appropriate feature selection
3. Normalize if needed
4. Monitor training progress
5. Compare model versions
6. Save best performing models

---

## Extension Possibilities

- Add cross-validation visualization
- Implement confusion matrix display
- Add feature importance plots
- Create prediction interface
- Add real-time data streaming
- Implement A/B testing interface
- Add time-series analysis
- Create explainability dashboard (SHAP/LIME)

---

## Support & Documentation

For issues or questions:
1. Check troubleshooting section
2. Review data format requirements
3. Verify Streamlit installation
4. Check Plotly version compatibility

---

## Version History

### v1.0.0 (Current)
- Initial release
- 6 main pages
- 10+ advanced visualizations
- Full model retraining pipeline
- Comprehensive data exploration

---
