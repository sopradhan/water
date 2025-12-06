# Dashboard - Quick Start Guide

## 30-Second Setup

### 1. Install Dependencies
```bash
pip install streamlit plotly pandas scikit-learn scipy
```

### 2. Start Dashboard
```bash
# From project root
python src/model/dashboard_launcher.py

# Or directly
streamlit run src/model/dashboard.py
```

### 3. Access Dashboard
```
Open: http://localhost:8501
```

---

## Navigation Overview

### Page 1: Overview [Data Summary]
- View dataset statistics
- Check data quality
- See feature counts
- Preview raw data

### Page 2: Exploratory Analysis [EDA]
- **Distributions**: Histogram plots
- **Box Plots**: Outlier detection
- **Bivariate**: Feature relationships
- **Scatter Matrix**: All feature pairs

### Page 3: Feature Engineering [3D + PCA]
- 3D scatter visualization (custom features)
- PCA analysis with loadings
- Feature variance ranking
- Skewness analysis

### Page 4: Statistical Analysis [Correlations]
- Correlation matrix heatmap
- Outlier statistics
- Distribution type analysis
- Strong correlation pairs

### Page 5: Advanced Visualizations [Professional Plots]
- 3D scatter (custom axes)
- PCA projection
- Correlation heatmap
- Full interactive plots

### Page 6: Model Retraining [Training Pipeline]
- Configure model type
- Prepare data
- Select features
- Train and evaluate
- Save trained model

---

## Common Tasks

### Explore Data Distribution
1. Go to **Exploratory Analysis**
2. Click **[DIST] Distributions** tab
3. View all feature histograms

### Find Feature Correlations
1. Go to **Statistical Analysis**
2. View correlation heatmap
3. Look for red/blue clusters
4. Check "Strong Correlations" list

### Visualize 3D Feature Space
1. Go to **Feature Engineering**
2. Select 3 features
3. Interact with 3D plot (rotate/zoom)
4. View feature statistics

### Train New Model
1. Go to **Model Retraining**
2. Select model type
3. Configure data preparation
4. Click **[START] Start Model Retraining**
5. Monitor progress
6. Save successful model

### Detect Outliers
1. Go to **Statistical Analysis**
2. Check outlier detection metrics
3. View distribution by feature
4. Consider removing in retraining

---

## Tips & Tricks

### Performance
- Dashboard caches data automatically
- First load may take 30-60 seconds
- Subsequent loads are instant
- Sampling used for large plots

### Visualization
- Hover over plots for details
- Zoom: scroll wheel
- Pan: drag plot area
- Reset: double-click

### Data Preparation
- Choose appropriate missing value handling
- Remove outliers for cleaner models
- Normalize for neural networks
- Use feature selection for efficiency

### Model Training
- Start with default settings
- Increase epochs for better accuracy
- Use ensemble for robustness
- Save multiple model versions

---

## File Locations

```
Project Root/
├── src/
│   ├── data/
│   │   └── training_dataset/
│   │       ├── water_batch_01.json
│   │       ├── water_batch_02.json
│   │       └── water_batch_03.json
│   └── model/
│       ├── dashboard.py           (Main app)
│       ├── dashboard_launcher.py  (Startup script)
│       ├── dashboard_requirements.txt
│       ├── DASHBOARD_GUIDE.md     (Full documentation)
│       ├── DASHBOARD_QUICKSTART.md (This file)
│       └── weights/               (Saved models)
```

---

## Sample Commands

### Run Dashboard
```bash
cd c:\Users\PRADHAN\OneDrive\Desktop\water
python src/model/dashboard_launcher.py
```

### Install All Requirements
```bash
pip install -r src/model/dashboard_requirements.txt
```

### Update Streamlit
```bash
pip install --upgrade streamlit
```

---

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Clear Cache | Ctrl+C (then Restart) |
| Refresh | F5 |
| Full Screen | F11 |
| Developer Tools | F12 |

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `streamlit not found` | `pip install streamlit` |
| `Data loading failed` | Check `src/data/training_dataset/` exists |
| `Memory error` | Close other apps, restart Streamlit |
| `Port 8501 already in use` | `streamlit run dashboard.py --server.port 8502` |
| `Plotly not installed` | `pip install plotly` |

---

## Next Steps

1. **Explore Data**: Start with Overview page
2. **Analyze Features**: Use Exploratory Analysis
3. **Engineer Features**: Go to Feature Engineering
4. **Train Model**: Go to Model Retraining
5. **Save Model**: Export trained model

---

## More Information

- Full documentation: `DASHBOARD_GUIDE.md`
- API documentation: Check RAG API docs
- Model API: Check `src/model/api.py`

---

Good luck with your water anomaly detection system! 💧
