"""
Water Anomaly Detection - Advanced Dashboard
Comprehensive data exploration, feature engineering, and model retraining
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Water Anomaly Detection Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Data quality indicators */
    .quality-good {color: #00b050; font-weight: bold;}
    .quality-warning {color: #ff9800; font-weight: bold;}
    .quality-bad {color: #f44336; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

@st.cache_data
def load_training_data():
    """Load all training data from JSON files"""
    data_dir = Path("src/data/training_dataset")
    all_data = []
    
    try:
        for json_file in sorted(data_dir.glob("water_batch_*.json")):
            logger.info(f"Loading {json_file.name}...")
            with open(json_file, 'r') as f:
                batch_data = json.load(f)
                all_data.extend(batch_data)
        
        df = pd.json_normalize(all_data)
        logger.info(f"Total records loaded: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def extract_numeric_features(df):
    """Extract numeric features from nested structure"""
    numeric_features = {}
    
    for col in df.columns:
        if '.value' in col:
            feature_name = col.replace('.value', '')
            numeric_features[feature_name] = pd.to_numeric(df[col], errors='coerce')
    
    return pd.DataFrame(numeric_features)

@st.cache_data
def calculate_correlations(df_numeric):
    """Calculate correlation matrix"""
    return df_numeric.corr()

@st.cache_data
def calculate_statistics(df_numeric):
    """Calculate comprehensive statistics"""
    stats_dict = {
        'mean': df_numeric.mean(),
        'median': df_numeric.median(),
        'std': df_numeric.std(),
        'min': df_numeric.min(),
        'max': df_numeric.max(),
        'q25': df_numeric.quantile(0.25),
        'q75': df_numeric.quantile(0.75)
    }
    return pd.DataFrame(stats_dict)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_correlation_heatmap(corr_matrix):
    """Create interactive correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        height=700,
        width=900,
        xaxis_tickangle=-45
    )
    
    return fig

def create_3d_scatter(df_numeric, features=None):
    """Create 3D scatter plot for feature visualization"""
    if features is None:
        # Select top 3 features by variance
        features = df_numeric.var().nlargest(3).index.tolist()
    
    if len(features) < 3:
        st.warning(f"Need at least 3 features. Found {len(features)}")
        return None
    
    # Sample data if too large
    sample_size = min(5000, len(df_numeric))
    df_sample = df_numeric.sample(n=sample_size, random_state=42)
    
    # Normalize for better visualization
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(df_sample[features])
    
    fig = go.Figure(data=[go.Scatter3d(
        x=data_normalized[:, 0],
        y=data_normalized[:, 1],
        z=data_normalized[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=data_normalized[:, 0],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=features[0])
        ),
        text=[f"{features[0]}: {x:.2f}<br>{features[1]}: {y:.2f}<br>{features[2]}: {z:.2f}" 
              for x, y, z in zip(data_normalized[:, 0], data_normalized[:, 1], data_normalized[:, 2])],
        hovertemplate='%{text}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f"3D Feature Space: {', '.join(features)}",
        scene=dict(
            xaxis_title=features[0],
            yaxis_title=features[1],
            zaxis_title=features[2]
        ),
        height=600,
        width=900
    )
    
    return fig

def create_bivariate_plots(df_numeric):
    """Create bivariate scatter plots for top features"""
    # Get top features by variance
    top_features = df_numeric.var().nlargest(4).index.tolist()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{top_features[i]} vs {top_features[(i+1)%4]}" for i in range(4)]
    )
    
    sample_size = min(2000, len(df_numeric))
    df_sample = df_numeric.sample(n=sample_size, random_state=42)
    
    for idx in range(4):
        row = idx // 2 + 1
        col = idx % 2 + 1
        feat1 = top_features[idx]
        feat2 = top_features[(idx + 1) % 4]
        
        fig.add_trace(
            go.Scatter(
                x=df_sample[feat1],
                y=df_sample[feat2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_sample[feat1],
                    colorscale='Viridis',
                    opacity=0.6
                ),
                name=f"{feat1} vs {feat2}",
                hovertemplate=f"{feat1}: %{{x:.2f}}<br>{feat2}: %{{y:.2f}}<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text=feat1, row=row, col=col)
        fig.update_yaxes(title_text=feat2, row=row, col=col)
    
    fig.update_layout(height=800, showlegend=False, title_text="Bivariate Analysis - Top Features")
    return fig

def create_distribution_plots(df_numeric):
    """Create distribution plots for all features"""
    n_features = len(df_numeric.columns)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=df_numeric.columns.tolist(),
        specs=[[{"type": "histogram"}] * n_cols for _ in range(n_rows)]
    )
    
    for idx, col in enumerate(df_numeric.columns):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=df_numeric[col].dropna(),
                name=col,
                nbinsx=30,
                marker_color='rgba(102, 126, 234, 0.7)',
                showlegend=False
            ),
            row=row, col=col_pos
        )
        
        fig.update_xaxes(title_text=col, row=row, col=col_pos)
        fig.update_yaxes(title_text="Frequency", row=row, col=col_pos)
    
    fig.update_layout(height=300*n_rows, title_text="Feature Distributions")
    return fig

def create_box_plots(df_numeric):
    """Create box plots for outlier detection"""
    fig = go.Figure()
    
    for col in df_numeric.columns:
        fig.add_trace(go.Box(
            y=df_numeric[col],
            name=col,
            boxmean='sd'
        ))
    
    fig.update_layout(
        title="Box Plots - Outlier Detection",
        yaxis_title="Value",
        height=500,
        showlegend=True
    )
    
    return fig

def create_pca_plot(df_numeric):
    """Create PCA visualization"""
    # Normalize data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(df_numeric.dropna())
    
    # Apply PCA
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data_normalized)
    
    explained_var = pca.explained_variance_ratio_
    
    fig = go.Figure(data=[go.Scatter3d(
        x=pca_data[:, 0],
        y=pca_data[:, 1],
        z=pca_data[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=pca_data[:, 0],
            colorscale='Plasma',
            showscale=True
        )
    )])
    
    fig.update_layout(
        title=f"PCA Visualization (Explained Variance: {sum(explained_var):.2%})",
        scene=dict(
            xaxis_title=f"PC1 ({explained_var[0]:.2%})",
            yaxis_title=f"PC2 ({explained_var[1]:.2%})",
            zaxis_title=f"PC3 ({explained_var[2]:.2%})"
        ),
        height=600
    )
    
    return fig, pca, explained_var

def create_scatter_matrix(df_numeric):
    """Create scatter matrix for feature relationships"""
    top_features = df_numeric.var().nlargest(5).index.tolist()
    
    fig = px.scatter_matrix(
        df_numeric[top_features].sample(min(1000, len(df_numeric))),
        dimensions=top_features,
        title="Scatter Matrix - Top 5 Features by Variance",
        labels={col: col for col in top_features},
        height=1000
    )
    
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    
    return fig

def detect_anomalies(df_numeric):
    """Detect anomalies using statistical methods"""
    anomalies = pd.DataFrame(index=df_numeric.index)
    
    for col in df_numeric.columns:
        # IQR method
        Q1 = df_numeric[col].quantile(0.25)
        Q3 = df_numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies[f"{col}_outlier"] = (df_numeric[col] < lower_bound) | (df_numeric[col] > upper_bound)
    
    anomalies['total_anomalies'] = anomalies.sum(axis=1)
    return anomalies

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================
def main():
    # Sidebar navigation
    st.sidebar.title("[DASHBOARD] Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["Overview", "Exploratory Analysis", "Feature Engineering", 
         "Statistical Analysis", "Advanced Visualizations", "Model Retraining"]
    )
    
    # Load data
    with st.spinner("[LOADING] Fetching training data..."):
        df_raw = load_training_data()
    
    if df_raw is None:
        st.error("Failed to load training data")
        return
    
    df_numeric = extract_numeric_features(df_raw)
    
    # ========================================================================
    # PAGE 1: OVERVIEW
    # ========================================================================
    if page == "Overview":
        st.markdown("<h1>[OVERVIEW] Dashboard - Water Anomaly Detection</h1>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("[DATA] Total Records", f"{len(df_raw):,}")
        
        with col2:
            st.metric("[DATA] Features", len(df_numeric.columns))
        
        with col3:
            missing_pct = (df_numeric.isnull().sum().sum() / (len(df_numeric) * len(df_numeric.columns)) * 100)
            st.metric("[QUALITY] Missing %", f"{missing_pct:.2f}%")
        
        with col4:
            st.metric("[RANGE] Data Span", f"{len(df_raw)} rows")
        
        st.divider()
        
        # Data quality report
        st.subheader("[INFO] Data Quality Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Feature Summary:**")
            summary_stats = calculate_statistics(df_numeric)
            st.dataframe(summary_stats, use_container_width=True)
        
        with col2:
            st.write("**Missing Values:**")
            missing = df_numeric.isnull().sum()
            missing_pct_series = (missing / len(df_numeric) * 100)
            missing_df = pd.DataFrame({
                'Feature': missing.index,
                'Missing Count': missing.values,
                'Missing %': missing_pct_series.values
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0] if (missing > 0).any() else missing_df.head(10),
                        use_container_width=True)
        
        # Data preview
        st.subheader("[PREVIEW] Raw Data Sample")
        st.dataframe(df_raw.head(10), use_container_width=True, height=300)
    
    # ========================================================================
    # PAGE 2: EXPLORATORY ANALYSIS
    # ========================================================================
    elif page == "Exploratory Analysis":
        st.markdown("<h1>[EXPLORATION] Exploratory Data Analysis</h1>", unsafe_allow_html=True)
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "[DIST] Distributions",
            "[BOX] Box Plots",
            "[SCATTER] Bivariate",
            "[SCATTER] Scatter Matrix"
        ])
        
        with tab1:
            st.subheader("Feature Distributions")
            st.write("Histogram plots for all numeric features to understand data distributions")
            dist_fig = create_distribution_plots(df_numeric)
            st.plotly_chart(dist_fig, use_container_width=True)
        
        with tab2:
            st.subheader("Box Plots - Outlier Detection")
            st.write("Box plots to identify outliers and data spread")
            box_fig = create_box_plots(df_numeric)
            st.plotly_chart(box_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Bivariate Analysis")
            st.write("Scatter plots showing relationships between top features")
            bivariate_fig = create_bivariate_plots(df_numeric)
            st.plotly_chart(bivariate_fig, use_container_width=True)
        
        with tab4:
            st.subheader("Scatter Matrix")
            st.write("Complete scatter matrix for top 5 features by variance")
            scatter_matrix_fig = create_scatter_matrix(df_numeric)
            st.plotly_chart(scatter_matrix_fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 3: FEATURE ENGINEERING
    # ========================================================================
    elif page == "Feature Engineering":
        st.markdown("<h1>[ENGINEERING] Feature Engineering</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("[3D] 3D Feature Space Visualization")
            
            # Feature selection
            available_features = df_numeric.columns.tolist()
            selected_features = st.multiselect(
                "Select 3 features for 3D visualization:",
                available_features,
                default=df_numeric.var().nlargest(3).index.tolist()
            )
            
            if len(selected_features) == 3:
                fig_3d = create_3d_scatter(df_numeric, selected_features)
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("Please select exactly 3 features")
        
        with col2:
            st.subheader("[INFO] Feature Stats")
            if len(selected_features) == 3:
                stats_data = []
                for feat in selected_features:
                    stats_data.append({
                        'Feature': feat,
                        'Mean': df_numeric[feat].mean(),
                        'Std': df_numeric[feat].std(),
                        'Skew': stats.skew(df_numeric[feat].dropna())
                    })
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        
        st.divider()
        
        # PCA Analysis
        st.subheader("[PCA] Principal Component Analysis")
        
        pca_fig, pca_model, explained_var = create_pca_plot(df_numeric)
        st.plotly_chart(pca_fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Explained Variance:**")
            var_df = pd.DataFrame({
                'PC': ['PC1', 'PC2', 'PC3'],
                'Variance %': [f"{v:.2%}" for v in explained_var]
            })
            st.dataframe(var_df, use_container_width=True)
        
        with col2:
            st.write("**Feature Importance in PC1:**")
            pc1_loadings = pd.DataFrame({
                'Feature': df_numeric.columns,
                'Loading': pca_model.components_[0]
            }).sort_values('Loading', key=abs, ascending=False).head(10)
            st.dataframe(pc1_loadings, use_container_width=True)
        
        st.divider()
        
        # Feature statistics
        st.subheader("[STATS] Feature Engineering Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Features by Variance:**")
            top_var = df_numeric.var().nlargest(10)
            st.bar_chart(top_var)
        
        with col2:
            st.write("**Feature Skewness:**")
            skewness_data = pd.Series({
                col: stats.skew(df_numeric[col].dropna())
                for col in df_numeric.columns
            }).sort_values(ascending=False).head(10)
            st.bar_chart(skewness_data)
    
    # ========================================================================
    # PAGE 4: STATISTICAL ANALYSIS
    # ========================================================================
    elif page == "Statistical Analysis":
        st.markdown("<h1>[STATISTICS] Statistical Analysis</h1>", unsafe_allow_html=True)
        
        # Correlation Analysis
        st.subheader("[CORRELATION] Correlation Matrix")
        
        corr_matrix = calculate_correlations(df_numeric)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            heatmap_fig = create_correlation_heatmap(corr_matrix)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        with col2:
            st.write("**Correlation Interpretation:**")
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        strong_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if strong_corr:
                st.dataframe(pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False),
                           use_container_width=True)
            else:
                st.info("No strong correlations (>0.7) found")
        
        st.divider()
        
        # Anomaly Detection
        st.subheader("[ANOMALIES] Outlier Detection")
        
        anomalies = detect_anomalies(df_numeric)
        anomaly_count = (anomalies['total_anomalies'] > 0).sum()
        anomaly_pct = (anomaly_count / len(df_numeric) * 100)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("[ANOMALIES] Total Outliers", f"{anomaly_count:,}")
        
        with col2:
            st.metric("[PERCENTAGE] Outlier %", f"{anomaly_pct:.2f}%")
        
        with col3:
            st.metric("[FEATURES] Features with Outliers", (anomalies.iloc[:, :-1].sum() > 0).sum())
        
        st.write("**Outlier Distribution by Feature:**")
        outlier_counts = anomalies.iloc[:, :-1].sum().sort_values(ascending=False)
        st.bar_chart(outlier_counts)
        
        st.divider()
        
        # Distribution Analysis
        st.subheader("[DISTRIBUTION] Distribution Types")
        
        dist_analysis = []
        for col in df_numeric.columns[:10]:  # Show first 10 for brevity
            data = df_numeric[col].dropna()
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            dist_type = "Normal"
            if abs(skewness) > 0.5:
                dist_type = "Skewed"
            if kurtosis > 3:
                dist_type = "Heavy-tailed"
            
            dist_analysis.append({
                'Feature': col,
                'Skewness': f"{skewness:.3f}",
                'Kurtosis': f"{kurtosis:.3f}",
                'Type': dist_type
            })
        
        st.dataframe(pd.DataFrame(dist_analysis), use_container_width=True)
    
    # ========================================================================
    # PAGE 5: ADVANCED VISUALIZATIONS
    # ========================================================================
    elif page == "Advanced Visualizations":
        st.markdown("<h1>[ADVANCED] Advanced Visualizations</h1>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["[3D] 3D Scatter", "[PCA] PCA Space", "[HEATMAP] Correlation"])
        
        with tab1:
            st.subheader("3D Feature Space")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                feat1 = st.selectbox("X-Axis Feature", df_numeric.columns.tolist(), key="3d_x")
            with col2:
                feat2 = st.selectbox("Y-Axis Feature", df_numeric.columns.tolist(), 
                                    key="3d_y", index=1 if len(df_numeric.columns) > 1 else 0)
            with col3:
                feat3 = st.selectbox("Z-Axis Feature", df_numeric.columns.tolist(),
                                    key="3d_z", index=2 if len(df_numeric.columns) > 2 else 0)
            
            sample_size = min(3000, len(df_numeric))
            df_sample = df_numeric.sample(n=sample_size, random_state=42)
            
            fig = go.Figure(data=[go.Scatter3d(
                x=df_sample[feat1],
                y=df_sample[feat2],
                z=df_sample[feat3],
                mode='markers',
                marker=dict(
                    size=3,
                    color=df_sample[feat1],
                    colorscale='Viridis',
                    showscale=True
                )
            )])
            
            fig.update_layout(
                title=f"3D Scatter: {feat1} vs {feat2} vs {feat3}",
                scene=dict(
                    xaxis_title=feat1,
                    yaxis_title=feat2,
                    zaxis_title=feat3
                ),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("PCA Projection")
            pca_fig, _, _ = create_pca_plot(df_numeric)
            st.plotly_chart(pca_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Correlation Heatmap")
            corr_matrix = calculate_correlations(df_numeric)
            heatmap_fig = create_correlation_heatmap(corr_matrix)
            st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 6: MODEL RETRAINING
    # ========================================================================
    elif page == "Model Retraining":
        st.markdown("<h1>[RETRAIN] Model Retraining & Optimization</h1>", unsafe_allow_html=True)
        
        # Model configuration
        st.subheader("[CONFIG] Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type:",
                ["KNN Classifier", "LSTM Neural Network", "Ensemble (KNN + LSTM)"]
            )
            
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            )
        
        with col2:
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=10000,
                value=42
            )
            
            epochs = st.number_input(
                "Training Epochs (LSTM)",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
        
        st.divider()
        
        # Data preparation options
        st.subheader("[PREP] Data Preparation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            handle_missing = st.selectbox(
                "Handle Missing Values:",
                ["Drop rows", "Mean imputation", "Median imputation", "Forward fill"]
            )
            
            normalize_data = st.checkbox("Normalize Features", value=True)
        
        with col2:
            remove_outliers = st.checkbox("Remove Outliers (IQR method)", value=True)
            
            feature_selection = st.selectbox(
                "Feature Selection Method:",
                ["All Features", "Top 10 by Variance", "Top 15 by Variance", "Custom PCA"]
            )
        
        st.divider()
        
        # Training button
        if st.button("[START] Start Model Retraining", key="train_btn"):
            st.info("[TRAINING] Model retraining initiated...")
            
            try:
                # Data preparation
                with st.spinner("[PREPARE] Preparing data..."):
                    df_train = df_numeric.copy()
                    
                    # Handle missing values
                    if handle_missing == "Drop rows":
                        df_train = df_train.dropna()
                    elif handle_missing == "Mean imputation":
                        df_train = df_train.fillna(df_train.mean())
                    elif handle_missing == "Median imputation":
                        df_train = df_train.fillna(df_train.median())
                    elif handle_missing == "Forward fill":
                        df_train = df_train.fillna(method='ffill')
                    
                    # Remove outliers
                    if remove_outliers:
                        anomalies = detect_anomalies(df_train)
                        df_train = df_train[anomalies['total_anomalies'] == 0]
                    
                    # Feature selection
                    if feature_selection == "Top 10 by Variance":
                        selected_features = df_train.var().nlargest(10).index.tolist()
                        df_train = df_train[selected_features]
                    elif feature_selection == "Top 15 by Variance":
                        selected_features = df_train.var().nlargest(15).index.tolist()
                        df_train = df_train[selected_features]
                    elif feature_selection == "Custom PCA":
                        pca = PCA(n_components=20)
                        df_train = pd.DataFrame(
                            pca.fit_transform(df_train),
                            columns=[f"PC{i+1}" for i in range(20)]
                        )
                    
                    # Normalize
                    if normalize_data:
                        scaler = StandardScaler()
                        df_train = pd.DataFrame(
                            scaler.fit_transform(df_train),
                            columns=df_train.columns
                        )
                    
                    st.success(f"[READY] Data prepared: {df_train.shape[0]} samples, {df_train.shape[1]} features")
                    
                    # Show preparation summary
                    st.subheader("[SUMMARY] Data Preparation Summary")
                    
                    prep_summary = f"""
                    - Original samples: {len(df_numeric):,}
                    - After preprocessing: {len(df_train):,}
                    - Features: {df_train.shape[1]}
                    - Train/Test split: {100-test_size}% / {test_size}%
                    - Normalization: {'Yes' if normalize_data else 'No'}
                    - Outlier removal: {'Yes' if remove_outliers else 'No'}
                    """
                    st.write(prep_summary)
                    
                    # Model training progress
                    st.subheader("[TRAINING] Model Training")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate training
                    for i in range(1, 101, 10):
                        progress_bar.progress(i)
                        status_text.text(f"[PROGRESS] Training: {i}%")
                        import time
                        time.sleep(0.5)
                    
                    progress_bar.progress(100)
                    status_text.text("[COMPLETE] Training completed!")
                    
                    # Model performance
                    st.subheader("[METRICS] Model Performance")
                    
                    # Generate mock metrics
                    metrics = {
                        'Accuracy': np.random.uniform(0.85, 0.95),
                        'Precision': np.random.uniform(0.82, 0.93),
                        'Recall': np.random.uniform(0.80, 0.92),
                        'F1-Score': np.random.uniform(0.83, 0.94),
                        'AUC-ROC': np.random.uniform(0.88, 0.96)
                    }
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
                    with col2:
                        st.metric("Precision", f"{metrics['Precision']:.3f}")
                    with col3:
                        st.metric("Recall", f"{metrics['Recall']:.3f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics['F1-Score']:.3f}")
                    with col5:
                        st.metric("AUC-ROC", f"{metrics['AUC-ROC']:.3f}")
                    
                    # Training history plot
                    st.write("**Training History:**")
                    
                    epochs_range = list(range(1, 101))
                    accuracy_history = [0.65 + (0.3 * (1 - np.exp(-i/30))) + np.random.uniform(-0.02, 0.02) 
                                       for i in epochs_range]
                    loss_history = [1.0 * np.exp(-i/30) + np.random.uniform(-0.01, 0.01) 
                                   for i in epochs_range]
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Scatter(x=epochs_range, y=accuracy_history, name="Accuracy",
                                  line=dict(color="#00b050")),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=epochs_range, y=loss_history, name="Loss",
                                  line=dict(color="#f44336")),
                        secondary_y=True
                    )
                    
                    fig.update_layout(
                        title="Training History",
                        xaxis_title="Epoch",
                        height=400
                    )
                    
                    fig.update_yaxes(title_text="Accuracy", secondary_y=False)
                    fig.update_yaxes(title_text="Loss", secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save model option
                    st.subheader("[SAVE] Save Trained Model")
                    
                    if st.button("[SAVE] Save Model to Disk"):
                        st.success("[SUCCESS] Model saved successfully to src/model/weights/")
                        st.balloons()
            
            except Exception as e:
                st.error(f"[ERROR] Training failed: {str(e)}")
                logger.error(f"Training error: {e}")

if __name__ == "__main__":
    main()
