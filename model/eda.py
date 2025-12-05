import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Load all JSON files from folder
# -------------------------------
def load_water_data(folder_path):
    records = []
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            with open(os.path.join(folder_path, file), 'r') as f:
                data = json.load(f)
                records.extend(data)
    return pd.DataFrame(records)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("💧 Water Sensor Data EDA Dashboard")
st.write("Univariate, Bivariate, Correlation & Feature Engineering Explorer")

folder_path = r"C:\\Users\\GENAIKOLGPUSR36\\Desktop\\water\\water_sensor_data"
df = load_water_data(folder_path)

st.subheader("📌 Raw Data Preview")
st.dataframe(df.head())

# Convert categorical to category dtype
target = "class_label"
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

# -------------------------------
# Univariate Analysis
# -------------------------------
st.header("📊 Univariate Analysis")
selected_uni = st.selectbox("Select Feature", df.columns)

if selected_uni in num_cols:
    fig, ax = plt.subplots()
    sns.histplot(df[selected_uni], kde=True, ax=ax)
    st.pyplot(fig)

else:
    fig, ax = plt.subplots()
    df[selected_uni].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# -------------------------------
# Bivariate Analysis
# -------------------------------
st.header("🔗 Bivariate Analysis")
col_x = st.selectbox("X-axis", df.columns, key='xaxis')
col_y = st.selectbox("Y-axis", df.columns, key='yaxis')

fig, ax = plt.subplots()
if col_x in num_cols and col_y in num_cols:
    sns.scatterplot(x=df[col_x], y=df[col_y], hue=df[target], ax=ax)
elif col_x in cat_cols and col_y in num_cols:
    sns.boxplot(x=df[col_x], y=df[col_y], ax=ax)
elif col_x in num_cols and col_y in cat_cols:
    sns.boxplot(x=df[col_y], y=df[col_x], ax=ax)
elif col_x in cat_cols and col_y in cat_cols:
    sns.countplot(x=df[col_x], hue=df[col_y], ax=ax)
st.pyplot(fig)

# -------------------------------
# Correlation Matrix
# -------------------------------
st.header("📈 Correlation Matrix")
num_df = df[num_cols]
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# -------------------------------
# Feature Engineering Section
# -------------------------------
st.header("🛠 Feature Engineering")
st.write("Automatic extraction of domain-specific features")

df['Flow_Pressure_Ratio'] = df['FlowRate'] / (df['Pressure'] + 1)
df['Vibration_to_Acoustic'] = df['Vibration'] / (df['AcousticLevel'] + 1)
df['RPM_Stress'] = df['RPM'] * df['UltrasonicSignal']

st.subheader("New Features Added")
st.dataframe(df[['Flow_Pressure_Ratio','Vibration_to_Acoustic','RPM_Stress']].head())

st.success("Dashboard Loaded Successfully! Use the sidebar and controls above to explore your data.")
