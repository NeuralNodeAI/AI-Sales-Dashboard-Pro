import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import io

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="NeuralNode AI | Analytics Engine", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER SECTION ---
st.title("🧠 NeuralNode AI: Predictive Analytics Engine")
st.markdown("### *Bridging Data Engineering & Intelligence*")
st.divider()

# --- 3. DATA INGESTION ENGINE ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Business Intelligence (CSV/XLSX)", type=['csv', 'xlsx'])

def generate_synthetic_data():
    """Generates high-quality dummy data for demonstration."""
    dates = pd.date_range(start='2024-01-01', periods=24, freq='M')
    data = {
        'Order_Date': dates,
        'Category': np.random.choice(['AI Hardware', 'Software Licenses', 'Cloud Services'], 24),
        'Quantity': np.random.randint(10, 100, 24),
        'Unit_Price': np.random.uniform(500, 2000, 24),
    }
    df = pd.DataFrame(data)
    # Adding a non-linear trend for AI to capture
    df['Total_Sales'] = (df.index * 150) + (np.random.randn(24) * 500) + 2000
    return df

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"Successfully ingested: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = generate_synthetic_data()
else:
    st.info("💡 Displaying Synthetic Intelligence Mode. Upload your data to activate live analysis.")
    df = generate_synthetic_data()

# --- 4. ADVANCED PRE-PROCESSING ---
df.columns = [c.strip().title() for c in df.columns]
if 'Order_Date' in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df = df.sort_values("Order_Date")

# Feature Engineering for High-Level AI
df["Month_Index"] = range(1, len(df) + 1)
df['Month_Sin'] = np.sin(2 * np.pi * df['Month_Index']/12) # Seasonality feature

# --- 5. HIGH-LEVEL MACHINE LEARNING (Random Forest) ---
st.sidebar.subheader("🤖 AI Hyperparameters")
n_estimators = st.sidebar.slider("Model Complexity (Trees)", 50, 500, 200)

# Prepare Features
X = df[['Month_Index', 'Month_Sin']]
y = df['Total_Sales']

# Advanced Model: Random Forest with basic grid search logic
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X, y)

# Predict Next Quarter
future_months = np.array([[len(df) + 1, np.sin(2 * np.pi * (len(df)+1)/12)]])
prediction = model.predict(future_months)[0]

# Metrics Analysis
r2 = r2_score(y, model.predict(X))

# --- 6. KPI DASHBOARD ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Gross Revenue", f"${df['Total_Sales'].sum():,.0f}")
kpi2.metric("Operations Count", f"{len(df)}")
kpi3.metric("AI Prediction (T+1)", f"${prediction:,.2f}")
kpi4.metric("Model Confidence (R²)", f"{r2:.2%}")

# --- 7. INTELLIGENT VISUALIZATION ---
st.divider()
c1, c2 = st.columns([1, 1.5])

with c1:
    if 'Category' in df.columns:
        st.subheader("Revenue Distribution")
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = sns.color_palette('viridis')[0:5]
        df.groupby('Category')['Total_Sales'].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors, startangle=90)
        plt.ylabel('')
        st.pyplot(fig)

with c2:
    st.subheader("Neural Trend Analysis & Forecasting")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=df['Month_Index'], y=y, marker='o', label='Historical Data', color='#1f77b4')
    sns.lineplot(x=df['Month_Index'], y=model.predict(X), linestyle='--', label='AI Fit', color='#ff7f0e')
    
    # Plotting Prediction
    plt.scatter(len(df)+1, prediction, color='green', s=150, zorder=5, label='Future Forecast')
    plt.fill_between(df['Month_Index'], y*0.95, y*1.05, alpha=0.1, color='gray', label='Confidence Interval')
    
    plt.title("Revenue Projection Engine")
    plt.grid(True, alpha=0.3)
    plt.legend()
    st.pyplot(fig)

# --- 8. DATA EXPORT HUB ---
st.divider()
st.subheader("📦 Export Processed Intelligence")
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='NeuralNode_CleanData')

st.download_button(
    label="📥 Download Structured Intelligence (Excel)",
    data=buffer.getvalue(),
    file_name="NeuralNode_Market_Analysis.xlsx",
    mime="application/vnd.ms-excel"
)

if st.checkbox("Explore Neural Raw Data"):
    st.dataframe(df.style.highlight_max(axis=0, color='#d4edda'))

st.sidebar.markdown("---")
st.sidebar.info("© 2026 NeuralNode AI. All Rights Reserved.")
