import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import importlib

st.set_page_config(page_title="Boston AQI & Pollen Dashboard", layout="wide")

# Data Loader

def safe_read_open_meteo(path):
    # Skip the metadata block at the top
    df = pd.read_csv(path, skiprows=3)

    # Fix timestamp immediately
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.floor("s")

    return df


@st.cache_data
def load_data():
    base = Path("Data")

    pollutants = pd.read_csv(base / "boston_pollutants.csv")
    pollen = pd.read_csv(base / "boston_pollen.csv")
    weather = safe_read_open_meteo(base / "boston_weather.csv")

    # Sanitize datetime columns globally
    for df in [pollutants, pollen, weather]:
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.floor("s")

    return {
        "pollutants": pollutants,
        "weather": weather,
        "pollen": pollen
    }


# Load sanitized data
data = load_data()

MODEL_REGISTRY = {
    "Linear Regression AQI": {
        "module": "models.linear_regression",
        "class": "LinearRegressionAQI"
    },
    "XGBoost AQI": {
        "module": "models.xgboost_aqi",
        "class": "XGBoostAQI"
    },
    "LightGBM AQI": {
        "module": "models.lightgbm_aqi",
        "class": "LightGBMAQI"
    },
    "Random Forest AQI": {
        "module": "models.random_forest_aqi",
        "class": "RandomForestAQI"
    },
    "Ensemble AQI": {
        "module": "models.ensemble_aqi",
        "class": "EnsembleAQI"
    },
    "Random Forest Pollen": {
        "module": "models.random_forest_pollen",
        "class": "RandomForestPollen"
    },
    "LightGBM Pollen": {
        "module": "models.lightgbm_pollen",
        "class": "LightGBMPollen"
    },
    "XGBoost Pollen": {
        "module": "models.xgboost_pollen",
        "class": "XGBoostPollen"
    }
}

# Dynamically load a model class
def load_model(model_info):
    module = importlib.import_module(model_info["module"])
    return getattr(module, model_info["class"])()

# Sidebar Navigation
st.sidebar.title("🔬 Model Dashboard")
page = st.sidebar.radio(
    "Select page:",
    ["Home", "Models"],
)

# Home Page
if page == "Home":
    st.title("🌤️ Boston AQI & Pollen Prediction Dashboard")

    st.markdown("""
    This dashboard aggregates models contributed by team members.  
    Implement a model inside **`models/`** and register it in the dict.
    """)

    st.header("📁 Loaded Data Samples")
    tabs = st.tabs(["Pollutants", "Weather", "Pollen"])

    with tabs[0]:
        st.dataframe(data["pollutants"].head())
    with tabs[1]:
        st.dataframe(data["weather"].head())
    with tabs[2]:
        st.dataframe(data["pollen"].head())

# Models Page
if page == "Models":

    st.title("🤖 Model Explorer")

    selected = st.selectbox(
        "Choose a model:",
        list(MODEL_REGISTRY.keys())
    )

    model_info = MODEL_REGISTRY[selected]
    model_instance = load_model(model_info)

    st.subheader(f"📌 {selected}")
    st.write(model_instance.short_description())

    # Generate predictions
    predictions = model_instance.predict(data)

    # Visualization Section (Moved to Top & Made Interactive)
    st.header("📊 Interactive Visualization")

    # Check if predictions df has the expected structure
    # We expect at least: 'date', 'actual', 'predicted' (names might vary slightly based on model)
    # We will try to detect them flexibly.
    
    cols = predictions.columns
    # Basic logic to identify columns for plotting
    date_col = next((c for c in cols if 'date' in c.lower() or 'time' in c.lower()), None)
    actual_col = next((c for c in cols if 'actual' in c.lower() or 'true' in c.lower()), None)
    pred_col = next((c for c in cols if 'pred' in c.lower()), None)

    if date_col and actual_col and pred_col:
        
        tab1, tab2 = st.tabs(["Time Series", "Actual vs Predicted"])

        with tab1:
            st.subheader("Time Series Analysis")
            # Interactive Time Series Plot using Plotly
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=predictions[date_col], y=predictions[actual_col],
                                mode='lines', name='Actual', line=dict(color='gray', width=1)))
            fig_ts.add_trace(go.Scatter(x=predictions[date_col], y=predictions[pred_col],
                                mode='lines', name='Predicted', line=dict(color='blue', width=2)))
            
            fig_ts.update_layout(
                title="Forecast vs Actual Over Time",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig_ts, width='stretch')

        with tab2:
            st.subheader("Scatter Analysis")
            # Interactive Scatter Plot
            fig_sc = px.scatter(predictions, x=actual_col, y=pred_col, 
                                title="Actual vs Predicted",
                                labels={actual_col: "Actual Value", pred_col: "Predicted Value"},
                                opacity=0.6)
            
            # Add a perfect prediction line (y=x)
            min_val = min(predictions[actual_col].min(), predictions[pred_col].min())
            max_val = max(predictions[actual_col].max(), predictions[pred_col].max())
            
            fig_sc.add_shape(
                type="line",
                x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash"),
            )
            st.plotly_chart(fig_sc, width='stretch')

    else:
        # Fallback to model's internal matplotlib plot if columns aren't standard
        fig = model_instance.plot_results(data)
        st.plotly_chart(fig, width='stretch')

    # Predictions Data Section
    st.header("📈 Prediction Data")
    st.dataframe(predictions.head())
    
    with st.expander("See full data statistics"):
        st.write(predictions.describe())