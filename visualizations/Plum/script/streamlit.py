import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

st.set_page_config(page_title="Boston AQI & Pollen Dashboard", layout="wide")

# =========================================================
# Data Loader
# =========================================================

def safe_read_open_meteo(path):
    # Skip the metadata block at the top
    return pd.read_csv(path, skiprows=3)

@st.cache_data
def load_data():
    base = Path("data")

    return {
        "pollutants": pd.read_csv(base / "boston_pollutants_with_aqi.csv"),
        "weather": safe_read_open_meteo(base / "boston-weather-data(open_meteo).csv"),
        "pollen": pd.read_csv(base / "EPHT_Pollen_Data.csv")
    }

data = load_data()

# =========================================================
# Model Registry — teammates add their models here
# =========================================================
# Format:
# "Model Name": {
#     "module": "models.model_filename",
#     "class": "ModelClassInside"
# }

MODEL_REGISTRY = {
    "Linear Regression AQI": {
        "module": "models.linear_regression",
        "class": "LinearRegressionAQI"
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

# =========================================================
# Sidebar Navigation
# =========================================================
st.sidebar.title("🔬 Model Dashboard")
page = st.sidebar.radio(
    "Select page:",
    ["Home", "Models"],
)

# =========================================================
# Home Page
# =========================================================
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

# =========================================================
# Models Page
# =========================================================
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

    # Placeholder model prediction example
    st.header("📈 Predictions (Example)")

    # Let the model generate predictions (even dummy)
    predictions = model_instance.predict(data)

    st.dataframe(predictions.head())

    # Plot section — model may override
    st.header("📊 Visualization")

    fig = plt.figure(figsize=(8,4))
    model_instance.plot_results(data, fig)
    st.pyplot(fig)
