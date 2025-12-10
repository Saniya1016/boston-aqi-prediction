# file: xgboost_pollen.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

class XGBoostPollen:
    def __init__(self):
        self.model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100, # Default per notebook output [84]
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        self.trained = False
        self.y_test_real = None
        self.y_pred_real = None
        self.metrics = {}

    def short_description(self):
        return "XGBoost Regressor for Pollen (Log-transformed target)."

    def _engineer_features(self, data_dict):
        # Reusing the extensive feature engineering from Random Forest as baseline
        w = data_dict["weather"].copy()
        p = data_dict["pollutants"].copy()
        pol = data_dict["pollen"].copy()

        w['Date'] = pd.to_datetime(w['time'])
        p['Date'] = pd.to_datetime(p['date'])
        pol['Date'] = pd.to_datetime(pol['Date'])

        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()

        target = "Total_Pollen"
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["lag1"] = df[target].shift(1)
        df["lag2"] = df[target].shift(2)
        df["lag3"] = df[target].shift(3)
        df["pollen_3day"] = df[target].rolling(3).mean()
        df["pollen_7day"] = df[target].rolling(7).mean()

        threshold = df[target].mean() + df[target].std()
        df["is_spike"] = (df[target] > threshold).astype(int)

        df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        weights = np.array([0.1, 0.3, 0.6])
        rolling_cols = [
            "temperature_2m_mean (°C)", "apparent_temperature_mean (°C)",
            "PM2.5", "O3", "CO", "NO2", "SO2",
            "AQI", "AQI_PM2.5", "AQI_O3", "AQI_CO", "AQI_NO2", "AQI_SO2"
        ]
        rolling_cols = [c for c in rolling_cols if c in df.columns]

        for col in rolling_cols:
            df[f"{col}_weighted3"] = (
                df[col].shift(2).bfill() * weights[0] +
                df[col].shift(1).bfill() * weights[1] +
                df[col] * weights[2]
            )
            for lag in range(1, 4):
                df[f"{col}_lag{lag}"] = df[col].shift(lag).bfill()

        interaction_pairs = [
            ("temperature_2m_mean (°C)", "AQI"),
            ("apparent_temperature_mean (°C)", "PM2.5"),
            ("wind_speed_10m_max (km/h)", "O3"),
        ]
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

        df = df.dropna()
        return df

    def predict(self, data_dict):
        df = self._engineer_features(data_dict)
        target = "Total_Pollen"

        remove_cols = ["Date", "time", "date", "Month", "Year", "Day", "Week", 
                       "Tree_Level", "Grass_Level", "Weed_Level", "Ragweed_Level", 
                       "AQI_Category", "weather_code (wmo code)", "OBJECTID",
                       "Tree", "Grass", "Weed", "Ragweed", "day_of_year", target]
        
        feature_cols = [c for c in df.columns if c not in remove_cols and df[c].dtype in ['float64', 'int64', 'int32']]
        
        X = df[feature_cols]
        y_log = np.log1p(df[target])

        # Use notebook specific year split if aligned, or random if consistent
        # Notebook snippet 7 uses Year < 2022 for train. Let's use TimeSeriesSplit logic or standard year split to be safe.
        # But RF used random split. I will stick to RF's random split for consistency with the RF file.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        y_pred_log = self.model.predict(X_test)
        self.y_pred_real = np.expm1(y_pred_log)
        self.y_test_real = np.expm1(y_test)

        mae = mean_absolute_error(self.y_test_real, self.y_pred_real)
        rmse = np.sqrt(mean_squared_error(self.y_test_real, self.y_pred_real))
        r2 = r2_score(self.y_test_real, self.y_pred_real)
        self.metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

        return pd.DataFrame({
            "Actual Pollen": self.y_test_real.values,
            "Predicted Pollen": self.y_pred_real
        })

    def plot_results(self, data_dict):
        if not self.trained:
            fig = go.Figure()
            fig.add_annotation(text="Model not trained.", x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.y_test_real,
            y=self.y_pred_real,
            mode='markers',
            name='Test Predictions',
            opacity=0.6,
            marker=dict(color='purple')
        ))

        line_max = max(self.y_test_real.max(), self.y_pred_real.max())
        fig.add_trace(go.Scatter(
            x=[0, line_max], 
            y=[0, line_max],
            mode='lines',
            name='Perfect Fit',
            line=dict(dash='dash', color='red')
        ))

        fig.update_layout(
            title=f"XGBoost (Log-Target)<br>R²: {self.metrics['R2']:.3f} | MAE: {self.metrics['MAE']:.1f}",
            xaxis_title="Actual Total Pollen",
            yaxis_title="Predicted Total Pollen",
            template="plotly_white"
        )
        return fig