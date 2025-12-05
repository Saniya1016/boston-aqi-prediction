import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class EnsembleAQI:
    def __init__(self):
        # Initialize all three models
        self.xgb = XGBRegressor(
            objective="reg:squarederror", random_state=42, n_jobs=-1,
            n_estimators=800, max_depth=4, learning_rate=0.03,
            subsample=1.0, colsample_bytree=0.7, reg_lambda=5.0
        )
        self.lgbm = LGBMRegressor(
            num_leaves=31, learning_rate=0.01, n_estimators=900,
            subsample=0.7, colsample_bytree=1.0, objective="regression",
            random_state=42, n_jobs=-1
        )
        self.rf = RandomForestRegressor(
            n_estimators=600, max_depth=10, min_samples_split=2,
            random_state=42, n_jobs=-1
        )
        self.trained = False
        self.test_data = None
        self.metrics = {}

    def short_description(self):
        return "Ensemble (Average) of XGBoost, LightGBM, and Random Forest models."

    def _build_features(self, df):
        df = df.sort_values("date").copy()
        
        # AQI Lags
        df['AQI_lag_2'] = df['AQI'].shift(2)
        df['AQI_lag_3'] = df['AQI'].shift(3)
        df['AQI_roll3'] = df['AQI'].shift(1).rolling(3).mean()
        df['AQI_roll7'] = df['AQI'].shift(1).rolling(7).mean()
        df['AQI_diff_1'] = df['AQI'].shift(1) - df['AQI'].shift(2)
        df['AQI_diff_7'] = df['AQI'].shift(1) - df['AQI'].shift(7)
        
        # Pollutants
        for p in ["PM2.5", "O3", "CO", "NO2", "SO2"]:
            if p in df.columns:
                df[f'{p}_lag1'] = df[p].shift(1)
                df[f'{p}_lag3'] = df[p].shift(3)
                df[f"{p}_lag7"] = df[p].shift(7)
                df[f'{p}_roll3'] = df[p].shift(1).rolling(3).mean()
                df[f'{p}_roll7'] = df[p].shift(1).rolling(7).mean()
        
        # Weather
        weather_map = {"temperature_2m_mean (°C)": "temp", "wind_speed_10m_max (km/h)": "wind", "precipitation_sum (mm)": "rain"}
        for old, new in weather_map.items():
            if old in df.columns:
                df[f"{new}_lag1"] = df[old].shift(1)
                df[f"{new}_roll3"] = df[old].shift(1).rolling(3).mean()
                df[f"{new}_roll7"] = df[old].shift(1).rolling(7).mean()

        # Interactions
        if "PM2.5_lag1" in df.columns and "wind_lag1" in df.columns: df["PM25_wind"] = df["PM2.5_lag1"] * df["wind_lag1"]
        if "O3_lag1" in df.columns and "temp_lag1" in df.columns: df["O3_temp"] = df["O3_lag1"] * df["temp_lag1"]
        if "NO2_lag1" in df.columns and "wind_lag1" in df.columns: df["NO2_wind"] = df["NO2_lag1"] * df["wind_lag1"]

        # Time
        df["month"] = df["date"].dt.month
        df["dayofyear"] = df["date"].dt.dayofyear
        df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
        df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
        df["doy_sin"] = np.sin(2*np.pi*df["dayofyear"]/365)
        df["doy_cos"] = np.cos(2*np.pi*df["dayofyear"]/365)
        df["dayofweek"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["dow_sin"] = np.sin(2*np.pi*df["dayofweek"]/7)
        df["dow_cos"] = np.cos(2*np.pi*df["dayofweek"]/7)
        
        return df.dropna()

    def predict(self, data_dict):
        pollutants = data_dict["pollutants"].copy()
        weather = data_dict["weather"].copy()
        pollutants['date'] = pd.to_datetime(pollutants['date'])
        w_date = 'time' if 'time' in weather.columns else 'date'
        weather[w_date] = pd.to_datetime(weather[w_date])
        weather = weather.rename(columns={w_date: 'date'})
        df = pd.merge(pollutants, weather, on='date', how='inner')
        
        df = self._build_features(df)

        exclude = ["date", "time", "date_local", "num_pollutants_available", "AQI", "AQI_Category", 
                   "AQI_PM2.5", "AQI_O3", "AQI_CO", "AQI_NO2", "AQI_SO2", "PM2.5", "O3", "CO", "NO2", "SO2",
                   "temperature_2m_mean (°C)", "precipitation_sum (mm)", "wind_speed_10m_max (km/h)", 
                   "wind_gusts_10m_max (km/h)", "apparent_temperature_mean (°C)", "wind_direction_10m_dominant (°)"]
        
        feats = [c for c in df.columns if c not in exclude]
        
        # Chronological Split
        latest = df['date'].max()
        test_start = latest - pd.DateOffset(years=1)
        
        X_train = df[df['date'] < test_start][feats]
        y_train = df[df['date'] < test_start]["AQI"]
        X_test = df[df['date'] >= test_start][feats]
        y_test = df[df['date'] >= test_start]["AQI"]

        # Train all 3
        self.xgb.fit(X_train, y_train)
        self.lgbm.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)
        self.trained = True
        
        # Predict all 3 and average
        pred_xgb = self.xgb.predict(X_test)
        pred_lgbm = self.lgbm.predict(X_test)
        pred_rf = self.rf.predict(X_test)
        
        final_preds = (pred_xgb + pred_lgbm + pred_rf) / 3
        
        self.test_data = df[df['date'] >= test_start].copy()
        self.test_data['actual_AQI'] = y_test
        self.test_data['predicted_AQI'] = final_preds
        self.metrics['MAE'] = mean_absolute_error(y_test, final_preds)
        self.metrics['R2'] = r2_score(y_test, final_preds)
        
        return self.test_data[['date', 'actual_AQI', 'predicted_AQI']]

    def plot_results(self, data_dict=None):
        if not self.trained:
            fig = go.Figure()
            fig.add_annotation(text="Model not trained.", x=0.5, y=0.5, showarrow=False)
            return fig

        df = self.test_data

        # Create subplot layout: 1 row, 2 columns
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Ensemble Model<br>MAE={self.metrics['MAE']:.2f}, R²={self.metrics['R2']:.2f}",
                "Actual vs Predicted AQI Over Time"
            )
        )

        # -------------------------------
        # SCATTER PLOT (Left)
        # -------------------------------
        fig.add_trace(
            go.Scatter(
                x=df["actual_AQI"], 
                y=df["predicted_AQI"],
                mode='markers',
                name="Predictions",
                opacity=0.6
            ),
            row=1, col=1
        )

        # Diagonal reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 200],
                y=[0, 200],
                mode='lines',
                name="Perfect Fit",
                line=dict(dash='dash', color='red')
            ),
            row=1, col=1
        )

        fig.update_xaxes(title="Actual", row=1, col=1)
        fig.update_yaxes(title="Predicted", row=1, col=1)

        # -------------------------------
        # TIME SERIES PLOT (Right)
        # -------------------------------
        fig.add_trace(
            go.Scatter(
                x=df["date"], 
                y=df["actual_AQI"],
                mode='lines',
                name="Actual",
                line=dict(width=2, color='gray')
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=df["date"], 
                y=df["predicted_AQI"],
                mode='lines',
                name="Ensemble Prediction",
                line=dict(width=2, color='purple')
            ),
            row=1, col=2
        )

        fig.update_xaxes(title="Date", row=1, col=2)
        fig.update_yaxes(title="AQI", row=1, col=2)

        # -------------------------------
        # Layout & Theme
        # -------------------------------
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            width=1200,
            height=500
        )

        return fig