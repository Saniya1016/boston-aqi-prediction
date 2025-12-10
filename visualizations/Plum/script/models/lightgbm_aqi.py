# file: lightgbm_aqi.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LightGBMAQI:
    def __init__(self):
        self.model = LGBMRegressor(
            num_leaves=31,
            learning_rate=0.01,
            n_estimators=900,
            subsample=0.7,
            colsample_bytree=1.0,
            objective="regression",
            random_state=42,
            n_jobs=-1
        )
        self.trained = False
        self.test_data = None
        self.metrics = {}

    def short_description(self):
        return "LightGBM Regressor for AQI (leaves=31, lr=0.01, n_estimators=900)."

    def _build_features(self, df):
        df = df.sort_values("date").copy()
        
        df['AQI_lag_2'] = df['AQI'].shift(2)
        df['AQI_lag_3'] = df['AQI'].shift(3)
        df['AQI_roll3'] = df['AQI'].shift(1).rolling(3).mean()
        df['AQI_roll7'] = df['AQI'].shift(1).rolling(7).mean()
        df['AQI_diff_1'] = df['AQI'].shift(1) - df['AQI'].shift(2)
        df['AQI_diff_7'] = df['AQI'].shift(1) - df['AQI'].shift(7)
        
        pollutants = ["PM2.5", "O3", "CO", "NO2", "SO2"]
        for p in pollutants:
            if p in df.columns:
                df[f'{p}_lag1'] = df[p].shift(1)
                df[f'{p}_lag3'] = df[p].shift(3)
                df[f"{p}_lag7"] = df[p].shift(7)
                df[f'{p}_roll3'] = df[p].shift(1).rolling(3).mean()
                df[f'{p}_roll7'] = df[p].shift(1).rolling(7).mean()
        
        weather_cols = {
            "temperature_2m_mean (°C)": "temp",
            "wind_speed_10m_max (km/h)": "wind",
            "precipitation_sum (mm)": "rain"
        }
        for old, base in weather_cols.items():
            if old in df.columns:
                df[f"{base}_lag1"] = df[old].shift(1)
                df[f"{base}_roll3"] = df[old].shift(1).rolling(3).mean()
                df[f"{base}_roll7"] = df[old].shift(1).rolling(7).mean()

        if "PM2.5_lag1" in df.columns and "wind_lag1" in df.columns:
            df["PM25_wind"] = df["PM2.5_lag1"] * df["wind_lag1"]
        if "O3_lag1" in df.columns and "temp_lag1" in df.columns:
            df["O3_temp"] = df["O3_lag1"] * df["temp_lag1"]
        if "NO2_lag1" in df.columns and "wind_lag1" in df.columns:
            df["NO2_wind"] = df["NO2_lag1"] * df["wind_lag1"]

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
        weather_date_col = 'time' if 'time' in weather.columns else 'date'
        weather[weather_date_col] = pd.to_datetime(weather[weather_date_col])
        weather = weather.rename(columns={weather_date_col: 'date'})
        
        df = pd.merge(pollutants, weather, on='date', how='inner')
        df_featured = self._build_features(df)

        exclude = [
            "date", "time", "date_local", "num_pollutants_available",
            "AQI", "AQI_Category", "AQI_PM2.5", "AQI_O3", "AQI_CO", "AQI_NO2", "AQI_SO2",
            "PM2.5", "O3", "CO", "NO2", "SO2",
            "temperature_2m_mean (°C)", "precipitation_sum (mm)",
            "wind_speed_10m_max (km/h)", "wind_gusts_10m_max (km/h)",
            "apparent_temperature_mean (°C)", "wind_direction_10m_dominant (°)"
        ]
        feature_cols = [c for c in df_featured.columns if c not in exclude]
        
        latest_date = df_featured['date'].max()
        test_start = latest_date - pd.DateOffset(years=1)
        
        train_val_df = df_featured[df_featured['date'] < test_start].copy()
        test_df = df_featured[df_featured['date'] >= test_start].copy()
        
        X_train = train_val_df[feature_cols]
        y_train = train_val_df["AQI"]
        X_test = test_df[feature_cols]
        y_test = test_df["AQI"]

        self.model.fit(X_train, y_train)
        self.trained = True

        preds = self.model.predict(X_test)
        
        self.test_data = test_df.copy()
        self.test_data['AQI_Predicted'] = preds
        self.test_data['AQI_Observed'] = y_test
        
        self.metrics['MAE'] = mean_absolute_error(y_test, preds)
        self.metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, preds))
        self.metrics['R2'] = r2_score(y_test, preds)

        return self.test_data[['date', 'AQI_Observed', 'AQI_Predicted']]

    def plot_results(self, data_dict=None):
        if not self.trained:
            fig = go.Figure()
            fig.add_annotation(text="Model not trained.", x=0.5, y=0.5, showarrow=False)
            return fig

        # UPDATED: 2 rows, 1 col
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"Actual vs Predicted (R²={self.metrics['R2']:.2f})", "LightGBM Time Series"),
            vertical_spacing=0.15
        )

        y_test = self.test_data['AQI_Observed']
        preds = self.test_data['AQI_Predicted']
        
        # 1. Scatter (Row 1)
        fig.add_trace(
            go.Scatter(x=y_test, y=preds, mode='markers', name='Predictions', marker=dict(color='green', opacity=0.5, size=6)),
            row=1, col=1
        )
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')),
            row=1, col=1
        )

        # 2. Time Series (Row 2)
        fig.add_trace(
            go.Scatter(x=self.test_data['date'], y=y_test, mode='lines', name='Actual', line=dict(color='gray', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.test_data['date'], y=preds, mode='lines', name='Predicted', line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        # UPDATED: showlegend=True, height=900
        fig.update_layout(template="plotly_white", showlegend=True, height=900)
        return fig