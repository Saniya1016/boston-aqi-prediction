# file: random_forest_aqi.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Reusing the structure from xgboost_aqi.py

class RandomForestAQI:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=600,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        self.trained = False
        self.test_data = None
        self.predictions = None
        self.metrics = {}

    def short_description(self):
        return "Random Forest Regressor for AQI (n_estimators=600, max_depth=10)."

    def _build_features(self, df):
        # Same feature engineering logic as XGBoostAQI
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
        self.test_data['predicted_AQI'] = preds
        self.test_data['actual_AQI'] = y_test
        
        self.metrics['MAE'] = mean_absolute_error(y_test, preds)
        self.metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, preds))
        self.metrics['R2'] = r2_score(y_test, preds)

        return self.test_data[['date', 'actual_AQI', 'predicted_AQI']]

    def plot_results(self, data_dict, fig):
        if not self.trained:
            plt.text(0.5, 0.5, "Model training failed or not run.", ha="center")
            return

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        y_test = self.test_data['actual_AQI']
        preds = self.test_data['predicted_AQI']
        
        ax1.scatter(y_test, preds, alpha=0.5, color='green', s=10)
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_title(f"Actual vs Predicted (R²={self.metrics['R2']:.2f})")
        ax1.set_xlabel("Actual AQI")
        ax1.set_ylabel("Predicted AQI")
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.test_data['date'], y_test, label="Actual", color='gray', alpha=0.7)
        ax2.plot(self.test_data['date'], preds, label="Predicted", color='green', alpha=0.7)
        ax2.set_title("AQI Forecast Over Time (Test Set)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("AQI")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()