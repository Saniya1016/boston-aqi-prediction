import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

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
        self.metrics = {}

    def short_description(self):
        return "Random Forest Regressor (n_estimators=600, max_depth=10)."

    def _build_features(self, df):
        # Consistent feature engineering
        df = df.sort_values("date").copy()
        
        # Lags
        df['AQI_lag_2'] = df['AQI'].shift(2)
        df['AQI_lag_3'] = df['AQI'].shift(3)
        df['AQI_roll3'] = df['AQI'].shift(1).rolling(3).mean()
        df['AQI_roll7'] = df['AQI'].shift(1).rolling(7).mean()
        df['AQI_diff_1'] = df['AQI'].shift(1) - df['AQI'].shift(2)
        df['AQI_diff_7'] = df['AQI'].shift(1) - df['AQI'].shift(7)
        
        for p in ["PM2.5", "O3", "CO", "NO2", "SO2"]:
            if p in df.columns:
                df[f'{p}_lag1'] = df[p].shift(1)
                df[f'{p}_lag3'] = df[p].shift(3)
                df[f"{p}_lag7"] = df[p].shift(7)
                df[f'{p}_roll3'] = df[p].shift(1).rolling(3).mean()
                df[f'{p}_roll7'] = df[p].shift(1).rolling(7).mean()
        
        weather_map = {"temperature_2m_mean (°C)": "temp", "wind_speed_10m_max (km/h)": "wind", "precipitation_sum (mm)": "rain"}
        for old, new in weather_map.items():
            if old in df.columns:
                df[f"{new}_lag1"] = df[old].shift(1)
                df[f"{new}_roll3"] = df[old].shift(1).rolling(3).mean()
                df[f"{new}_roll7"] = df[old].shift(1).rolling(7).mean()

        if "PM2.5_lag1" in df.columns and "wind_lag1" in df.columns: df["PM25_wind"] = df["PM2.5_lag1"] * df["wind_lag1"]
        if "O3_lag1" in df.columns and "temp_lag1" in df.columns: df["O3_temp"] = df["O3_lag1"] * df["temp_lag1"]
        if "NO2_lag1" in df.columns and "wind_lag1" in df.columns: df["NO2_wind"] = df["NO2_lag1"] * df["wind_lag1"]

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
        
        latest = df['date'].max()
        test_start = latest - pd.DateOffset(years=1)
        
        X_train = df[df['date'] < test_start][feats]
        y_train = df[df['date'] < test_start]["AQI"]
        X_test = df[df['date'] >= test_start][feats]
        y_test = df[df['date'] >= test_start]["AQI"]

        self.model.fit(X_train, y_train)
        self.trained = True
        
        preds = self.model.predict(X_test)
        
        self.test_data = df[df['date'] >= test_start].copy()
        self.test_data['actual_AQI'] = y_test
        self.test_data['predicted_AQI'] = preds
        self.metrics['R2'] = r2_score(y_test, preds)
        
        return self.test_data[['date', 'actual_AQI', 'predicted_AQI']]

    def plot_results(self, data_dict, fig):
        if not self.trained: return
        ax1, ax2 = fig.subplots(1, 2)
        
        ax1.scatter(self.test_data['actual_AQI'], self.test_data['predicted_AQI'], alpha=0.5, color='orange')
        lims = [0, 200]
        ax1.plot(lims, lims, 'r--')
        ax1.set_title(f"Random Forest (R²={self.metrics['R2']:.2f})")
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        
        ax2.plot(self.test_data['date'], self.test_data['actual_AQI'], label="Actual", color='gray', alpha=0.5)
        ax2.plot(self.test_data['date'], self.test_data['predicted_AQI'], label="RF", color='orange', alpha=0.8)
        ax2.set_title("Timeline")
        ax2.legend()
        plt.tight_layout()