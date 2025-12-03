import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RandomForestPollen:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=4,
            max_features=None,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        self.trained = False
        self.y_test_real = None
        self.y_pred_real = None
        self.metrics = {}

    def short_description(self):
        return "Random Forest Regressor predicting Total_Pollen using Log-Transformed target, seasonal features (sin/cos), weighted rolling averages, and interaction terms."

    def _engineer_features(self, data_dict):
        # 1. Merge Data
        w = data_dict["weather"].copy()
        p = data_dict["pollutants"].copy()
        pol = data_dict["pollen"].copy()

        # Standardize Date columns
        w['Date'] = pd.to_datetime(w['time'])
        p['Date'] = pd.to_datetime(p['date'])
        pol['Date'] = pd.to_datetime(pol['Date'])

        # Merge
        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')

        # Filter Season (March - Oct)
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()

        # 2. Base Features
        df["day_of_year"] = df["Date"].dt.dayofyear
        
        # Pollen Lags/Rolling
        target = "Total_Pollen"
        df["lag1"] = df[target].shift(1)
        df["lag2"] = df[target].shift(2)
        df["lag3"] = df[target].shift(3)
        df["pollen_3day"] = df[target].rolling(3).mean()
        df["pollen_7day"] = df[target].rolling(7).mean()

        # Spike detection
        threshold = df[target].mean() + df[target].std()
        df["is_spike"] = (df[target] > threshold).astype(int)

        # 3. Seasonal Features
        df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # 4. Weighted Rolling & Lags for Pollutants/Weather
        weights = np.array([0.1, 0.3, 0.6])
        rolling_cols = [
            "temperature_2m_mean (°C)", "apparent_temperature_mean (°C)",
            "PM2.5", "O3", "CO", "NO2", "SO2",
            "AQI_PM2.5", "AQI_O3", "AQI_CO", "AQI_NO2", "AQI_SO2", "AQI"
        ]

        # Ensure columns exist before processing
        rolling_cols = [c for c in rolling_cols if c in df.columns]

        for col in rolling_cols:
            # Weighted Rolling
            df[f"{col}_weighted3"] = (
                df[col].shift(2).bfill() * weights[0] +
                df[col].shift(1).bfill() * weights[1] +
                df[col] * weights[2]
            )
            # Lags
            for lag in range(1, 4):
                df[f"{col}_lag{lag}"] = df[col].shift(lag).bfill()

        # 5. Interactions
        if 'temperature_2m_mean (°C)' in df.columns and 'AQI' in df.columns:
            df["temp_x_aqi"] = df['temperature_2m_mean (°C)'] * df['AQI']
        if 'apparent_temperature_mean (°C)' in df.columns and 'PM2.5' in df.columns:
            df["app_temp_x_pm25"] = df['apparent_temperature_mean (°C)'] * df['PM2.5']
        if 'wind_speed_10m_max (km/h)' in df.columns and 'O3' in df.columns:
            df["wind_x_o3"] = df['wind_speed_10m_max (km/h)'] * df['O3']

        # Cleanup
        df = df.dropna()
        return df

    def predict(self, data_dict):
        df = self._engineer_features(data_dict)
        
        target = "Total_Pollen"
        # Drop non-numeric/identifier columns for X
        drop_cols = ["Date", "time", "date", "Month", "Year", "Day", "Week", 
                     "Tree_Level", "Grass_Level", "Weed_Level", "Ragweed_Level", 
                     "AQI_Category", "weather_code (wmo code)", "OBJECTID",
                     "Tree", "Grass", "Weed", "Ragweed", target]
        
        feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in ['float64', 'int64', 'int32']]
        
        X = df[feature_cols]
        # Log Transform Target (as per notebook)
        y_log = np.log1p(df[target])

        X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        self.trained = True

        # Predict and Inverse Log Transform
        y_pred_log = self.model.predict(X_test)
        self.y_pred_real = np.expm1(y_pred_log)
        self.y_test_real = np.expm1(y_test)

        # Calculate Metrics
        mae = mean_absolute_error(self.y_test_real, self.y_pred_real)
        rmse = np.sqrt(mean_squared_error(self.y_test_real, self.y_pred_real))
        r2 = r2_score(self.y_test_real, self.y_pred_real)
        self.metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

        return pd.DataFrame({
            "Actual Pollen": self.y_test_real.values,
            "Predicted Pollen": self.y_pred_real
        })

    def plot_results(self, data_dict, fig):
        if not self.trained:
            plt.text(0.5, 0.5, "Model not trained.", ha="center")
            return

        ax = fig.add_subplot(111)
        ax.scatter(self.y_test_real, self.y_pred_real, alpha=0.6, color='green', label="Predictions")
        
        # Perfect fit line
        line_max = max(self.y_test_real.max(), self.y_pred_real.max())
        ax.plot([0, line_max], [0, line_max], 'k--', label="Perfect Fit")

        ax.set_xlabel("Actual Total Pollen")
        ax.set_ylabel("Predicted Total Pollen")
        ax.set_title(f"Random Forest (Log-Transformed) Results\nR²: {self.metrics['R2']:.3f} | RMSE: {self.metrics['RMSE']:.1f}")
        ax.legend()
        fig.tight_layout()