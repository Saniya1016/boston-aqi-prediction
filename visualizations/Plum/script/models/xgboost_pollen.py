import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class XGBoostPollen:
    def __init__(self):
        # Using sklearn API wrapper for easier integration
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=1200,
            early_stopping_rounds=50,
            eval_metric='rmse',
            seed=42
        )
        self.trained = False
        self.y_test = None
        self.y_pred = None
        self.metrics = {}

    def short_description(self):
        return "XGBoost Regressor using extensive feature engineering (Seasonality, Weighted Rolling, Lags, Interactions)."

    def _engineer_features(self, data_dict):
        # 1. Merge
        w = data_dict["weather"].copy()
        p = data_dict["pollutants"].copy()
        pol = data_dict["pollen"].copy()

        w['Date'] = pd.to_datetime(w['time'])
        p['Date'] = pd.to_datetime(p['date'])
        pol['Date'] = pd.to_datetime(pol['Date'])

        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()

        # 2. Features matching Notebook Section 5
        target = "Total_Pollen"
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["lag1"] = df[target].shift(1)
        df["lag2"] = df[target].shift(2)
        df["lag3"] = df[target].shift(3)
        df["pollen_3day"] = df[target].rolling(3).mean()
        df["pollen_7day"] = df[target].rolling(7).mean()

        threshold = df[target].mean() + df[target].std()
        df["is_spike"] = (df[target] > threshold).astype(int)

        # Seasonal
        df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # Weighted rolling + Lags
        weights = np.array([0.1, 0.3, 0.6])
        rolling_cols = [
            "temperature_2m_mean (°C)", "apparent_temperature_mean (°C)",
            "PM2.5", "O3", "CO", "NO2", "SO2",
            "AQI_PM2.5", "AQI_O3", "AQI_CO", "AQI_NO2", "AQI_SO2", "AQI"
        ]
        rolling_cols = [c for c in rolling_cols if c in df.columns]

        for col in rolling_cols:
            df[f'{col}_weighted3'] = (
                df[col].shift(2).bfill() * weights[0] +
                df[col].shift(1).bfill() * weights[1] +
                df[col] * weights[2]
            )
            for lag in range(1, 4):
                df[f"{col}_lag{lag}"] = df[col].shift(lag).bfill()

        # Interactions
        if 'temperature_2m_mean (°C)' in df.columns and 'AQI' in df.columns:
            df["temp_x_aqi"] = df['temperature_2m_mean (°C)'] * df['AQI']
        if 'apparent_temperature_mean (°C)' in df.columns and 'PM2.5' in df.columns:
            df["app_temp_x_pm25"] = df['apparent_temperature_mean (°C)'] * df['PM2.5']
        if 'wind_speed_10m_max (km/h)' in df.columns and 'O3' in df.columns:
            df["wind_x_o3"] = df['wind_speed_10m_max (km/h)'] * df['O3']

        df = df.dropna()
        return df

    def predict(self, data_dict):
        df = self._engineer_features(data_dict)
        target = "Total_Pollen"

        drop_cols = ["Date", "time", "date", "Month", "Year", "Day", "Week", 
                     "Tree_Level", "Grass_Level", "Weed_Level", "Ragweed_Level", 
                     "AQI_Category", "OBJECTID", "Tree", "Grass", "Weed", "Ragweed", target]
        
        feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in ['float64', 'int64', 'int32']]
        
        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        self.trained = True
        
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2 = r2_score(self.y_test, self.y_pred)
        self.metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

        return pd.DataFrame({
            "Actual Pollen": self.y_test.values,
            "Predicted Pollen": self.y_pred
        })

    def plot_results(self, data_dict, fig):
        if not self.trained:
            plt.text(0.5, 0.5, "Model not trained.", ha="center")
            return

        ax = fig.add_subplot(111)
        ax.scatter(self.y_test, self.y_pred, alpha=0.6, color='orange', label="Predictions")
        
        line_max = max(self.y_test.max(), self.y_pred.max())
        ax.plot([0, line_max], [0, line_max], 'k--', label="Perfect Fit")

        ax.set_xlabel("Actual Total Pollen")
        ax.set_ylabel("Predicted Total Pollen")
        ax.set_title(f"XGBoost Results\nR²: {self.metrics['R2']:.3f} | RMSE: {self.metrics['RMSE']:.1f}")
        ax.legend()
        fig.tight_layout()