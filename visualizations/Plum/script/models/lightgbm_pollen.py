import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LightGBMPollen:
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=42
        )
        self.trained = False
        self.y_test = None
        self.y_pred = None
        self.metrics = {}

    def short_description(self):
        return "LightGBM Regressor on raw target using early stopping. Features include basic weather/pollution, lags, rolling means, and spike indicators."

    def _engineer_features(self, data_dict):
        # 1. Merge Data
        w = data_dict["weather"].copy()
        p = data_dict["pollutants"].copy()
        pol = data_dict["pollen"].copy()

        w['Date'] = pd.to_datetime(w['time'])
        p['Date'] = pd.to_datetime(p['date'])
        pol['Date'] = pd.to_datetime(pol['Date'])

        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()

        # 2. Features matching Notebook Section 4
        target = "Total_Pollen"
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["lag1"] = df[target].shift(1)
        df["lag2"] = df[target].shift(2)
        df["lag3"] = df[target].shift(3)
        df["pollen_3day"] = df[target].rolling(3).mean()
        df["pollen_7day"] = df[target].rolling(7).mean()

        # Spike: > 75th percentile of recent 3-day mean
        # Note: Notebook logic was slightly complex, using a simplified version based on notebook cell 189
        df["is_spike"] = ((df[target] - df["pollen_3day"]) > df["pollen_3day"].quantile(0.75)).astype(int)

        df = df.dropna()
        return df

    def predict(self, data_dict):
        df = self._engineer_features(data_dict)
        target = "Total_Pollen"

        # Filter strictly numeric features available
        remove_cols = ["Date", "time", "date", "Month", "Year", "Day", "Week", 
                       "Tree_Level", "Grass_Level", "Weed_Level", "Ragweed_Level", 
                       "AQI_Category", "OBJECTID", "Tree", "Grass", "Weed", "Ragweed", target]
        
        feature_cols = [c for c in df.columns if c not in remove_cols and df[c].dtype in ['float64', 'int64', 'int32']]
        
        X = df[feature_cols]
        y = df[target]

        # Rename columns to avoid LightGBM JSON errors with special characters
        X.columns = ["".join (c if c.isalnum() else "_" for c in col) for col in X.columns]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50)]
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
        ax.scatter(self.y_test, self.y_pred, alpha=0.6, color='purple', label="Predictions")
        
        line_max = max(self.y_test.max(), self.y_pred.max())
        ax.plot([0, line_max], [0, line_max], 'k--', label="Perfect Fit")

        ax.set_xlabel("Actual Total Pollen")
        ax.set_ylabel("Predicted Total Pollen")
        ax.set_title(f"LightGBM Results\nR²: {self.metrics['R2']:.3f} | RMSE: {self.metrics['RMSE']:.1f}")
        ax.legend()
        fig.tight_layout()