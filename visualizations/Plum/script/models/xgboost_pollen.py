import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from functools import reduce

class XGBoostPollen:
    def __init__(self):
        self.params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'eval_metric': 'rmse',
            'seed': 42
        }
        self.num_boost_round = 1200
        self.early_stopping_rounds = 50
        
        self.model = None
        self.trained = False
        self.test_data = None
        self.metrics = {}
        self.YEAR_SPLIT = 2023

        # --- FIX: EXPLICIT FEATURE LIST (Pre-Normalized for XGBoost) ---
        # XGBoost code typically normalizes spaces/parens, so we match that here.
        self.EXPLICIT_NUMERIC_COLS = [
            'temperature_2m_mean_°C', 'apparent_temperature_mean_°C', 
            'precipitation_sum_mm', 'wind_gusts_10m_max_km_h', 
            'wind_speed_10m_max_km_h', 'wind_direction_10m_dominant_°', 
            'PM2.5', 'O3', 'CO', 'NO2', 'SO2', 
            'AQI_PM2.5', 'AQI_O3', 'AQI_CO', 'AQI_NO2', 'AQI_SO2', 'AQI', 
            'num_pollutants_available'
        ]

    def short_description(self):
        return "XGBoost Regressor | Features: Weighted Rolling, Interactions (Fixed Features)"

    def _merge_data(self, data_dict):
        dfs = []
        if "pollen" in data_dict:
            df_p = data_dict["pollen"].copy()
            df_p = df_p.rename(columns={"Date": "date", "time": "date"})
            dfs.append(df_p)
        if "weather" in data_dict:
            df_w = data_dict["weather"].copy()
            df_w = df_w.rename(columns={"time": "date"})
            dfs.append(df_w)
        if "pollutants" in data_dict:
            df_pol = data_dict["pollutants"].copy()
            df_pol = df_pol.rename(columns={"Date": "date", "time": "date"})
            dfs.append(df_pol)

        if not dfs:
            raise RuntimeError("No data found in data_dict")

        df_merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), dfs)
        return df_merged

    def _normalize_columns(self, df):
        df.columns = (
            df.columns
            .str.replace(" ", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("/", "_", regex=False)
        )
        if "date" in df.columns:
            df = df.rename(columns={"date": "Date"})
        return df

    def _build_features(self, df):
        df = df.sort_values("Date").reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Year"] = df["Date"].dt.year
        
        # 1. USE EXPLICIT NUMERIC COLS
        numeric_cols = [c for c in self.EXPLICIT_NUMERIC_COLS if c in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Target 
        target = "Total_Pollen"
        
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["lag1"] = df[target].shift(1).ffill()
        df["lag2"] = df[target].shift(2).ffill()
        df["lag3"] = df[target].shift(3).ffill()

        df["pollen_3day"] = df[target].shift(1).rolling(3).mean().ffill()
        df["pollen_7day"] = df[target].shift(1).rolling(7).mean().ffill()

        df["is_spike"] = (df["lag1"] > df["pollen_3day"]).astype(int)
        df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # Weighted Rolling
        weights = np.array([0.1, 0.3, 0.6])
        rolling_cols = [
            'temperature_2m_mean_°C', 'apparent_temperature_mean_°C',
            'PM2.5', 'O3', 'CO', 'NO2', 'SO2',
            'AQI_PM2.5', 'AQI_O3', 'AQI_CO', 'AQI_NO2', 'AQI_SO2', 'AQI'
        ]
        
        for col in rolling_cols:
            if col in df.columns:
                df[f'{col}_weighted3'] = (
                    df[col].shift(3).ffill() * weights[0] +
                    df[col].shift(2).ffill() * weights[1] +
                    df[col].shift(1).ffill() * weights[2]
                )
                for lag in range(1, 4):
                    df[f"{col}_lag{lag}"] = df[col].shift(lag).ffill()

        # Interactions
        interaction_pairs = [
            ('temperature_2m_mean_°C', 'AQI'),
            ('apparent_temperature_mean_°C', 'PM2.5'),
            ('wind_speed_10m_max_km_h', 'O3'),
        ]
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

        # Drop NaNs
        required_lags = ["lag1", "lag2", "lag3", "pollen_3day", "pollen_7day"]
        df = df.dropna(subset=required_lags).reset_index(drop=True)

        # Define Features
        base_features = numeric_cols 
        extra_features = [
            "lag1", "lag2", "lag3",
            "pollen_3day", "pollen_7day",
            "is_spike", "sin_day", "cos_day"
        ]
        weighted_features = [c for c in df.columns if "_weighted3" in c]
        lag_features = [c for c in df.columns if "_lag" in c and c not in ["lag1", "lag2", "lag3"]]
        inter_features = [c for c in df.columns if "_x_" in c]

        features = base_features + extra_features + weighted_features + lag_features + inter_features
        features = [f for f in features if f in df.columns]

        return df, features, target

    def predict(self, data_dict):
        # 1. Merge & Normalize
        df_merged = self._merge_data(data_dict)
        df_norm = self._normalize_columns(df_merged)
        
        # 2. Build Features
        df_processed, features, target = self._build_features(df_norm)

        # 3. Split
        train_idx = df_processed['Year'] < self.YEAR_SPLIT
        test_idx = df_processed['Year'] == self.YEAR_SPLIT

        X_train = df_processed.loc[train_idx, features]
        y_train = df_processed.loc[train_idx, target]
        X_test = df_processed.loc[test_idx, features]
        y_test = df_processed.loc[test_idx, target]
        test_dates = df_processed.loc[test_idx, "Date"]

        if len(X_test) == 0:
            raise RuntimeError(f"No test data found for Year {self.YEAR_SPLIT}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        evals = [(dtrain, 'train'), (dtest, 'eval')]
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        self.trained = True

        y_pred = self.model.predict(dtest)

        self.metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)
        }

        results = pd.DataFrame({
            "date": test_dates.values,
            "Observed": y_test.values,
            "Forecast": y_pred
        })
        self.test_data = results
        return results

    def plot_results(self, data=None):
        if not self.trained or self.test_data is None:
            fig = go.Figure()
            fig.add_annotation(text="Model not trained.", x=0.5, y=0.5, showarrow=False)
            return fig

        y_test = self.test_data['Observed']
        preds = self.test_data['Forecast']
        dates = self.test_data['date']

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"Actual vs Predicted (R²={self.metrics['R2']:.2f})", "Time Series Forecast"),
            vertical_spacing=0.15
        )

        fig.add_trace(
            go.Scatter(x=y_test, y=preds, mode='markers', name='Predictions', 
                       marker=dict(color='blue', opacity=0.5, size=6)),
            row=1, col=1
        )
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                       name='Perfect Fit', line=dict(color='red', dash='dash')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=dates, y=y_test, mode='lines', name='Actual', line=dict(color='gray', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=preds, mode='lines', name='Predicted', line=dict(color='green', width=2)),
            row=2, col=1
        )

        fig.update_layout(template="plotly_white", showlegend=True, height=900)
        return fig