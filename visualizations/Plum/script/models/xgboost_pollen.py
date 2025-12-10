# file: xgboost_pollen_streamlit.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

class XGBoostPollen:
    def __init__(self):
        self.model = None
        self.trained = False
        self.y_test_real = None
        self.y_pred_real = None
        self.metrics = {}
        self.features = []

    def short_description(self):
        return "XGBoost Regressor for Pollen using chronological split and causal features."

    def _normalize_columns(self, df):
        df = df.copy()
        df.columns = (
            df.columns
            .str.replace(" ", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("/", "_", regex=False)
        )
        return df

    def _engineer_features(self, data_dict):
        # Merge and normalize columns
        w = self._normalize_columns(data_dict["weather"].copy())
        p = self._normalize_columns(data_dict["pollutants"].copy())
        pol = self._normalize_columns(data_dict["pollen"].copy())
        pol['Date'] = pd.to_datetime(pol['Date'])
        w['Date'] = pd.to_datetime(w['time'])
        p['Date'] = pd.to_datetime(p['date'])

        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()
        df['Year'] = df['Date'].dt.year

        target = "Total_Pollen"

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Fill numeric NaNs
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Feature engineering (causal)
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["lag1"] = df[target].shift(1).ffill()
        df["lag2"] = df[target].shift(2).ffill()
        df["lag3"] = df[target].shift(3).ffill()
        df["pollen_3day"] = df[target].shift(1).rolling(3).mean().ffill()
        df["pollen_7day"] = df[target].shift(1).rolling(7).mean().ffill()
        df["is_spike"] = (df["lag1"] > df["pollen_3day"]).astype(int)
        df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # Weighted rolling features
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

        # Interaction features
        interaction_pairs = [
            ('temperature_2m_mean_°C', 'AQI'),
            ('apparent_temperature_mean_°C', 'PM2.5'),
            ('wind_speed_10m_max_km_h', 'O3'),
        ]
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

        # Drop initial NaNs from lags
        df = df.dropna(subset=["lag1", "lag2", "lag3", "pollen_3day", "pollen_7day"])

        # Define features
        remove_cols = ["Tree", "Grass", "Weed", "Ragweed", target, "Year", "Date"]
        base_features = [c for c in numeric_cols if c not in remove_cols]
        extra_features = ["lag1", "lag2", "lag3", "pollen_3day", "pollen_7day",
                          "is_spike", "sin_day", "cos_day"]
        weighted_features = [f"{col}_weighted3" for col in rolling_cols if f"{col}_weighted3" in df.columns]
        lag_features = [f"{col}_lag{lag}" for col in rolling_cols for lag in range(1, 4)]
        interaction_features = [f'{a}_x_{b}' for a, b in interaction_pairs]

        self.features = [f for f in (base_features + extra_features + weighted_features + lag_features + interaction_features) if f in df.columns]

        return df, target

    def predict(self, data_dict):
        df, target = self._engineer_features(data_dict)

        # Chronological split
        X_train = df.loc[df['Year'] < 2023, self.features]
        y_train = df.loc[df['Year'] < 2023, target]
        X_test = df.loc[df['Year'] >= 2023, self.features]
        y_test = df.loc[df['Year'] >= 2023, target]

        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'eval_metric': 'rmse',
            'seed': 42
        }

        evals = [(dtrain, 'train'), (dtest, 'eval')]
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1200,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        self.trained = True
        self.y_test_real = y_test
        self.y_pred_real = self.model.predict(dtest)

        # Metrics
        self.metrics = {
            "MAE": mean_absolute_error(y_test, self.y_pred_real),
            "RMSE": np.sqrt(mean_squared_error(y_test, self.y_pred_real)),
            "R2": r2_score(y_test, self.y_pred_real)
        }

        return pd.DataFrame({
            "Actual Pollen": self.y_test_real.values,
            "Predicted Pollen": self.y_pred_real
        })

    def plot_results(self):
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
            title=f"XGBoost — R²: {self.metrics['R2']:.3f} | MAE: {self.metrics['MAE']:.1f}",
            xaxis_title="Actual Total Pollen",
            yaxis_title="Predicted Total Pollen",
            template="plotly_white"
        )
        return fig
