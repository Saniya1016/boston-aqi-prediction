import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class LightGBMPollen:
    def __init__(self):
        # FIX: Initialize the actual model here, otherwise self.model.fit fails
        self.model = LGBMRegressor(
            n_estimators=1000, 
            learning_rate=0.05, 
            random_state=42, 
            verbose=-1
        )
        self.trained = False
        self.y_test_real = None
        self.y_pred_real = None
        self.metrics = {}
        self.features = []
        self.output_df = None # FIX: Initialize this to avoid attribute errors

    def short_description(self):
        # FIX: Corrected description to match the model used
        return "LightGBM Regressor for Pollen using chronological split and causal features."

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
        
        # Ensure datetime format
        pol['Date'] = pd.to_datetime(pol['Date'])
        w['Date'] = pd.to_datetime(w['time'])
        p['Date'] = pd.to_datetime(p['date'])

        # Merge
        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        
        # Filter for Pollen Season (March - October)
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()
        df['Year'] = df['Date'].dt.year

        target = "Total_Pollen"

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Fill numeric NaNs
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # --- Feature Engineering ---
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
        
        # Only process columns that actually exist in the merged dataframe
        rolling_cols = [c for c in rolling_cols if c in df.columns]

        for col in rolling_cols:
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

        # Drop initial NaNs created by lags
        df = df.dropna(subset=["lag1", "lag2", "lag3", "pollen_3day", "pollen_7day"])

        # --- Define Feature Lists ---
        remove_cols = ["Tree", "Grass", "Weed", "Ragweed", target, "Year", "Date"]
        
        # Base numeric features (excluding targets and dates)
        base_features = [c for c in numeric_cols if c not in remove_cols]
        
        # Engineered features
        extra_features = ["lag1", "lag2", "lag3", "pollen_3day", "pollen_7day",
                          "is_spike", "sin_day", "cos_day"]
        
        weighted_features = [f"{col}_weighted3" for col in rolling_cols if f"{col}_weighted3" in df.columns]
        lag_features = [f"{col}_lag{lag}" for col in rolling_cols for lag in range(1, 4)]
        interaction_features = [f'{a}_x_{b}' for a, b in interaction_pairs if f'{a}_x_{b}' in df.columns]

        # Combine all features and ensure they exist in DF
        candidates = base_features + extra_features + weighted_features + lag_features + interaction_features
        self.features = list(set([f for f in candidates if f in df.columns]))

        return df, target

    def predict(self, data_dict):
        # 1. Run Engineering
        df, target = self._engineer_features(data_dict)
        
        # 2. Sanitize Column Names for LightGBM
        # LightGBM errors on special chars in col names. Clean them *before* split.
        clean_map = {c: re.sub(r'[^\w]', '_', c) for c in df.columns}
        df = df.rename(columns=clean_map)
        
        # Update self.features to match the cleaned names
        self.features = [clean_map[f] for f in self.features if f in clean_map]
        
        # 3. Time Split
        df['Year'] = df['Date'].dt.year
        train_df = df[df['Year'] < 2023].copy()
        test_df = df[df['Year'] >= 2023].copy()
        
        # Drop residual NaNs
        train_df = train_df.dropna().reset_index(drop=True)
        test_df = test_df.dropna().reset_index(drop=True)
        
        # 4. Select Features
        X_train = train_df[self.features]
        y_train = train_df[target]
        X_test = test_df[self.features]
        y_test = test_df[target]
        
        # 5. Log Transform Target (handling 0s with log1p)
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        
        # 6. Fit Model
        self.model.fit(
            X_train, y_train_log,
            eval_set=[(X_test, y_test_log)],
            eval_metric='rmse',
        )
        self.trained = True
        
        # 7. Predict
        preds_log = self.model.predict(X_test)
        preds = np.expm1(preds_log) # Inverse log
        
        # 8. Calculate Metrics
        self.metrics['MAE'] = mean_absolute_error(y_test, preds)
        self.metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, preds))
        self.metrics['R2'] = r2_score(y_test, preds)
        
        # 9. Store Output
        self.output_df = test_df[['Date']].copy()
        self.output_df['Total_Pollen'] = y_test
        self.output_df['Predicted_Pollen'] = preds
        
        return self.output_df

    def plot_results(self, data_dict):
        # Ensure we have predictions
        if self.output_df is None:
            self.predict(data_dict)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Actual vs Predicted (R²={self.metrics.get('R2', 0):.2f})", "LightGBM Time Series")
        )
        
        y_test = self.output_df['Total_Pollen']
        preds = self.output_df['Predicted_Pollen']
        
        # Scatter Plot
        fig.add_trace(
            go.Scatter(x=y_test, y=preds, mode='markers', name='Predictions', marker=dict(color='purple', opacity=0.5)),
            row=1, col=1
        )
        # Perfect fit line
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')),
            row=1, col=1
        )

        # Time Series Plot
        fig.add_trace(
            go.Scatter(x=self.output_df['Date'], y=y_test, mode='lines', name='Actual', line=dict(color='white', width=1)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.output_df['Date'], y=preds, mode='lines', name='Predicted', line=dict(color='purple', width=2)),
            row=1, col=2
        )
        
        # FIX: Changed showlegend to True so we can distinguish lines
        fig.update_layout(template="plotly_white", showlegend=True, height=500)
        return fig