import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from functools import reduce

class LightGBMLogPollen:
    def __init__(self):
        self.model = None
        self.trained = False
        self.test_data = None
        self.metrics = {}
        self.selected_features = []

    def short_description(self):
        return "LightGBM Regressor (Log-Target, Spike Weighted, Season-Filtered Mar-Oct) - Matches Notebook Logic."

    def _clean_column_name(self, col):
        """Replicates: c.replace(' ', '_').replace('(', '').replace(')', '')"""
        return col.replace(" ", "_").replace("(", "").replace(")", "")

    def _merge_data(self, data_dict):
        """
        Replicates the notebook merging logic:
        1. Load datasets.
        2. Parse dates.
        3. Rename 'time'/'date' -> 'Date'.
        4. Inner merge Weather -> Pollution -> Pollen.
        """
        # Extract DataFrames
        weather = data_dict.get("weather").copy()
        pollution = data_dict.get("pollutants").copy()
        pollen = data_dict.get("pollen").copy()

        # Parse Dates (Already handled by Streamlit loader, but ensuring consistency)
        # Rename columns to match Notebook exactly
        if 'time' in weather.columns:
            weather.rename(columns={'time': 'Date'}, inplace=True)
        
        if 'date' in pollution.columns:
            pollution.rename(columns={'date': 'Date'}, inplace=True)
            
        if 'Date' in pollen.columns:
            pollen['Date'] = pd.to_datetime(pollen['Date'])

        # Ensure datetime objects
        weather['Date'] = pd.to_datetime(weather['Date'])
        pollution['Date'] = pd.to_datetime(pollution['Date'])
        pollen['Date'] = pd.to_datetime(pollen['Date'])

        # Merge Chain (Weather -> Pollution -> Pollen)
        # merged = weather.merge(pollution, on='Date', how='inner').merge(pollen, on='Date', how='inner')
        merged = weather.merge(pollution, on='Date', how='inner')
        merged = merged.merge(pollen, on='Date', how='inner')

        return merged

    def _engineer_features(self, df):
        """
        Replicates the exact feature engineering pipeline from the notebook.
        """
        # 1. Filter Season (March - October)
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()

        # 2. Select Relevant Numeric Columns (Notebook 'cols' list)
        notebook_cols = [
            # Weather
            'weather_code (wmo code)', 'temperature_2m_mean (°C)', 
            'apparent_temperature_mean (°C)', 'precipitation_sum (mm)', 
            'wind_gusts_10m_max (km/h)', 'wind_speed_10m_max (km/h)', 
            'wind_direction_10m_dominant (°)',
            # Pollution
            'PM2.5', 'O3', 'CO', 'NO2', 'SO2',
            'AQI_PM2.5', 'AQI_O3', 'AQI_CO', 'AQI_NO2', 'AQI_SO2',
            'AQI', 'num_pollutants_available',
            # Pollen
            'Tree', 'Grass', 'Weed', 'Ragweed', 'Total_Pollen'
        ]

        # Filter strictly for columns present in the dataframe
        present_cols = [c for c in notebook_cols if c in df.columns]
        
        # 3. Clean Column Names
        # Map old names to new clean names so we can rename the DF
        rename_map = {c: self._clean_column_name(c) for c in df.columns}
        df = df.rename(columns=rename_map)
        
        # Update our list of numeric target columns to the cleaned versions
        numeric_cols = [self._clean_column_name(c) for c in present_cols]
        
        # 4. Numeric Conversion & Fill NaNs (0)
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # 5. Feature Engineering
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["month"] = df["Date"].dt.month

        # Cyclical encoding
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Rolling features (shift(1) ensures no look-ahead/leakage)
        target_col = "Total_Pollen"
        df["pollen_2day"] = df[target_col].shift(1).rolling(2).mean()
        df["pollen_3day"] = df[target_col].shift(1).rolling(3).mean()
        df["pollen_7day"] = df[target_col].shift(1).rolling(7).mean()
        df["pollen_14day"] = df[target_col].shift(1).rolling(14).mean()
        df["pollen_30day"] = df[target_col].shift(1).rolling(30).mean()

        # Lagged features
        for lag in [1, 2, 3, 4, 7, 14, 21, 30]:
            df[f"lag_{lag}"] = df[target_col].shift(lag)

        # Spike ratio feature
        df["pollen_ratio_1d_7d"] = df["lag_1"] / df["pollen_7day"]

        # Drop rows with NaNs (based on max rolling window)
        # Notebook: df = df.dropna(subset=required_cols).copy() where required_cols=["pollen_30day"]
        df = df.dropna(subset=["pollen_30day"]).reset_index(drop=True)

        return df, numeric_cols

    def predict(self, data_dict):
        # 1. Merge & Engineer
        df_merged = self._merge_data(data_dict)
        df, numeric_cols = self._engineer_features(df_merged)

        # 2. Split (Train/Val/Test) by Year
        df["Year"] = df["Date"].dt.year
        
        # Strict Notebook Logic:
        # Train: < 2022
        # Val: == 2022
        # Test: >= 2023
        train_df = df[df["Year"] < 2022].copy()
        val_df   = df[df["Year"] == 2022].copy()
        test_df  = df[df["Year"] >= 2023].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("Data split resulted in empty sets. Check Date/Year ranges.")

        # 3. Transformations (Calculated on Train, Applied to All)
        # Notebook: train_cap = train_df["Total_Pollen"].quantile(0.99)
        train_cap = train_df["Total_Pollen"].quantile(0.99)
        # Notebook: spike_threshold = train_df["pollen_ratio_1d_7d"].quantile(0.95)
        spike_threshold = train_df["pollen_ratio_1d_7d"].quantile(0.95)

        def apply_transformations(sub_df):
            sub_df = sub_df.copy()
            sub_df["Total_Pollen_capped"] = sub_df["Total_Pollen"].clip(upper=train_cap)
            sub_df["log_total_pollen"] = np.log1p(sub_df["Total_Pollen_capped"])
            sub_df["is_spike"] = (sub_df["pollen_ratio_1d_7d"] > spike_threshold).astype(int)
            return sub_df

        train_df = apply_transformations(train_df)
        val_df   = apply_transformations(val_df)
        test_df  = apply_transformations(test_df)

        # 4. Define Features
        target = "log_total_pollen"
        
        remove_cols = [
            "Tree", "Grass", "Weed", "Ragweed", "Total_Pollen", 
            "Total_Pollen_capped", "log_total_pollen", "Date", 
            "Year", "day_of_year", "month"
        ]
        
        base_features = [c for c in numeric_cols if c not in remove_cols]
        extra_features = [
            "day_sin", "day_cos", "month_sin", "month_cos",
            "pollen_2day", "pollen_3day", "pollen_7day", "pollen_14day", "pollen_30day",
            "lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "lag_14", "lag_21", "lag_30",
            "pollen_ratio_1d_7d", "is_spike"
        ]
        
        features = base_features + extra_features
        # Ensure features exist in dataframe
        features = [f for f in features if f in train_df.columns]

        # Prepare X and y
        X_train = train_df[features]
        y_train = train_df[target]
        
        X_val   = val_df[features]
        y_val   = val_df[target]
        
        X_test  = test_df[features]
        y_test_original = test_df["Total_Pollen"]

        # Sample Weights (10x for spikes in training)
        train_weights = np.where(train_df["is_spike"] == 1, 10, 1)

        # 5. Stage 1: Feature Selection
        # Notebook: lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05...)
        lgb_reg = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        
        lgb_reg.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val)], # Validating on 2022 data
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Select Features > 50 importance
        importances = pd.DataFrame({"feature": features, "importance": lgb_reg.feature_importances_})
        self.selected_features = importances[importances["importance"] > 50]["feature"].tolist()
        
        # Fallback if too few features selected
        if not self.selected_features:
            self.selected_features = importances.sort_values("importance", ascending=False).head(10)["feature"].tolist()

        # 6. Stage 2: Final Model Training (Trimmed Features)
        X_train_trim = X_train[self.selected_features]
        X_val_trim   = X_val[self.selected_features]
        X_test_trim  = X_test[self.selected_features]
        
        # Weights array must match index of training set (it does, derived from same df)
        train_weights_trim = train_weights

        self.model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        self.model.fit(
            X_train_trim, y_train,
            sample_weight=train_weights_trim,
            eval_set=[(X_val_trim, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        self.trained = True

        # 7. Predictions & Metrics (Inverse Log Transform)
        y_pred_log = self.model.predict(X_test_trim)
        y_pred = np.expm1(y_pred_log) # Exp(x) - 1 to reverse Log1p

        self.metrics["MAE"]  = mean_absolute_error(y_test_original, y_pred)
        self.metrics["RMSE"] = np.sqrt(mean_squared_error(y_test_original, y_pred))
        self.metrics["R2"]   = r2_score(y_test_original, y_pred)

        # Prepare Output for Streamlit
        # Must return dataframe with Date, Observed, Forecast
        self.test_data = test_df[["Date"]].copy()
        self.test_data = self.test_data.rename(columns={"Date": "date"}) # Lowercase for Streamlit consistency
        self.test_data["Observed"] = y_test_original.values
        self.test_data["Forecast"] = y_pred

        return self.test_data

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

        # Scatter
        fig.add_trace(
            go.Scatter(x=y_test, y=preds, mode='markers', name='Predictions', 
                       marker=dict(color='blue', opacity=0.5, size=6)),
            row=1, col=1
        )
        
        # Perfect Fit Line
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                       name='Perfect Fit', line=dict(color='red', dash='dash')),
            row=1, col=1
        )

        # Time Series
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