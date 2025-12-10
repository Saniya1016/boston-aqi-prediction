import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightgbm as lgb

class LightGBMPollen:
    def __init__(self):
        self.model = LGBMRegressor(
            n_estimators=1000, 
            learning_rate=0.05, 
            random_state=42, 
            verbose=-1,
            n_jobs=-1
        )
        self.trained = False
        self.y_test_real = None
        self.y_pred_real = None
        self.metrics = {}
        self.features = []
        self.output_df = None
        self.train_cap = None
        self.spike_threshold = None
        self.selected_features = []

    def short_description(self):
        return "LightGBM Regressor for Pollen using chronological split, causal features, and feature selection."

    def _normalize_columns(self, df):
        df = df.copy()
        df.columns = (
            df.columns
            .str.replace(" ", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("/", "_", regex=False)
            .str.replace("°C", "_C", regex=False)
            .str.replace("km/h", "km_h", regex=False)
        )
        return df

    def _engineer_features(self, data_dict):
        # --- Data Merge and Cleaning ---
        w = self._normalize_columns(data_dict["weather"].copy())
        p = self._normalize_columns(data_dict["pollutants"].copy())
        pol = self._normalize_columns(data_dict["pollen"].copy())
        
        # Ensure datetime format
        pol['Date'] = pd.to_datetime(pol['Date'])
        w['Date'] = pd.to_datetime(w['time']) # Keep 'time' in w for merging
        p['Date'] = pd.to_datetime(p['date'])

        # Merge
        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        
        # Filter for Pollen Season (March - October)
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()
        df['Year'] = df['Date'].dt.year

        target_name = "Total_Pollen"

        # Sanitize Column Names *Early*
        clean_map = {c: re.sub(r'[^\w]', '_', c) for c in df.columns}
        df = df.rename(columns=clean_map)
        target_name = clean_map.get("Total_Pollen", "Total_Pollen")
        
        # --- FIX: Define STRICTLY EXCLUDED columns first ---
        # This list ensures these columns NEVER become a feature.
        STRICT_REMOVE_COLS = [
            "Tree", "Grass", "Weed", "Ragweed", # Other pollen types, not target
            target_name,                        # Target column
            "Year", "Date", "time", "date",     # Date/time columns
            "day_of_year", "month"              # Components of cyclical features
        ]
        
        # Select numeric columns (including time-related cols before explicit removal)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Fill numeric NaNs
        for col in numeric_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df[numeric_cols] = df[numeric_cols].fillna(0)


        # --- Feature Engineering (Matching Second Script) ---
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["month"] = df["Date"].dt.month
        
        # Cyclical encoding
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Rolling features (shift(1) ensures no look-ahead/leakage)
        df["pollen_2day"] = df[target_name].shift(1).rolling(2).mean()
        df["pollen_3day"] = df[target_name].shift(1).rolling(3).mean()
        df["pollen_7day"] = df[target_name].shift(1).rolling(7).mean()
        df["pollen_14day"] = df[target_name].shift(1).rolling(14).mean()
        df["pollen_30day"] = df[target_name].shift(1).rolling(30).mean()

        # Lagged features
        for lag in [1,2,3,4,7,14,21,30]:
            df[f"lag_{lag}"] = df[target_name].shift(lag)

        # Spike ratio feature
        df["pollen_ratio_1d_7d"] = df["lag_1"] / df["pollen_7day"]

        # Drop rows with NaNs from rolling/lags (max window is 30 days)
        df = df.dropna(subset=["lag_30", "pollen_30day"]).copy()

        # --- Define Feature Lists (STRICTLY Filtered) ---
        
        # Base numeric features: Filter numeric_cols by removing STRICT_REMOVE_COLS
        base_features = [c for c in numeric_cols if c not in STRICT_REMOVE_COLS]
        
        # Engineered features
        extra_features = [
            "day_sin","day_cos","month_sin","month_cos",
            "pollen_2day","pollen_3day","pollen_7day","pollen_14day","pollen_30day",
            "lag_1","lag_2","lag_3","lag_4","lag_7","lag_14","lag_21","lag_30",
            "pollen_ratio_1d_7d"
        ]

        # Combine all features and ensure they exist in DF
        candidates = base_features + extra_features
        self.features = list(set([f for f in candidates if f in df.columns]))

        return df, target_name

    def _apply_transformations(self, data_df, target_name):
        """Applies capping, log transform, and spike feature based on stored training stats."""
        data_df["Total_Pollen_capped"] = data_df[target_name].clip(upper=self.train_cap)
        data_df["log_total_pollen"] = np.log1p(data_df["Total_Pollen_capped"])
        # Only create is_spike if pollen_ratio_1d_7d is present and we have a threshold
        if "pollen_ratio_1d_7d" in data_df.columns and self.spike_threshold is not None:
             data_df["is_spike"] = (data_df["pollen_ratio_1d_7d"] > self.spike_threshold).astype(int)
        else:
             data_df["is_spike"] = 0 
        return data_df

    def predict(self, data_dict):
        # 1. Run Engineering
        df, target_name = self._engineer_features(data_dict)
        
        # 2. Time Split
        df['Year'] = df['Date'].dt.year
        train_df = df[df['Year'] < 2022].copy()
        val_df = df[df['Year'] == 2022].copy()
        test_df = df[df['Year'] >= 2023].copy()

        # FIX: Check for empty dataframes after split
        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("Time split resulted in empty training, validation, or test dataframes. Check your input data dates.")

        # 3. Calculate Target/Spike Transformations on Training Data
        self.train_cap = train_df[target_name].quantile(0.99)
        if "pollen_ratio_1d_7d" in train_df.columns:
            self.spike_threshold = train_df["pollen_ratio_1d_7d"].quantile(0.95)
        
        train_df = self._apply_transformations(train_df, target_name)
        val_df = self._apply_transformations(val_df, target_name)
        test_df = self._apply_transformations(test_df, target_name)

        # 4. Define Final Training Variables
        target = "log_total_pollen"
        
        if 'is_spike' not in self.features and 'is_spike' in train_df.columns:
            self.features.append('is_spike')
            
        X_train = train_df[self.features]
        y_train = train_df[target]
        X_val = val_df[self.features]
        y_val = val_df[target]
        y_test_original = test_df[target_name] 

        # 5. Sample Weights
        train_weights = np.where(train_df.get("is_spike")==1, 10, 1)
        
        # 6. Train Initial Model for Feature Selection
        lgb_reg_fs = LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_reg_fs.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val)], 
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
        )

        # Feature Selection
        importances = pd.DataFrame({
            "feature": self.features, 
            "importance": lgb_reg_fs.feature_importances_
        }).sort_values(by="importance", ascending=False)
        
        # Select features with importance > 50
        self.selected_features = importances[importances["importance"]>50]["feature"].tolist()
        
        # Trim DataFrames to selected features
        X_train_trim = X_train[self.selected_features]
        X_val_trim = X_val[self.selected_features]
        # FIX: Ensure X_test_trim is created from the test_df rows, using self.features, then trimmed.
        X_test_trim = test_df[self.features][self.selected_features]
        train_weights_trim = train_weights[:X_train_trim.shape[0]]

        # 7. Retrain Final Model
        self.model.fit(
            X_train_trim, y_train,
            sample_weight=train_weights_trim,
            eval_set=[(X_val_trim, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
        )
        self.trained = True
        
        # 8. Predict on Test Set
        preds_log = self.model.predict(X_test_trim)
        preds = np.expm1(preds_log)
        
        # 9. Calculate Metrics
        self.metrics['MAE'] = mean_absolute_error(y_test_original, preds)
        self.metrics['RMSE'] = np.sqrt(mean_squared_error(y_test_original, preds))
        self.metrics['R2'] = r2_score(y_test_original, preds)
        
        # 10. Store Output
        self.output_df = test_df[['Date']].copy()
        self.output_df['Total_Pollen'] = y_test_original
        self.output_df['Predicted_Pollen'] = preds
        
        return self.output_df

    # ... (plot_results method remains the same)
    def plot_results(self, data_dict):
        # ... (plot_results content) ...
        # Ensure we have predictions
        if self.output_df is None:
            self.predict(data_dict)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Actual vs Predicted (R²={self.metrics.get('R2', 0):.4f})", "LightGBM Time Series")
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
            go.Scatter(x=self.output_df['Date'], y=y_test, mode='lines', name='Actual', line=dict(color='gray', width=1)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.output_df['Date'], y=preds, mode='lines', name='Predicted', line=dict(color='purple', width=2)),
            row=1, col=2
        )
        
        fig.update_layout(template="plotly_white", showlegend=True, height=500)
        return fig