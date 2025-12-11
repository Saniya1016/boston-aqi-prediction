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
        
        # State for transformations
        self.train_cap = None
        self.spike_threshold = None
        self.selected_features = []
        
        # Split Years
        self.VAL_YEAR = 2022
        self.TEST_YEAR = 2023

    def short_description(self):
        return "LightGBM Regressor (Log-Target, Spike Weighted, Two-Stage Feature Selection)."

    def _clean_column_name(self, col):
        """Replicates: c.replace(' ', '_').replace('(', '').replace(')', '')"""
        return col.replace(" ", "_").replace("(", "").replace(")", "")

    def _merge_data(self, data_dict):
        dfs = []
        # Standardize 'date' column name across all inputs
        if "pollen" in data_dict:
            df = data_dict["pollen"].copy()
            if "Date" in df.columns: df = df.rename(columns={"Date": "date"})
            if "time" in df.columns: df = df.rename(columns={"time": "date"})
            dfs.append(df)
            
        if "weather" in data_dict:
            df = data_dict["weather"].copy()
            if "time" in df.columns: df = df.rename(columns={"time": "date"})
            dfs.append(df)
            
        if "pollutants" in data_dict:
            df = data_dict["pollutants"].copy()
            if "Date" in df.columns: df = df.rename(columns={"Date": "date"})
            if "time" in df.columns: df = df.rename(columns={"time": "date"})
            dfs.append(df)

        if not dfs:
            raise RuntimeError("No data found in data_dict")

        # Merge all available dataframes
        df_merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), dfs)
        return df_merged

    def _engineer_features(self, df):
        # 1. Clean Column Names
        df.columns = [self._clean_column_name(c) for c in df.columns]
        
        # Ensure Date/Year exist
        date_col = 'date' if 'date' in df.columns else 'Date'
        df["Date"] = pd.to_datetime(df[date_col], errors='coerce')
        df["Year"] = df["Date"].dt.year

        # 2. Numeric Conversion (Force Coerce)
        # CRITICAL FIX: The script forces ALL non-Date columns to numeric. 
        # Previous class version used select_dtypes, which skipped 'object' columns that should have been coerced.
        exclude_cols = ['Date', 'date', 'Year']
        numeric_candidates = [c for c in df.columns if c not in exclude_cols]
        
        for col in numeric_candidates:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaNs with 0 (Matches script logic)
        df[numeric_candidates] = df[numeric_candidates].fillna(0)
        
        # Capture the list of valid numeric columns for feature selection later
        # (This matches script's `numeric_cols` list)
        numeric_cols = numeric_candidates

        # 3. Time Features
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["month"] = df["Date"].dt.month
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # 4. Rolling Features (Shifted to avoid leakage)
        target_col = "Total_Pollen"
        if target_col not in df.columns:
            # Fallback if cleaning changed casing, though strictly following script logic implies "Total_Pollen"
            possible = [c for c in df.columns if "total_pollen" in c.lower()]
            if possible: target_col = possible[0]
            else: raise ValueError(f"Target '{target_col}' not found in dataframe.")

        df["pollen_2day"] = df[target_col].shift(1).rolling(2).mean()
        df["pollen_3day"] = df[target_col].shift(1).rolling(3).mean()
        df["pollen_7day"] = df[target_col].shift(1).rolling(7).mean()
        df["pollen_14day"] = df[target_col].shift(1).rolling(14).mean()
        df["pollen_30day"] = df[target_col].shift(1).rolling(30).mean()

        # 5. Lagged Features
        for lag in [1, 2, 3, 4, 7, 14, 21, 30]:
            df[f"lag_{lag}"] = df[target_col].shift(lag)

        # 6. Spike Ratio
        df["pollen_ratio_1d_7d"] = df["lag_1"] / df["pollen_7day"]

        # 7. Drop NaNs (based on max rolling window)
        df = df.dropna(subset=["pollen_30day"]).reset_index(drop=True)

        return df, numeric_cols

    def predict(self, data_dict):
        # 1. Merge & Initial Engineering
        df_merged = self._merge_data(data_dict)
        df, numeric_cols = self._engineer_features(df_merged)

        # 2. Split (Train/Val/Test)
        train_df = df[df["Year"] < self.VAL_YEAR].copy()
        val_df   = df[df["Year"] == self.VAL_YEAR].copy()
        test_df  = df[df["Year"] >= self.TEST_YEAR].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("Data split resulted in empty sets. Check Date/Year ranges.")

        # 3. Target & Spike Transformation (Fit on Train, Apply to All)
        self.train_cap = train_df["Total_Pollen"].quantile(0.99)
        self.spike_threshold = train_df["pollen_ratio_1d_7d"].quantile(0.95)

        def apply_transformations(sub_df):
            sub_df = sub_df.copy()
            sub_df["Total_Pollen_capped"] = sub_df["Total_Pollen"].clip(upper=self.train_cap)
            sub_df["log_total_pollen"] = np.log1p(sub_df["Total_Pollen_capped"])
            sub_df["is_spike"] = (sub_df["pollen_ratio_1d_7d"] > self.spike_threshold).astype(int)
            return sub_df

        train_df = apply_transformations(train_df)
        val_df   = apply_transformations(val_df)
        test_df  = apply_transformations(test_df)

        # 4. Define Features
        target = "log_total_pollen"
        remove_cols = [
            "Tree", "Grass", "Weed", "Ragweed", "Total_Pollen", 
            "Total_Pollen_capped", "log_total_pollen", "Date", "date", 
            "Year", "day_of_year", "month"
        ]
        
        base_features = [c for c in numeric_cols if c not in remove_cols and c in train_df.columns]
        extra_features = [
            "day_sin", "day_cos", "month_sin", "month_cos",
            "pollen_2day", "pollen_3day", "pollen_7day", "pollen_14day", "pollen_30day",
            "lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "lag_14", "lag_21", "lag_30",
            "pollen_ratio_1d_7d", "is_spike"
        ]
        
        features = base_features + extra_features
        features = [f for f in features if f in train_df.columns]

        X_train = train_df[features]
        y_train = train_df[target]
        X_val   = val_df[features]
        y_val   = val_df[target]
        X_test  = test_df[features]
        y_test_original = test_df["Total_Pollen"]

        # Sample Weights
        train_weights = np.where(train_df["is_spike"] == 1, 10, 1)

        # 5. Training Stage 1: Feature Selection
        lgb_reg = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        lgb_reg.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Filter Features
        importances = pd.DataFrame({"feature": features, "importance": lgb_reg.feature_importances_})
        self.selected_features = importances[importances["importance"] > 50]["feature"].tolist()
        
        if not self.selected_features:
            self.selected_features = importances.sort_values("importance", ascending=False).head(10)["feature"].tolist()

        # 6. Training Stage 2: Final Model
        X_train_trim = X_train[self.selected_features]
        X_val_trim   = X_val[self.selected_features]
        X_test_trim  = X_test[self.selected_features]
        
        self.model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        self.model.fit(
            X_train_trim, y_train,
            sample_weight=train_weights,
            eval_set=[(X_val_trim, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        self.trained = True

        # 7. Predictions & Evaluation
        y_pred_log = self.model.predict(X_test_trim)
        y_pred = np.expm1(y_pred_log)

        self.metrics["MAE"]  = mean_absolute_error(y_test_original, y_pred)
        self.metrics["RMSE"] = np.sqrt(mean_squared_error(y_test_original, y_pred))
        self.metrics["R2"]   = r2_score(y_test_original, y_pred)

        self.test_data = test_df[["Date"]].copy()
        self.test_data = self.test_data.rename(columns={"Date": "date"})
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