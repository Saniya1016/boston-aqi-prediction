import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from functools import reduce

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
        self.test_data = None
        self.metrics = {}
        self.YEAR_SPLIT = 2023
        
        # --- FIX: HARDCODED FEATURE LIST FROM YOUR LOCAL SCRIPT ---
        # This ensures the model sees EXACTLY the same columns as your local script.
        self.EXPLICIT_NUMERIC_COLS = [
            'temperature_2m_mean (°C)', 
            'apparent_temperature_mean (°C)', 
            'precipitation_sum (mm)', 
            'wind_gusts_10m_max (km/h)', 
            'wind_speed_10m_max (km/h)', 
            'wind_direction_10m_dominant (°)', 
            'PM2.5', 'O3', 'CO', 'NO2', 'SO2', 
            'AQI_PM2.5', 'AQI_O3', 'AQI_CO', 'AQI_NO2', 'AQI_SO2', 'AQI', 
            'num_pollutants_available'
        ]

    def short_description(self):
        return "Random Forest (Exact Features) | Features: Lags + Weather Rolling Means"

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

    def _build_features(self, df):
        df = df.sort_values("date").reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        
        # Target check
        target_col = "Total_Pollen"
        if target_col not in df.columns:
             candidates = [c for c in df.columns if "total" in c.lower() and "pollen" in c.lower()]
             if candidates:
                 target_col = candidates[0]
                 df["Total_Pollen"] = df[target_col]

        # 1. USE EXPLICIT NUMERIC COLS (The Fix)
        # We verify which of your list actually exist in the merged file to prevent KeyErrors
        # (e.g. if 'num_pollutants_available' was calculated locally but not in the CSV)
        numeric_cols = [c for c in self.EXPLICIT_NUMERIC_COLS if c in df.columns]
        
        # If 'num_pollutants_available' is missing, we proceed without it but print a warning if you like
        # For now, we assume standard behavior.

        # Coerce safely
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 2. Basic engineered features
        df["lag1"] = df["Total_Pollen"].shift(1)
        df["lag2"] = df["Total_Pollen"].shift(2)
        df["lag3"] = df["Total_Pollen"].shift(3)
        df["pollen_3day"] = df["Total_Pollen"].shift(1).rolling(window=3, min_periods=1).mean()
        df["pollen_7day"] = df["Total_Pollen"].shift(1).rolling(window=7, min_periods=1).mean()

        df["day_of_year"] = df["Date"].dt.dayofyear
        df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # 3. Rolling & Weighted features
        # We use the columns from your list that match the 'rolling' candidates
        rolling_candidates = [
            "temperature_2m_mean (°C)", "apparent_temperature_mean (°C)",
            "PM2.5", "O3", "CO", "NO2", "SO2", "AQI",
            "AQI_PM2.5", "AQI_O3", "AQI_CO", "AQI_NO2", "AQI_SO2",
            "wind_speed_10m_max (km/h)"
        ]
        
        rolling_cols = [c for c in rolling_candidates if c in numeric_cols]
        weights = np.array([0.1, 0.3, 0.6])

        for col in rolling_cols:
            s0 = df[col].shift(2)
            s1 = df[col].shift(1)
            s2 = df[col]
            df[f"{col}_weighted3"] = (weights[0] * s0) + (weights[1] * s1) + (weights[2] * s2)
            
            for lag in range(1, 4):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        # Interactions
        candidate_interactions = [
            ("temperature_2m_mean (°C)", "AQI"),
            ("apparent_temperature_mean (°C)", "PM2.5"),
            ("wind_speed_10m_max (km/h)", "O3"),
        ]
        for c1, c2 in candidate_interactions:
            if c1 in df.columns and c2 in df.columns:
                df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

        # 4. Spike indicator
        train_mask = df["Year"] < self.YEAR_SPLIT
        train_vals = df.loc[train_mask, "Total_Pollen"].dropna()
        if len(train_vals) >= 5:
            threshold = train_vals.mean() + train_vals.std()
        else:
            threshold = train_vals.median() + 1.0 * train_vals.std() if len(train_vals) > 0 else df["Total_Pollen"].median()
        df["is_spike"] = (df["Total_Pollen"].shift(1) > threshold).astype(int)

        # 5. Build Final Feature List
        exclude_if_present = {"Total_Pollen", "Tree", "Grass", "Weed", "Ragweed", "Date", "Year", "date"}
        
        # Base features is EXACTLY your list (minus target/date if they somehow got in there)
        base_features = [c for c in numeric_cols if c not in exclude_if_present]
        
        extra_features = ["lag1", "lag2", "lag3", "pollen_3day", "pollen_7day", "sin_day", "cos_day", "is_spike"]
        generated = [c for c in df.columns if ("_weighted3" in c) or (c.endswith("_lag1") or c.endswith("_lag2") or c.endswith("_lag3")) or ("_x_" in c)]
        generated = [c for c in generated if c not in {"lag1", "lag2", "lag3"}]

        features = []
        # Order matters for reproducibility in some edge cases, so we stick to a predictable order
        for c in base_features + extra_features + generated:
            if c in df.columns and c not in features:
                features.append(c)

        # 6. Drop Rows
        X_all = df[features].copy()
        y_all = df["Total_Pollen"].copy()
        dates_all = df["date"].copy()
        years_all = df["Year"].copy()

        required_cols = ["lag1", "lag2", "lag3", "pollen_3day", "pollen_7day"]
        required_cols = [c for c in required_cols if c in X_all.columns]
        
        if required_cols:
            mask_valid = X_all[required_cols].notna().all(axis=1) & y_all.notna()
            X_all = X_all.loc[mask_valid].reset_index(drop=True)
            y_all = y_all.loc[mask_valid].reset_index(drop=True)
            dates_all = dates_all.loc[mask_valid].reset_index(drop=True)
            years_all = years_all.loc[mask_valid].reset_index(drop=True)

        return X_all, y_all, dates_all, years_all

    def predict(self, data_dict):
        df_merged = self._merge_data(data_dict)
        X_all, y_all, dates_all, years_all = self._build_features(df_merged)

        train_idx = years_all < self.YEAR_SPLIT
        test_idx = years_all >= self.YEAR_SPLIT

        X_train = X_all.loc[train_idx]
        X_test  = X_all.loc[test_idx]
        y_train = y_all.loc[train_idx]
        y_test  = y_all.loc[test_idx]
        test_dates = dates_all.loc[test_idx]

        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)
        X_test  = X_test.fillna(train_medians)

        y_train_log = np.log1p(y_train)
        self.model.fit(X_train, y_train_log)
        self.trained = True

        preds_log = self.model.predict(X_test)
        preds = np.expm1(preds_log)

        self.metrics = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        }

        results = pd.DataFrame({
            "date": test_dates.values,
            "Observed": y_test.values,
            "Forecast": preds
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