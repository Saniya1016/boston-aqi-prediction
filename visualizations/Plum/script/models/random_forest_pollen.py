import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        self.test_df = None

    def short_description(self):
        return "Random Forest Regressor (Notebook Setup). Features: Weighted Rolling Avgs, Seasonality (Sin/Cos), Interactions. Target is Log-Transformed."

    def _engineer_features(self, data_dict):
        """
        Replicates the data merging and extensive feature engineering 
        from the 'Advanced Features' section of the notebook.
        """
        # 1. Load and Merge Data
        w = data_dict["weather"].copy()
        p = data_dict["pollutants"].copy()
        pol = data_dict["pollen"].copy()

        # Standardize Date columns for merging
        w['Date'] = pd.to_datetime(w['time'])
        p['Date'] = pd.to_datetime(p['date'])
        pol['Date'] = pd.to_datetime(pol['Date'])

        # Merge (Inner Join)
        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')

        # Filter for pollen season (March through October)
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()

        # 2. Base Pollen Features (Lags & Rolling)
        target = "Total_Pollen"
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["lag1"] = df[target].shift(1)
        df["lag2"] = df[target].shift(2)
        df["lag3"] = df[target].shift(3)
        df["pollen_3day"] = df[target].rolling(3).mean()
        df["pollen_7day"] = df[target].rolling(7).mean()

        # Spike indicator
        threshold = df[target].mean() + df[target].std()
        df["is_spike"] = (df[target] > threshold).astype(int)

        # 3. Seasonal Features (Sin/Cos)
        df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # 4. Weighted Rolling Averages (3-day)
        # Weights: 0.1 (t-2), 0.3 (t-1), 0.6 (t)
        weights = np.array([0.1, 0.3, 0.6])
        
        rolling_cols = [
            "temperature_2m_mean (°C)", "apparent_temperature_mean (°C)",
            "PM2.5", "O3", "CO", "NO2", "SO2",
            "AQI_PM2.5", "AQI_O3", "AQI_CO", "AQI_NO2", "AQI_SO2", "AQI"
        ]
        
        # Only process columns that actually exist in the data
        rolling_cols = [c for c in rolling_cols if c in df.columns]

        for col in rolling_cols:
            df[f"{col}_weighted3"] = (
                df[col].shift(2).bfill() * weights[0] +
                df[col].shift(1).bfill() * weights[1] +
                df[col] * weights[2]
            )
            # 5. Lag features for pollutants (t-1, t-2, t-3)
            for lag in range(1, 4):
                df[f"{col}_lag{lag}"] = df[col].shift(lag).bfill()

        # 6. Interaction Features
        interaction_pairs = [
            ("temperature_2m_mean (°C)", "AQI"),
            ("apparent_temperature_mean (°C)", "PM2.5"),
            ("wind_speed_10m_max (km/h)", "O3"),
        ]

        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

        # Drop NAs created by lags/rolling
        df = df.dropna()
        return df

    def predict(self, data_dict):
        # Generate the specific dataframe structure from the notebook
        df = self._engineer_features(data_dict)
        target = "Total_Pollen"

        # Define Features (Numeric columns only, excluding metadata and raw pollen splits)
        remove_cols = ["Date", "time", "date", "Month", "Year", "Day", "Week", 
                       "Tree_Level", "Grass_Level", "Weed_Level", "Ragweed_Level", 
                       "AQI_Category", "weather_code (wmo code)", "OBJECTID",
                       "Tree", "Grass", "Weed", "Ragweed", "day_of_year", target]
        
        feature_cols = [c for c in df.columns if c not in remove_cols and df[c].dtype in ['float64', 'int64', 'int32']]
        
        X = df[feature_cols]
        
        # Log-Transform Target (Notebook Step 4)
        y_log = np.log1p(df[target])

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=0.2, random_state=42
        )

        # Fit Model
        self.model.fit(X_train, y_train)
        self.trained = True

        # Predict
        y_pred_log = self.model.predict(X_test)
        
        # Inverse Log Transform (Notebook Step 7)
        self.y_pred_real = np.expm1(y_pred_log)
        self.y_test_real = np.expm1(y_test)

        # Capture Dates for Time Series Plot (sorted chronologically)
        self.test_df = pd.DataFrame({
            "Date": df.loc[X_test.index, "Date"],
            "Actual": self.y_test_real,
            "Predicted": self.y_pred_real
        }).sort_values("Date")

        # Metrics
        mae = mean_absolute_error(self.y_test_real, self.y_pred_real)
        rmse = np.sqrt(mean_squared_error(self.y_test_real, self.y_pred_real))
        r2 = r2_score(self.y_test_real, self.y_pred_real)
        self.metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

        return pd.DataFrame({
            "Actual Pollen": self.y_test_real.values,
            "Predicted Pollen": self.y_pred_real
        })

    def plot_results(self, data_dict=None):
        if not self.trained:
            fig = go.Figure()
            fig.add_annotation(text="Model not trained.", x=0.5, y=0.5, showarrow=False)
            return fig

        # Create subplots: 1 row, 2 cols
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Actual vs Predicted (R²: {self.metrics['R2']:.2f})", 
                "Pollen Forecast Over Time"
            )
        )

        # Plot 1: Scatter
        fig.add_trace(go.Scatter(
            x=self.y_test_real,
            y=self.y_pred_real,
            mode='markers',
            name='Test Predictions',
            marker=dict(opacity=0.6, color='orange'),
            showlegend=False
        ), row=1, col=1)

        # Perfect diagonal reference line
        line_max = max(self.y_test_real.max(), self.y_pred_real.max())
        fig.add_trace(go.Scatter(
            x=[0, line_max], 
            y=[0, line_max],
            mode='lines',
            name='Perfect Fit',
            line=dict(dash='dash', color='red'),
            showlegend=False
        ), row=1, col=1)

        # Plot 2: Time Series
        fig.add_trace(go.Scatter(
            x=self.test_df["Date"],
            y=self.test_df["Actual"],
            mode='lines+markers',
            name='Actual',
            line=dict(color='gray'),
            opacity=0.7
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=self.test_df["Date"],
            y=self.test_df["Predicted"],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='orange'),
            opacity=0.7
        ), row=1, col=2)

        fig.update_layout(
            title=f"Random Forest (Log-Target) | MAE: {self.metrics['MAE']:.1f}",
            template="plotly_white",
            height=500,
            showlegend=True
        )

        # Update axis titles
        fig.update_xaxes(title_text="Actual Pollen", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Pollen", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Total Pollen", row=1, col=2)

        return fig