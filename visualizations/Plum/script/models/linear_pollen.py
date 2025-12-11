import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class LinearPollen:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.metrics = {}
        # Changed to match your plotting code requirements
        self.test_data = None
        self.trained = False

    def short_description(self):
        return "Linear Regression for Pollen (Standard Scaled features, Time-split 2023)."

    def _merge_data(self, data):
        """
        Internal helper to merge the dictionary of DataFrames 
        (pollutants, weather, pollen) into one DataFrame.
        """
        pollen_df = data['pollen'].copy()
        weather_df = data['weather'].copy()
        
        # Normalize Date columns for merging
        if 'Date' in pollen_df.columns:
            pollen_df = pollen_df.rename(columns={'Date': 'date'})
        
        if 'time' in weather_df.columns:
            weather_df = weather_df.rename(columns={'time': 'date'})
            
        # Merge Weather and Pollen
        merged = pd.merge(pollen_df, weather_df, on='date', how='inner')
        merged = merged.sort_values('date').reset_index(drop=True)
        return merged

    def _engineer_features(self, df):
        df = df.copy()
        target = "Total_Pollen"
        
        # Auto-select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        remove_cols = [
            "Tree", "Grass", "Weed", "Ragweed", "Total_Pollen",
            "Year", "Month", "Day", "Week",
            "OBJECTID", "AQI_Category", "index", "level_0"
        ]
        
        feature_cols = [c for c in numeric_cols if c not in remove_cols and c in df.columns]
        df = df.dropna(subset=feature_cols)

        return df, feature_cols, target

    def predict(self, data):
        # 1. Merge and Engineer
        merged_df = self._merge_data(data)
        df, feature_cols, target = self._engineer_features(merged_df)

        # 2. Split
        df["Year"] = df["date"].dt.year
        train_df = df[df["Year"] < 2023].copy()
        test_df  = df[df["Year"] >= 2023].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(f"Split failed. Train rows: {len(train_df)}, Test rows: {len(test_df)}")

        X_train = train_df[feature_cols]
        y_train = train_df[target]
        X_test  = test_df[feature_cols]
        y_test  = test_df[target]

        # 3. Train
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.trained = True  # Set trained flag for plot_results

        preds = self.model.predict(X_test_scaled)

        # 4. Metrics
        self.metrics["MAE"]  = mean_absolute_error(y_test, preds)
        self.metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, preds))
        self.metrics["R2"]   = r2_score(y_test, preds)

        # 5. Output Construction
        # Using 'Observed' and 'Forecast' to match your plot_results code
        self.test_data = test_df[["date"]].copy()
        self.test_data["Observed"] = y_test.values
        self.test_data["Forecast"] = preds

        return self.test_data

    def plot_results(self, data=None):
        """
        Custom plotting logic provided by user.
        """
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

        # Scatter Plot
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