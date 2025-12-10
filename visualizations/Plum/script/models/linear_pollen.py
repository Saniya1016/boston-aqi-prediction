# file: linear_pollen.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

class LinearPollen:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.trained = False
        self.y_test = None
        self.y_pred = None
        self.metrics = {}

    def short_description(self):
        return "Linear Regression for Pollen (Standard Scaled features)."

    def _engineer_features(self, data_dict):
        # Using basic numeric features as implied by the notebook's initial simple model
        w = data_dict["weather"].copy()
        p = data_dict["pollutants"].copy()
        pol = data_dict["pollen"].copy()

        w['Date'] = pd.to_datetime(w['time'])
        p['Date'] = pd.to_datetime(p['date'])
        pol['Date'] = pd.to_datetime(pol['Date'])

        merged = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        merged = merged[merged['Date'].dt.month.isin(range(3, 11))].copy()
        
        # Drop rows with missing values
        merged = merged.dropna()
        
        return merged

    def predict(self, data_dict):
        df = self._engineer_features(data_dict)
        target = "Total_Pollen"
        
        # Select numeric features excluding target and specific non-predictors
        remove_cols = ["Date", "time", "date", "Month", "Year", "Day", "Week", 
                       "Tree_Level", "Grass_Level", "Weed_Level", "Ragweed_Level", 
                       "AQI_Category", "weather_code (wmo code)", "OBJECTID",
                       "Tree", "Grass", "Weed", "Ragweed", target]
        
        feature_cols = [c for c in df.columns if c not in remove_cols and df[c].dtype in ['float64', 'int64', 'int32']]
        
        # Split by year (Notebook logic: < 2023 train, >= 2023 test)
        df = df.sort_values("Date")
        train_df = df[df['Date'].dt.year < 2023]
        test_df = df[df['Date'].dt.year >= 2023]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target]
        X_test = test_df[feature_cols]
        self.y_test = test_df[target]
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.trained = True
        
        self.y_pred = self.model.predict(X_test_scaled)
        
        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2 = r2_score(self.y_test, self.y_pred)
        self.metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

        return pd.DataFrame({
            "Actual Pollen": self.y_test.values,
            "Predicted Pollen": self.y_pred
        })

    def plot_results(self, data_dict):
        if not self.trained:
            fig = go.Figure()
            fig.add_annotation(text="Model not trained.", x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.y_test,
            y=self.y_pred,
            mode='markers',
            name='Test Predictions',
            opacity=0.6,
            marker=dict(color='blue')
        ))

        line_max = max(self.y_test.max(), self.y_pred.max())
        fig.add_trace(go.Scatter(
            x=[0, line_max], 
            y=[0, line_max],
            mode='lines',
            name='Perfect Fit',
            line=dict(dash='dash', color='red')
        ))

        fig.update_layout(
            title=f"Linear Regression<br>R²: {self.metrics['R2']:.3f} | MAE: {self.metrics['MAE']:.1f}",
            xaxis_title="Actual Total Pollen",
            yaxis_title="Predicted Total Pollen",
            template="plotly_white"
        )
        return fig