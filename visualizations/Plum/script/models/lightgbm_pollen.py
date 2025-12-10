import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class LightGBMPollen:
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=42
        )
        self.trained = False
        self.y_test = None
        self.y_pred = None
        self.metrics = {}
        self.test_df = None

    def short_description(self):
        return "LightGBM Regressor (Notebook Setup). Features: Lags, Rolling, Seasonality, Weighted Pollutants. Uses Early Stopping."

    def _engineer_features(self, data_dict):
        # Replicating the exact setup from the notebook
        w = data_dict["weather"].copy()
        p = data_dict["pollutants"].copy()
        pol = data_dict["pollen"].copy()

        w['Date'] = pd.to_datetime(w['time'])
        p['Date'] = pd.to_datetime(p['date'])
        pol['Date'] = pd.to_datetime(pol['Date'])

        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()

        target = "Total_Pollen"
        df["day_of_year"] = df["Date"].dt.dayofyear
        
        # Standard Lags
        df["lag1"] = df[target].shift(1)
        df["lag2"] = df[target].shift(2)
        df["lag3"] = df[target].shift(3)
        df["pollen_3day"] = df[target].rolling(3).mean()
        df["pollen_7day"] = df[target].rolling(7).mean()

        # Spike detection (Notebook logic)
        df["is_spike"] = ((df[target] - df["pollen_3day"]) > df["pollen_3day"].quantile(0.75)).astype(int)

        df = df.dropna()
        return df

    def predict(self, data_dict):
        df = self._engineer_features(data_dict)
        target = "Total_Pollen"

        remove_cols = ["Date", "time", "date", "Month", "Year", "Day", "Week", 
                       "Tree_Level", "Grass_Level", "Weed_Level", "Ragweed_Level", 
                       "AQI_Category", "weather_code (wmo code)", "OBJECTID",
                       "Tree", "Grass", "Weed", "Ragweed", target]
        
        # Select numeric features
        feature_cols = [c for c in df.columns if c not in remove_cols and df[c].dtype in ['float64', 'int64', 'int32']]
        
        X = df[feature_cols]
        y = df[target]

        # LightGBM crashes with special characters in column names like (°)
        # We sanitize them here
        X.columns = ["".join(c if c.isalnum() else "_" for c in col) for col in X.columns]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit with Early Stopping (Notebook logic)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50)]
        )
        self.trained = True
        
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)

        # Capture Dates for Time Series Plot (sorted chronologically)
        self.test_df = pd.DataFrame({
            "Date": df.loc[X_test.index, "Date"],
            "Actual": self.y_test.values,
            "Predicted": self.y_pred
        }).sort_values("Date")

        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2 = r2_score(self.y_test, self.y_pred)
        self.metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

        return pd.DataFrame({
            "Actual Pollen": self.y_test.values,
            "Predicted Pollen": self.y_pred
        })

    def plot_results(self, data_dict=None):
        if not self.trained:
            fig = go.Figure()
            fig.add_annotation(text="Model not trained.", x=0.5, y=0.5, showarrow=False)
            return fig

        actual = self.y_test
        predicted = self.y_pred

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
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(opacity=0.6, color='blue'),
            showlegend=False
        ), row=1, col=1)

        # Perfect Fit Line
        line_max = max(actual.max(), predicted.max())
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
            line=dict(color='blue'),
            opacity=0.7
        ), row=1, col=2)

        fig.update_layout(
            title_text=f"LightGBM Results | RMSE: {self.metrics['RMSE']:.1f}",
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