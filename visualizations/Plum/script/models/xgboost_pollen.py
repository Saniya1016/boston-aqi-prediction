import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class XGBoostPollen:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist', # Matches 'hist' usually implied by fast execution
            n_estimators=1200,
            early_stopping_rounds=50,
            eval_metric='rmse',
            seed=42
        )
        self.trained = False
        self.y_test = None
        self.y_pred = None
        self.metrics = {}
        self.test_df = None

    def short_description(self):
        return "XGBoost Regressor (Notebook Setup). Full feature set: Seasonality, Weighted Rolling, Lags, Interactions."

    def _engineer_features(self, data_dict):
        # 1. Merge
        w = data_dict["weather"].copy()
        p = data_dict["pollutants"].copy()
        pol = data_dict["pollen"].copy()

        w['Date'] = pd.to_datetime(w['time'])
        p['Date'] = pd.to_datetime(p['date'])
        pol['Date'] = pd.to_datetime(pol['Date'])

        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()

        # 2. Base Features
        target = "Total_Pollen"
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["lag1"] = df[target].shift(1)
        df["lag2"] = df[target].shift(2)
        df["lag3"] = df[target].shift(3)
        df["pollen_3day"] = df[target].rolling(3).mean()
        df["pollen_7day"] = df[target].rolling(7).mean()

        threshold = df[target].mean() + df[target].std()
        df["is_spike"] = (df[target] > threshold).astype(int)

        # 3. Seasonal (Sin/Cos)
        df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # 4. Weighted Rolling + Lags
        weights = np.array([0.1, 0.3, 0.6])
        rolling_cols = [
            "temperature_2m_mean (°C)", "apparent_temperature_mean (°C)",
            "PM2.5", "O3", "CO", "NO2", "SO2",
            "AQI_PM2.5", "AQI_O3", "AQI_CO", "AQI_NO2", "AQI_SO2", "AQI"
        ]
        # Filter existing
        rolling_cols = [c for c in rolling_cols if c in df.columns]

        for col in rolling_cols:
            df[f'{col}_weighted3'] = (
                df[col].shift(2).bfill() * weights[0] +
                df[col].shift(1).bfill() * weights[1] +
                df[col] * weights[2]
            )
            for lag in range(1, 4):
                df[f"{col}_lag{lag}"] = df[col].shift(lag).bfill()

        # 5. Interactions
        interaction_pairs = [
            ('temperature_2m_mean (°C)', 'AQI'),
            ('apparent_temperature_mean (°C)', 'PM2.5'),
            ('wind_speed_10m_max (km/h)', 'O3'),
        ]
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

        df = df.dropna()
        return df

    def predict(self, data_dict):
        df = self._engineer_features(data_dict)
        target = "Total_Pollen"

        drop_cols = ["Date", "time", "date", "Month", "Year", "Day", "Week", 
                     "Tree_Level", "Grass_Level", "Weed_Level", "Ragweed_Level", 
                     "AQI_Category", "OBJECTID", "Tree", "Grass", "Weed", "Ragweed", "day_of_year", target]
        
        feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in ['float64', 'int64', 'int32']]
        
        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
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
            marker=dict(opacity=0.6, color='green'),
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
            line=dict(color='green'),
            opacity=0.7
        ), row=1, col=2)

        fig.update_layout(
            title=f"XGBoost Results | RMSE: {self.metrics['RMSE']:.1f}",
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