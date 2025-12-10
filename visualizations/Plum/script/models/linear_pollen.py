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
        self.output_df = None

    def short_description(self):
        return "Linear Regression for Pollen (Standard Scaled features)."

    def _engineer_features(self, data_dict):
        w = data_dict["weather"].copy()
        p = data_dict["pollutants"].copy()
        pol = data_dict["pollen"].copy()

        # Rename and Format Dates
        w = w.rename(columns={'time': 'Date'})
        p = p.rename(columns={'date': 'Date'})
        
        if 'Date' not in pol.columns:
            date_col = next((c for c in pol.columns if 'date' in c.lower()), None)
            if date_col:
                pol = pol.rename(columns={date_col: 'Date'})

        w['Date'] = pd.to_datetime(w['Date'])
        p['Date'] = pd.to_datetime(p['Date'])
        pol['Date'] = pd.to_datetime(pol['Date'])

        # Merge
        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        
        # Season Filter (March - Oct)
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        target = "Total_Pollen"
        exclude = ["Tree", "Grass", "Weed", "Ragweed", "Total_Pollen", "Date", "Year", "Month", "Day", "Week", "AQI_Category", "date_local"]
        feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]
        
        df = df.dropna(subset=feature_cols + [target])
        
        return df, feature_cols, target

    def predict(self, data_dict):
        df, feature_cols, target = self._engineer_features(data_dict)
        
        df['Year'] = df['Date'].dt.year
        train_df = df[df['Year'] < 2023].copy()
        test_df = df[df['Year'] >= 2023].copy()
        
        if len(train_df) == 0 or len(test_df) == 0:
            return pd.DataFrame(columns=['Date', 'Total_Pollen', 'Predicted_Pollen'])

        X_train = train_df[feature_cols]
        y_train = train_df[target]
        X_test = test_df[feature_cols]
        y_test = test_df[target]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        preds = self.model.predict(X_test_scaled)
        
        self.metrics['MAE'] = mean_absolute_error(y_test, preds)
        self.metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, preds))
        self.metrics['R2'] = r2_score(y_test, preds)

        self.output_df = test_df[['Date']].copy()
        self.output_df['Total_Pollen'] = y_test
        self.output_df['Predicted_Pollen'] = preds
        
        return self.output_df

    def plot_results(self, data_dict):
        if self.output_df is None:
            self.predict(data_dict)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Actual vs Predicted (R²={self.metrics.get('R2', 0):.2f})", "Time Series Prediction")
        )
        
        y_test = self.output_df['Total_Pollen']
        preds = self.output_df['Predicted_Pollen']
        
        # 1. Scatter
        fig.add_trace(
            go.Scatter(x=y_test, y=preds, mode='markers', name='Predictions', marker=dict(color='blue', opacity=0.5)),
            row=1, col=1
        )
        # Identity line
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')),
            row=1, col=1
        )

        # 2. Time Series
        fig.add_trace(
            go.Scatter(x=self.output_df['Date'], y=y_test, mode='lines', name='Actual', line=dict(color='black', width=1)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.output_df['Date'], y=preds, mode='lines', name='Predicted', line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        fig.update_layout(template="plotly_white", showlegend=False)
        return fig