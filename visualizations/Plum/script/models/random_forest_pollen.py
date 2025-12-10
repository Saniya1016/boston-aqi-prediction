import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RandomForestPollen:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            min_samples_split=2,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42
        )
        self.metrics = {}
        self.output_df = None

    def short_description(self):
        return "Random Forest (Log-target, Rolling/Lag Features, Interactions)."

    def _engineer_features(self, data_dict):
        w = data_dict["weather"].copy()
        p = data_dict["pollutants"].copy()
        pol = data_dict["pollen"].copy()
        
        w = w.rename(columns={'time': 'Date'})
        p = p.rename(columns={'date': 'Date'})
        if 'Date' not in pol.columns:
            date_col = next((c for c in pol.columns if 'date' in c.lower()), None)
            if date_col:
                pol = pol.rename(columns={date_col: 'Date'})

        w['Date'] = pd.to_datetime(w['Date'])
        p['Date'] = pd.to_datetime(p['Date'])
        pol['Date'] = pd.to_datetime(pol['Date'])
        
        df = w.merge(p, on='Date', how='inner').merge(pol, on='Date', how='inner')
        df = df.sort_values('Date').reset_index(drop=True)
        
        target = "Total_Pollen"
        
        # Features
        df['lag1'] = df[target].shift(1)
        df['lag2'] = df[target].shift(2)
        df['pollen_3day'] = df[target].shift(1).rolling(3).mean()
        df['pollen_7day'] = df[target].shift(1).rolling(7).mean()
        
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        
        weights = np.array([0.1, 0.3, 0.6])
        exclude_cols = ["Tree", "Grass", "Weed", "Ragweed", target, "Date", "Year", "Month", "Day", "Week", "AQI_Category", "date_local"]
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude_cols]
        
        for col in numeric_cols:
            s0 = df[col].shift(2)
            s1 = df[col].shift(1)
            s2 = df[col]
            df[f'{col}_weighted3'] = (weights[0]*s0 + weights[1]*s1 + weights[2]*s2)
            df[f'{col}_lag1'] = df[col].shift(1)

        if 'temperature_2m_mean (°C)' in df.columns and 'AQI' in df.columns:
            df['temp_x_AQI'] = df['temperature_2m_mean (°C)'] * df['AQI']
        
        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()
        return df, target

    def predict(self, data_dict):
        df, target = self._engineer_features(data_dict)
        
        df['Year'] = df['Date'].dt.year
        train_df = df[df['Year'] < 2023].copy()
        test_df = df[df['Year'] >= 2023].copy()
        
        # Spike
        spike_thresh = train_df[target].mean() + train_df[target].std()
        train_df['is_spike'] = (train_df[target].shift(1) > spike_thresh).astype(int)
        test_df['is_spike'] = (test_df[target].shift(1) > spike_thresh).astype(int)
        
        train_df = train_df.dropna().reset_index(drop=True)
        test_df = test_df.dropna().reset_index(drop=True)
        
        exclude_final = [target, "Date", "Year", "Month", "Day", "Week", "AQI_Category", "date_local", "Tree", "Grass", "Weed", "Ragweed"]
        features = [c for c in train_df.columns if c not in exclude_final]
        
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]
        
        y_train_log = np.log1p(y_train)
        
        self.model.fit(X_train, y_train_log)
        
        preds_log = self.model.predict(X_test)
        preds = np.expm1(preds_log)
        
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
            subplot_titles=(f"Actual vs Predicted (R²={self.metrics.get('R2', 0):.2f})", "Random Forest Time Series")
        )
        
        y_test = self.output_df['Total_Pollen']
        preds = self.output_df['Predicted_Pollen']
        
        fig.add_trace(
            go.Scatter(x=y_test, y=preds, mode='markers', name='Predictions', marker=dict(color='green', opacity=0.5)),
            row=1, col=1
        )
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.output_df['Date'], y=y_test, mode='lines', name='Actual', line=dict(color='black', width=1)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.output_df['Date'], y=preds, mode='lines', name='Predicted', line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        fig.update_layout(template="plotly_white", showlegend=False)
        return fig