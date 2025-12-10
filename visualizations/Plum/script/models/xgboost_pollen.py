import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class XGBoostPollen:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        self.metrics = {}
        self.output_df = None

    def short_description(self):
        return "XGBoost (Rolling Means, Weighted Features, Early Stopping)."

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
        
        df['lag1'] = df[target].shift(1)
        df['roll3'] = df[target].shift(1).rolling(3).mean()
        df['roll7'] = df[target].shift(1).rolling(7).mean()
        df['roll30'] = df[target].shift(1).rolling(30).mean()
        
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

        df = df[df['Date'].dt.month.isin(range(3, 11))].copy()
        return df, target

    def predict(self, data_dict):
        df, target = self._engineer_features(data_dict)
        
        df['Year'] = df['Date'].dt.year
        train_df = df[df['Year'] < 2023].copy()
        test_df = df[df['Year'] >= 2023].copy()
        
        thresh = train_df[target].mean() + train_df[target].std()
        train_df['is_spike'] = (train_df[target].shift(1) > thresh).astype(int)
        test_df['is_spike'] = (test_df[target].shift(1) > thresh).astype(int)
        
        train_df = train_df.dropna().reset_index(drop=True)
        test_df = test_df.dropna().reset_index(drop=True)
        
        exclude_final = [target, "Date", "Year", "Month", "Day", "Week", "AQI_Category", "date_local", "Tree", "Grass", "Weed", "Ragweed"]
        features = [c for c in train_df.columns if c not in exclude_final]
        
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        preds = self.model.predict(X_test)
        
        self.metrics['MAE'] = mean_absolute_error(y_test, preds)
        self.metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, preds))
        self.metrics['R2'] = r2_score(y_test, preds)
        
        self.output_df = test_df[['Date']].copy()
        self.output_df['Actual_Pollen'] = y_test
        self.output_df['Predicted_Pollen'] = preds
        
        return self.output_df

    def plot_results(self, data_dict):
        if self.output_df is None:
            self.predict(data_dict)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        y_test = self.output_df['Actual_Pollen']
        preds = self.output_df['Predicted_Pollen']
        
        ax1.scatter(y_test, preds, alpha=0.5, color='orange')
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_title(f"Actual vs Predicted (R²={self.metrics.get('R2', 0):.2f})")
        ax1.set_xlabel("Actual Pollen")
        ax1.set_ylabel("Predicted Pollen")
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.output_df['Date'], y_test, label='Actual', color='black', alpha=0.7)
        ax2.plot(self.output_df['Date'], preds, label='Predicted', color='orange', alpha=0.7)
        ax2.set_title("XGBoost Time Series")
        ax2.set_xlabel("Date")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        
        plt.tight_layout()
        return fig
