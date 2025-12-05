import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

class LinearRegressionAQI:
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def short_description(self):
        return "Linear Regression baseline using pollutants (PM2.5, O3, CO, NO2, SO2) to predict AQI with 80/20 train-test split."

    def predict(self, data_dict):
        df = data_dict["pollutants"].copy()

        # ==== Select features ====
        features = ["PM2.5", "O3", "CO", "NO2", "SO2"]
        target = "AQI"

        # Drop null rows
        df = df.dropna(subset=features + [target])

        X = df[features]
        y = df[target]

        # ==== Train-test split ====
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )

        # ==== Fit model ====
        self.model.fit(X_train, y_train)
        self.trained = True

        # Store for plotting
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)

        results = pd.DataFrame({
            "actual_AQI": y_test.values,
            "predicted_AQI": self.y_pred
        })

        return results

    def plot_results(self, data_dict=None):
        if not self.trained:
            fig = go.Figure()
            fig.add_annotation(text="Model not trained yet.", x=0.5, y=0.5, showarrow=False)
            return fig

        actual = self.y_test.values
        predicted = self.y_pred

        fig = go.Figure()

        # Scatter points
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            opacity=0.6
        ))

        # Perfect fit reference line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Fit (y=x)',
            line=dict(dash='dash', color='red')
        ))

        fig.update_layout(
            title="Actual vs Predicted AQI (Test Set)",
            xaxis_title="Actual AQI",
            yaxis_title="Predicted AQI",
            template="plotly_white",
        )

        return fig

