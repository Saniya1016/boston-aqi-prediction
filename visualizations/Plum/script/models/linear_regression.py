import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

    def plot_results(self, data_dict, fig):
        if not self.trained:
            plt.text(0.5, 0.5, "Model not trained yet.", ha="center")
            return

        actual = self.y_test.values
        predicted = self.y_pred

        # Scatter plot
        plt.scatter(actual, predicted, alpha=0.6, label="Predictions")

        # Diagonal line (perfect prediction)
        line_min = min(actual.min(), predicted.min())
        line_max = max(actual.max(), predicted.max())
        plt.plot([line_min, line_max], [line_min, line_max], 'r--', label="Perfect Fit (y=x)")

        # Labels and style
        plt.xlabel("Actual AQI")
        plt.ylabel("Predicted AQI")
        plt.title("Actual vs Predicted AQI (Test Set)")
        plt.legend()
        plt.tight_layout()
