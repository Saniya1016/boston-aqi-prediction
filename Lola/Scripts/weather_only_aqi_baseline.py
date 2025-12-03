import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#Load merged dataset
df = pd.read_csv("Lola/merged_data.csv")

print("Columns in merged dataframe:")
print(df.columns)

#Define target and weather-only features
target_col = "AQI"

weather_cols = [
    "temperature_2m_mean (°C)",
    "precipitation_sum (mm)",
    "apparent_temperature_mean (°C)",
    "wind_gusts_10m_max (km/h)",
    "wind_speed_10m_max (km/h)",
    "wind_direction_10m_dominant (°)",
]

#Drop rows with missing target or weather features
df_model = df.dropna(subset=[target_col] + weather_cols)

X = df_model[weather_cols]
y = df_model[target_col]

print(f"\nUsing {len(weather_cols)} weather features:")
for c in weather_cols:
    print("  -", c)
print(f"\nNumber of rows after dropping NA: {len(df_model)}")

#Time-aware train/validation split (no shuffle)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"\nTrain size: {len(X_train)}, Val size: {len(X_val)}")

#Linear Regression baseline
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_train_pred_lr = linreg.predict(X_train)
y_val_pred_lr = linreg.predict(X_val)

r2_train_lr = r2_score(y_train, y_train_pred_lr)
r2_val_lr = r2_score(y_val, y_val_pred_lr)

print("\n=== Weather-only AQI baseline: Linear Regression ===")
print(f"Train R^2: {r2_train_lr:.3f}")
print(f"Val   R^2: {r2_val_lr:.3f}")

#Random Forest 
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=0,
    n_jobs=-1,
)

rf.fit(X_train, y_train)
y_val_pred_rf = rf.predict(X_val)
r2_val_rf = r2_score(y_val, y_val_pred_rf)

print("\n=== Weather-only AQI baseline: Random Forest ===")
print(f"Val R^2: {r2_val_rf:.3f}")

#Save summary to file
with open("Lola/weather_only_aqi_results.txt", "w") as f:
    f.write("Weather-only AQI baseline (no lagged AQI, no pollen, no pollutants)\n")
    f.write(f"Features: {weather_cols}\n")
    f.write(f"Train R^2 (LinearRegression): {r2_train_lr:.3f}\n")
    f.write(f"Val   R^2 (LinearRegression): {r2_val_lr:.3f}\n")
    f.write(f"Val   R^2 (RandomForest): {r2_val_rf:.3f}\n")

print("\nSaved results to Lola/weather_only_aqi_results.txt")
