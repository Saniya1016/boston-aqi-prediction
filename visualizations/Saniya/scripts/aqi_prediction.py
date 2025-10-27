import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from functions import init_aqi_data
import matplotlib.pyplot as plt

# Load merged AQI + weather data
df = init_aqi_data()

# Additional feature engineering
print("Creating additional features...")

# More lag features
df['AQI_lag_2'] = df['AQI'].shift(2)
df['AQI_lag_7'] = df['AQI'].shift(7)  # Weekly pattern
df['AQI_rolling_7'] = df['AQI'].rolling(7, min_periods=1).mean()

# Temporal features (seasonal patterns are important for AQI)
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['day_of_week'] = df['date'].dt.dayofweek
df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3)  # 1=Winter, 2=Spring, 3=Summer, 4=Fall

# Weather lags - these capture multi-day atmospheric conditions
if 'wind_speed_10m_max' in df.columns:
    df['wind_lag_1'] = df['wind_speed_10m_max'].shift(1)
    df['wind_rolling_3'] = df['wind_speed_10m_max'].rolling(3).mean()  # Stagnant air conditions
    
if 'temperature_2m_mean' in df.columns:
    df['temp_lag_1'] = df['temperature_2m_mean'].shift(1)
    
if 'precipitation_sum' in df.columns:
    df['is_rainy'] = (df['precipitation_sum'] > 0).astype(int)
    df['precip_lag_1'] = df['precipitation_sum'].shift(1)
    df['precip_rolling_3'] = df['precipitation_sum'].rolling(3).sum()  # Rain clears pollutants
    # Days since last rain (cumulative dry days)
    df['days_since_rain'] = (df['precipitation_sum'] == 0).astype(int).groupby(
        (df['precipitation_sum'] > 0).cumsum()
    ).cumsum()

# Weather interaction features
if 'temperature_2m_mean' in df.columns and 'wind_speed_10m_max' in df.columns:
    df['temp_wind_interaction'] = df['temperature_2m_mean'] * df['wind_speed_10m_max']

# Drop rows with NaNs from new features
df = df.dropna()

print(f"Final dataset: {len(df)} rows after feature engineering")

# Define feature set
features = [
    # Current weather
    'temperature_2m_mean',
    'apparent_temperature_mean',
    'precipitation_sum',
    'wind_speed_10m_max',
    'wind_gusts_10m_max',
    # AQI lags
    'AQI_lag_1',
    'AQI_lag_2',
    'AQI_lag_7',
    'AQI_rolling_3',
    'AQI_rolling_7',
    # Weather lags (test if these add value)
    'wind_lag_1',
    'wind_rolling_3',
    'temp_lag_1',
    'precip_lag_1',
    'precip_rolling_3',
    'days_since_rain',
    # Temporal
    'month',
    'day_of_year',
    'day_of_week',
    'season',
    # Interactions
    'temp_wind_interaction',
    'is_rainy'
]

# Filter to only existing columns
features = [f for f in features if f in df.columns]
print(f"\nUsing {len(features)} features: {features}")

X = df[features]
y = df['AQI']

# Chronological train/test split
split_date = '2019-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

X_train, y_train = train[features], train['AQI']
X_test, y_test = test[features], test['AQI']

print(f"\nTrain set: {len(X_train)} samples ({train['date'].min()} to {train['date'].max()})")
print(f"Test set: {len(X_test)} samples ({test['date'].min()} to {test['date'].max()})")

# =============================================================================
# Model 1: Random Forest (Your original)
# =============================================================================
print("\n" + "="*60)
print("MODEL 1: RANDOM FOREST")
print("="*60)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"R²: {r2_score(y_test, y_pred_rf):.4f}")

# =============================================================================
# Model 2: Gradient Boosting
# =============================================================================
print("\n" + "="*60)
print("MODEL 2: GRADIENT BOOSTING")
print("="*60)

gb = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_gb)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_gb):.2f}")
print(f"R²: {r2_score(y_test, y_pred_gb):.4f}")

# =============================================================================
# Feature Importance Analysis
# =============================================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE (Gradient Boosting)")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.to_string(index=False))

# Analysis: Are weather lags useful?
print("\n" + "="*60)
print("WEATHER LAG USEFULNESS ANALYSIS")
print("="*60)

weather_lag_features = ['wind_lag_1', 'wind_rolling_3', 'temp_lag_1', 'precip_lag_1', 
                        'precip_rolling_3', 'days_since_rain']
weather_lag_features = [f for f in weather_lag_features if f in features]

if weather_lag_features:
    weather_lag_importance = feature_importance[
        feature_importance['feature'].isin(weather_lag_features)
    ].sort_values('importance', ascending=False)
    
    print("\nWeather Lag Feature Importance:")
    print(weather_lag_importance.to_string(index=False))
    
    # Compare to AQI lags
    aqi_lag_features = ['AQI_lag_1', 'AQI_lag_2', 'AQI_lag_7', 'AQI_rolling_3', 'AQI_rolling_7']
    aqi_lag_importance = feature_importance[
        feature_importance['feature'].isin(aqi_lag_features)
    ].sort_values('importance', ascending=False)
    
    print("\nAQI Lag Feature Importance (for comparison):")
    print(aqi_lag_importance.to_string(index=False))
    
    # Verdict
    avg_weather_lag_imp = weather_lag_importance['importance'].mean()
    avg_aqi_lag_imp = aqi_lag_importance['importance'].mean()
    
    print(f"\nAverage Importance:")
    print(f"  Weather lags: {avg_weather_lag_imp:.4f}")
    print(f"  AQI lags:     {avg_aqi_lag_imp:.4f}")
    
    if avg_weather_lag_imp > 0.02:  # Threshold for "useful"
        print("\n✓ VERDICT: Weather lags ARE useful! They add predictive value.")
    else:
        print("\n✗ VERDICT: Weather lags are redundant. AQI lags capture enough info.")

# =============================================================================
# Performance by AQI Category
# =============================================================================
print("\n" + "="*60)
print("PERFORMANCE BY AQI CATEGORY (Gradient Boosting)")
print("="*60)

categories = [
    (0, 50, "Good"),
    (50, 100, "Moderate"),
    (100, 150, "Unhealthy for Sensitive"),
    (150, 300, "Unhealthy+")
]

for low, high, label in categories:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() > 0:
        rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred_gb[mask]))
        mae = mean_absolute_error(y_test[mask], y_pred_gb[mask])
        print(f"{label:30s} ({low:3d}-{high:3d}): RMSE={rmse:5.2f}, MAE={mae:5.2f}, n={mask.sum():4d}")

# =============================================================================
# Visualizations
# =============================================================================
fig = plt.figure(figsize=(16, 10))

# 1. Predicted vs Actual (Random Forest)
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_pred_rf, alpha=0.5, s=20)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel("Actual AQI", fontsize=11)
ax1.set_ylabel("Predicted AQI", fontsize=11)
ax1.set_title("Random Forest: Predicted vs Actual", fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.text(0.05, 0.95, f"R² = {r2_score(y_test, y_pred_rf):.3f}", 
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Predicted vs Actual (Gradient Boosting)
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test, y_pred_gb, alpha=0.5, s=20)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel("Actual AQI", fontsize=11)
ax2.set_ylabel("Predicted AQI", fontsize=11)
ax2.set_title("Gradient Boosting: Predicted vs Actual", fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.text(0.05, 0.95, f"R² = {r2_score(y_test, y_pred_gb):.3f}", 
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Residuals Plot (Gradient Boosting)
ax3 = plt.subplot(2, 3, 3)
residuals = y_test - y_pred_gb
ax3.scatter(y_pred_gb, residuals, alpha=0.5, s=20)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel("Predicted AQI", fontsize=11)
ax3.set_ylabel("Residuals (Actual - Predicted)", fontsize=11)
ax3.set_title("Residual Plot (Gradient Boosting)", fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# 4. Feature Importance
ax4 = plt.subplot(2, 3, 4)
top_features = feature_importance.head(10)
ax4.barh(range(len(top_features)), top_features['importance'])
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['feature'])
ax4.set_xlabel("Importance", fontsize=11)
ax4.set_title("Top 10 Feature Importance", fontsize=12, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

# 5. Time Series Comparison
ax5 = plt.subplot(2, 3, 5)
# Plot last 100 days of test set
plot_slice = slice(-100, None)
dates_plot = test['date'].iloc[plot_slice]
ax5.plot(dates_plot, y_test.iloc[plot_slice], 'o-', label='Actual', alpha=0.7, markersize=4)
ax5.plot(dates_plot, y_pred_gb[plot_slice], 's-', label='Predicted', alpha=0.7, markersize=4)
ax5.set_xlabel("Date", fontsize=11)
ax5.set_ylabel("AQI", fontsize=11)
ax5.set_title("Last 100 Days: Actual vs Predicted", fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

# 6. Error Distribution
ax6 = plt.subplot(2, 3, 6)
ax6.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax6.axvline(x=0, color='r', linestyle='--', lw=2)
ax6.set_xlabel("Residual (Actual - Predicted)", fontsize=11)
ax6.set_ylabel("Frequency", fontsize=11)
ax6.set_title("Distribution of Prediction Errors", fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)
ax6.text(0.05, 0.95, f"Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}", 
         transform=ax6.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('aqi_prediction_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved comprehensive analysis to 'aqi_prediction_analysis.png'")
plt.show()

# =============================================================================
# Prediction Examples
# =============================================================================
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

# Show some predictions with largest errors
errors = np.abs(y_test.values - y_pred_gb)
worst_indices = np.argsort(errors)[-5:]

print("\n5 Worst Predictions (Gradient Boosting):")
for idx in worst_indices:
    actual = y_test.iloc[idx]
    predicted = y_pred_gb[idx]
    date = test.iloc[idx]['date']
    error = actual - predicted
    print(f"Date: {date.strftime('%Y-%m-%d')}, Actual: {actual:.1f}, Predicted: {predicted:.1f}, Error: {error:+.1f}")