import pandas as pd

# Load datasets
weather = pd.read_csv("boston-weather-data(open_meteo).csv")
pollution = pd.read_csv("boston_pollutants_with_aqi.csv")
pollen = pd.read_csv("EPHT_Pollen_Data.csv")

# Parse dates for merging
weather['time'] = pd.to_datetime(weather['time'])
pollution['date'] = pd.to_datetime(pollution['date'])
pollen['Date'] = pd.to_datetime(pollen['Date'])

# Standardize date column names
weather.rename(columns={'time': 'Date'}, inplace=True)
pollution.rename(columns={'date': 'Date'}, inplace=True)

# Merge all
merged = weather.merge(pollution, on='Date', how='inner').merge(pollen, on='Date', how='inner')

merged.head()


cols = [
    'temperature_2m_mean (°C)',
    'precipitation_sum (mm)',
    'wind_speed_10m_max (km/h)',
    'PM2.5', 'O3', 'CO', 'NO2', 'SO2', 'AQI',
    'Tree', 'Grass', 'Weed', 'Ragweed', 'Total_Pollen'
]

# Only keep those that exist
numeric = merged[[c for c in cols if c in merged.columns]]

corr = numeric.corr()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap: Weather, Air Quality, and Pollen")
plt.show()







