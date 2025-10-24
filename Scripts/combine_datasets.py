import pandas as pd
import numpy as np
'''when all the csvs are merged to main, 
we can replace the dummy data with them'''

#for real data
# weather_df = pd.read_csv('../Data/boston-weather-data(open_meteo).csv')
# pollen_df = pd.read_csv('../Data/EPHT_Pollen_Data.csv')
# aqi_df = pd.read_csv('../Data/boston_pollutants_with_aqi.csv')

#rename 'time' col to 'date'
#weather_df.rename(columns={'time': 'date'}, inplace=True)
#aqi_df.rename(columns={'time': 'date'}, inplace=True)

#put in datetime format
#weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
#aqi_df['date'] = pd.to_datetime(aqi_df['date'], errors='coerce')
#pollen_df['date'] = pd.to_datetime(pollen_df['Date'], errors='coerce')

#choose relevant cols
#pollen_df = pollen_df[['date', 'Tree', 'Grass', 'Weed', 'Ragweed', 'Total_Pollen']]

#fill missing vals w/ 0s
#pollen_df.fillna(0, inplace=True)

#dummy data

#weather
weather_df = pd.DataFrame({
    'date': pd.date_range('2009-01-01', periods=10),
    'temperature_2m_mean': np.random.uniform(-10, 30, 10),
    'precipitation_sum': np.random.uniform(0, 10, 10),
    'wind_speed_10m_max': np.random.uniform(0, 50, 10),
    'wind_direction_10m_dominant': np.random.randint(0, 360, 10)
})

#pollen
pollen_df = pd.DataFrame({
     'date': pd.date_range('2009-01-01', periods=10),
    'tree_pollen': np.random.randint(0, 300, 10),
    'grass_pollen': np.random.randint(0, 200, 10),
    'weed_pollen': np.random.randint(0, 150, 10)
})

#AQI(Airnow)
aqi_df = pd.DataFrame({
    'date': pd.date_range('2009-01-01', periods=10),
    'PM2.5': np.random.uniform(5, 25, 10),
    'O3': np.random.uniform(0.01, 0.05, 10),
    'CO': np.random.uniform(0.2, 1.0, 10),
    'NO2': np.random.uniform(10, 40, 10),
    'SO2': np.random.uniform(1, 10, 10),
    'AQI': np.random.randint(20, 80, 10),
    'AQI_Category': np.random.choice(['Good', 'Moderate', 'Unhealthy'], 10)
})

#merge data (based on date)
merged_df = aqi_df.merge(weather_df, on='date', how='outer')
merged_df = merged_df.merge(pollen_df, on='date', how='outer')

#filling the missing data with column mean
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
merged_df[numeric_cols] = merged_df[numeric_cols].apply(lambda col: col.fillna(col.mean()))

#save to new file
merged_df.to_csv('merged_data.csv', index=False)
print("file saved as 'merged_data.csv'")