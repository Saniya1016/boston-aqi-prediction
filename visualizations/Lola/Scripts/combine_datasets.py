import pandas as pd
import numpy as np
'''when all the csvs are merged to main, 
we can replace the dummy data with them'''

#read in data
weather_df = pd.read_csv(r'Data/boston-weather-data(open_meteo).csv', skiprows=2 ) #skip metadata lines
pollen_df = pd.read_csv(r'Data/EPHT_Pollen_Data.csv')
aqi_df = pd.read_csv(r'Data/boston_pollutants_with_aqi.csv')

#rename 'time' col to 'date'
weather_df.rename(columns={'time': 'date'}, inplace=True)
aqi_df.rename(columns={'time': 'date'}, inplace=True)

#put in datetime format
weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
aqi_df['date'] = pd.to_datetime(aqi_df['date'], errors='coerce')
pollen_df['date'] = pd.to_datetime(pollen_df['Date'], errors='coerce')

#choose relevant cols
pollen_df = pollen_df[['date', 'Tree', 'Grass', 'Weed', 'Ragweed', 'Total_Pollen']]

#fill missing vals w/ 0s
pollen_df.fillna(0, inplace=True)

#merge data (based on date)
merged_df = aqi_df.merge(weather_df, on='date', how='outer')
merged_df = merged_df.merge(pollen_df, on='date', how='outer')

#filling the missing data with column mean
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
merged_df[numeric_cols] = merged_df[numeric_cols].apply(lambda col: col.fillna(col.mean()))

#save to new file
# save to new file
merged_df.to_csv('visualizations/Lola/Scripts/merged_data.csv', index=False)
print("file saved as 'visualizations/Lola/Scripts/merged_data.csv'")