import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#load data
df = pd.read_csv('merged_data.csv')

#datetime format
df['date'] = pd.to_datetime(df['date'])

#set cols to be plotted
variables = [
    'AQI',
    'temperature_2m_mean',
    'Tree',
    'Grass',
    'Weed',
    'Ragweed',
    'Total_Pollen'
]

#output folder
output_dir = './Plots'
os.makedirs(output_dir, exist_ok=True)

#plot and save variables
for var in variables:
    if var not in df.columns:
        print(f"Skipping '{var}' (not found in data)")
        continue

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='date', y=var)
    plt.title(f'{var} Over Time')
    plt.xlabel('Date')
    plt.ylabel(var)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{var}_timeseries.png')
    plt.close()
    print(f"Saved {var}_timeseries.png")
print("charts saved to 'Plots' folder")