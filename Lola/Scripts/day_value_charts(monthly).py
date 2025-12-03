import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('merged_data.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

#filter to last 3 years
df = df[df['date'] >= '2022-01-01']

#group by month (take monthly mean)
df_monthly = (
    df.set_index('date')
      .resample('M')  #monthly frequency
      .mean(numeric_only=True)
      .reset_index()
)

variables = [
    'AQI',
    'temperature_2m_mean',
    'Tree',
    'Grass',
    'Weed',
    'Ragweed',
    'Total_Pollen'
]
output_dir = './Plots (Monthly)'
os.makedirs(output_dir, exist_ok=True)

for var in variables:
    if var not in df_monthly.columns:
        print(f"Skipping '{var}' (not found in data)")
        continue

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_monthly, x='date', y=var, marker='o')
    plt.title(f'{var} (Monthly Average, 2022–Present)')
    plt.xlabel('Date')
    plt.ylabel(var)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{var}_monthly_timeseries.png')
    plt.close()
    print(f"Saved {var}_monthly_timeseries.png")

print("Charts saved to 'Plots (Monthly)' folder.")
