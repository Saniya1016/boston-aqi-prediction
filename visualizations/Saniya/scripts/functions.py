import pandas as pd
import numpy as np


def load_csv(file, skiprows=0, date_col=None):
    df = pd.read_csv(file, skiprows=skiprows)
    if date_col is None:
        date_col = 'time'  # default for weather files
    
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in {file}")
    
    df['date'] = pd.to_datetime(df[date_col])
    return df


def merge_data(df_1, df_2):
    merged_df = pd.merge(df_1, df_2, on='date', how='inner')
    print(f"✓ Merged dataset: {len(merged_df)} days")
    print(f"  Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    return merged_df


def add_3_day_lagged_column(df, col):
    rolling_col = f"{col}_rolling_3"
    df[rolling_col] = df[col].rolling(window=3, min_periods=1).mean()
    return df


def add_column_yesterdays_aqi(df, col):
    lag_col = f"{col}_lag_1"
    df[lag_col] = df[col].shift(1)
    return df


def init_aqi_data():
    df1 = load_csv('../data/boston-weather-data(open_meteo).csv', skiprows=3) #change location of file to match current
    df2 = load_csv('../data/boston_pollutants_with_aqi.csv', date_col='date') #change location of file to match current

    merged_df = merge_data(df1, df2)

    merged_df = add_column_yesterdays_aqi(merged_df, 'AQI')
    merged_df = add_3_day_lagged_column(merged_df, 'AQI')

    # # If you want to do the same for PM2.5
    # merged_df = add_column_yesterdays_aqi(merged_df, 'PM2.5')
    # merged_df = add_3_day_lagged_column(merged_df, 'PM2.5')

    # Drop NaNs created by lag
    merged_df = merged_df.dropna()

    return merged_df