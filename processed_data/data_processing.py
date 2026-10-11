import os
import numpy as np
import pandas as pd


def step1_load_and_filter_data(df):
    print('Step 1: Time Scoping and Initial Filtering...')

    # Standardize and Sort datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Time Scoping (March 2024 to December 2025)
    df.set_index('datetime', inplace=True)
    df_scoped = df.loc['2024-03-01':'2025-12-31'].copy()
    df_scoped = df_scoped.reset_index()

    # Initial Feature Selection
    keeped_cols = [
        'datetime', 'hour', 'day', 'month', 'year', 'day_of_week',
        'is_weekend', 'pm25', 'pm25_lag1', 'temperature', 'humidity',
        'wind_speed', 'wind_direction', 'precipitation', 'is_raining',
        'is_dry_season',
    ]
    actual_keep_cols = [col for col in keeped_cols if col in df_scoped.columns]

    return df_scoped[actual_keep_cols].copy()


def step2_add_features(df):
    print('Step 2: Adding Categorical Features and Binning...')

    # WHO Exceedance
    df['is_who_exceeded'] = (df['pm25'] > 15.0).astype(int)

    # AQI Category
    def get_aqi_category(pm):
        if pd.isna(pm): return 'Unknown'
        if pm <= 12.0:
            return 'Good'
        elif pm <= 35.4:
            return 'Moderate'
        elif pm <= 55.4:
            return 'Unhealthy for Sensitive Groups'
        elif pm <= 150.4:
            return 'Unhealthy'
        elif pm <= 250.4:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'

    df['aqi_category'] = df['pm25'].apply(get_aqi_category)

    # Wind Direction Cardinal
    def get_cardinal_wind(deg):
        if pd.isna(deg): return 'N/A'
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = int(((deg % 360) + 22.5) // 45) % 8
        return directions[idx]

    df['wind_direction_cardinal'] = df['wind_direction'].apply(get_cardinal_wind)

    # Wind Speed Group
    def get_wind_speed_group(speed):
        if pd.isna(speed): return 'Unknown'
        if speed < 2.0:
            return 'Low'
        elif speed <= 5.0:
            return 'Medium'
        else:
            return 'High'

    df['wind_speed_group'] = df['wind_speed'].apply(get_wind_speed_group)

    # Time of Day Group
    def get_time_of_day(hour):
        if 7 <= hour <= 9:
            return 'Morning Rush'
        elif 10 <= hour <= 15:
            return 'Mid-day'
        elif 17 <= hour <= 19:
            return 'Evening Rush'
        else:
            return 'Night & Early Morning'

    df['time_of_day_group'] = df['hour'].apply(get_time_of_day)

    # Season Label
    df['season_label'] = df['is_dry_season'].map({1: 'Dry Season', 0: 'Wet Season'})

    # Cleansing Delta (Handling rain effect)
    if 'pm25_lag1' in df.columns and 'is_raining' in df.columns:
        raw_delta = np.where(df['is_raining'] == 1, df['pm25_lag1'] - df['pm25'], 0.0)
        df['cleansing_delta'] = np.clip(raw_delta, a_min=0, a_max=None).round(2)
        df.drop(columns=['pm25_lag1'], inplace=True)
    else:
        df['cleansing_delta'] = 0.0

    # Temperature Binning
    df['temp_bin'] = pd.cut(
        df['temperature'],
        bins=[-np.inf, 15, 20, 25, 30, np.inf],
        labels=['<15°C', '15-20°C', '20-25°C', '25-30°C', '>30°C'],
    )

    # Humidity Binning
    df['humidity_bin'] = pd.cut(
        df['humidity'],
        bins=[-np.inf, 50, 65, 80, 90, np.inf],
        labels=['<50%', '50-65%', '65-80%', '80-90%', '>90%'],
    )

    return df


def step3_clean_and_export(df):
    print('Step 3: Final Cleaning, Rounding, and Export Prep...')

    # Rounding float columns
    float_cols = ['pm25', 'temperature', 'humidity', 'wind_speed', 'precipitation', 'cleansing_delta']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].round(2)

    # Final Feature Selection
    keep_cols = [
        'datetime', 'hour', 'day', 'month', 'year', 'day_of_week',
        'is_weekend', 'is_dry_season', 'season_label', 'pm25',
        'is_who_exceeded', 'aqi_category', 'temperature', 'temp_bin',
        'humidity', 'humidity_bin', 'wind_speed', 'wind_speed_group',
        'wind_direction', 'wind_direction_cardinal', 'precipitation',
        'is_raining', 'cleansing_delta', 'time_of_day_group',
    ]

    actual_cols = [col for col in keep_cols if col in df.columns]

    return df[actual_cols].copy()


def run_data_pipeline(input_path, output_path):
    print("--- STARTING DATA PIPELINE ---")
    print(f"Reading from: {input_path}")

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Could not find '{input_path}'. Please check the file path.")
        return

    # Read the raw data once
    raw_df = pd.read_csv(input_path)

    # Run the pipeline sequentially in memory (avoids saving intermediate CSVs)
    df_step1 = step1_load_and_filter_data(raw_df)
    df_step2 = step2_add_features(df_step1)
    final_df = step3_clean_and_export(df_step2)

    # Export final result
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Pipeline complete! Saved {len(final_df)} rows to: {output_path}")
    print("--- DONE ---")


if __name__ == '__main__':
    # Update these filenames if necessary
    INPUT_FILE = 'hanoi_aqi_ml_ready_fixed.csv'
    OUTPUT_FILE = 'hanoi_air_quality_clean.csv'

    run_data_pipeline(INPUT_FILE, OUTPUT_FILE)