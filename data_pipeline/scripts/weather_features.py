# external_features.py
import pandas as pd
import holidays

def add_external_features(df: pd.DataFrame, weather_path=None):
    """Add weather and holiday flags."""
    df['date'] = df['hour_timestamp'].dt.date
    df['is_holiday'] = df['date'].isin(holidays.US(years=[2023, 2024, 2025])).astype(int)

    if weather_path:
        weather = pd.read_csv(weather_path, parse_dates=['datetime'])
        weather['date'] = weather['datetime'].dt.date
        df = df.merge(weather[['date', 'temperature', 'precipitation']], on='date', how='left')

    print("✅ Added external (holiday/weather) features.")
    return df
