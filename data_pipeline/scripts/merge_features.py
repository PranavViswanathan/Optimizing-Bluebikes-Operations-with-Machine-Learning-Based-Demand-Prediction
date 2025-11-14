# feature_merger.py
import pandas as pd
from .spatial_features import add_spatial_features
from .weather_features import add_external_features

def merge_all_features(base_df: pd.DataFrame):
    df = base_df.copy()
    df = add_spatial_features(df)
    df = add_external_features(df, weather_path="data/weather_boston.csv")
    # Optional: merge user behavior if applicable
    # df = df.merge(add_user_behavior_features(original_trips_df), on=['station_name', 'date'], how='left')
    return df
