# Code to generate features for the data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import logging
from datetime import datetime
import wandb
import argparse
import sys

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    logger.info("Loading data files.")
    bluebike_data = pd.read_pickle('D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\after_duplicates.pkl')
    weather_data = pd.read_pickle('D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\noaa_weather\\after_duplicates.pkl')
    college_data = pd.read_pickle('D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\boston_clg\\after_duplicates.pkl')
    
    logger.info(f"Data loaded: {len(bluebike_data):,} rides")
    
    logger.info("Processing bike ride data.")
    bluebike_data['start_time'] = pd.to_datetime(bluebike_data['start_time']).dt.tz_localize(None)
    bluebike_data['stop_time'] = pd.to_datetime(bluebike_data['stop_time']).dt.tz_localize(None)
    
    bluebike_data['date'] = bluebike_data['start_time'].dt.date
    bluebike_data['hour'] = bluebike_data['start_time'].dt.hour
    bluebike_data['day_of_week'] = bluebike_data['start_time'].dt.dayofweek
    bluebike_data['month'] = bluebike_data['start_time'].dt.month
    bluebike_data['year'] = bluebike_data['start_time'].dt.year
    bluebike_data['day'] = bluebike_data['start_time'].dt.day
    
    bluebike_data['duration_minutes'] = (bluebike_data['stop_time'] - bluebike_data['start_time']).dt.total_seconds() / 60
    
    original_count = len(bluebike_data)
    bluebike_data = bluebike_data[(bluebike_data['duration_minutes'] > 0) & 
                                    (bluebike_data['duration_minutes'] < 1440)]
    logger.info(f"Filtered invalid durations: {original_count:,} -> {len(bluebike_data):,}")
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    bluebike_data['distance_km'] = haversine_distance(
        bluebike_data['start_station_latitude'],
        bluebike_data['start_station_longitude'],
        bluebike_data['end_station_latitude'],
        bluebike_data['end_station_longitude']
    )
    
    bluebike_data['is_member'] = (bluebike_data['user_type'] == 'member').astype(int)
    
    logger.info("Creating hourly aggregations...")
    hourly_rides = bluebike_data.groupby(['date', 'hour']).agg({
        'ride_id': 'count',
        'duration_minutes': ['mean', 'std', 'median'],
        'distance_km': ['mean', 'std', 'median'],
        'is_member': 'mean',
        'day_of_week': 'first',
        'month': 'first',
        'year': 'first',
        'day': 'first'
    }).reset_index()
    
    hourly_rides.columns = ['date', 'hour', 'ride_count', 'duration_mean', 'duration_std', 
                            'duration_median', 'distance_mean', 'distance_std', 'distance_median',
                            'member_ratio', 'day_of_week', 'month', 'year', 'day']
    
    hourly_rides['duration_std'] = hourly_rides['duration_std'].fillna(0)
    hourly_rides['distance_std'] = hourly_rides['distance_std'].fillna(0)
    
    date_range = pd.date_range(
        start=hourly_rides['date'].min(),
        end=hourly_rides['date'].max(),
        freq='D'
    ).date
    
    all_hours = range(24)
    full_index = pd.MultiIndex.from_product([date_range, all_hours], names=['date', 'hour'])
    
    hourly_rides_complete = hourly_rides.set_index(['date', 'hour']).reindex(full_index, fill_value=0)
    
    temporal_cols = ['day_of_week', 'month', 'year', 'day']
    for col in temporal_cols:
        hourly_rides_complete[col] = hourly_rides_complete.groupby(level='date')[col].transform(
            lambda x: x.replace(0, np.nan).ffill().bfill().fillna(0)
        )
    
    hourly_rides_complete = hourly_rides_complete.reset_index()
    
    logger.info("Merging with weather data...")
    weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date
    model_data = pd.merge(hourly_rides_complete, weather_data, on='date', how='left')
    model_data = model_data.dropna(subset=['TMAX', 'TMIN', 'PRCP'])
    logger.info(f"Final dataset: {len(model_data):,} records")
    
    logger.info("Creating features...")
    model_data['hour_sin'] = np.sin(2 * np.pi * model_data['hour'] / 24)
    model_data['hour_cos'] = np.cos(2 * np.pi * model_data['hour'] / 24)
    model_data['dow_sin'] = np.sin(2 * np.pi * model_data['day_of_week'] / 7)
    model_data['dow_cos'] = np.cos(2 * np.pi * model_data['day_of_week'] / 7)
    model_data['month_sin'] = np.sin(2 * np.pi * model_data['month'] / 12)
    model_data['month_cos'] = np.cos(2 * np.pi * model_data['month'] / 12)
    
    model_data['is_morning_rush'] = model_data['hour'].isin([7, 8, 9]).astype(int)
    model_data['is_evening_rush'] = model_data['hour'].isin([17, 18, 19]).astype(int)
    model_data['is_night'] = ((model_data['hour'] >= 22) | (model_data['hour'] <= 5)).astype(int)
    model_data['is_midday'] = model_data['hour'].isin([11, 12, 13, 14]).astype(int)
    model_data['is_weekend'] = model_data['day_of_week'].isin([5, 6]).astype(int)
    
    model_data['weekend_night'] = model_data['is_weekend'] * model_data['is_night']
    model_data['weekday_morning_rush'] = (1 - model_data['is_weekend']) * model_data['is_morning_rush']
    model_data['weekday_evening_rush'] = (1 - model_data['is_weekend']) * model_data['is_evening_rush']
    
    model_data['temp_range'] = model_data['TMAX'] - model_data['TMIN']
    model_data['temp_avg'] = (model_data['TMAX'] + model_data['TMIN']) / 2
    model_data['is_rainy'] = (model_data['PRCP'] > 0).astype(int)
    model_data['is_heavy_rain'] = (model_data['PRCP'] > 10).astype(int)
    model_data['is_cold'] = (model_data['temp_avg'] < 10).astype(int)
    model_data['is_hot'] = (model_data['temp_avg'] > 25).astype(int)
    
    model_data = model_data.sort_values(['date', 'hour']).reset_index(drop=True)
    model_data['rides_last_hour'] = model_data['ride_count'].shift(1).fillna(0)
    model_data['rides_same_hour_yesterday'] = model_data['ride_count'].shift(24).fillna(0)
    model_data['rides_same_hour_last_week'] = model_data['ride_count'].shift(24*7).fillna(0)
    model_data['rides_rolling_3h'] = model_data['ride_count'].shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
    model_data['rides_rolling_24h'] = model_data['ride_count'].shift(1).rolling(window=24, min_periods=1).mean().fillna(0)
    
    feature_columns = [
        'hour', 'day_of_week', 'month', 'year', 'day',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'is_morning_rush', 'is_evening_rush', 'is_night', 'is_midday', 'is_weekend',
        'weekend_night', 'weekday_morning_rush', 'weekday_evening_rush',
        'TMAX', 'TMIN', 'PRCP', 'temp_range', 'temp_avg',
        'is_rainy', 'is_heavy_rain', 'is_cold', 'is_hot',
        'rides_last_hour', 'rides_same_hour_yesterday', 'rides_same_hour_last_week',
        'rides_rolling_3h', 'rides_rolling_24h',
        'duration_mean', 'duration_std', 'duration_median',
        'distance_mean', 'distance_std', 'distance_median', 'member_ratio'
    ]
    
    X = model_data[feature_columns]
    y = model_data['ride_count']
    
    return X, y, feature_columns
