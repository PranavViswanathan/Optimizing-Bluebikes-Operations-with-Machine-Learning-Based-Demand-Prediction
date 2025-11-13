import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# LOAD THE TRAINED MODEL
# ============================================
print("=" * 80)
print("BLUEBIKE DEMAND PREDICTION - SAMPLE DATA")
print("=" * 80)

print("\nLoading trained model...")
lgb_model = joblib.load('lightgbm_bikeshare_model.pkl')
metadata = joblib.load('model_metadata.pkl')

print("✓ Model loaded successfully")
print(f"✓ Model uses {len(metadata['features'])} features")

# ============================================
# CREATE SAMPLE DATA IN EXACT PKL FORMAT
# ============================================
print("\n" + "=" * 80)
print("CREATING SAMPLE DATA (Same format as your PKL files)")
print("=" * 80)

# Let's create sample data for next week's predictions
# This matches EXACTLY the format of your bluebike PKL file

# Generate sample rides for a few days
sample_start_date = datetime(2025, 9, 1, 0, 0, 0)  # September 1, 2025
num_days = 7  # One week of data
rides_per_day = 3000  # Approximate rides per day

print(f"\nGenerating sample data for {num_days} days starting {sample_start_date.date()}")

# Create sample bluebike data matching your exact columns
sample_rides = []

for day in range(num_days):
    current_date = sample_start_date + timedelta(days=day)
    
    # Generate rides throughout the day with realistic patterns
    for hour in range(24):
        # More rides during rush hours
        if hour in [7, 8, 9]:  # Morning rush
            num_rides = np.random.poisson(200)
        elif hour in [17, 18, 19]:  # Evening rush
            num_rides = np.random.poisson(250)
        elif hour in [11, 12, 13, 14]:  # Lunch time
            num_rides = np.random.poisson(150)
        elif hour >= 22 or hour <= 5:  # Night
            num_rides = np.random.poisson(30)
        else:  # Regular hours
            num_rides = np.random.poisson(100)
        
        # Generate individual rides for this hour
        for ride_num in range(num_rides):
            start_time = current_date + timedelta(hours=hour, 
                                                 minutes=np.random.randint(0, 60),
                                                 seconds=np.random.randint(0, 60))
            
            # Duration between 5 and 45 minutes
            duration_minutes = np.random.gamma(3, 5)  # Roughly 15 min average
            duration_minutes = np.clip(duration_minutes, 5, 45)
            stop_time = start_time + timedelta(minutes=duration_minutes)
            
            # Sample station IDs (matching your data range)
            start_station_id = np.random.randint(1, 585)
            end_station_id = np.random.randint(1, 585)
            
            # Boston area coordinates (matching your data range)
            start_lat = np.random.uniform(42.30, 42.40)
            start_lon = np.random.uniform(-71.15, -71.05)
            end_lat = np.random.uniform(42.30, 42.40)
            end_lon = np.random.uniform(-71.15, -71.05)
            
            sample_rides.append({
                'ride_id': f'SAMPLE_{day:02d}_{hour:02d}_{ride_num:04d}',
                'rideable_type': np.random.choice(['classic_bike', 'electric_bike'], p=[0.7, 0.3]),
                'start_time': start_time,
                'stop_time': stop_time,
                'start_station_name': f'Station {start_station_id}',
                'start_station_id': float(start_station_id),
                'end_station_name': f'Station {end_station_id}',
                'end_station_id': float(end_station_id),
                'start_station_latitude': start_lat,
                'start_station_longitude': start_lon,
                'end_station_latitude': end_lat,
                'end_station_longitude': end_lon,
                'user_type': np.random.choice(['member', 'casual'], p=[0.7, 0.3])
            })

# Create DataFrame matching exact structure
sample_bluebike_data = pd.DataFrame(sample_rides)

print(f"✓ Generated {len(sample_bluebike_data):,} sample rides")

# Create sample weather data
sample_weather_data = []
for day in range(num_days):
    current_date = sample_start_date + timedelta(days=day)
    
    # September weather in Boston (realistic ranges)
    base_temp = 20  # 20°C base for September
    temp_variation = np.random.normal(0, 3)
    
    sample_weather_data.append({
        'date': current_date.date(),
        'TMAX': base_temp + temp_variation + np.random.uniform(3, 7),  # Max temp
        'TMIN': base_temp + temp_variation - np.random.uniform(3, 7),  # Min temp
        'PRCP': np.random.choice([0, 0, 0, 0, 5, 10, 20], p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02])  # Rain
    })

sample_weather_data = pd.DataFrame(sample_weather_data)
sample_weather_data['date'] = sample_weather_data['date'].astype(str)

print(f"✓ Generated {len(sample_weather_data)} days of weather data")

# ============================================
# PROCESS SAMPLE DATA (SAME AS TRAINING)
# ============================================
print("\n" + "=" * 80)
print("PROCESSING SAMPLE DATA")
print("=" * 80)

# Process exactly like in training
bluebike_data = sample_bluebike_data.copy()

# Extract temporal features
bluebike_data['date'] = pd.to_datetime(bluebike_data['start_time']).dt.date
bluebike_data['hour'] = pd.to_datetime(bluebike_data['start_time']).dt.hour
bluebike_data['day_of_week'] = pd.to_datetime(bluebike_data['start_time']).dt.dayofweek
bluebike_data['month'] = pd.to_datetime(bluebike_data['start_time']).dt.month
bluebike_data['year'] = pd.to_datetime(bluebike_data['start_time']).dt.year
bluebike_data['day'] = pd.to_datetime(bluebike_data['start_time']).dt.day

# Calculate duration
bluebike_data['duration_minutes'] = (
    pd.to_datetime(bluebike_data['stop_time']) - 
    pd.to_datetime(bluebike_data['start_time'])
).dt.total_seconds() / 60

# Calculate distance
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

# Create hourly aggregations
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

print(f"✓ Created {len(hourly_rides)} hourly records")

# Fill missing hours
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

# Merge with weather
sample_weather_data['date'] = pd.to_datetime(sample_weather_data['date']).dt.date
model_data = pd.merge(hourly_rides_complete, sample_weather_data, on='date', how='left')

# Create features
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
model_data['is_cold'] = (model_data['temp_avg'] < 5).astype(int)
model_data['is_hot'] = (model_data['temp_avg'] > 25).astype(int)

# Lag features
model_data = model_data.sort_values(['date', 'hour']).reset_index(drop=True)
model_data['rides_last_hour'] = model_data['ride_count'].shift(1).fillna(0)
model_data['rides_same_hour_yesterday'] = model_data['ride_count'].shift(24).fillna(0)
model_data['rides_same_hour_last_week'] = model_data['ride_count'].shift(24*7).fillna(0)
model_data['rides_rolling_3h'] = model_data['ride_count'].shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
model_data['rides_rolling_24h'] = model_data['ride_count'].shift(1).rolling(window=24, min_periods=1).mean().fillna(0)

print("✓ All features created")

# ============================================
# MAKE PREDICTIONS
# ============================================
print("\n" + "=" * 80)
print("MAKING PREDICTIONS")
print("=" * 80)

# Prepare features for prediction
X_predict = model_data[metadata['features']]

# Make predictions
predictions = lgb_model.predict(X_predict, num_iteration=lgb_model.best_iteration)

# Add predictions to dataframe
model_data['predicted_rides'] = predictions
model_data['actual_rides'] = model_data['ride_count']

# ============================================
# DISPLAY RESULTS
# ============================================
print("\n" + "=" * 80)
print("PREDICTION RESULTS")
print("=" * 80)

# Show daily summary
daily_summary = model_data.groupby('date').agg({
    'actual_rides': 'sum',
    'predicted_rides': 'sum'
}).round(0).astype(int)

print("\nDaily Predictions:")
print("-" * 40)
for date, row in daily_summary.iterrows():
    day_name = pd.to_datetime(date).strftime('%A')
    print(f"{date} ({day_name}):")
    print(f"  Actual rides: {row['actual_rides']:,}")
    print(f"  Predicted rides: {row['predicted_rides']:,}")
    print(f"  Difference: {row['predicted_rides'] - row['actual_rides']:+,}")

# Show hourly predictions for first day
print("\n" + "=" * 80)
print("HOURLY PREDICTIONS - FIRST DAY")
print("=" * 80)

first_day = model_data['date'].min()
first_day_data = model_data[model_data['date'] == first_day][
    ['hour', 'actual_rides', 'predicted_rides', 'temp_avg', 'is_rainy']
].round(1)

print(f"\nDate: {first_day}")
print("-" * 60)
print(f"{'Hour':>4} {'Actual':>8} {'Predicted':>10} {'Temp(°C)':>10} {'Rain':>6}")
print("-" * 60)

for _, row in first_day_data.iterrows():
    rain_status = '☔' if row['is_rainy'] else ''
    print(f"{int(row['hour']):4d} {int(row['actual_rides']):8d} {int(row['predicted_rides']):10d} "
          f"{row['temp_avg']:10.1f} {rain_status:>6}")

# Calculate overall metrics
mae = np.mean(np.abs(model_data['actual_rides'] - model_data['predicted_rides']))
rmse = np.sqrt(np.mean((model_data['actual_rides'] - model_data['predicted_rides'])**2))

print("\n" + "=" * 80)
print("SAMPLE PREDICTION METRICS")
print("=" * 80)
print(f"Mean Absolute Error: {mae:.1f} rides")
print(f"Root Mean Square Error: {rmse:.1f} rides")
print(f"Total actual rides: {model_data['actual_rides'].sum():,}")
print(f"Total predicted rides: {model_data['predicted_rides'].sum():,.0f}")

# ============================================
# SAVE SAMPLE DATA
# ============================================
print("\n" + "=" * 80)
print("SAVING SAMPLE DATA")
print("=" * 80)

# Save the sample data in PKL format (same as your original)
sample_bluebike_data.to_pickle('sample_bluebike_data.pkl')
sample_weather_data.to_pickle('sample_weather_data.pkl')

# Save predictions
model_data[['date', 'hour', 'actual_rides', 'predicted_rides', 
            'temp_avg', 'is_rainy', 'is_weekend']].to_csv('predictions.csv', index=False)

print("✓ Saved sample_bluebike_data.pkl (matches your bluebike PKL format)")
print("✓ Saved sample_weather_data.pkl (matches your weather PKL format)")
print("✓ Saved predictions.csv (prediction results)")

print("\n" + "=" * 80)
print("You can now use these PKL files as input to test your pipeline!")
print("=" * 80)