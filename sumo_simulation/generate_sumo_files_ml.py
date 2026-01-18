#!/usr/bin/env python3
"""
Enhanced SUMO file generator with ML model integration.

This script generates SUMO simulation files with demand predictions from the trained XGBoost model.
It creates color-coded station markers based on predicted demand levels.
"""

import pandas as pd
import numpy as np
import datetime
import os
import joblib
import subprocess
import xml.sax.saxutils
import xml.etree.ElementTree as ET
from datetime import datetime as dt, timedelta

# Configuration
PARQUET_PATH = '/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/data_pipeline/data/raw/bluebikes/trips_2024.parquet'
STATIONS_CSV = '/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/sumo_simulation/stations.csv'
MODEL_PATH = '/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/model_pipeline/models/production/current_model.pkl'
OUTPUT_DIR = '/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/sumo_simulation'
SIMULATION_DATE = '2024-05-01'
START_HOUR = 8
END_HOUR = 9
SKIP_TRIPS = ['D01ACF58E87D2E9F'] # Trips that fail map matching

def load_model():
    """Load the trained XGBoost model."""
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Will use historical data instead of predictions.")
        return None

def create_features_for_prediction(station_id, timestamp, historical_data):
    """
    Create feature vector for model prediction.
    
    The model expects 48 features:
    ['hour', 'day_of_week', 'month', 'year', 'day', 'hour_sin', 'hour_cos', 'dow_sin',
     'dow_cos', 'month_sin', 'month_cos', 'is_morning_rush', 'is_evening_rush',
     'is_night', 'is_midday', 'is_weekend', 'weekend_night',
     'weekday_morning_rush', 'weekday_evening_rush', 'TMAX', 'TMIN', 'PRCP',
     'temp_range', 'temp_avg', 'is_rainy', 'is_heavy_rain', 'is_cold', 'is_hot',
     'rides_last_hour', 'rides_same_hour_yesterday', 'rides_same_hour_last_week',
     'rides_rolling_3h', 'rides_rolling_24h', 'duration_mean', 'duration_std',
     'duration_median', 'distance_mean', 'distance_std', 'distance_median',
     'member_ratio', 'is_hour_8', 'is_hour_17_18', 'rush_intensity',
     'high_demand_flag', 'low_demand_flag', 'demand_volatility', 'problem_period',
     'hour_group']
    
    For simplicity, we'll create a basic feature set with reasonable defaults.
    """
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month
    year = timestamp.year
    day = timestamp.day
    
    # Cyclical features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Time-based flags
    is_morning_rush = 1 if 7 <= hour <= 9 else 0
    is_evening_rush = 1 if 16 <= hour <= 19 else 0
    is_night = 1 if hour < 6 or hour >= 22 else 0
    is_midday = 1 if 11 <= hour <= 14 else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    weekend_night = is_weekend * is_night
    weekday_morning_rush = (1 - is_weekend) * is_morning_rush
    weekday_evening_rush = (1 - is_weekend) * is_evening_rush
    
    # Weather features (using defaults - would need actual weather data)
    TMAX = 70  # Default temp
    TMIN = 55
    PRCP = 0
    temp_range = TMAX - TMIN
    temp_avg = (TMAX + TMIN) / 2
    is_rainy = 0
    is_heavy_rain = 0
    is_cold = 0
    is_hot = 0
    
    # Historical features (simplified - would need actual historical data)
    rides_last_hour = 10
    rides_same_hour_yesterday = 10
    rides_same_hour_last_week = 10
    rides_rolling_3h = 30
    rides_rolling_24h = 200
    
    # Trip statistics (defaults)
    duration_mean = 600  # 10 minutes
    duration_std = 300
    duration_median = 500
    distance_mean = 2.0  # km
    distance_std = 1.0
    distance_median = 1.5
    member_ratio = 0.8
    
    # Additional flags
    is_hour_8 = 1 if hour == 8 else 0
    is_hour_17_18 = 1 if hour in [17, 18] else 0
    rush_intensity = is_morning_rush + is_evening_rush
    high_demand_flag = 0
    low_demand_flag = 0
    demand_volatility = 0.5
    problem_period = 0
    hour_group = hour // 6  # 0-3 for 4 groups
    
    features = [
        hour, day_of_week, month, year, day,
        hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos,
        is_morning_rush, is_evening_rush, is_night, is_midday, is_weekend,
        weekend_night, weekday_morning_rush, weekday_evening_rush,
        TMAX, TMIN, PRCP, temp_range, temp_avg,
        is_rainy, is_heavy_rain, is_cold, is_hot,
        rides_last_hour, rides_same_hour_yesterday, rides_same_hour_last_week,
        rides_rolling_3h, rides_rolling_24h,
        duration_mean, duration_std, duration_median,
        distance_mean, distance_std, distance_median,
        member_ratio,
        is_hour_8, is_hour_17_18, rush_intensity,
        high_demand_flag, low_demand_flag, demand_volatility, problem_period,
        hour_group
    ]
    
    return features

def get_demand_color(demand_level):
    """
    Return RGB color based on demand level.
    Low demand: Green (0,1,0)
    Medium demand: Yellow (1,1,0)
    High demand: Red (1,0,0)
    """
    if demand_level < 5:
        return "0,1,0"  # Green
    else:
        return "1,0,0"  # Red

def get_net_boundary(net_file):
    """Parse net.xml to get original boundary."""
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        location = root.find('location')
        if location is not None:
            boundary_str = location.get('origBoundary')
            if boundary_str:
                parts = boundary_str.split(',')
                # xmin, ymin, xmax, ymax
                return [float(p) for p in parts]
    except Exception as e:
        print(f"Warning: Could not read network boundary: {e}")
    return None

def is_within_boundary(lon, lat, boundary):
    if boundary is None:
        return True
    xmin, ymin, xmax, ymax = boundary
    return xmin <= lon <= xmax and ymin <= lat <= ymax

def generate_sumo_files_with_ml():
    print("Loading stations...")
    stations = pd.read_csv(STATIONS_CSV)
    
    print("Loading model...")
    model = load_model()
    
    # 1. Generate stations.add.xml with demand predictions
    print("Generating stations with demand predictions...")
    
    simulation_time = pd.Timestamp(f"{SIMULATION_DATE} {START_HOUR}:00:00", tz="UTC")
    
    with open(os.path.join(OUTPUT_DIR, 'stations_ml.add.xml'), 'w') as f:
        f.write('<additional>\n')
        f.write('    <!-- Bluebikes stations color-coded by predicted demand -->\n')
        
        for _, row in stations.iterrows():
            # SUMO IDs should be alphanumeric/safe
            station_id = "".join(filter(str.isalnum, str(row['id'])))
            if not station_id or station_id.lower() == 'nan':
                continue
                
            if model is not None:
                # Generate features and predict
                features = create_features_for_prediction(row['id'], simulation_time, None)
                prediction = model.predict([features])[0]
                demand_level = max(0, prediction)  # Ensure non-negative
            else:
                # Fallback to random for demo
                demand_level = np.random.randint(0, 20)
            
            color = get_demand_color(demand_level)
            
            f.write(f'    <poi id="{station_id}" type="station" x="{row["lon"]}" y="{row["lat"]}" color="{color}" layer="100">\n')
            f.write(f'        <param key="demand" value="{demand_level:.1f}"/>\n')
            safe_name = xml.sax.saxutils.escape(str(row["name"]))
            f.write(f'        <param key="name" value="{safe_name}"/>\n')
            f.write(f'    </poi>\n')
        
        f.write('</additional>\n')
    
    print("Generated stations_ml.add.xml with ML predictions")

    # 2. Generate trips.xml (same as before)
    print(f"Loading trips for {SIMULATION_DATE} {START_HOUR}:00 to {END_HOUR}:00...")
    
    # Get network boundary to filter trips
    net_file = os.path.join(OUTPUT_DIR, 'boston.net.xml')
    boundary = get_net_boundary(net_file)
    if boundary:
        print(f"Filtering trips to network boundary: {boundary}")
    
    df = pd.read_parquet(PARQUET_PATH)
    
    if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
        df['start_time'] = pd.to_datetime(df['start_time'])

    target_start = pd.Timestamp(f"{SIMULATION_DATE} {START_HOUR}:00:00", tz="UTC")
    target_end = pd.Timestamp(f"{SIMULATION_DATE} {END_HOUR}:00:00", tz="UTC")
    
    mask = (df['start_time'] >= target_start) & (df['start_time'] < target_end)
    trips = df[mask].sort_values('start_time')
    
    print(f"Found {len(trips)} trips initially.")
    
    with open(os.path.join(OUTPUT_DIR, 'trips.trips.xml'), 'w') as f:
        f.write('<routes>\n')
        f.write('    <vType id="bike" accel="0.8" decel="1.5" sigma="0.5" length="1.6" minGap="0.5" maxSpeed="5.5"/>\n')
        
        valid_trips_count = 0
        for idx, row in trips.iterrows():
            if row['ride_id'] in SKIP_TRIPS:
                continue
                
            # Check if within boundary
            if not is_within_boundary(row['start_station_longitude'], row['start_station_latitude'], boundary):
                continue
            if not is_within_boundary(row['end_station_longitude'], row['end_station_latitude'], boundary):
                continue
                
            depart = (row['start_time'] - target_start).total_seconds()
            f.write(f'    <trip id="{row["ride_id"]}" type="bike" depart="{depart:.2f}" fromXY="{row["start_station_longitude"]},{row["start_station_latitude"]}" toXY="{row["end_station_longitude"]},{row["end_station_latitude"]}" />\n')
            valid_trips_count += 1
            
        f.write('</routes>\n')

    print(f"Generated trips.trips.xml with {valid_trips_count} valid trips.")
    
    # Run DUAROUTER to filter out non-routable trips
    print("Running DUAROUTER to filter invalid trips...")
    try:
        duarouter_cmd = [
            'duarouter', 
            '-n', os.path.join(OUTPUT_DIR, 'boston.net.xml'),
            '-t', os.path.join(OUTPUT_DIR, 'trips.trips.xml'),
            '-o', os.path.join(OUTPUT_DIR, 'trips.rou.xml'),
            '--ignore-errors',
            '--no-step-log',
            '--no-warnings'
        ]
        subprocess.run(duarouter_cmd, check=True)
        print("Generated trips.rou.xml (Filtered)")
        route_file = "trips.rou.xml"
    except Exception as e:
        print(f"Warning: DUAROUTER failed: {e}")
        print("Falling back to raw trips file.")
        route_file = "trips.trips.xml"

    # 3. Generate config with filtered routes
    with open(os.path.join(OUTPUT_DIR, 'simulation_ml.sumocfg'), 'w') as f:
        f.write(f'''<configuration>
    <input>
        <net-file value="boston.net.xml"/>
        <route-files value="{route_file}"/>
        <additional-files value="stations_ml.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
    </processing>
    <routing>
        <junction-taz value="true"/>
    </routing>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="false"/>
        <error-log value="simulation_ml.errors.log"/>
    </report>
    <gui_only>
        <gui-settings-file value="gui-settings.xml"/>
    </gui_only>
</configuration>
''')
    print("Generated simulation_ml.sumocfg")
    
    # 4. Generate GUI settings for better visualization
    with open(os.path.join(OUTPUT_DIR, 'gui-settings.xml'), 'w') as f:
        f.write('''<viewsettings>
    <scheme name="real world"/>
    <delay value="100"/>
    <viewport zoom="1000" x="0" y="0"/>
</viewsettings>
''')
    print("Generated gui-settings.xml")

if __name__ == "__main__":
    generate_sumo_files_with_ml()
    print("\n✓ All files generated successfully!")
    print("\nTo run the ML-enhanced simulation:")
    print("  sumo-gui -c simulation_ml.sumocfg")
