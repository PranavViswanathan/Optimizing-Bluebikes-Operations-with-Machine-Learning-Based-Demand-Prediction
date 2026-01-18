
import pandas as pd
import os

PARQUET_PATH = '/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/data_pipeline/data/raw/bluebikes/trips_2024.parquet'
OUTPUT_FILE = '/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/sumo_simulation/stations.csv'

def extract_stations():
    print(f"Reading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    
    # Extract start stations
    start_stations = df[['start_station_id', 'start_station_name', 'start_station_latitude', 'start_station_longitude']].rename(
        columns={
            'start_station_id': 'id',
            'start_station_name': 'name',
            'start_station_latitude': 'lat',
            'start_station_longitude': 'lon'
        }
    )
    
    # Extract end stations
    end_stations = df[['end_station_id', 'end_station_name', 'end_station_latitude', 'end_station_longitude']].rename(
        columns={
            'end_station_id': 'id',
            'end_station_name': 'name',
            'end_station_latitude': 'lat',
            'end_station_longitude': 'lon'
        }
    )
    
    # Combine and drop duplicates
    all_stations = pd.concat([start_stations, end_stations])
    unique_stations = all_stations.drop_duplicates(subset=['id'])
    
    print(f"Found {len(unique_stations)} unique stations.")
    
    unique_stations.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved stations to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_stations()
