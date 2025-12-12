from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache
import logging

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
PORT = int(os.getenv('HISTORICAL_DATA_SERVICE_PORT', 5003))
DATA_PATH = os.getenv('HISTORICAL_DATA_PATH', '../../data_pipeline/data/raw/bluebikes')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for loaded data (in-memory)
_data_cache = {}

def load_trip_data():
    """Load trip data from parquet files with caching"""
    global _data_cache
    
    if 'trips' in _data_cache:
        logger.info("Using cached trip data")
        return _data_cache['trips']
    
    try:
        data_dir = Path(DATA_PATH)
        parquet_files = sorted(data_dir.glob('trips_*.parquet'))
        
        if not parquet_files:
            logger.error(f"No parquet files found in {data_dir}")
            return None
        
        logger.info(f"Loading {len(parquet_files)} parquet files...")
        
        # Load all parquet files and concatenate
        dfs = []
        for file in parquet_files:
            logger.info(f"Reading {file.name}...")
            df = pd.read_parquet(file)
            dfs.append(df)
        
        trips_df = pd.concat(dfs, ignore_index=True)
        
        # Convert datetime columns if they're strings
        if 'start_time' in trips_df.columns:
            trips_df['start_time'] = pd.to_datetime(trips_df['start_time'], utc=True)
        if 'stop_time' in trips_df.columns:
            trips_df['stop_time'] = pd.to_datetime(trips_df['stop_time'], utc=True)
        
        # Cache the data and finding max date
        max_date = trips_df['start_time'].max()
        _data_cache['trips'] = trips_df
        _data_cache['max_date'] = max_date
        
        logger.info(f"Loaded {len(trips_df):,} trips from {trips_df['start_time'].min()} to {max_date}")
        logger.info(f"Using {max_date} as the anchor for relative time queries")
        return trips_df
    
    except Exception as e:
        logger.error(f"Error loading trip data: {str(e)}")
        return None

@lru_cache(maxsize=128)
def get_station_hourly_data(station_id, days=7):
    """Get hourly aggregated data for a station over the past N days"""
    try:
        trips_df = load_trip_data()
        if trips_df is None:
            return None
        
        # Calculate time range based on data max date
        end_date = _data_cache.get('max_date', datetime.now(timezone.utc))
        start_date = end_date - timedelta(days=days)
        
        # Filter for the station and time range
        station_starts = trips_df[
            (trips_df['start_station_id'] == station_id) & 
            (trips_df['start_time'] >= start_date) &
            (trips_df['start_time'] <= end_date)
        ].copy()
        
        station_ends = trips_df[
            (trips_df['end_station_id'] == station_id) & 
            (trips_df['stop_time'] >= start_date) &
            (trips_df['stop_time'] <= end_date)
        ].copy()
        
        # Group by hour
        station_starts['hour'] = station_starts['start_time'].dt.floor('H')
        station_ends['hour'] = station_ends['stop_time'].dt.floor('H')
        
        pickups = station_starts.groupby('hour').size().reset_index(name='pickups')
        dropoffs = station_ends.groupby('hour').size().reset_index(name='dropoffs')
        
        # Merge and fill missing hours with 0
        all_hours = pd.date_range(start=start_date.replace(minute=0, second=0, microsecond=0), 
                                   end=end_date, 
                                   freq='H')
        
        result = pd.DataFrame({'hour': all_hours})
        result = result.merge(pickups, on='hour', how='left')
        result = result.merge(dropoffs, on='hour', how='left')
        result = result.fillna(0)
        
        # Format for frontend
        data = []
        for _, row in result.iterrows():
            data.append({
                'time': row['hour'].isoformat(),
                'pickups': int(row['pickups']),
                'dropoffs': int(row['dropoffs']),
                'total': int(row['pickups'] + row['dropoffs'])
            })
        
        return data
    
    except Exception as e:
        logger.error(f"Error getting hourly data: {str(e)}")
        return None

@lru_cache(maxsize=128)
def get_station_daily_data(station_id, days=30):
    """Get daily aggregated data for a station over the past N days"""
    try:
        trips_df = load_trip_data()
        if trips_df is None:
            return None
        
        # Calculate time range based on data max date
        end_date = _data_cache.get('max_date', datetime.now(timezone.utc))
        start_date = end_date - timedelta(days=days)
        
        # Filter for the station and time range
        station_starts = trips_df[
            (trips_df['start_station_id'] == station_id) & 
            (trips_df['start_time'] >= start_date) &
            (trips_df['start_time'] <= end_date)
        ].copy()
        
        station_ends = trips_df[
            (trips_df['end_station_id'] == station_id) & 
            (trips_df['stop_time'] >= start_date) &
            (trips_df['stop_time'] <= end_date)
        ].copy()
        
        # Group by day
        station_starts['date'] = station_starts['start_time'].dt.date
        station_ends['date'] = station_ends['stop_time'].dt.date
        
        pickups = station_starts.groupby('date').size().reset_index(name='pickups')
        dropoffs = station_ends.groupby('date').size().reset_index(name='dropoffs')
        
        # Merge and fill missing days with 0
        all_days = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        
        result = pd.DataFrame({'date': all_days.date})
        result = result.merge(pickups, on='date', how='left')
        result = result.merge(dropoffs, on='date', how='left')
        result = result.fillna(0)
        
        # Format for frontend
        data = []
        for _, row in result.iterrows():
            data.append({
                'time': pd.Timestamp(row['date']).isoformat(),
                'pickups': int(row['pickups']),
                'dropoffs': int(row['dropoffs']),
                'total': int(row['pickups'] + row['dropoffs'])
            })
        
        return data
    
    except Exception as e:
        logger.error(f"Error getting daily data: {str(e)}")
        return None

@lru_cache(maxsize=128)
def get_station_weekly_data(station_id, weeks=12):
    """Get weekly aggregated data for a station over the past N weeks"""
    try:
        trips_df = load_trip_data()
        if trips_df is None:
            return None
        
        # Calculate time range based on data max date
        end_date = _data_cache.get('max_date', datetime.now(timezone.utc))
        start_date = end_date - timedelta(weeks=weeks)
        
        # Filter for the station and time range
        station_starts = trips_df[
            (trips_df['start_station_id'] == station_id) & 
            (trips_df['start_time'] >= start_date) &
            (trips_df['start_time'] <= end_date)
        ].copy()
        
        station_ends = trips_df[
            (trips_df['end_station_id'] == station_id) & 
            (trips_df['stop_time'] >= start_date) &
            (trips_df['stop_time'] <= end_date)
        ].copy()
        
        # Group by week (week starts on Monday)
        station_starts['week'] = station_starts['start_time'].dt.to_period('W-MON').dt.start_time
        station_ends['week'] = station_ends['stop_time'].dt.to_period('W-MON').dt.start_time
        
        pickups = station_starts.groupby('week').size().reset_index(name='pickups')
        dropoffs = station_ends.groupby('week').size().reset_index(name='dropoffs')
        
        # Merge
        result = pickups.merge(dropoffs, on='week', how='outer').fillna(0)
        result = result.sort_values('week')
        
        # Format for frontend
        data = []
        for _, row in result.iterrows():
            data.append({
                'time': row['week'].isoformat(),
                'pickups': int(row['pickups']),
                'dropoffs': int(row['dropoffs']),
                'total': int(row['pickups'] + row['dropoffs'])
            })
        
        return data
    
    except Exception as e:
        logger.error(f"Error getting weekly data: {str(e)}")
        return None

@app.route('/api/historical/<station_id>/hourly', methods=['GET'])
def get_hourly(station_id):
    """Get hourly data for the past 7 days"""
    days = request.args.get('days', 7, type=int)
    
    logger.info(f"Fetching hourly data for station {station_id} (past {days} days)")
    
    data = get_station_hourly_data(station_id, days)
    
    if data is None:
        return jsonify({'error': 'Failed to fetch historical data'}), 500
    
    if len(data) == 0:
        return jsonify({
            'station_id': station_id,
            'time_range': 'hourly',
            'days': days,
            'data': [],
            'message': 'No data available for this station'
        })
    
    return jsonify({
        'station_id': station_id,
        'time_range': 'hourly',
        'days': days,
        'data': data
    })

@app.route('/api/historical/<station_id>/daily', methods=['GET'])
def get_daily(station_id):
    """Get daily data for the past 30 days"""
    days = request.args.get('days', 30, type=int)
    
    logger.info(f"Fetching daily data for station {station_id} (past {days} days)")
    
    data = get_station_daily_data(station_id, days)
    
    if data is None:
        return jsonify({'error': 'Failed to fetch historical data'}), 500
    
    if len(data) == 0:
        return jsonify({
            'station_id': station_id,
            'time_range': 'daily',
            'days': days,
            'data': [],
            'message': 'No data available for this station'
        })
    
    return jsonify({
        'station_id': station_id,
        'time_range': 'daily',
        'days': days,
        'data': data
    })

@app.route('/api/historical/<station_id>/weekly', methods=['GET'])
def get_weekly(station_id):
    """Get weekly data for the past 12 weeks"""
    weeks = request.args.get('weeks', 12, type=int)
    
    logger.info(f"Fetching weekly data for station {station_id} (past {weeks} weeks)")
    
    data = get_station_weekly_data(station_id, weeks)
    
    if data is None:
        return jsonify({'error': 'Failed to fetch historical data'}), 500
    
    if len(data) == 0:
        return jsonify({
            'station_id': station_id,
            'time_range': 'weekly',
            'weeks': weeks,
            'data': [],
            'message': 'No data available for this station'
        })
    
    return jsonify({
        'station_id': station_id,
        'time_range': 'weekly',
        'weeks': weeks,
        'data': data
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    trips_df = load_trip_data()
    
    return jsonify({
        'status': 'ok',
        'data_loaded': trips_df is not None,
        'trips_count': len(trips_df) if trips_df is not None else 0,
        'data_path': DATA_PATH,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting Historical Data Service...")
    logger.info(f"Data path: {DATA_PATH}")
    
    # Preload data on startup
    logger.info("Preloading trip data...")
    load_trip_data()
    
    logger.info(f"Historical Data Service running on port {PORT}")
    logger.info(f"\nAvailable endpoints:")
    logger.info(f"  GET  /api/historical/:station_id/hourly")
    logger.info(f"  GET  /api/historical/:station_id/daily")
    logger.info(f"  GET  /api/historical/:station_id/weekly")
    logger.info(f"  GET  /health")
    logger.info("")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
