import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =====================================
# QUICK DATA PREPROCESSING
# =====================================

def quick_preprocess(filepath, sample_size=50000, top_n_stations=10):
    """
    Fast preprocessing for a trial model
    - Uses only a sample of data for speed
    - Focuses on top stations only
    - Creates station-level demand features
    """
    
    print("="*50)
    print("QUICK DATA PREPROCESSING")
    print("="*50)
    
    # Load sample of data
    print(f"\nLoading {sample_size:,} random rows...")
    
    # First, get total rows to sample properly
    total_rows = sum(1 for line in open(filepath)) - 1
    skip_rows = sorted(np.random.choice(range(1, total_rows), 
                                       total_rows - sample_size, 
                                       replace=False))
    
    # Load data with sampling
    df = pd.read_csv(filepath, skiprows=skip_rows)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    print(f"✓ Loaded {len(df):,} rows")
    
    # Convert datetime columns
    print("\nConverting datetime columns...")
    df['starttime'] = pd.to_datetime(df['starttime'])
    df['stoptime'] = pd.to_datetime(df['stoptime'])
    
    # Filter to top stations only (for faster processing)
    print(f"\nFiltering to top {top_n_stations} stations...")
    top_start_stations = df['start_station_id'].value_counts().head(top_n_stations).index
    df_filtered = df[df['start_station_id'].isin(top_start_stations)]
    print(f"✓ Reduced to {len(df_filtered):,} trips from top stations")
    
    return df_filtered

def create_station_demand_features(df):
    """
    Transform trip-level data to station-hour level demand data
    """
    
    print("\n" + "="*50)
    print("CREATING DEMAND FEATURES")
    print("="*50)
    
    # Extract time features
    df['hour'] = df['starttime'].dt.hour
    df['day_of_week'] = df['starttime'].dt.dayofweek
    df['date'] = df['starttime'].dt.date
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['starttime'].dt.month
    
    # Create hourly demand aggregation
    print("\nAggregating to hourly station demand...")
    
    hourly_demand = df.groupby(['start_station_id', 'start_station_name', 
                                'date', 'hour', 'day_of_week', 
                                'is_weekend', 'month']).agg({
        'bikeid': 'count',  # Number of departures
        'tripduration': 'mean'  # Average trip duration
    }).reset_index()
    
    hourly_demand.rename(columns={'bikeid': 'departures', 
                                 'tripduration': 'avg_trip_duration'}, inplace=True)
    
    # Sort by station and time
    hourly_demand = hourly_demand.sort_values(['start_station_id', 'date', 'hour'])
    
    print(f"✓ Created {len(hourly_demand):,} station-hour records")
    
    # Create lag features (previous hours' demand)
    print("\nCreating lag features...")
    
    for lag in [1, 2, 3, 24]:  # 1hr, 2hr, 3hr, and same hour yesterday
        hourly_demand[f'departures_lag_{lag}h'] = \
            hourly_demand.groupby('start_station_id')['departures'].shift(lag)
    
    # Rolling averages
    hourly_demand['departures_rolling_mean_6h'] = \
        hourly_demand.groupby('start_station_id')['departures'].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )
    
    hourly_demand['departures_rolling_mean_24h'] = \
        hourly_demand.groupby('start_station_id')['departures'].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
    
    # Historical averages (same hour, same day of week)
    historical_avg = hourly_demand.groupby(['start_station_id', 'hour', 
                                           'day_of_week'])['departures'].mean().reset_index()
    historical_avg.rename(columns={'departures': 'historical_avg_departures'}, inplace=True)
    
    hourly_demand = hourly_demand.merge(historical_avg, 
                                       on=['start_station_id', 'hour', 'day_of_week'], 
                                       how='left')
    
    # Time-based features
    hourly_demand['is_rush_hour'] = hourly_demand['hour'].isin([7,8,9,17,18,19]).astype(int)
    hourly_demand['is_night'] = hourly_demand['hour'].isin(range(0,6)).astype(int)
    
    # Drop rows with NaN (from lag features)
    print(f"\n🧹 Cleaning data...")
    print(f"   Records before cleaning: {len(hourly_demand):,}")
    hourly_demand = hourly_demand.dropna()
    print(f"   Records after cleaning: {len(hourly_demand):,}")
    
    return hourly_demand

def prepare_model_data(hourly_demand):
    """
    Prepare features and target for modeling
    """
    
    print("\n" + "="*50)
    print("PREPARING MODEL DATA")
    print("="*50)
    
    # Define features
    feature_columns = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'departures_lag_1h', 'departures_lag_2h', 'departures_lag_3h', 
        'departures_lag_24h',
        'departures_rolling_mean_6h', 'departures_rolling_mean_24h',
        'historical_avg_departures',
        'is_rush_hour', 'is_night',
        'avg_trip_duration'
    ]
    
    # Target variable
    target = 'departures'
    
    # Create feature matrix and target vector
    X = hourly_demand[feature_columns]
    y = hourly_demand[target]
    
    # Add station ID as a categorical feature (one-hot encoding)
    station_dummies = pd.get_dummies(hourly_demand['start_station_id'], 
                                    prefix='station', 
                                    drop_first=True)
    X = pd.concat([X, station_dummies], axis=1)
    
    print(f"\nFeature Matrix:")
    print(f"   Shape: {X.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    
    # Split data (80% train, 20% test)
    # Using time-based split for time series
    split_index = int(len(X) * 0.8)
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    print(f"\nTrain/Test Split:")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")
    
    # Store metadata for interpretation
    metadata = {
        'station_names': hourly_demand[['start_station_id', 'start_station_name']].drop_duplicates(),
        'test_data': hourly_demand.iloc[split_index:][['start_station_id', 'start_station_name', 
                                                       'date', 'hour', 'departures']],
        'feature_names': X.columns.tolist()
    }
    
    return X_train, X_test, y_train, y_test, metadata

def train_quick_model(X_train, X_test, y_train, y_test):
    """
    Train a simple Random Forest model
    """
    
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    print("\nTraining Random Forest...")
    print("   (Using small parameters for speed)")
    
    # Simple Random Forest with small parameters for speed
    model = RandomForestRegressor(
        n_estimators=50,  # Fewer trees for speed
        max_depth=10,     # Limit depth for speed
        min_samples_split=10,
        random_state=42,
        n_jobs=-1         # Use all cores
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    print("\nModel Performance:")
    print("\nTraining Set:")
    print(f"   MAE: {mean_absolute_error(y_train, y_pred_train):.2f} bikes")
    print(f"   RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f} bikes")
    print(f"   R²: {r2_score(y_train, y_pred_train):.3f}")
    
    print("\nTest Set:")
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    
    print(f"   MAE: {mae:.2f} bikes")
    print(f"   RMSE: {rmse:.2f} bikes")
    print(f"   R²: {r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    
    return model, y_pred_test

def analyze_predictions(y_test, y_pred_test, metadata):
    """
    Analyze model predictions
    """
    
    print("\n" + "="*50)
    print("PREDICTION ANALYSIS")
    print("="*50)
    
    # Add predictions to test data
    test_results = metadata['test_data'].copy()
    test_results['predicted_departures'] = y_pred_test
    test_results['error'] = test_results['departures'] - test_results['predicted_departures']
    test_results['abs_error'] = abs(test_results['error'])
    
    # Best predictions
    print("\n Best Predictions (Lowest Error):")
    best = test_results.nsmallest(5, 'abs_error')
    for _, row in best.iterrows():
        print(f"   Station: {row['start_station_name'][:30]:30s} | "
              f"Hour: {row['hour']:02d} | "
              f"Actual: {row['departures']:3.0f} | "
              f"Predicted: {row['predicted_departures']:3.0f} | "
              f"Error: {row['error']:+.1f}")
    
    # Worst predictions
    print("\nWorst Predictions (Highest Error):")
    worst = test_results.nlargest(5, 'abs_error')
    for _, row in worst.iterrows():
        print(f"   Station: {row['start_station_name'][:30]:30s} | "
              f"Hour: {row['hour']:02d} | "
              f"Actual: {row['departures']:3.0f} | "
              f"Predicted: {row['predicted_departures']:3.0f} | "
              f"Error: {row['error']:+.1f}")
    
    # Performance by station
    print("\nPerformance by Station:")
    station_performance = test_results.groupby('start_station_name')['abs_error'].agg(['mean', 'std']).sort_values('mean')
    print("\nBest Performing Stations:")
    for station, row in station_performance.head(3).iterrows():
        print(f"   {station[:40]:40s} - MAE: {row['mean']:.2f} ± {row['std']:.2f}")
    
    # Performance by hour
    hourly_performance = test_results.groupby('hour')['abs_error'].mean().sort_values()
    print("\nBest Predicted Hours:")
    for hour, error in hourly_performance.head(3).items():
        print(f"   {hour:02d}:00 - MAE: {error:.2f} bikes")
    
    return test_results

# =====================================
# MAIN EXECUTION FUNCTION
# =====================================

def run_quick_model(filepath):
    """
    Main function to run the entire pipeline
    """
    
    print("\n" + "🚴"*25)
    print("\n   BLUEBIKES DEMAND PREDICTION - QUICK TRIAL MODEL")
    print("\n" + "🚴"*25 + "\n")
    
    # 1. Preprocess data
    df = quick_preprocess(filepath, sample_size=50000, top_n_stations=10)
    
    # 2. Create features
    hourly_demand = create_station_demand_features(df)
    
    # 3. Prepare model data
    X_train, X_test, y_train, y_test, metadata = prepare_model_data(hourly_demand)
    
    # 4. Train model
    model, y_pred_test = train_quick_model(X_train, X_test, y_train, y_test)
    
    # 5. Analyze results
    results = analyze_predictions(y_test, y_pred_test, metadata)
    
    print("\n" + "="*50)
    print("MODEL COMPLETE!")
    print("="*50)
    print("\nNext Steps:")
    print("   1. Try with more data (increase sample_size)")
    print("   2. Add weather features")
    print("   3. Try different algorithms (XGBoost, LSTM)")
    print("   4. Add more sophisticated features")
    print("   5. Implement real-time predictions")
    
    return model, results

# =====================================
# RUN THE MODEL
# =====================================

if __name__ == "__main__":
    # Update this path to your CSV file
    filepath = "D:/MLOps_Coursework/ML-OPs/DatasetGeneration/bluebikes/combined_bluebikes.csv"

    
    # Run the quick model
    model, results = run_quick_model(filepath)
    
    # Optional: Save the model
    # import joblib
    # joblib.dump(model, 'bluebikes_demand_model.pkl')
    
    print("\nDone! Model trained successfully.")