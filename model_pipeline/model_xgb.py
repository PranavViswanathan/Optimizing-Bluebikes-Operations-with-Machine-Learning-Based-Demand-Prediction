import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ================== DATA LOADING ==================
print("Loading data...")
bluebike_data = pd.read_pickle('D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\after_duplicates.pkl')
weather_data = pd.read_pickle('D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\noaa_weather\\after_duplicates.pkl')
college_data = pd.read_pickle('D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\boston_clg\\after_duplicates.pkl')

print(f"Data loaded: {len(bluebike_data):,} rides")

# ================== TIME PREPROCESSING ==================
bluebike_data['start_time'] = pd.to_datetime(bluebike_data['start_time']).dt.tz_localize(None)
bluebike_data['stop_time'] = pd.to_datetime(bluebike_data['stop_time']).dt.tz_localize(None)

# Extract temporal features
bluebike_data['date'] = bluebike_data['start_time'].dt.date
bluebike_data['hour'] = bluebike_data['start_time'].dt.hour
bluebike_data['day_of_week'] = bluebike_data['start_time'].dt.dayofweek
bluebike_data['month'] = bluebike_data['start_time'].dt.month
bluebike_data['year'] = bluebike_data['start_time'].dt.year
bluebike_data['day'] = bluebike_data['start_time'].dt.day

# Calculate duration
bluebike_data['duration_minutes'] = (bluebike_data['stop_time'] - bluebike_data['start_time']).dt.total_seconds() / 60

# Filter outliers
bluebike_data = bluebike_data[(bluebike_data['duration_minutes'] > 0) & 
                                (bluebike_data['duration_minutes'] < 1440)]

# ================== DISTANCE CALCULATION ==================
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

# User type encoding
bluebike_data['is_member'] = (bluebike_data['user_type'] == 'member').astype(int)

# ================== HOURLY AGGREGATION ==================
print("Aggregating to hourly level...")
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

# Flatten column names
hourly_rides.columns = ['date', 'hour', 'ride_count', 'duration_mean', 'duration_std', 
                        'duration_median', 'distance_mean', 'distance_std', 'distance_median',
                        'member_ratio', 'day_of_week', 'month', 'year', 'day']

# Fill NaN values
hourly_rides['duration_std'] = hourly_rides['duration_std'].fillna(0)
hourly_rides['distance_std'] = hourly_rides['distance_std'].fillna(0)

# ================== COMPLETE TIME SERIES ==================
# Create complete hourly time series
date_range = pd.date_range(
    start=hourly_rides['date'].min(),
    end=hourly_rides['date'].max(),
    freq='D'
).date

all_hours = range(24)
full_index = pd.MultiIndex.from_product([date_range, all_hours], names=['date', 'hour'])

hourly_rides_complete = hourly_rides.set_index(['date', 'hour']).reindex(full_index, fill_value=0)

# Forward/backward fill temporal columns
temporal_cols = ['day_of_week', 'month', 'year', 'day']
for col in temporal_cols:
    hourly_rides_complete[col] = hourly_rides_complete.groupby(level='date')[col].transform(
        lambda x: x.replace(0, np.nan).ffill().bfill().fillna(0)
    )

hourly_rides_complete = hourly_rides_complete.reset_index()

# ================== MERGE WITH WEATHER ==================
weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date
model_data = pd.merge(hourly_rides_complete, weather_data, on='date', how='left')

# Drop rows with missing weather data
model_data = model_data.dropna(subset=['TMAX', 'TMIN', 'PRCP'])

# ================== FEATURE ENGINEERING ==================
print("Engineering features...")

# Cyclical encoding for temporal features
model_data['hour_sin'] = np.sin(2 * np.pi * model_data['hour'] / 24)
model_data['hour_cos'] = np.cos(2 * np.pi * model_data['hour'] / 24)
model_data['dow_sin'] = np.sin(2 * np.pi * model_data['day_of_week'] / 7)
model_data['dow_cos'] = np.cos(2 * np.pi * model_data['day_of_week'] / 7)
model_data['month_sin'] = np.sin(2 * np.pi * model_data['month'] / 12)
model_data['month_cos'] = np.cos(2 * np.pi * model_data['month'] / 12)

# Time period indicators
model_data['is_morning_rush'] = model_data['hour'].isin([7, 8, 9]).astype(int)
model_data['is_evening_rush'] = model_data['hour'].isin([17, 18, 19]).astype(int)
model_data['is_night'] = ((model_data['hour'] >= 22) | (model_data['hour'] <= 5)).astype(int)
model_data['is_midday'] = model_data['hour'].isin([11, 12, 13, 14]).astype(int)
model_data['is_weekend'] = model_data['day_of_week'].isin([5, 6]).astype(int)

# Interaction features
model_data['weekend_night'] = model_data['is_weekend'] * model_data['is_night']
model_data['weekday_morning_rush'] = (1 - model_data['is_weekend']) * model_data['is_morning_rush']
model_data['weekday_evening_rush'] = (1 - model_data['is_weekend']) * model_data['is_evening_rush']

# Weather features
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

# ================== PREPARE FEATURES ==================
feature_columns = [
    # Temporal features
    'hour', 'day_of_week', 'month', 'year', 'day',
    # Cyclical features
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    # Time period indicators
    'is_morning_rush', 'is_evening_rush', 'is_night', 'is_midday', 'is_weekend',
    # Interaction features
    'weekend_night', 'weekday_morning_rush', 'weekday_evening_rush',
    # Weather features
    'TMAX', 'TMIN', 'PRCP', 'temp_range', 'temp_avg',
    'is_rainy', 'is_heavy_rain', 'is_cold', 'is_hot',
    # Lag features
    'rides_last_hour', 'rides_same_hour_yesterday', 'rides_same_hour_last_week',
    'rides_rolling_3h', 'rides_rolling_24h',
    # Ride characteristics
    'duration_mean', 'duration_std', 'duration_median',
    'distance_mean', 'distance_std', 'distance_median', 'member_ratio'
]

X = model_data[feature_columns]
y = model_data['ride_count']

# ================== TRAIN-TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training with {len(X_train):,} samples")
print(f"Testing with {len(X_test):,} samples")

# ================== XGBOOST MODEL TRAINING ==================
print("\nTraining XGBoost model...")

# XGBoost parameters (tuned for similar performance to LightGBM)
xgb_params = {
    'objective': 'reg:squarederror',  # Regression task
    'max_depth': 8,                    # Maximum tree depth (similar to num_leaves)
    'learning_rate': 0.05,             # Same as LightGBM
    'n_estimators': 1000,              # Number of boosting rounds
    'subsample': 0.8,                  # Similar to bagging_fraction
    'colsample_bytree': 0.9,           # Similar to feature_fraction
    'min_child_weight': 20,            # Similar to min_child_samples
    'reg_alpha': 0.1,                  # L1 regularization
    'reg_lambda': 0.1,                 # L2 regularization
    'random_state': 42,
    'n_jobs': -1,                      # Use all CPU cores
    'tree_method': 'hist',             # Fast histogram optimized approximate greedy algorithm
    'enable_categorical': False,        # We're handling categoricals ourselves
    'early_stopping_rounds': 50,       # Early stopping rounds
    'verbosity': 0                     # Silent mode
}

# Create XGBoost model with early stopping
xgb_model = xgb.XGBRegressor(**xgb_params)

# Train with early stopping
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=True  # Print progress
)

print(f"\nBest iteration: {xgb_model.best_iteration}")
print(f"Best score: {xgb_model.best_score:.4f}")
# ================== MODEL EVALUATION ==================
print("\nEvaluating model performance...")

# Make predictions
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Calculate metrics
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)
print(f"Training Set:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.2f} rides")
print(f"  MAE: {train_mae:.2f} rides")
print(f"\nTest Set:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.2f} rides")
print(f"  MAE: {test_mae:.2f} rides")

# ================== FEATURE IMPORTANCE ==================
print("\n" + "="*50)
print("TOP 15 FEATURE IMPORTANCES")
print("="*50)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Display top 15 features
for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['feature']:30s}: {row['importance']:.4f}")

# ================== VISUALIZATIONS ==================
print("\nGenerating visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Feature Importance Plot
ax1 = axes[0, 0]
top_features = feature_importance.head(15)
ax1.barh(range(len(top_features)), top_features['importance'].values)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'].values)
ax1.set_xlabel('Importance')
ax1.set_title('Top 15 Feature Importances (XGBoost)')
ax1.invert_yaxis()

# 2. Actual vs Predicted Scatter Plot
ax2 = axes[0, 1]
sample_idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
ax2.scatter(y_test.iloc[sample_idx], y_pred_test[sample_idx], alpha=0.5, s=10)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Rides')
ax2.set_ylabel('Predicted Rides')
ax2.set_title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.4f}')

# 3. Residual Plot
ax3 = axes[0, 2]
residuals = y_test - y_pred_test
ax3.scatter(y_pred_test, residuals, alpha=0.5, s=10)
ax3.axhline(y=0, color='r', linestyle='--')
ax3.set_xlabel('Predicted Rides')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot')

# 4. Prediction Error Distribution
ax4 = axes[1, 0]
ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax4.set_xlabel('Prediction Error')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Prediction Error Distribution\nMean: {residuals.mean():.2f}, Std: {residuals.std():.2f}')

# 5. Hourly Average Predictions vs Actual
ax5 = axes[1, 1]
hourly_actual = model_data.groupby('hour')['ride_count'].mean()
model_data['predictions'] = xgb_model.predict(X)
hourly_predicted = model_data.groupby('hour')['predictions'].mean()
ax5.plot(hourly_actual.index, hourly_actual.values, label='Actual', marker='o')
ax5.plot(hourly_predicted.index, hourly_predicted.values, label='Predicted', marker='s')
ax5.set_xlabel('Hour of Day')
ax5.set_ylabel('Average Rides')
ax5.set_title('Hourly Pattern: Actual vs Predicted')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Weekly Average Predictions vs Actual
ax6 = axes[1, 2]
weekly_actual = model_data.groupby('day_of_week')['ride_count'].mean()
weekly_predicted = model_data.groupby('day_of_week')['predictions'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
x_pos = np.arange(len(days))
width = 0.35
ax6.bar(x_pos - width/2, weekly_actual.values, width, label='Actual', alpha=0.8)
ax6.bar(x_pos + width/2, weekly_predicted.values, width, label='Predicted', alpha=0.8)
ax6.set_xlabel('Day of Week')
ax6.set_ylabel('Average Rides')
ax6.set_title('Weekly Pattern: Actual vs Predicted')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(days)
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_model_performance.png', dpi=100, bbox_inches='tight')
plt.show()

# ================== SAVE MODEL ==================
print("\nSaving model and metadata...")

# Save XGBoost model
joblib.dump(xgb_model, './models/xgboost_bikeshare_model.pkl')

# Save metadata
metadata = {
    'features': feature_columns,
    'model_params': xgb_params,
    'performance': {
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    },
    'best_iteration': xgb_model.best_iteration,
    'feature_importance': feature_importance.to_dict()
}

joblib.dump(metadata, 'xgboost_model_metadata.pkl')

# Save feature importance separately
feature_importance.to_csv('xgboost_feature_importance.csv', index=False)

print("\nModel saved: xgboost_bikeshare_model.pkl")
print("Metadata saved: xgboost_model_metadata.pkl")
print("Feature importance saved: xgboost_feature_importance.csv")
print("\n" + "="*50)
print("XGBoost Model Training Complete!")
print("="*50)

# ================== COMPARISON WITH LIGHTGBM ==================
print("="*50)
print(f"  XGBoost Test R²: {test_r2:.4f}")
print(f"  XGBoost Test MAE: {test_mae:.2f} rides")

