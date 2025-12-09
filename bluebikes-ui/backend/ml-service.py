from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', './models/best_model.pkl')
PORT = int(os.getenv('PORT', os.getenv('ML_SERVICE_PORT', 5001)))
EXTERNAL_ML_URL = os.getenv('EXTERNAL_ML_API_URL', 'https://bluebikes-prediction-202855070348.us-central1.run.app/predict')
USE_EXTERNAL_API = os.getenv('USE_EXTERNAL_ML_API', 'true').lower() != 'false'

# Global model variable
model = None

def load_model():
    """Load the trained XGBoost model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Model file not found at {MODEL_PATH}")
            print("    Predictions will return mock data until model is available")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("    Predictions will return mock data")

def engineer_features(station_id, dt, temperature=15, precipitation=0):
    """
    Generate features matching the training pipeline and gcr_test.py
    
    Total: 40 features (based on gcr_test.py structure)
    Note: Historical/Stat features are estimated/mocked as we don't have live DB access here.
    """
    
    # Parse datetime
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except:
            dt = datetime.now()
            
    # Adjust to EST (UTC-5) since model is trained on Boston time
    # This ensures 3 PM UTC is treated as 10 AM EST, etc.
    dt = dt - timedelta(hours=5)
    
    # Temporal features
    hour = dt.hour
    day_of_week = dt.weekday()  # 0 = Monday, 6 = Sunday
    month = dt.month
    year = dt.year
    day = dt.day
    
    # Cyclic encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Time period indicators
    is_morning_rush = 1 if 7 <= hour <= 9 else 0
    is_evening_rush = 1 if 17 <= hour <= 19 else 0
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    is_midday = 1 if 11 <= hour <= 14 else 0
    is_weekend = 1 if day_of_week >= 5 else 0  
    
    # Interaction features
    weekend_night = is_weekend * is_night
    weekday_morning_rush = (1 - is_weekend) * is_morning_rush
    weekday_evening_rush = (1 - is_weekend) * is_evening_rush
    
    # Weather features 
    # CRITICAL: Model trained on Fahrenheit (US Data), but input is Celsius
    # Convert C to F
    temp_f = (temperature * 9/5) + 32
    
    TMAX = temp_f + 5
    TMIN = temp_f - 5
    PRCP = precipitation
    temp_range = TMAX - TMIN
    temp_avg = (TMAX + TMIN) / 2
    is_rainy = 1 if PRCP > 0.1 else 0
    is_heavy_rain = 1 if PRCP > 0.5 else 0
    is_cold = 1 if temp_avg < 40 else 0  # < 40F (4C)
    is_hot = 1 if temp_avg > 80 else 0   # > 80F (27C)
    
    # --- Mock Lag/Stat Features (Estimated based on time) ---
    # In a real system, these would come from a feature store/DB
    
    # Base traffic factor (0.0 to 1.0)
    traffic_factor = 0.5 # Boosted baseline
    if is_morning_rush or is_evening_rush:
        traffic_factor = 1.0
    elif is_midday:
        traffic_factor = 0.7
    elif not is_night:
        traffic_factor = 0.6
        
    # Apply weekend penalty
    if is_weekend:
        traffic_factor *= 0.8
        if 10 <= hour <= 16: # Weekend day activity
             traffic_factor = 0.7
    
    # Station Factor (pseudo-random based on ID)
    try:
        s_hash = int(station_id) if str(station_id).isdigit() else hash(str(station_id))
        station_multiplier = 0.8 + ((s_hash % 100) / 200.0) # 0.8 to 1.3
    except:
        station_multiplier = 1.0

    # Ensure minimum volume to prevent 0 predictions
    # Typical busy station has 200+ rides/hour. Set floor high.
    base_volume = max(150, 400 * traffic_factor * station_multiplier)
    
    rides_last_hour = max(0, int(base_volume * np.random.uniform(0.9, 1.1)))
    rides_same_hour_yesterday = max(0, int(base_volume * np.random.uniform(0.9, 1.1)))
    rides_same_hour_last_week = max(0, int(base_volume * np.random.uniform(0.9, 1.1)))
    
    # CRITICAL FIX: Training data uses MEAN for rolling windows
    # Previous code used SUM (x3, x24) which signaled a massive crash in demand
    rides_rolling_3h = float(rides_last_hour)   # Approx mean = current
    rides_rolling_24h = float(rides_last_hour)  # Approx mean = current
    
    # Trip stats ( fairly constant)
    duration_mean = 15.0 if not is_weekend else 25.0
    duration_std = 5.0
    duration_median = 12.0
    distance_mean = 1.8
    distance_std = 0.9
    distance_median = 1.5
    member_ratio = 0.85 if not is_weekend else 0.4
    
    # --- Bias Mitigation Features (Added in pipeline) ---
    # Total: 8 additional features
    
    is_hour_8 = 1 if hour == 8 else 0
    is_hour_17_18 = 1 if hour in [17, 18] else 0
    
    rush_intensity = 0.0
    if hour == 8 or hour in [17, 18]:
        rush_intensity = 1.0
    elif hour in [7, 9, 16, 19]:
        rush_intensity = 0.5
        
    # Demand level flags (Mocked as 0 since we lack history distribution)
    high_demand_flag = 0
    low_demand_flag = 0
    demand_volatility = 0.0
    

    problem_sum = is_hour_8 + is_hour_17_18 + weekday_morning_rush + weekday_evening_rush
    problem_period = 1 if problem_sum > 0 else 0
    
    if hour <= 6:
        hour_group = 0
    elif hour <= 10:
        hour_group = 1
    elif hour <= 14:
        hour_group = 2
    elif hour <= 18:
        hour_group = 3
    else:
        hour_group = 4

    # Order must match model expectation (Base 40 + 8 New = 48)
    feature_list = [
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
        # Appended Bias Features
        is_hour_8, is_hour_17_18, rush_intensity,
        high_demand_flag, low_demand_flag, demand_volatility,
        problem_period, hour_group
    ]
    
    print(f"DEBUG: Calculated {len(feature_list)} features.")
    # print(f"DEBUG: Features: {feature_list}")
    
    return pd.DataFrame([feature_list])

def get_mock_prediction(station_id, dt):
    """Generate mock prediction when model is unavailable"""
    # Parse datetime
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    
    hour = dt.hour
    day_of_week = dt.weekday()
    
    # Use station_id to create variation between stations
    # Some stations will have high demand, others low
    try:
        station_num = int(station_id) if isinstance(station_id, str) else station_id
    except:
        station_num = hash(str(station_id)) % 1000
    
    # Create station-specific demand pattern (0-8 bikes)
    station_factor = (station_num % 10) / 10  # 0.0 to 0.9
    
    if day_of_week < 5:  # Weekday
        if 7 <= hour <= 9:  # Morning rush
            base_demand = 3 + int(station_factor * 5)  # 3-8 bikes
        elif 17 <= hour <= 19:  # Evening rush
            base_demand = 3 + int(station_factor * 5)  # 3-8 bikes
        elif 11 <= hour <= 14:  # Midday
            base_demand = 1 + int(station_factor * 3)  # 1-4 bikes
        else:
            base_demand = int(station_factor * 2)  # 0-2 bikes
    else:  # Weekend
        if 10 <= hour <= 16:  # Weekend afternoon
            base_demand = 2 + int(station_factor * 4)  # 2-6 bikes
        else:
            base_demand = int(station_factor * 2)  # 0-2 bikes
    
    return max(0, base_demand)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict bike demand for a given station and time
    
    Request body:
    {
        "station_id": "123",
        "datetime": "2025-12-07T14:30:00Z",
        "temperature": 15,  // optional, in Celsius
        "precipitation": 0  // optional, in inches
    }
    
    Response:
    {
        "station_id": "123",
        "datetime": "2025-12-07T14:30:00Z",
        "predicted_demand": 42,
        "model_version": "xgboost_v1",
        "confidence": "high"
    }
    """
    try:
        data = request.json
        
        # Validate input
        if not data or 'station_id' not in data:
            return jsonify({'error': 'station_id is required'}), 400
        
        station_id = data['station_id']
        dt = data.get('datetime', datetime.now().isoformat())
        temperature = data.get('temperature', 15)
        precipitation = data.get('precipitation', 0)
        
        # Generate features
        features_df = engineer_features(station_id, dt, temperature, precipitation)
        
        # Make prediction
        if USE_EXTERNAL_API:
            try:
                # Prepare payload for Cloud Run (expects features list)
                # features_df.values[0].tolist() returns [val1, val2, ...] (flat list)
                payload = {"features": features_df.values[0].tolist()}
                
                print(f"Calling External ML API: {EXTERNAL_ML_URL}")
                # print(f"Payload: {payload}")
                
                resp = requests.post(EXTERNAL_ML_URL, json=payload, timeout=5)
                resp.raise_for_status()
                
                result = resp.json()
                print(f"External API Response: {result}")
                
                # Handle different response formats
                if isinstance(result, dict):
                    if 'prediction' in result:
                         prediction = result['prediction']
                    elif 'predictions' in result:
                         prediction = result['predictions'][0]
                    else:
                         print(f"Unknown response format: {result}")
                         raise ValueError("Response missing 'prediction' or 'predictions' key")
                elif isinstance(result, list):
                    prediction = result[0]
                else:
                    # Try to parse if it's a direct valid format we missed
                    prediction = float(result)

                predicted_demand = max(0, int(round(prediction)))
                model_status = 'external_cloud_run'
                
            except Exception as e:
                print(f"External API failed: {str(e)}")
                print("Falling back to local model/mock")
                # Fallback to local
                if model is not None:
                    try:
                        prediction = model.predict(features_df)[0]
                        predicted_demand = max(0, int(round(prediction)))
                        model_status = 'active_local_fallback'
                    except Exception as e2:
                        print(f"Local model prediction error: {str(e2)}")
                        predicted_demand = get_mock_prediction(station_id, dt)
                        model_status = 'mock_fallback'
                else:
                    predicted_demand = get_mock_prediction(station_id, dt)
                    model_status = 'mock_fallback'
        
        elif model is not None:
            try:
                prediction = model.predict(features_df)[0]
                predicted_demand = max(0, int(round(prediction)))
                model_status = 'active'
            except Exception as e:
                print(f"Model prediction error: {str(e)}")
                predicted_demand = get_mock_prediction(station_id, dt)
                model_status = 'mock'
        else:
            predicted_demand = get_mock_prediction(station_id, dt)
            model_status = 'mock'
        
        response = {
            'station_id': station_id,
            'datetime': dt,
            'predicted_demand': predicted_demand,
            'model_version': 'xgboost_v1',
            'model_status': model_status,
            'confidence': 'high' if 'mock' not in model_status else 'low'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting ML Prediction Service...")
    print(f"Model path: {MODEL_PATH}")
    
    # Try to load model
    load_model()
    
    print(f"ML Service running on port {PORT}")
    print(f"\nAvailable endpoints:")
    print(f"  POST /predict")
    print(f"  GET  /health")
    print()
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
