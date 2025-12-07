from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', './models/best_model.pkl')
PORT = int(os.getenv('ML_SERVICE_PORT', 5001))

# Global model variable
model = None

def load_model():
    """Load the trained XGBoost model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"⚠️  Model file not found at {MODEL_PATH}")
            print("    Predictions will return mock data until model is available")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("    Predictions will return mock data")

def engineer_features(station_id, dt, temperature=15, precipitation=0):
    """
    Generate features matching the training pipeline
    
    Features expected by the model (based on model_pipeline/README.md):
    - Temporal: hour, day_of_week, month, year, day
    - Cyclic: hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
    - Time periods: is_morning_rush, is_evening_rush, is_night, is_midday, is_weekend
    - Interactions: weekend_night, weekday_morning_rush, weekday_evening_rush
    - Weather: TMAX, TMIN, PRCP, temp_range, temp_avg, is_rainy, is_heavy_rain, is_cold, is_hot
    - Station: station_id (as numeric)
    """
    
    # Parse datetime
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    
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
    is_weekend = 1 if day_of_week >= 5 else 0  # Saturday=5, Sunday=6
    
    # Interaction features
    weekend_night = is_weekend * is_night
    weekday_morning_rush = (1 - is_weekend) * is_morning_rush
    weekday_evening_rush = (1 - is_weekend) * is_evening_rush
    
    # Weather features (using provided temperature and precipitation)
    # Estimate min/max from current temp (simplified)
    TMAX = temperature + 2
    TMIN = temperature - 2
    PRCP = precipitation
    temp_range = TMAX - TMIN
    temp_avg = (TMAX + TMIN) / 2
    is_rainy = 1 if PRCP > 0.1 else 0
    is_heavy_rain = 1 if PRCP > 0.5 else 0
    is_cold = 1 if temp_avg < 5 else 0
    is_hot = 1 if temp_avg > 25 else 0
    
    # Convert station_id to numeric (if string)
    try:
        station_id_num = int(station_id) if isinstance(station_id, str) else station_id
    except:
        station_id_num = hash(str(station_id)) % 10000  # Fallback hash
    
    # Create feature dictionary
    features = {
        'station_id': station_id_num,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'year': year,
        'day': day,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'is_morning_rush': is_morning_rush,
        'is_evening_rush': is_evening_rush,
        'is_night': is_night,
        'is_midday': is_midday,
        'is_weekend': is_weekend,
        'weekend_night': weekend_night,
        'weekday_morning_rush': weekday_morning_rush,
        'weekday_evening_rush': weekday_evening_rush,
        'TMAX': TMAX,
        'TMIN': TMIN,
        'PRCP': PRCP,
        'temp_range': temp_range,
        'temp_avg': temp_avg,
        'is_rainy': is_rainy,
        'is_heavy_rain': is_heavy_rain,
        'is_cold': is_cold,
        'is_hot': is_hot
    }
    
    return pd.DataFrame([features])

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
        if model is not None:
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
            'confidence': 'high' if model_status == 'active' else 'low'
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
    print("🤖 Starting ML Prediction Service...")
    print(f"📂 Model path: {MODEL_PATH}")
    
    # Try to load model
    load_model()
    
    print(f"🚀 ML Service running on port {PORT}")
    print(f"\nAvailable endpoints:")
    print(f"  POST /predict")
    print(f"  GET  /health")
    print()
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
