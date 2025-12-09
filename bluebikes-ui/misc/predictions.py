# app.py - Backend service that proxies predictions to Cloud Run
"""
Backend service that prepares features and calls the deployed Cloud Run model
Acts as a bridge between your frontend and the Cloud Run prediction service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
CLOUD_RUN_URL = os.getenv('CLOUD_RUN_URL', 'https://bluebikes-prediction-202855070348.us-central1.run.app')
PORT = int(os.getenv('ML_SERVICE_PORT', 5001))
RELOAD_TOKEN = os.getenv('RELOAD_TOKEN', None)  # Optional token for reload endpoint

# Cache for model metadata
model_metadata_cache = {
    'last_checked': None,
    'metadata': None
}

def check_cloud_run_health():
    """Check if Cloud Run service is healthy"""
    try:
        response = requests.get(f"{CLOUD_RUN_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            model_metadata_cache['last_checked'] = datetime.now()
            model_metadata_cache['metadata'] = data.get('model_metadata', {})
            return True, data
        return False, None
    except Exception as e:
        print(f"Cloud Run health check failed: {e}")
        return False, None

def engineer_features_for_cloud_run(station_id, dt, temperature=15, precipitation=0):
    """
    Generate 48 features expected by the Cloud Run model
    Based on your deployed model's requirements
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
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Interaction features
    weekend_night = is_weekend * is_night
    weekday_morning_rush = (1 - is_weekend) * is_morning_rush
    weekday_evening_rush = (1 - is_weekend) * is_evening_rush
    
    # Weather features
    TMAX = temperature + 2
    TMIN = temperature - 2
    PRCP = precipitation
    temp_range = TMAX - TMIN
    temp_avg = (TMAX + TMIN) / 2
    is_rainy = 1 if PRCP > 0.1 else 0
    is_heavy_rain = 1 if PRCP > 0.5 else 0
    is_cold = 1 if temp_avg < 5 else 0
    is_hot = 1 if temp_avg > 25 else 0
    
    # Historical features (mock/placeholder values since we don't have real historical data)
    # In production, these would come from a database
    rides_last_hour = 100 + np.random.randint(-20, 20)
    rides_same_hour_yesterday = 95 + np.random.randint(-20, 20)
    rides_same_hour_last_week = 98 + np.random.randint(-20, 20)
    rides_rolling_3h = 280 + np.random.randint(-50, 50)
    rides_rolling_24h = 2200 + np.random.randint(-200, 200)
    
    # Statistical features (mock values)
    duration_mean = 12.5 + np.random.randn()
    duration_std = 3.2 + np.random.randn() * 0.5
    duration_median = 11.0 + np.random.randn()
    distance_mean = 1.8 + np.random.randn() * 0.2
    distance_std = 0.9 + np.random.randn() * 0.1
    distance_median = 1.5 + np.random.randn() * 0.2
    member_ratio = 0.7 + np.random.randn() * 0.1
    member_ratio = max(0, min(1, member_ratio))  # Clamp between 0 and 1
    
    # Station-specific features (could be one-hot encoded or embedding)
    # For now, using numeric representation
    try:
        station_num = int(station_id) if isinstance(station_id, str) else station_id
    except:
        station_num = hash(str(station_id)) % 1000
    
    # Create feature array - 48 features total
    features = [
        # Temporal (11)
        hour, day_of_week, month, year, day,
        hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos,
        
        # Time periods (8)
        is_morning_rush, is_evening_rush, is_night, is_midday, is_weekend,
        weekend_night, weekday_morning_rush, weekday_evening_rush,
        
        # Weather (9)
        TMAX, TMIN, PRCP, temp_range, temp_avg,
        is_rainy, is_heavy_rain, is_cold, is_hot,
        
        # Historical (5)
        rides_last_hour, rides_same_hour_yesterday, rides_same_hour_last_week,
        rides_rolling_3h, rides_rolling_24h,
        
        # Statistical (7)
        duration_mean, duration_std, duration_median,
        distance_mean, distance_std, distance_median, member_ratio,
        
        # Additional features to reach 48 (8)
        # These could be station-specific features, embeddings, etc.
        station_num % 100,  # Station ID feature
        station_num % 10,   # Station cluster/group
        0, 0, 0, 0, 0, 0    # Padding/reserved for future features
    ]
    
    return features

def get_mock_prediction(station_id, dt):
    """Fallback mock prediction when Cloud Run is unavailable"""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    
    hour = dt.hour
    day_of_week = dt.weekday()
    
    try:
        station_num = int(station_id) if isinstance(station_id, str) else station_id
    except:
        station_num = hash(str(station_id)) % 1000
    
    station_factor = (station_num % 10) / 10
    
    if day_of_week < 5:  # Weekday
        if 7 <= hour <= 9:  # Morning rush
            base_demand = 30 + int(station_factor * 50)
        elif 17 <= hour <= 19:  # Evening rush
            base_demand = 35 + int(station_factor * 55)
        elif 11 <= hour <= 14:  # Midday
            base_demand = 20 + int(station_factor * 30)
        else:
            base_demand = 10 + int(station_factor * 20)
    else:  # Weekend
        if 10 <= hour <= 16:  # Weekend afternoon
            base_demand = 25 + int(station_factor * 40)
        else:
            base_demand = 10 + int(station_factor * 20)
    
    return max(0, base_demand)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict bike demand by calling Cloud Run model
    
    Request body:
    {
        "station_id": "123",
        "datetime": "2025-12-07T14:30:00Z",
        "temperature": 15,  // optional, in Celsius
        "precipitation": 0  // optional, in inches
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
        
        # Generate 48 features for Cloud Run model
        features = engineer_features_for_cloud_run(station_id, dt, temperature, precipitation)
        
        # Call Cloud Run prediction endpoint
        try:
            cloud_run_request = {
                "features": features
            }
            
            response = requests.post(
                f"{CLOUD_RUN_URL}/predict",
                json=cloud_run_request,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted_demand = max(0, int(round(result['prediction'])))
                model_version = result.get('model_version', 'unknown')
                model_type = result.get('model_type', 'unknown')
                
                # Check for warnings
                confidence = 'high'
                if 'warning' in result:
                    confidence = 'medium'
                    print(f"Model warning: {result['warning']}")
                
                response_data = {
                    'station_id': station_id,
                    'datetime': dt,
                    'predicted_demand': predicted_demand,
                    'model_version': model_version,
                    'model_type': model_type,
                    'model_status': 'cloud_run',
                    'confidence': confidence
                }
                
            else:
                # Cloud Run returned an error
                print(f"Cloud Run error: {response.status_code} - {response.text}")
                predicted_demand = get_mock_prediction(station_id, dt)
                
                response_data = {
                    'station_id': station_id,
                    'datetime': dt,
                    'predicted_demand': predicted_demand,
                    'model_version': 'fallback',
                    'model_status': 'mock',
                    'confidence': 'low',
                    'error': 'Cloud Run model error'
                }
                
        except requests.exceptions.Timeout:
            print("Cloud Run request timeout")
            predicted_demand = get_mock_prediction(station_id, dt)
            
            response_data = {
                'station_id': station_id,
                'datetime': dt,
                'predicted_demand': predicted_demand,
                'model_version': 'fallback',
                'model_status': 'mock',
                'confidence': 'low',
                'error': 'Cloud Run timeout'
            }
            
        except Exception as e:
            print(f"Cloud Run request failed: {e}")
            predicted_demand = get_mock_prediction(station_id, dt)
            
            response_data = {
                'station_id': station_id,
                'datetime': dt,
                'predicted_demand': predicted_demand,
                'model_version': 'fallback',
                'model_status': 'mock',
                'confidence': 'low',
                'error': str(e)
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction for multiple stations/times
    Calls Cloud Run batch endpoint
    """
    try:
        data = request.json
        
        if not data or 'predictions' not in data:
            return jsonify({'error': 'predictions array is required'}), 400
        
        # Prepare batch features
        all_features = []
        metadata = []
        
        for pred_request in data['predictions']:
            station_id = pred_request['station_id']
            dt = pred_request.get('datetime', datetime.now().isoformat())
            temperature = pred_request.get('temperature', 15)
            precipitation = pred_request.get('precipitation', 0)
            
            features = engineer_features_for_cloud_run(station_id, dt, temperature, precipitation)
            all_features.append(features)
            metadata.append({
                'station_id': station_id,
                'datetime': dt
            })
        
        # Call Cloud Run batch endpoint
        try:
            cloud_run_request = {
                "instances": all_features
            }
            
            response = requests.post(
                f"{CLOUD_RUN_URL}/batch_predict",
                json=cloud_run_request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result['predictions']
                
                # Combine predictions with metadata
                response_data = {
                    'predictions': [
                        {
                            'station_id': meta['station_id'],
                            'datetime': meta['datetime'],
                            'predicted_demand': max(0, int(round(pred))),
                            'confidence': 'high'
                        }
                        for meta, pred in zip(metadata, predictions)
                    ],
                    'model_status': 'cloud_run',
                    'count': len(predictions)
                }
            else:
                # Fallback to mock predictions
                response_data = {
                    'predictions': [
                        {
                            'station_id': meta['station_id'],
                            'datetime': meta['datetime'],
                            'predicted_demand': get_mock_prediction(meta['station_id'], meta['datetime']),
                            'confidence': 'low'
                        }
                        for meta in metadata
                    ],
                    'model_status': 'mock',
                    'count': len(metadata),
                    'error': 'Cloud Run batch prediction failed'
                }
                
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            response_data = {
                'predictions': [
                    {
                        'station_id': meta['station_id'],
                        'datetime': meta['datetime'],
                        'predicted_demand': get_mock_prediction(meta['station_id'], meta['datetime']),
                        'confidence': 'low'
                    }
                    for meta in metadata
                ],
                'model_status': 'mock',
                'count': len(metadata),
                'error': str(e)
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint that also checks Cloud Run status"""
    
    # Check Cloud Run health
    cloud_run_healthy, cloud_run_data = check_cloud_run_health()
    
    health_response = {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'cloud_run_url': CLOUD_RUN_URL,
        'cloud_run_healthy': cloud_run_healthy,
        'cloud_run_model': None
    }
    
    if cloud_run_healthy and cloud_run_data:
        health_response['cloud_run_model'] = {
            'loaded': cloud_run_data.get('model_loaded', False),
            'type': cloud_run_data.get('model_metadata', {}).get('model_type'),
            'features_expected': cloud_run_data.get('model_metadata', {}).get('n_features'),
            'status': cloud_run_data.get('model_metadata', {}).get('status')
        }
    
    return jsonify(health_response)

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """
    Trigger Cloud Run to reload its model from GCS
    This is useful after Airflow updates the model
    """
    try:
        headers = {}
        if RELOAD_TOKEN:
            headers['X-Auth-Token'] = RELOAD_TOKEN
        
        response = requests.post(
            f"{CLOUD_RUN_URL}/reload",
            headers=headers,
            json={},
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify({
                'status': 'success',
                'message': 'Cloud Run model reloaded',
                'details': response.json()
            })
        elif response.status_code == 401:
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized - check reload token'
            }), 401
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to reload model',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        print(f"Reload failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the deployed model"""
    try:
        response = requests.get(f"{CLOUD_RUN_URL}/metrics", timeout=5)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Could not fetch model info'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting ML Prediction Service (Cloud Run Proxy)...")
    print(f"Cloud Run URL: {CLOUD_RUN_URL}")
    print(f"Service running on port {PORT}")
    
    # Check Cloud Run health on startup
    healthy, _ = check_cloud_run_health()
    if healthy:
        print("✅ Cloud Run service is healthy")
    else:
        print("⚠️  Cloud Run service is not responding - will use fallback predictions")
    
    print(f"\nAvailable endpoints:")
    print(f"  POST /predict         - Single prediction")
    print(f"  POST /predict/batch   - Batch predictions")
    print(f"  GET  /health          - Health check")
    print(f"  POST /reload-model    - Trigger model reload")
    print(f"  GET  /model-info      - Get model information")
    print()
    
    app.run(host='0.0.0.0', port=PORT, debug=False)