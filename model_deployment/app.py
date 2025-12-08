# app.py
"""
Cloud Run service for Bluebikes demand prediction
Loads models from GCS and serves predictions via REST API
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging
from datetime import datetime
from google.cloud import storage
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration from environment variables
GCS_BUCKET = os.environ.get('GCS_BUCKET', 'mlruns234')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/production/current_model.pkl')

# Global variables for model and metadata
model = None
model_metadata = {
    'loaded_at': None,
    'model_path': None,
    'model_type': None,
    'version': None,
    'error': None
}

def download_model_from_gcs():
    """
    Download and load model from GCS
    Returns the loaded model or None if failed
    """
    try:
        logger.info(f"Downloading model from gs://{GCS_BUCKET}/{MODEL_PATH}")
        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(MODEL_PATH)
        
        # Check if model exists
        if not blob.exists():
            raise FileNotFoundError(f"Model not found at gs://{GCS_BUCKET}/{MODEL_PATH}")
        
        # Download to temporary file
        temp_model_path = '/tmp/model.pkl'
        blob.download_to_filename(temp_model_path)
        logger.info(f"Model downloaded to {temp_model_path}")
        
        # Get model metadata
        blob_metadata = blob.metadata or {}
        version = blob_metadata.get('version', blob.updated.isoformat() if blob.updated else 'unknown')
        
        # Load model
        loaded_model = joblib.load(temp_model_path)
        
        # Basic validation - just check it has predict method
        if not hasattr(loaded_model, 'predict'):
            raise ValueError("Model does not have 'predict' method")
        
        # Get model info
        model_type = type(loaded_model).__name__
        n_features = getattr(loaded_model, 'n_features_in_', 'unknown')
        
        logger.info(f"Model loaded successfully")
        logger.info(f"  Type: {model_type}")
        logger.info(f"  Expected features: {n_features}")
        logger.info(f"  Version: {version}")
        
        # Clean up temp file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        
        return loaded_model, version
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def initialize_model():
    """
    Initialize model on startup or refresh
    """
    global model, model_metadata
    
    loaded_model, version = download_model_from_gcs()
    
    if loaded_model is not None:
        model = loaded_model
        model_metadata = {
            'loaded_at': datetime.now().isoformat(),
            'model_path': f"gs://{GCS_BUCKET}/{MODEL_PATH}",
            'model_type': type(model).__name__,
            'version': version,
            'n_features': getattr(model, 'n_features_in_', 'unknown'),
            'error': None,
            'status': 'ready'
        }
        logger.info("Model initialization complete")
        return True
    else:
        model_metadata['error'] = "Failed to load model from GCS"
        model_metadata['status'] = 'error'
        logger.error("Model initialization failed")
        return False

# Initialize model on startup
logger.info("Starting Bluebikes prediction service...")
initialization_success = initialize_model()
if initialization_success:
    logger.info("Service ready to accept requests")
else:
    logger.warning("Service started but model not loaded - predictions will fail")

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with service information"""
    return jsonify({
        'service': 'Bluebikes Demand Prediction API',
        'version': '2.1.0',
        'status': model_metadata.get('status', 'unknown'),
        'model_metadata': model_metadata,
        'endpoints': {
            '/': 'Service information',
            '/health': 'Health check with detailed status',
            '/predict': 'Single prediction (POST)',
            '/batch_predict': 'Batch predictions (POST)',
            '/reload': 'Force model reload (POST)',
            '/metrics': 'Service metrics'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint with detailed status
    Returns 200 if healthy, 503 if unhealthy
    """
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'service': 'bluebikes-prediction',
        'model_loaded': model is not None,
        'model_metadata': model_metadata
    }
    
    if model is not None:
        health_status['status'] = 'healthy'
        return jsonify(health_status), 200
    else:
        health_status['status'] = 'unhealthy'
        health_status['error'] = model_metadata.get('error', 'Model not loaded')
        return jsonify(health_status), 503

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint
    Accepts JSON with 'features' array
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'details': model_metadata.get('error', 'Unknown error')
        }), 503
    
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Get features
        if 'features' not in data:
            return jsonify({
                'error': 'Missing "features" in request',
                'expected_format': {
                    'features': '[list of feature values]'
                },
                'model_expects': f"{model_metadata.get('n_features', 'unknown')} features"
            }), 400
        
        features = np.array([data['features']])
        
        # Log request info
        logger.info(f"Prediction request received with {features.shape[1]} features")
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Log prediction
        logger.info(f"Prediction made: {prediction:.2f}")
        
        # Prepare response
        response = {
            'prediction': float(prediction),
            'model_version': model_metadata.get('version', 'unknown'),
            'model_type': model_metadata.get('model_type', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add warning if prediction seems unusual
        if prediction < 0:
            response['warning'] = 'Negative prediction - model may need retraining'
            logger.warning(f"Negative prediction: {prediction}")
        elif prediction > 1000:
            response['warning'] = 'Unusually high prediction - please verify input features'
            logger.warning(f"High prediction: {prediction}")
        
        return jsonify(response), 200
        
    except ValueError as ve:
        logger.error(f"Value error in prediction: {str(ve)}")
        return jsonify({
            'error': 'Invalid input',
            'details': str(ve),
            'model_expects': f"{model_metadata.get('n_features', 'unknown')} features"
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    Accepts JSON with 'instances' array of arrays
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'details': model_metadata.get('error', 'Unknown error')
        }), 503
    
    try:
        # Parse request
        data = request.get_json()
        if not data or 'instances' not in data:
            return jsonify({
                'error': 'Missing "instances" in request',
                'expected_format': {
                    'instances': '[[features1], [features2], ...]'
                },
                'model_expects': f"{model_metadata.get('n_features', 'unknown')} features per instance"
            }), 400
        
        instances = np.array(data['instances'])
        
        # Log request info
        logger.info(f"Batch prediction request for {len(instances)} instances")
        
        # Make predictions
        predictions = model.predict(instances)
        
        # Log results
        logger.info(f"Batch predictions made: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
        
        # Check for anomalies
        if np.any(predictions < 0):
            logger.warning(f"Negative predictions detected in batch: {np.sum(predictions < 0)} instances")
        
        return jsonify({
            'predictions': predictions.tolist(),
            'count': len(predictions),
            'model_version': model_metadata.get('version', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except ValueError as ve:
        logger.error(f"Value error in batch prediction: {str(ve)}")
        return jsonify({
            'error': 'Invalid input',
            'details': str(ve),
            'model_expects': f"{model_metadata.get('n_features', 'unknown')} features per instance"
        }), 400
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': 'Batch prediction failed',
            'details': str(e)
        }), 500

@app.route('/reload', methods=['POST'])
def reload_model():
    """
    Force model reload from GCS
    Useful when Airflow uploads a new model
    """
    logger.info("Model reload requested")
    
    # Check for optional auth token
    auth_token = request.headers.get('X-Auth-Token')
    expected_token = os.environ.get('RELOAD_AUTH_TOKEN')
    
    if expected_token and auth_token != expected_token:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Reload model
    success = initialize_model()
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Model reloaded successfully',
            'model_metadata': model_metadata
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to reload model',
            'error': model_metadata.get('error', 'Unknown error')
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Service metrics endpoint for monitoring
    """
    metrics_data = {
        'service': 'bluebikes-prediction',
        'model_loaded': model is not None,
        'model_loaded_at': model_metadata.get('loaded_at'),
        'model_version': model_metadata.get('version'),
        'model_type': model_metadata.get('model_type'),
        'model_features': model_metadata.get('n_features'),
        'model_path': model_metadata.get('model_path'),
        'status': model_metadata.get('status')
    }
    
    # Add uptime if model is loaded
    if model_metadata.get('loaded_at'):
        try:
            loaded_time = datetime.fromisoformat(model_metadata['loaded_at'])
            uptime_seconds = (datetime.now() - loaded_time).total_seconds()
            metrics_data['uptime_seconds'] = uptime_seconds
        except:
            pass
    
    return jsonify(metrics_data), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/health', '/predict', '/batch_predict', '/reload', '/metrics']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)