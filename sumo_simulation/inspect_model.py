
import joblib
import pandas as pd
import numpy as np
import sys

MODEL_PATH = '/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/model_pipeline/models/production/current_model.pkl'

def inspect_model():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model type: {type(model)}")
        if hasattr(model, 'feature_names_in_'):
            print("Feature names:", model.feature_names_in_)
        elif hasattr(model, 'get_booster'):
             print("Feature names:", model.get_booster().feature_names)
        else:
            print("Could not find feature names.")
            print(f"n_features_in_: {getattr(model, 'n_features_in_', 'Unknown')}")

    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    inspect_model()
