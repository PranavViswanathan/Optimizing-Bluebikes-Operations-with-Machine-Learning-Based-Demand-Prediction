# # test_cloud_run_endpoint.py
# import requests
# import numpy as np
# from datetime import datetime
# import json

# # Configuration - Your Cloud Run endpoint
# SERVICE_URL = "https://bluebikes-prediction-202855070348.us-central1.run.app"
# PREDICT_ENDPOINT = f"{SERVICE_URL}/predict"
# HEALTH_ENDPOINT = f"{SERVICE_URL}/health"

# print("🚴 BlueBikes Cloud Run Endpoint Test")
# print("="*50)
# print(f"Service URL: {SERVICE_URL}")
# print("-"*50)

# # First, check health
# print("\n  Checking service health...")
# try:
#     health_response = requests.get(HEALTH_ENDPOINT, timeout=10)
#     if health_response.status_code == 200:
#         health_data = health_response.json()
#         print(f"  Service is healthy")
#         print(f"   Model type: {health_data.get('model_type', 'Unknown')}")
#     else:
#         print(f"  Health check returned status {health_response.status_code}")
#         print(f"   Response: {health_response.text}")
# except Exception as e:
#     print(f"  Health check failed: {e}")
#     print("   Service might not be running or URL might be incorrect")
#     exit(1)

# # Your actual features based on your training code
# # Total: 43 features in this exact order
# test_samples = [
#     # Sample 1: Tuesday afternoon in June, moderate weather
#     [
#         14,      # hour
#         2,       # day_of_week (0=Monday, 6=Sunday)
#         6,       # month
#         2025,    # year
#         15,      # day
#         0.781,   # hour_sin
#         0.625,   # hour_cos
#         0.975,   # dow_sin
#         -0.223,  # dow_cos
#         0.0,     # month_sin
#         1.0,     # month_cos
#         0,       # is_morning_rush
#         0,       # is_evening_rush
#         0,       # is_night
#         1,       # is_midday
#         0,       # is_weekend
#         0,       # weekend_night
#         0,       # weekday_morning_rush
#         0,       # weekday_evening_rush
#         75,      # TMAX (temperature max)
#         60,      # TMIN (temperature min)
#         0.0,     # PRCP (precipitation)
#         15,      # temp_range
#         67.5,    # temp_avg
#         0,       # is_rainy
#         0,       # is_heavy_rain
#         0,       # is_cold
#         0,       # is_hot
#         150,     # rides_last_hour
#         145,     # rides_same_hour_yesterday
#         155,     # rides_same_hour_last_week
#         148,     # rides_rolling_3h
#         2500,    # rides_rolling_24h
#         12.5,    # duration_mean
#         3.2,     # duration_std
#         11.0,    # duration_median
#         1.8,     # distance_mean
#         0.9,     # distance_std
#         1.5,     # distance_median
#         0.71     # member_ratio
#     ],
    
#     # Sample 2: Monday morning rush hour in July, hot day
#     [
#         8,       # hour (morning rush)
#         1,       # day_of_week (Monday)
#         7,       # month
#         2025,    # year
#         7,       # day
#         0.951,   # hour_sin
#         0.309,   # hour_cos
#         0.782,   # dow_sin
#         0.623,   # dow_cos
#         0.866,   # month_sin
#         0.5,     # month_cos
#         1,       # is_morning_rush
#         0,       # is_evening_rush
#         0,       # is_night
#         0,       # is_midday
#         0,       # is_weekend
#         0,       # weekend_night
#         1,       # weekday_morning_rush (interaction feature)
#         0,       # weekday_evening_rush
#         88,      # TMAX
#         72,      # TMIN
#         0.0,     # PRCP
#         16,      # temp_range
#         80,      # temp_avg
#         0,       # is_rainy
#         0,       # is_heavy_rain
#         0,       # is_cold
#         1,       # is_hot
#         250,     # rides_last_hour
#         240,     # rides_same_hour_yesterday
#         245,     # rides_same_hour_last_week
#         235,     # rides_rolling_3h
#         3200,    # rides_rolling_24h
#         10.2,    # duration_mean
#         2.8,     # duration_std
#         9.5,     # duration_median
#         1.5,     # distance_mean
#         0.7,     # distance_std
#         1.3,     # distance_median
#         0.85     # member_ratio (high member usage during commute)
#     ],
    
#     # Sample 3: Saturday night in December, cold and rainy
#     [
#         22,      # hour (night)
#         6,       # day_of_week (Saturday)
#         12,      # month
#         2025,    # year
#         21,      # day
#         -0.781,  # hour_sin
#         0.625,   # hour_cos
#         0.0,     # dow_sin
#         -1.0,    # dow_cos
#         0.0,     # month_sin
#         -1.0,    # month_cos
#         0,       # is_morning_rush
#         0,       # is_evening_rush
#         1,       # is_night
#         0,       # is_midday
#         1,       # is_weekend
#         1,       # weekend_night (interaction: weekend + night)
#         0,       # weekday_morning_rush
#         0,       # weekday_evening_rush
#         38,      # TMAX
#         28,      # TMIN
#         0.3,     # PRCP (light rain)
#         10,      # temp_range
#         33,      # temp_avg
#         1,       # is_rainy
#         0,       # is_heavy_rain
#         1,       # is_cold
#         0,       # is_hot
#         45,      # rides_last_hour (low due to weather)
#         50,      # rides_same_hour_yesterday
#         120,     # rides_same_hour_last_week
#         48,      # rides_rolling_3h
#         1200,    # rides_rolling_24h
#         18.5,    # duration_mean (longer trips at night)
#         5.2,     # duration_std
#         17.0,    # duration_median
#         2.5,     # distance_mean
#         1.2,     # distance_std
#         2.2,     # distance_median
#         0.35     # member_ratio (more casual users on weekend nights)
#     ]
# ]

# print(f"\n  Sending {len(test_samples)} test predictions...")
# print("-" * 50)

# # Prepare request payload
# payload = {
#     "instances": test_samples
# }

# # Make predictions
# try:
#     print("\n🔮 Making predictions...")
#     response = requests.post(
#         PREDICT_ENDPOINT,
#         json=payload,
#         headers={"Content-Type": "application/json"},
#         timeout=30
#     )
    
#     if response.status_code == 200:
#         result = response.json()
#         predictions = result.get('predictions', [])
        
#         # Display results
#         scenarios = [
#             "Tuesday afternoon, moderate weather",
#             "Monday morning rush, hot day",
#             "Saturday night, cold & rainy"
#         ]
        
#         print("\n  Prediction Results:")
#         print("-" * 30)
#         for i, (scenario, prediction) in enumerate(zip(scenarios, predictions)):
#             print(f"\nScenario {i+1}: {scenario}")
#             print(f"  Predicted bike demand: {prediction:.2f} bikes")
            
#         # Quick statistics
#         print(f"\n  Prediction Statistics:")
#         print(f"  Average: {np.mean(predictions):.2f} bikes")
#         print(f"  Min: {np.min(predictions):.2f} bikes")
#         print(f"  Max: {np.max(predictions):.2f} bikes")
#         print(f"  Std Dev: {np.std(predictions):.2f}")
        
#     else:
#         print(f"  Prediction failed with status {response.status_code}")
#         print(f"   Response: {response.text}")
        
# except requests.exceptions.Timeout:
#     print("  Request timed out. The model might be taking too long to respond.")
#     print("   This could happen on the first request if the model is loading.")
#     print("   Try running the script again.")
    
# except Exception as e:
#     print(f"  Error making prediction: {e}")
#     print("\nTroubleshooting:")
#     print("1. Check if the service URL is correct")
#     print("2. Verify the model expects exactly 43 features")
#     print("3. Check Cloud Run logs for details:")
#     print(f"   gcloud run services logs read bluebikes-prediction --region us-central1")

# # Test single prediction with simpler format
# print("\n" + "="*50)
# print("🧪 Testing single prediction with 'features' format...")

# single_sample = {
#     "features": test_samples[0]  # Just the first sample
# }

# try:
#     response = requests.post(
#         PREDICT_ENDPOINT,
#         json=single_sample,
#         headers={"Content-Type": "application/json"},
#         timeout=10
#     )
    
#     if response.status_code == 200:
#         result = response.json()
#         print(f"  Single prediction successful: {result}")
#     else:
#         print(f"  Single prediction returned status {response.status_code}")
#         print(f"   Response: {response.text}")
        
# except Exception as e:
#     print(f"  Single prediction failed: {e}")

# print("\n" + "="*50)
# print("  Test complete!")
# print(f"\n🔗 Your API is live at: {SERVICE_URL}")
# print(f"   You can share this URL for others to use your model!")

# # Show curl command examples
# print("\n📝 Example curl commands:")
# print(f"curl {HEALTH_ENDPOINT}")
# print(f"""curl -X POST {PREDICT_ENDPOINT} \\
#   -H "Content-Type: application/json" \\
#   -d '{{"features": [14, 2, 6, 2025, 15, 0.781, 0.625, 0.975, -0.223, 0.0, 1.0, 0, 0, 0, 1, 0, 0, 0, 0, 75, 60, 0.0, 15, 67.5, 0, 0, 0, 0, 150, 145, 155, 148, 2500, 12.5, 3.2, 11.0, 1.8, 0.9, 1.5, 0.71]}}'""")
import json
import requests
import numpy as np

# Your Cloud Run endpoint
URL = "https://bluebikes-prediction-3xb5qzo57a-uc.a.run.app/predict"

# Load test cases
with open("test.json") as f:
    data = json.load(f)

print("="*60)
print("BLUEBIKES MODEL PREDICTION TESTS")
print("="*60)
print(f"Endpoint: {URL}")
print(f"Number of test cases: {len(data['cases'])}")
print("="*60 + "\n")

# Track results
successful = 0
failed = 0

# Run each test case
for i, case in enumerate(data["cases"], 1):
    print(f"Test {i}: {case['name']}")
    print(f"  Description: {case['description']}")
    
    # Make prediction request
    try:
        resp = requests.post(URL, json={"features": case["instances"][0]}, timeout=10)
        
        if resp.status_code == 200:
            result = resp.json()
            prediction = result['prediction']
            successful += 1
            
            print(f"  ✓ SUCCESS - Predicted demand: {prediction:.2f} bikes")
            
            # Add context based on prediction value
            if prediction < 50:
                print(f"    → Low demand expected")
            elif prediction < 150:
                print(f"    → Moderate demand expected")
            elif prediction < 300:
                print(f"    → High demand expected")
            else:
                print(f"    → Very high demand expected")
                
            # Check for warnings
            if 'warning' in result:
                print(f"    ⚠ Warning: {result['warning']}")
                
        else:
            failed += 1
            print(f"  ✗ FAILED - Status: {resp.status_code}")
            print(f"    Error: {resp.json()}")
            
    except requests.exceptions.Timeout:
        failed += 1
        print(f"  ✗ TIMEOUT - Request took too long")
    except Exception as e:
        failed += 1
        print(f"  ✗ ERROR - {str(e)}")
    
    print()

# Summary
print("="*60)
print("TEST SUMMARY")
print("="*60)
print(f"Total tests: {len(data['cases'])}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Success rate: {(successful/len(data['cases'])*100):.1f}%")
print("="*60)

# If all tests passed, show statistics
if successful > 0:
    print("\nPREDICTION STATISTICS:")
    
    # Collect all successful predictions
    predictions = []
    for case in data["cases"]:
        try:
            resp = requests.post(URL, json={"features": case["instances"][0]}, timeout=5)
            if resp.status_code == 200:
                predictions.append(resp.json()['prediction'])
        except:
            pass
    
    if predictions:
        print(f"  Average: {np.mean(predictions):.2f} bikes")
        print(f"  Min: {np.min(predictions):.2f} bikes")
        print(f"  Max: {np.max(predictions):.2f} bikes")
        print(f"  Std Dev: {np.std(predictions):.2f}")
        
        # Show which scenarios have highest/lowest demand
        sorted_cases = sorted(
            [(case["name"], pred) for case, pred in zip(data["cases"], predictions) if pred],
            key=lambda x: x[1]
        )
        
        print(f"\n  Lowest demand: {sorted_cases[0][0]} ({sorted_cases[0][1]:.2f} bikes)")
        print(f"  Highest demand: {sorted_cases[-1][0]} ({sorted_cases[-1][1]:.2f} bikes)")