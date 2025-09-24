import pandas as pd
from scipy.spatial import cKDTree
import numpy as np

trips_file = "bluebikes/bluebikes_data/combined_dataset/combined_bluebikes.csv"      
weather_file = "NOAA/boston_daily_weather.csv"  
schools_file = "Boston_GIS/boston_schools.csv"       
output_file = "bluebikes_weather_schools.csv"


trips = pd.read_csv(trips_file, parse_dates=["starttime"])
weather = pd.read_csv(weather_file, parse_dates=["date"])
schools = pd.read_csv(schools_file)

print(f"Trips: {trips.shape}, Weather: {weather.shape}, Schools: {schools.shape}")


trips["date"] = trips["starttime"].dt.date
weather["date"] = weather["date"].dt.date
trips = trips.merge(weather, on="date", how="left")
print("Merged weather. Trips shape:", trips.shape)

school_coords = schools[["latitude", "longitude"]].to_numpy()
school_tree = cKDTree(school_coords)
trip_coords = trips[["start station latitude", "start station longitude"]].to_numpy()
distances, indices = school_tree.query(trip_coords, k=1)
trips["nearest_school_name"] = schools.iloc[indices]["name"].values
trips["nearest_school_type"] = schools.iloc[indices]["type"].values
trips["distance_to_school_meters"] = distances * 111000  # approximate conversion lat/lon -> meters

trips.to_csv(output_file, index=False)
print(f"Saved combined dataset to {output_file}")
