import pandas as pd

# File paths
trips_file = "bluebikes/combined_bluebikes.csv"
weather_file = "NOAA/boston_daily_weather.csv"
output_file = "bluebikes/processed_bike_weather.csv"

# -------------------
# 1. Load Data
# -------------------
trips = pd.read_csv(trips_file, parse_dates=["starttime", "stoptime"])
weather = pd.read_csv(weather_file, parse_dates=["date"])

# -------------------
# 2. Aggregate trips by station & hour
# -------------------
# Extract temporal features
trips["date"] = trips["starttime"].dt.date
trips["hour"] = trips["starttime"].dt.hour
trips["day_of_week"] = trips["starttime"].dt.dayofweek  # 0=Mon, 6=Sun
trips["month"] = trips["starttime"].dt.month
trips["is_weekend"] = trips["day_of_week"].isin([5, 6]).astype(int)

# Group by start station + date + hour
agg_trips = (
    trips.groupby(["start station id", "date", "hour"])
    .size()
    .reset_index(name="trip_count")
)

# -------------------
# 3. Merge with weather data
# -------------------
agg_trips["date"] = pd.to_datetime(agg_trips["date"])
weather["date"] = pd.to_datetime(weather["date"])

merged = agg_trips.merge(weather, on="date", how="left")

# -------------------
# 4. Handle missing values
# -------------------
merged["trip_count"] = merged["trip_count"].fillna(0)
merged = merged.fillna(method="ffill").fillna(method="bfill")

# -------------------
# 5. Encode categorical variables
# -------------------
station_dummies = pd.get_dummies(merged["start station id"], prefix="station")
processed = pd.concat([merged.drop(columns=["start station id"]), station_dummies], axis=1)

# -------------------
# 6. Save to output file
# -------------------
processed.to_csv(output_file, index=False)

print(f"✅ Processed dataset saved to {output_file}")
print(processed.head())
