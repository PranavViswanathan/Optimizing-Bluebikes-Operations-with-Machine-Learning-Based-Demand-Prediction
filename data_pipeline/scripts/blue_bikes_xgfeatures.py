import pandas as pd
import numpy as np
import holidays
from pathlib import Path

def extract_bluebikes_features_with_weather(
    bikes_pickle_path,
    weather_pickle_path=None,
    output_pickle_path=None,
    output_csv_path=None
):
    """
    Extract full Bluebikes hourly features aligned with the LightGBM training pipeline:
      - Temporal & cyclic features
      - Ride duration/distance/member ratio aggregates
      - Weather and derived weather flags
      - Time-of-day indicators (rush hours, weekend, etc.)
      - Lag and rolling features on ride_count
    """
    print("🔹 Loading bike data...")
    df = pd.read_pickle(bikes_pickle_path)
    print(f"✅ Loaded {len(df):,} trips")

    # ---- Clean and basic preprocessing ----
    df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize(None)
    df["stop_time"] = pd.to_datetime(df["stop_time"]).dt.tz_localize(None)

    df["duration_minutes"] = (df["stop_time"] - df["start_time"]).dt.total_seconds() / 60
    df = df[(df["duration_minutes"] > 0) & (df["duration_minutes"] < 1440)]

    # ---- Compute haversine distance ----
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    df["distance_km"] = haversine(
        df["start_station_latitude"], df["start_station_longitude"],
        df["end_station_latitude"], df["end_station_longitude"]
    )

    df["is_member"] = (df["user_type"] == "member").astype(int)

    # ---- Temporal breakdown ----
    df["date"] = df["start_time"].dt.date
    df["hour"] = df["start_time"].dt.hour
    df["day_of_week"] = df["start_time"].dt.dayofweek
    df["month"] = df["start_time"].dt.month
    df["year"] = df["start_time"].dt.year
    df["day"] = df["start_time"].dt.day

    # ---- Aggregate hourly metrics ----
    hourly = df.groupby(["date", "hour"]).agg({
        "ride_id": "count",
        "duration_minutes": ["mean", "std", "median"],
        "distance_km": ["mean", "std", "median"],
        "is_member": "mean",
        "day_of_week": "first",
        "month": "first",
        "year": "first",
        "day": "first"
    }).reset_index()

    hourly.columns = [
        "date", "hour", "ride_count",
        "duration_mean", "duration_std", "duration_median",
        "distance_mean", "distance_std", "distance_median",
        "member_ratio", "day_of_week", "month", "year", "day"
    ]

    hourly["duration_std"] = hourly["duration_std"].fillna(0)
    hourly["distance_std"] = hourly["distance_std"].fillna(0)

    # ---- Fill missing hours for continuity ----
    date_range = pd.date_range(hourly["date"].min(), hourly["date"].max(), freq="D").date
    all_hours = range(24)
    full_index = pd.MultiIndex.from_product([date_range, all_hours], names=["date", "hour"])
    hourly_full = hourly.set_index(["date", "hour"]).reindex(full_index, fill_value=0).reset_index()

    # ---- Forward-fill temporal columns ----
    for col in ["day_of_week", "month", "year", "day"]:
        hourly_full[col] = hourly_full.groupby("date")[col].transform(lambda x: x.replace(0, np.nan).ffill().bfill().fillna(0))

    # ---- Merge weather ----
    if weather_pickle_path:
        print("🔹 Loading weather data...")
        weather = pd.read_pickle(weather_pickle_path)
        weather["date"] = pd.to_datetime(weather["date"]).dt.date
        hourly_full = hourly_full.merge(weather, on="date", how="left")
        hourly_full = hourly_full.dropna(subset=["TMAX", "TMIN", "PRCP"])

    # ---- Cyclic encodings ----
    hourly_full["hour_sin"] = np.sin(2 * np.pi * hourly_full["hour"] / 24)
    hourly_full["hour_cos"] = np.cos(2 * np.pi * hourly_full["hour"] / 24)
    hourly_full["dow_sin"] = np.sin(2 * np.pi * hourly_full["day_of_week"] / 7)
    hourly_full["dow_cos"] = np.cos(2 * np.pi * hourly_full["day_of_week"] / 7)
    hourly_full["month_sin"] = np.sin(2 * np.pi * hourly_full["month"] / 12)
    hourly_full["month_cos"] = np.cos(2 * np.pi * hourly_full["month"] / 12)

    # ---- Time-of-day and weekday flags ----
    hourly_full["is_morning_rush"] = hourly_full["hour"].isin([7, 8, 9]).astype(int)
    hourly_full["is_evening_rush"] = hourly_full["hour"].isin([17, 18, 19]).astype(int)
    hourly_full["is_night"] = ((hourly_full["hour"] >= 22) | (hourly_full["hour"] <= 5)).astype(int)
    hourly_full["is_midday"] = hourly_full["hour"].isin([11, 12, 13, 14]).astype(int)
    hourly_full["is_weekend"] = hourly_full["day_of_week"].isin([5, 6]).astype(int)

    hourly_full["weekend_night"] = hourly_full["is_weekend"] * hourly_full["is_night"]
    hourly_full["weekday_morning_rush"] = (1 - hourly_full["is_weekend"]) * hourly_full["is_morning_rush"]
    hourly_full["weekday_evening_rush"] = (1 - hourly_full["is_weekend"]) * hourly_full["is_evening_rush"]

    # ---- Weather-derived features ----
    hourly_full["temp_range"] = hourly_full["TMAX"] - hourly_full["TMIN"]
    hourly_full["temp_avg"] = (hourly_full["TMAX"] + hourly_full["TMIN"]) / 2
    hourly_full["is_rainy"] = (hourly_full["PRCP"] > 0).astype(int)
    hourly_full["is_heavy_rain"] = (hourly_full["PRCP"] > 10).astype(int)
    hourly_full["is_cold"] = (hourly_full["temp_avg"] < 5).astype(int)
    hourly_full["is_hot"] = (hourly_full["temp_avg"] > 25).astype(int)

    # ---- Lag and rolling features ----
    hourly_full = hourly_full.sort_values(["date", "hour"]).reset_index(drop=True)
    hourly_full["rides_last_hour"] = hourly_full["ride_count"].shift(1).fillna(0)
    hourly_full["rides_same_hour_yesterday"] = hourly_full["ride_count"].shift(24).fillna(0)
    hourly_full["rides_same_hour_last_week"] = hourly_full["ride_count"].shift(24 * 7).fillna(0)
    hourly_full["rides_rolling_3h"] = hourly_full["ride_count"].shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
    hourly_full["rides_rolling_24h"] = hourly_full["ride_count"].shift(1).rolling(window=24, min_periods=1).mean().fillna(0)

    # ---- Save output ----
    if output_pickle_path:
        Path(output_pickle_path).parent.mkdir(parents=True, exist_ok=True)
        hourly_full.to_pickle(output_pickle_path)
        print(f"✅ Saved features to pickle: {output_pickle_path}")

    if output_csv_path:
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        hourly_full.to_csv(output_csv_path, index=False)
        print(f"✅ Saved features to CSV: {output_csv_path}")

    print(f"✅ Final feature dataframe: {hourly_full.shape}")
    return hourly_full


if __name__ == "__main__":
    bikes_pkl = "D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\after_duplicates.pkl"
    weather_pkl = "D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\noaa_weather\\after_duplicates.pkl"
    out_pkl = "D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\features_full.pkl"
    out_csv = "D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\features_full.csv"

    df_features = extract_bluebikes_features_with_weather(
        bikes_pkl,
        weather_pickle_path=weather_pkl,
        output_pickle_path=out_pkl,
        output_csv_path=out_csv
    )

    print("\n🔹 Sample output:")
    print(df_features.head())
    print(df_features.columns.tolist())
