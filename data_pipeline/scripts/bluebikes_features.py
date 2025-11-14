# # # bluebikes_features.py
# # import pandas as pd
# # import numpy as np
# # from pathlib import Path
# # import openpyxl


# # def most_common(series):
# #     """Return most common value or NaN if empty."""
# #     if len(series) == 0:
# #         return np.nan
# #     vc = series.value_counts()
# #     return vc.idxmax()


# # def extract_bluebikes_features(input_pickle_path, output_pickle_path):
# #     """
# #     Extract hourly demand features for Bluebikes dataset.
# #     Output: num_trips per station per hour (across months)
# #     """
# #     print("🔹 Loading data...")
# #     df = pd.read_pickle(input_pickle_path)

# #     # Validate column
# #     if "start_time" not in df.columns:
# #         raise ValueError("❌ 'start_time' column not found in dataset.")

# #     # Convert start_time to datetime
# #     df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

# #     # Create hour-level timestamps
# #     df["hour_timestamp"] = df["start_time"].dt.floor("h")

# #     # --- Aggregate trips per station per hour ---
# #     print("🔹 Aggregating hourly demand per station...")
# #     if "ride_id" in df.columns:
# #         agg_df = (
# #             df.groupby(["start_station_name", "hour_timestamp"])
# #             .agg(
# #                 num_trips=("ride_id", "count"),
# #                 rideable_type=("rideable_type", lambda x: most_common(x.dropna())),
# #             )
# #             .reset_index()
# #         )
# #     else:
# #         agg_df = (
# #             df.groupby(["start_station_name", "hour_timestamp"])
# #             .agg(
# #                 num_trips=("start_time", "count"),
# #                 rideable_type=("rideable_type", lambda x: most_common(x.dropna())),
# #             )
# #             .reset_index()
# #         )

# #     # --- Optional: Fill missing hours for time continuity ---
# #     print("🔹 Filling missing station-hour combinations...")
# #     stations = df["start_station_name"].unique()
# #     full_hours = pd.date_range(
# #         df["start_time"].min().floor("h"), df["start_time"].max().ceil("h"), freq="H"
# #     )

# #     full_index = pd.MultiIndex.from_product(
# #         [stations, full_hours], names=["start_station_name", "hour_timestamp"]
# #     )

# #     agg_df = (
# #         agg_df.set_index(["start_station_name", "hour_timestamp"])
# #         .reindex(full_index, fill_value=0)
# #         .reset_index()
# #     )

# #     # --- Add derived temporal features ---
# #     agg_df["hour"] = agg_df["hour_timestamp"].dt.hour
# #     agg_df["day_of_week"] = agg_df["hour_timestamp"].dt.dayofweek
# #     agg_df["month"] = agg_df["hour_timestamp"].dt.month
# #     agg_df["is_weekend"] = (agg_df["day_of_week"] >= 5).astype(int)

# #     # --- Save output ---
# #     Path(output_pickle_path).parent.mkdir(parents=True, exist_ok=True)
# #     agg_df.to_pickle(output_pickle_path)

# #     print(f"✅ Saved features to pickle: {output_pickle_path}")
# #     return agg_df


# # if __name__ == "__main__":
# #     in_pkl = r"D:\\MLOps_Coursework\\ML-OPs\data_pipeline\data\\processed\bluebikes\\after_duplicates.pkl"
# #     out_pkl = r"D:\\MLOps_Coursework\\ML-OPs\data_pipeline\data\\processed\bluebikes\\features.pkl"

# #     print("🚴 Extracting Bluebikes features (hourly demand)...")
# #     df_features = extract_bluebikes_features(in_pkl, out_pkl)

# #     # Save to CSV and Excel
# #     # out_csv = out_pkl.replace(".pkl", ".csv")
# #     # out_xlsx = out_pkl.replace(".pkl", ".xlsx")

# #     # df_features.to_csv(out_csv, index=False)
# #     # df_features.to_excel(out_xlsx, index=False)

# #     # print(f"✅ Features saved to CSV:   {out_csv}")
# #     # print(f"✅ Features saved to Excel: {out_xlsx}")

# #     # Preview
# #     print("\n🔹 Sample output:")
# #     print(df_features.head())
# #     print(df_features.shape)
# #     print(df_features.columns.tolist())



# import pandas as pd
# import numpy as np
# from pathlib import Path
# import openpyxl


# def most_common(series):
#     """Return most common value or NaN if empty."""
#     if len(series) == 0:
#         return np.nan
#     vc = series.value_counts()
#     return vc.idxmax()


# def extract_bluebikes_features(input_pickle_path, output_pickle_path):
#     """
#     Extract hourly net flow features for Bluebikes dataset.
#     Output: trips_started, trips_ended, and net_flow per station per hour.
#     Positive net_flow → more bikes arrived than left.
#     """
#     print("🔹 Loading data...")
#     df = pd.read_pickle(input_pickle_path)

#     # --- Validate columns ---
#     required_cols = {"start_time", "stop_time", "start_station_name", "end_station_name"}
#     missing = required_cols - set(df.columns)
#     if missing:
#         raise ValueError(f"❌ Missing required columns: {missing}")

#     # --- Convert to datetime ---
#     df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
#     df["stop_time"] = pd.to_datetime(df["stop_time"], errors="coerce")

#     # --- Create hour-level timestamps ---
#     df["start_hour"] = df["start_time"].dt.floor("h")
#     df["end_hour"] = df["stop_time"].dt.floor("h")

#     print("🔹 Aggregating trips started and ended per station-hour...")
#     # Trips started
#     starts = (
#         df.groupby(["start_station_name", "start_hour"])
#         .size()
#         .reset_index(name="trips_started")
#         .rename(columns={"start_station_name": "station_name", "start_hour": "hour_timestamp"})
#     )

#     # Trips ended
#     ends = (
#         df.groupby(["end_station_name", "end_hour"])
#         .size()
#         .reset_index(name="trips_ended")
#         .rename(columns={"end_station_name": "station_name", "end_hour": "hour_timestamp"})
#     )

#     # --- Merge starts and ends ---
#     agg_df = pd.merge(starts, ends, on=["station_name", "hour_timestamp"], how="outer")

#     # --- Fill missing counts ---
#     agg_df["trips_started"] = agg_df["trips_started"].fillna(0)
#     agg_df["trips_ended"] = agg_df["trips_ended"].fillna(0)

#     # --- Compute net flow ---
#     agg_df["net_flow"] = agg_df["trips_ended"] - agg_df["trips_started"]

#     # --- Fill missing station-hour combinations for continuity ---
#     print("🔹 Filling missing station-hour combinations...")
#     all_stations = pd.unique(df[["start_station_name", "end_station_name"]].values.ravel("K"))
#     full_hours = pd.date_range(
#         min(df["start_time"].min(), df["stop_time"].min()).floor("h"),
#         max(df["start_time"].max(), df["stop_time"].max()).ceil("h"),
#         freq="H",
#     )

#     full_index = pd.MultiIndex.from_product(
#         [all_stations, full_hours], names=["station_name", "hour_timestamp"]
#     )

#     agg_df = (
#         agg_df.set_index(["station_name", "hour_timestamp"])
#         .reindex(full_index, fill_value=0)
#         .reset_index()
#     )

#     # --- Add temporal features ---
#     agg_df["hour"] = agg_df["hour_timestamp"].dt.hour
#     agg_df["day_of_week"] = agg_df["hour_timestamp"].dt.dayofweek
#     agg_df["month"] = agg_df["hour_timestamp"].dt.month
#     agg_df["is_weekend"] = (agg_df["day_of_week"] >= 5).astype(int)

#     # --- Save output ---
#     Path(output_pickle_path).parent.mkdir(parents=True, exist_ok=True)
#     agg_df.to_pickle(output_pickle_path)

#     print(f"✅ Saved features to pickle: {output_pickle_path}")
#     return agg_df


# if __name__ == "__main__":
#     in_pkl = r"D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\after_duplicates.pkl"
#     out_pkl = r"D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\features_netflow.pkl"

#     print("🚴 Extracting Bluebikes net flow features...")
#     df_features = extract_bluebikes_features(in_pkl, out_pkl)

#     # --- Optional: Save to CSV/Excel ---
#     # out_csv = out_pkl.replace(".pkl", ".csv")
#     # out_xlsx = out_pkl.replace(".pkl", ".xlsx")
#     # df_features.to_csv(out_csv, index=False)
#     # df_features.to_excel(out_xlsx, index=False)
#     # print(f"✅ Features saved to CSV:   {out_csv}")
#     # print(f"✅ Features saved to Excel: {out_xlsx}")

#     print("\n🔹 Sample output:")
#     print(df_features.head())
#     print(df_features.shape)
#     print(df_features.columns.tolist())


# bluebikes_features_trips_no_lag.py
import pandas as pd
import holidays
from pathlib import Path

def extract_bluebikes_features_with_weather(
    bikes_pickle_path,
    weather_pickle_path=None,
    output_pickle_path=None,
    output_csv_path=None
):
    """
    Extract features for Bluebikes trips per hour (without lag features):
    - Temporal features (hour, day_of_week, month, weekend)
    - Optional weather features (TMAX, TMIN, PRCP)
    - Holiday flags
    """
    print("🔹 Loading bike data...")
    df = pd.read_pickle(bikes_pickle_path)

    # Validate columns
    required_cols = {"start_time", "start_station_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    # Aggregate trips started per station per hour
    df["hour_timestamp"] = df["start_time"].dt.floor("h")
    df_start = (
        df.groupby(["start_station_name", "hour_timestamp"])
          .size().reset_index(name="trips_started")
          .rename(columns={"start_station_name": "station_name"})
    )

    # Fill missing hours for continuity
    stations = df_start["station_name"].unique()
    full_hours = pd.date_range(df_start["hour_timestamp"].min(), df_start["hour_timestamp"].max(), freq="H")
    full_index = pd.MultiIndex.from_product([stations, full_hours], names=["station_name", "hour_timestamp"])
    agg_df = df_start.set_index(["station_name", "hour_timestamp"]).reindex(full_index, fill_value=0).reset_index()

    # Temporal features
    agg_df["hour"] = agg_df["hour_timestamp"].dt.hour
    agg_df["day_of_week"] = agg_df["hour_timestamp"].dt.dayofweek
    agg_df["month"] = agg_df["hour_timestamp"].dt.month
    agg_df["is_weekend"] = (agg_df["day_of_week"] >= 5).astype(int)

    # Holiday feature
    agg_df["is_holiday"] = agg_df["hour_timestamp"].dt.date.isin(holidays.US(years=[2023, 2024, 2025])).astype(int)

    # Weather features (optional)
    if weather_pickle_path:
        print("🔹 Loading weather data...")
        weather_df = pd.read_pickle(weather_pickle_path)
        if "date" not in weather_df.columns:
            raise ValueError("❌ Weather pickle must have 'date' column")
        weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date
        agg_df["date"] = agg_df["hour_timestamp"].dt.date
        agg_df = agg_df.merge(weather_df[["date", "TMAX", "TMIN", "PRCP"]], on="date", how="left")
        agg_df.drop(columns=["date"], inplace=True)
        print("✅ Weather features merged.")

    # Save output
    if output_pickle_path:
        Path(output_pickle_path).parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_pickle(output_pickle_path)
        print(f"✅ Saved features to pickle: {output_pickle_path}")
    if output_csv_path:
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(output_csv_path, index=False)
        print(f"✅ Saved features to CSV: {output_csv_path}")

    return agg_df


if __name__ == "__main__":
    bikes_pkl = "D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\after_duplicates.pkl"
    weather_pkl = "D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\noaa_weather\\after_duplicates.pkl"
    out_pkl = "D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\features_trips_no_lag.pkl"
    out_csv = "D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\features_trips_no_lag.csv"

    df_features = extract_bluebikes_features_with_weather(
        bikes_pkl,
        weather_pickle_path=weather_pkl,
        output_pickle_path=out_pkl,
        output_csv_path=out_csv
    )

    print("\n🔹 Sample output:")
    print(df_features.head())
    print(df_features.shape)
    print(df_features.columns.tolist())
