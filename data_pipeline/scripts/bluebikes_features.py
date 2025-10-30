# bluebikes_features.py
import pandas as pd
import numpy as np
from pathlib import Path
import openpyxl


def most_common(series):
    """Return most common value or NaN if empty."""
    if len(series) == 0:
        return np.nan
    vc = series.value_counts()
    return vc.idxmax()


def extract_bluebikes_features(input_pickle_path, output_pickle_path):
    """
    Extract hourly demand features for Bluebikes dataset.
    Output: num_trips per station per hour (across months)
    """
    print("🔹 Loading data...")
    df = pd.read_pickle(input_pickle_path)

    # Validate column
    if "start_time" not in df.columns:
        raise ValueError("❌ 'start_time' column not found in dataset.")

    # Convert start_time to datetime
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    # Create hour-level timestamps
    df["hour_timestamp"] = df["start_time"].dt.floor("h")

    # --- Aggregate trips per station per hour ---
    print("🔹 Aggregating hourly demand per station...")
    if "ride_id" in df.columns:
        agg_df = (
            df.groupby(["start_station_name", "hour_timestamp"])
            .agg(
                num_trips=("ride_id", "count"),
                rideable_type=("rideable_type", lambda x: most_common(x.dropna())),
            )
            .reset_index()
        )
    else:
        agg_df = (
            df.groupby(["start_station_name", "hour_timestamp"])
            .agg(
                num_trips=("start_time", "count"),
                rideable_type=("rideable_type", lambda x: most_common(x.dropna())),
            )
            .reset_index()
        )

    # --- Optional: Fill missing hours for time continuity ---
    print("🔹 Filling missing station-hour combinations...")
    stations = df["start_station_name"].unique()
    full_hours = pd.date_range(
        df["start_time"].min().floor("h"), df["start_time"].max().ceil("h"), freq="H"
    )

    full_index = pd.MultiIndex.from_product(
        [stations, full_hours], names=["start_station_name", "hour_timestamp"]
    )

    agg_df = (
        agg_df.set_index(["start_station_name", "hour_timestamp"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # --- Add derived temporal features ---
    agg_df["hour"] = agg_df["hour_timestamp"].dt.hour
    agg_df["day_of_week"] = agg_df["hour_timestamp"].dt.dayofweek
    agg_df["month"] = agg_df["hour_timestamp"].dt.month
    agg_df["is_weekend"] = (agg_df["day_of_week"] >= 5).astype(int)

    # --- Save output ---
    Path(output_pickle_path).parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_pickle(output_pickle_path)

    print(f"✅ Saved features to pickle: {output_pickle_path}")
    return agg_df


if __name__ == "__main__":
    in_pkl = r"D:\MLOps_Coursework\ML-OPs\data_pipeline\data\processed\bluebikes\after_duplicates.pkl"
    out_pkl = r"D:\MLOps_Coursework\ML-OPs\data_pipeline\data\processed\bluebikes\features.pkl"

    print("🚴 Extracting Bluebikes features (hourly demand)...")
    df_features = extract_bluebikes_features(in_pkl, out_pkl)

    # Save to CSV and Excel
    # out_csv = out_pkl.replace(".pkl", ".csv")
    # out_xlsx = out_pkl.replace(".pkl", ".xlsx")

    # df_features.to_csv(out_csv, index=False)
    # df_features.to_excel(out_xlsx, index=False)

    # print(f"✅ Features saved to CSV:   {out_csv}")
    # print(f"✅ Features saved to Excel: {out_xlsx}")

    # Preview
    print("\n🔹 Sample output:")
    print(df_features.head())
    print(df_features.shape)
    print(df_features.columns.tolist())
