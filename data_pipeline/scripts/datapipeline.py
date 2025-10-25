# datapipeline.py

import os
from data_collection import collect_bluebikes_data, collect_boston_college_data, collect_NOAA_Weather_data
from data_loader import load_data
from missing_value import handle_missing
from duplicate_data import handle_duplicates

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Dataset configuration ---
DATASETS = [
    {
        "name": "bluebikes",
        "raw_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "raw", "bluebikes"),
        "processed_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "processed", "bluebikes"),
        "preprocessing": {
            "missing_config": {
                "drop_columns": ["end_station_latitude", "end_station_longitude"],
                "fill_strategies": {"start_station_name": "mode", "end_station_name": "mode"},
                "raise_on_remaining": False
            }
        }
    },
    {
        "name": "boston_clg",
        "raw_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "raw", "boston_clg"),
        "processed_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "processed", "boston_clg")
    },
    {
        "name": "NOAA_weather",
        "raw_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "raw", "NOAA_weather"),
        "processed_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "processed", "NOAA_weather")
    }
]

def process_missing(input_path, output_path, config):
    handle_missing(
        input_pickle_path=os.path.join(input_path, "raw_data.pkl"),
        output_pickle_path=os.path.join(output_path, "raw_data.pkl"),
        drop_columns=config.get("drop_columns"),
        fill_strategies=config.get("fill_strategies"),
        raise_on_remaining=config.get("raise_on_remaining", True)
    )

def process_duplicates(input_path, output_path, config):
    handle_duplicates(
        input_pickle_path=os.path.join(input_path, "raw_data.pkl"),
        output_pickle_path=os.path.join(output_path, "raw_data.pkl"),
        subset=config.get("subset"),
        keep=config.get("keep", "first"),
        consider_all_columns=config.get("consider_all_columns", False),
        raise_on_remaining=config.get("raise_on_remaining", False)
    )

if __name__ == "__main__":
    # --- Data Collection ---
    collect_bluebikes_data(
        index_url="https://s3.amazonaws.com/hubway-data/index.html",
        years=["2025"],
        download_dir=os.path.join(PROJECT_DIR, "data_pipeline", "data", "temp", "bluebikes"),
        parquet_dir=os.path.join(PROJECT_DIR, "data_pipeline", "data", "raw", "bluebikes"),
        log_path="read_log.csv"
    )
    collect_boston_college_data()
    collect_NOAA_Weather_data()

    # --- Data Loading & Conditional Preprocessing ---
    for dataset in DATASETS:
        try:
            print(f"\nProcessing dataset: {dataset['name']}")

            # Load data
            load_data(
                pickle_path=dataset["processed_path"],
                data_paths=[dataset["raw_path"]],
                dataset_name=dataset["name"]
            )

            preprocessing = dataset.get("preprocessing", {})

            if "missing_config" in preprocessing:
                print(f"  -> Handling missing values for {dataset['name']}")
                process_missing(dataset["processed_path"], dataset["processed_path"], preprocessing["missing_config"])

            if "duplicates" in preprocessing:
                print(f"  -> Handling duplicates for {dataset['name']}")
                process_duplicates(dataset["processed_path"], dataset["processed_path"], preprocessing["duplicates"])

        except Exception as e:
            print(f"FAILED: Processing of {dataset['name']} failed")
            print(f"ERROR: {e}")

    print("\nAll datasets processed successfully.")
