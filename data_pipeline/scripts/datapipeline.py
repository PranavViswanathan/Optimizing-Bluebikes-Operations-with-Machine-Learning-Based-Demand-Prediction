from data_collection import run, collect_additional_data
from data_loader import load_data
from missing_value import handle_missing
from duplicate_data import handle_duplicates
import os

PROJECT_DIR="F:/MS in CS/MlOps/Project/Optimizing-Bluebikes-Operations-with-Machine-Learning-Based-Demand-Prediction/"
if __name__ == "__main__":
    run("https://s3.amazonaws.com/hubway-data/index.html", ["2025"], "bluebikes_zips", "parquet", "read_log.csv")
    # collect_additional_data()
    load_data(pickle_path="../data/processed/bluebikes", data_paths=["./parquet/trips_2025.parquet"], dataset_name="bluebikes")
    handle_missing(input_pickle_path=PROJECT_DIR+"data_pipeline/data/processed/bluebikes/raw_data.pkl", output_pickle_path=PROJECT_DIR+"data_pipeline/data/processed/bluebikes/raw_data.pkl", drop_columns=["end_station_latitude", "end_station_longitude"], fill_strategies={"start_station_name": "mode", "end_station_name": "mode"}, raise_on_remaining=False)
    handle_duplicates(
        input_pickle_path=PROJECT_DIR+"data_pipeline/data/processed/bluebikes/raw_data.pkl",
        output_pickle_path=PROJECT_DIR+"data_pipeline/data/processed/bluebikes/raw_data.pkl",
        subset=None,
        keep='first',
        consider_all_columns=False,
        raise_on_remaining=False
    )
    
    