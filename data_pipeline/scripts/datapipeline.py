import os
import sys
from pathlib import Path

# ---------- PATH SETUP ----------
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- IMPORTS ----------
from data_collection import (
    collect_bluebikes_data,
    collect_boston_college_data,
    collect_NOAA_Weather_data
)
from data_loader import load_data
from missing_value import handle_missing
from duplicate_data import handle_duplicates
from correlation_matrix import correlation_matrix
from station_ids_mapping import assign_station_ids

from logger import get_logger
logger = get_logger("datapipeline")

# ---------- PROJECT PATHS ----------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = [
    {
        "name": "bluebikes",
        "raw_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "raw", "bluebikes"),
        "processed_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "processed", "bluebikes"),
        "preprocessing": {
            "assign_station_ids": True, 
            "missing_config": {
                "drop_columns": ["end_station_latitude", "end_station_longitude", 'start_station_id', 'end_station_id'],
                "fill_strategies": {"start_station_name": "mode", "end_station_name": "mode"},
                "raise_on_remaining": False
            },
            "duplicates": {
                "subset": ["ride_id"],
                "keep": "first",
                "consider_all_columns": False,
                "raise_on_remaining": False
            }
        }
    },
    {
        "name": "boston_clg",
        "raw_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "raw", "boston_clg"),
        "processed_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "processed", "boston_clg"),
        "preprocessing": {
            "missing_config": {
                "fill_strategies": {
                    "Cost": "median",
                    "NumStudent": "median",
                    "BackupPowe": "median",
                    "Latitude": "median",
                    "Longitude": "median",
                    "X": "median",
                    "Y": "median",
                    "NumStudents13": "median"
                },
                "drop_columns": [],
                "raise_on_remaining": False
            },
            "duplicates": {
                "subset": ["Ref_ID", "ID1", "SchoolId", "Name", "Address"],
                "keep": "first",
                "consider_all_columns": False,
                "raise_on_remaining": False
            }
        }
    },
    {
        "name": "NOAA_weather",
        "raw_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "raw", "NOAA_weather"),
        "processed_path": os.path.join(PROJECT_DIR, "data_pipeline", "data", "processed", "NOAA_weather"),
        "preprocessing": {
            "missing_config": {
                "fill_strategies": {
                    "PRCP": "median",
                    "TMAX": "median",
                    "TMIN": "median"
                },
                "drop_columns": [],
                "raise_on_remaining": False,
                "duplicates": {
                "subset": ["date", "PRCP", "TMAX", "TMIN"],
                "keep": "first",
                "consider_all_columns": False,
                "raise_on_remaining": False
            }
            }
        }
    }
]


# ---------- PREPROCESSING FUNCTIONS ----------
def process_assign_station_ids(input_path, output_path):
    assign_station_ids(
        input_pickle_path=os.path.join(input_path, "raw_data.pkl"),
        output_pickle_path=os.path.join(output_path, "station_id_mapping.pkl")
    )

def process_missing(input_path, output_path, config, dataset):
    if(dataset=="bluebikes"):
        input_path=os.path.join(input_path,"station_id_mapping.pkl")
    else:
        input_path=os.path.join(input_path, "raw_data.pkl")
    handle_missing(
        input_pickle_path=input_path,
        output_pickle_path=os.path.join(output_path, "after_missing_data.pkl"),
        drop_columns=config.get("drop_columns"),
        fill_strategies=config.get("fill_strategies"),
        raise_on_remaining=config.get("raise_on_remaining", True)
    )

def process_duplicates(input_path, output_path, config):
    handle_duplicates(
        input_pickle_path=os.path.join(input_path, "after_missing_data.pkl"),
        output_pickle_path=os.path.join(output_path, "after_duplicates.pkl"),
        subset=config.get("subset"),
        keep=config.get("keep", "first"),
        consider_all_columns=config.get("consider_all_columns", False),
        raise_on_remaining=config.get("raise_on_remaining", False)
    )

# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    logger.info("Starting Data Pipeline...")

    # --- Data Collection ---
    try:
        logger.info("Collecting Bluebikes, Boston College, and NOAA datasets...")
        collect_bluebikes_data(
            index_url="https://s3.amazonaws.com/hubway-data/index.html",
            years=["2025"],
            download_dir=os.path.join(PROJECT_DIR, "data_pipeline", "data", "temp", "bluebikes"),
            parquet_dir=os.path.join(PROJECT_DIR, "data_pipeline", "data", "raw", "bluebikes"),
            log_path="read_log.csv"
        )
        collect_boston_college_data()
        collect_NOAA_Weather_data()
    except Exception as e:
        logger.exception(f"Data collection failed: {e}")

    # --- Data Loading & Preprocessing ---
    for dataset in DATASETS:
        try:
            logger.info(f"\nProcessing dataset: {dataset['name']}")

            # Load data
            pickle_path = load_data(
                pickle_path=dataset["processed_path"],
                data_paths=[dataset["raw_path"]],
                dataset_name=dataset["name"]
            )

            preprocessing = dataset.get("preprocessing", {})

            # --- Assign station IDs first for Bluebikes ---
            if preprocessing.get("assign_station_ids", False):
                logger.info(f"Assigning station IDs for {dataset['name']}")
                process_assign_station_ids(dataset["processed_path"], dataset["processed_path"])

            # --- Missing values ---
            if "missing_config" in preprocessing:
                logger.info(f"Handling missing values for {dataset['name']}")
                process_missing(dataset["processed_path"], dataset["processed_path"], preprocessing["missing_config"], dataset.get("name"))

            # --- Duplicates ---
            if "duplicates" in preprocessing:
                logger.info(f"Handling duplicates for {dataset['name']}")
                process_duplicates(dataset["processed_path"], dataset["processed_path"], preprocessing["duplicates"])

            # --- Correlation Matrix Generation ---
            logger.info(f"Generating correlation matrix for {dataset['name']}")
            if(dataset.get("name")=="NOAA_weather"):
                pkl_path=pkl_path=os.path.join(dataset["processed_path"], "after_missing_data.pkl")
            else:
                pkl_path=os.path.join(dataset["processed_path"], "after_duplicates.pkl")
            correlation_matrix(
                pkl_path=pkl_path,
                dataset_name=dataset["name"],
                method='pearson'
            )

        except Exception as e:
            logger.exception(f"FAILED: Processing of {dataset['name']} failed - {e}")

    logger.info("All datasets processed successfully.")
