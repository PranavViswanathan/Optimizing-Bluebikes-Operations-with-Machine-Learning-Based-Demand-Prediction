# assign_station_ids.py

"""
A module to assign numeric IDs to start and end stations in Bluebikes datasets
based on the distinct station names. It updates the 'start_station_id' and
'end_station_id' columns in the pickle file.
"""

import os
import sys
from pathlib import Path
import pickle
from typing import Optional
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from logger import get_logger
logger = get_logger("assign_station_ids")

# ---------------------- DEFAULT PATHS ----------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'bluebikes', 'raw_data.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'bluebikes', 'raw_data_with_ids.pkl')


def assign_station_ids(
    input_pickle_path: str = INPUT_PICKLE_PATH,
    output_pickle_path: Optional[str] = OUTPUT_PICKLE_PATH
) -> pd.DataFrame:
    """
    Assign numeric IDs to start and end stations based on distinct station names.

    Parameters
    ----------
    input_pickle_path : str
        Path to the pickle file containing the DataFrame.
    output_pickle_path : str, optional
        Path to save the updated DataFrame. If None, overwrites the input file.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with 'start_station_id' and 'end_station_id'.
    """

    if not os.path.exists(input_pickle_path):
        logger.error(f"Pickle file not found: {input_pickle_path}")
        raise FileNotFoundError(f"Pickle file not found: {input_pickle_path}")

    # --- Load DataFrame ---
    with open(input_pickle_path, "rb") as file:
        df = pickle.load(file)

    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected pandas DataFrame but got {type(df).__name__}")
        raise ValueError(f"Expected pandas DataFrame but got {type(df).__name__}")

    logger.info(f"Loaded DataFrame with shape {df.shape}")
    
    required_columns = ['start_station_id', 'end_station_id', 'start_station_name', 'end_station_name']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # --- Combine distinct station names ---
    all_stations = pd.concat([df['start_station_name'], df['end_station_name']]).dropna().unique()
    station_to_id = {name: idx+1 for idx, name in enumerate(all_stations)}
    logger.info(f"Found {len(station_to_id)} distinct stations")

    # --- Map IDs to the ID columns ---
    df['start_station_id'] = df['start_station_name'].map(station_to_id)
    df['end_station_id'] = df['end_station_name'].map(station_to_id)
    logger.info("Assigned numeric IDs to start and end stations")

    # --- Save updated DataFrame ---
    if output_pickle_path is None:
        output_pickle_path = input_pickle_path

    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)

    logger.info(f"Updated DataFrame saved to: {output_pickle_path}")
    return df


if __name__ == "__main__":
    logger.info("********** Assigning Station IDs **********")
    try:
        df_updated = assign_station_ids()
        logger.info(f"Station IDs assigned successfully. DataFrame shape: {df_updated.shape}")
        logger.info(f"Sample rows:\n{df_updated.head()}")
    except Exception as e:
        logger.error(f"Error during station ID assignment: {e}", exc_info=True)
