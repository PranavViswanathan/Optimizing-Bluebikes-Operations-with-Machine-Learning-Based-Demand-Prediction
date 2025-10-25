# data_loader.py
import os
import sys
from pathlib import Path
import pickle
import pandas as pd
from typing import List, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent 
PROJECT_ROOT = Path(__file__).resolve().parents[1] 

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
logger = get_logger("data_loader")

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_FOLDER_PATH = os.path.join(PROJECT_DIR, 'data', 'processed')
DEFAULT_DATA_PATHS = [
    os.path.join(PROJECT_DIR, 'data', 'raw', 'bluebikes'),
    os.path.join(PROJECT_DIR, 'data', 'raw', 'Boston_GIS'),
    os.path.join(PROJECT_DIR, 'data', 'raw', 'NOAA')
]

SUPPORTED_EXTENSIONS = ['.csv', '.parquet', '.xlsx', '.xls']


def load_single_file(file_path: str) -> pd.DataFrame:
    """Load a single data file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_folder(folder_path: str) -> pd.DataFrame:
    """Load all files of the same type from a folder and concatenate into a single DataFrame."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])
    
    if not files:
        raise FileNotFoundError(f"No supported files found in folder: {folder_path}")

    file_exts = {os.path.splitext(f)[1].lower() for f in files}
    if len(file_exts) != 1:
        raise ValueError(f"All files in the folder must have the same extension. Found: {file_exts}")

    logger.info(f"Loading {len(files)} files from folder: {folder_path}")
    df_list = []
    for f in files:
        try:
            df_list.append(load_single_file(f))
        except Exception as e:
            logger.error(f"Failed to load file {f}: {e}")
    return pd.concat(df_list, ignore_index=True)


def load_data(pickle_path: Optional[str] = None,
              data_paths: Optional[List[str]] = None,
              dataset_name: str = "bluebikes") -> str:
    """Load data from pickle if available, else from files/folders, and save as pickle."""
    if pickle_path is None:
        pickle_path = os.path.join(PROCESSED_FOLDER_PATH, dataset_name, "raw_data.pkl")
    else:
        pickle_path = os.path.join(PROCESSED_FOLDER_PATH, dataset_name, "raw_data.pkl")

    if os.path.exists(pickle_path):
        logger.info(f"Loading data from pickle: {pickle_path}")
        with open(pickle_path, "rb") as f:
            df = pickle.load(f)
    else:
        if data_paths is None:
            data_paths = DEFAULT_DATA_PATHS

        df_list = []
        for path in data_paths:
            if not os.path.exists(path):
                logger.warning(f"Path does not exist: {path}")
                continue
            try:
                if os.path.isfile(path):
                    df_list.append(load_single_file(path))
                elif os.path.isdir(path):
                    df_list.append(load_folder(path))
                else:
                    logger.warning(f"Unsupported path type: {path}")
            except Exception as e:
                logger.error(f"Failed to load data from {path}: {e}")

        if not df_list:
            raise FileNotFoundError(f"No data found in paths: {data_paths}")

        df = pd.concat(df_list, ignore_index=True)

        # Save to pickle
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(df, f)
        logger.info(f"Data saved to pickle: {pickle_path}")

    return pickle_path