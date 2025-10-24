# data_loader.py

import os
import pickle
import pandas as pd
from typing import List, Optional

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
    """
    Load a single data file based on its extension.
    """
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
    """
    Load all files of the same type from a folder and concatenate into a single DataFrame.
    Assumes all files have the same columns and extension.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])
    
    if not files:
        raise FileNotFoundError(f"No supported files found in folder: {folder_path}")

    # Check that all files have the same extension
    file_exts = {os.path.splitext(f)[1].lower() for f in files}
    if len(file_exts) != 1:
        raise ValueError(f"All files in the folder must have the same extension. Found: {file_exts}")

    print(f"Loading {len(files)} files from folder {folder_path}")
    df_list = [load_single_file(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

def load_data(pickle_path: Optional[str] = None,
              data_paths: Optional[List[str]] = None,
              dataset_name: str = "bluebikes") -> str:
    """
    Load data from pickle if available, else from files/folders, and save as pickle.
    Returns the pickle file path.
    """
    if pickle_path is None:
        pickle_path = os.path.join(PROCESSED_FOLDER_PATH, dataset_name, "raw_data.pkl")
    else:
        pickle_path = os.path.join(PROCESSED_FOLDER_PATH, dataset_name, "raw_data.pkl")

    if os.path.exists(pickle_path):
        print(f"Loading data from pickle: {pickle_path}")
        with open(pickle_path, "rb") as f:
            df = pickle.load(f)
    else:
        if data_paths is None:
            data_paths = DEFAULT_DATA_PATHS

        df_list = []
        for path in data_paths:
            if not os.path.exists(path):
                print(f"Path does not exist: {path}")
                continue
            if os.path.isfile(path):
                df_list.append(load_single_file(path))
            elif os.path.isdir(path):
                df_list.append(load_folder(path))
            else:
                print(f"Unsupported path type: {path}")

        if not df_list:
            raise FileNotFoundError(f"No data found in paths: {data_paths}")

        df = pd.concat(df_list, ignore_index=True)

        # Save to pickle
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(df, f)
        print(f"Data saved to pickle: {pickle_path}")

    return pickle_path