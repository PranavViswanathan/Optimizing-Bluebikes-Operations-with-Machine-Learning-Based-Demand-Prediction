"""
Module to handle the loading of dataset from multiple file formats.
Supports pickle, CSV, Parquet, and Excel file formats.
"""
import pickle
import os
import pandas as pd

# Determine the absolute path of the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use the project directory to construct paths to other directories
PROCESSED_FOLDER_PATH = os.path.join(PROJECT_DIR, 'data',
                                   'processed')

# !IMPORTANT: Need to fix data paths for all datasets while integrating the gathering code
DEFAULT_DATA_PATHS = [
    os.path.join(PROJECT_DIR, 'data', 'raw', 'bluebikes'),
    os.path.join(PROJECT_DIR, 'data', 'raw', 'Boston_GIS'),
    os.path.join(PROJECT_DIR, 'data', 'raw', 'NOAA')
]


def load_data(pickle_path=None, data_paths=None, dataset_name="bluebikes"):
    """
    Load the dataset.
    
    First, try to load from the pickle file. If it doesn't exist, load from the 
    first available data file (CSV, Parquet, or Excel).
    Regardless of the source, save the loaded data as a pickle for future use and
    return the path to that pickle.
    
    :param pickle_path: Path to the pickle file.
    :param data_paths: List of paths to check for data files (CSV, Parquet, Excel).
                       If None, uses DEFAULT_DATA_PATHS.
    :return: Path to the saved pickle file.
    """
    if pickle_path is None:
        pickle_path=os.path.join(PROCESSED_FOLDER_PATH, "bluebikes", "raw_data.pkl")
    else:
        pickle_path=os.path.join(PROCESSED_FOLDER_PATH, dataset_name, "raw_data.pkl")

    if data_paths is None:
        data_paths = DEFAULT_DATA_PATHS
    
    # Placeholder for the DataFrame
    df = None
    
    # Check if pickle file exists
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            df = pickle.load(file)
        print(f"Data loaded successfully from {pickle_path}.")
    else:
        # Try to load from available data files
        for data_path in data_paths:
            if os.path.exists(data_path):
                file_ext = os.path.splitext(data_path)[1].lower()
                
                try:
                    if file_ext == '.csv':
                        df = pd.read_csv(data_path)
                    elif file_ext == '.parquet':
                        df = pd.read_parquet(data_path)
                    elif file_ext in ['.xlsx', '.xls']:
                        df = pd.read_excel(data_path)
                    else:
                        print(f"Unsupported file format: {file_ext}")
                        continue
                    
                    print(f"Data loaded from {data_path}.")
                    break
                except Exception as e:
                    print(f"Error loading {data_path}: {str(e)}")
                    continue
        
        # If no data was loaded, raise an error
        if df is None:
            error_message = f"No data found in the specified paths: {pickle_path} or {data_paths}"
            print(error_message)
            raise FileNotFoundError(error_message)
    
    # Save the data to pickle for future use (or re-save it if loaded from existing pickle)
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, "wb") as file:
        pickle.dump(df, file)
    print(f"Data saved to {pickle_path} for future use.")
    
    return pickle_path

if __name__=="__main__":
    load_data(pickle_path="weather", data_paths=["D:\MLOps_Coursework\ML-OPs\data\\raw\\NOAA\\boston_daily_weather.csv"], dataset_name="weather")