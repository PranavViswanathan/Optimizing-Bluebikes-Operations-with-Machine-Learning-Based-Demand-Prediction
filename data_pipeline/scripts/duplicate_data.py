"""
A module for handling duplicate values in datasets with flexible options for
identifying and removing duplicates based on various strategies.
"""

import os
import pickle
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

# !IMPORTANT: Determine the absolute path of the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'colleges', 'raw_data.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'colleges' 'after_duplicates.pkl')

# Supported duplicate handling strategies
SUPPORTED_KEEP_OPTIONS = ['first', 'last', False]  # False means drop all duplicates
SUPPORTED_AGGREGATION_STRATEGIES = ['mean', 'median', 'mode', 'min', 'max', 'sum', 'count']


def handle_duplicates(
    input_pickle_path: str = INPUT_PICKLE_PATH,
    output_pickle_path: str = OUTPUT_PICKLE_PATH,
    subset: Optional[List[str]] = None,
    keep: Union[str, bool] = 'first',
    consider_all_columns: bool = True,
    aggregation_rules: Optional[Dict[str, str]] = None,
    report_only: bool = False,
    raise_on_remaining: bool = False
) -> str:
    """
    Load the DataFrame from the input pickle, handle duplicate values by either
    dropping rows or aggregating them based on specified strategies, then save to output pickle.
    
    :param input_pickle_path: Path to the input pickle file.
    :param output_pickle_path: Path to the output pickle file.
    :param subset: List of column names to consider when identifying duplicates.
                   If None and consider_all_columns is False, will detect key columns.
    :param keep: Determines which duplicates to keep:
                 - 'first': Keep first occurrence (default)
                 - 'last': Keep last occurrence
                 - False: Remove all duplicates
    :param consider_all_columns: If True, consider all columns for duplicate detection.
                                 If False, use subset or auto-detect key columns.
    :param aggregation_rules: Dictionary mapping column names to aggregation strategies
                             for handling duplicates. If provided, duplicates will be
                             aggregated instead of dropped.
                             Example: {'duration': 'mean', 'user_type': 'mode'}
    :param report_only: If True, only report duplicates without removing them.
    :param raise_on_remaining: If True, raise ValueError if duplicates remain after processing.
    :return: Path to the saved pickle file.
    
    :raises FileNotFoundError: If input pickle file doesn't exist.
    :raises ValueError: If invalid parameters provided or columns don't exist.
    """
    
    # Load DataFrame from input pickle
    if not os.path.exists(input_pickle_path):
        raise FileNotFoundError(
            f"No data found at the specified path: {input_pickle_path}"
        )
    
    with open(input_pickle_path, "rb") as file:
        df = pickle.load(file)
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"Expected a pandas DataFrame, but got {type(df).__name__}"
        )
    
    print(f"Initial shape: {df.shape}")
    print(f"Initial memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Determine columns to check for duplicates
    if consider_all_columns:
        check_columns = None  # Check all columns
        print("Checking for duplicates across all columns")
    else:
        if subset is not None:
            _validate_columns_exist(df, subset, "subset")
            check_columns = subset
            print(f"Checking for duplicates based on columns: {check_columns}")
        else:
            # Auto-detect potential key columns (non-numeric columns or ID columns)
            check_columns = _auto_detect_key_columns(df)
            print(f"Auto-detected key columns for duplicate check: {check_columns}")
    
    # Identify duplicates
    duplicates_mask = df.duplicated(subset=check_columns, keep=False)
    num_duplicate_rows = duplicates_mask.sum()
    
    if num_duplicate_rows == 0:
        print("No duplicate rows found!")
        _save_pickle(df, output_pickle_path)
        return output_pickle_path
    
    # Report duplicate statistics
    print(f"\nDuplicate Statistics:")
    print(f"Total duplicate rows: {num_duplicate_rows}")
    print(f"Percentage of duplicates: {(num_duplicate_rows / len(df)) * 100:.2f}%")
    
    if check_columns:
        duplicate_groups = df[duplicates_mask].groupby(check_columns).size()
        print(f"Number of duplicate groups: {len(duplicate_groups)}")
        print(f"Average duplicates per group: {duplicate_groups.mean():.2f}")
        print(f"Max duplicates in a group: {duplicate_groups.max()}")
    
    if report_only:
        print("\nReport only mode - no changes made to data")
        _save_pickle(df, output_pickle_path)
        return output_pickle_path
    
    df = _drop_duplicates(df, check_columns, keep)
    print(f"\nDropped duplicates with keep='{keep}'")

    # Check for remaining duplicates
    remaining_duplicates = df.duplicated(subset=check_columns, keep=False).sum()
    
    if remaining_duplicates > 0:
        message = f"There are {remaining_duplicates} duplicate rows remaining in the dataframe."
        
        if raise_on_remaining:
            print(message)
            raise ValueError(message)
        else:
            print(f"WARNING: {message}")
    
    print(f"\nFinal shape: {df.shape}")
    print(f"Rows removed: {len(df) - df.shape[0] if not aggregation_rules else num_duplicate_rows}")
    print(f"Final memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Save the data to output pickle
    _save_pickle(df, output_pickle_path)
    return output_pickle_path


def _validate_columns_exist(df: pd.DataFrame, columns: List[str], param_name: str) -> None:
    """
    Validate that all specified columns exist in the DataFrame.
    
    :param df: DataFrame to check.
    :param columns: List of column names to validate.
    :param param_name: Name of the parameter (for error message).
    :raises ValueError: If any column doesn't exist.
    """
    missing_columns = [col for col in columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"The following columns in '{param_name}' do not exist in the DataFrame: "
            f"{missing_columns}. Available columns: {list(df.columns)}"
        )


def _auto_detect_key_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect potential key columns for duplicate detection.
    Prioritizes ID columns, time columns, and categorical columns.
    Falls back to the first column if no suitable keys are found.
    
    :param df: DataFrame to analyze.
    :return: List of column names to use for duplicate detection.
    """
    key_columns = []
    
    # Check for ID columns
    id_keywords = ['id', 'key', 'code', 'identifier']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in id_keywords):
            key_columns.append(col)
    
    # Check for datetime columns
    time_keywords = ['time', 'date', 'timestamp', 'datetime']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in time_keywords):
            key_columns.append(col)
    
    # If no key columns found, use categorical columns with reasonable cardinality
    if not key_columns:
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                unique_ratio = df[col].nunique() / len(df)
                # Use columns with 0.01% to 50% unique values
                if 0.0001 < unique_ratio < 0.5:
                    key_columns.append(col)
    
    # If still no columns, use all non-float columns
    if not key_columns:
        for col in df.columns:
            if df[col].dtype not in ['float32', 'float64']:
                key_columns.append(col)
    
    # ✅ Fallback: if still empty, use the first column
    if not key_columns and len(df.columns) > 0:
        key_columns = [df.columns[0]]
    
    return key_columns[:10]  # Limit to 10 columns max


def _drop_duplicates(df: pd.DataFrame, subset: Optional[List[str]], keep: Union[str, bool]) -> pd.DataFrame:
    """
    Drop duplicate rows based on the specified strategy.
    
    :param df: DataFrame to modify.
    :param subset: Columns to consider for duplicate detection.
    :param keep: Which duplicates to keep ('first', 'last', or False).
    :return: DataFrame with duplicates removed.
    """
    if keep not in SUPPORTED_KEEP_OPTIONS:
        raise ValueError(
            f"Invalid 'keep' value: {keep}. "
            f"Supported options: {SUPPORTED_KEEP_OPTIONS}"
        )
    
    initial_rows = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    dropped_rows = initial_rows - len(df)
    
    print(f"Dropped {dropped_rows} duplicate rows")
    
    return df


def _save_pickle(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to pickle file.
    
    :param df: DataFrame to save.
    :param output_path: Path to save the pickle file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(df, file)
    print(f"Data saved to {output_path}.")


if __name__ == "__main__":
    print("*****************Before Handling Duplicates************************")
    df = pd.read_pickle("../data/processed/bluebikes/raw_data.pkl")
    print("Shape of data:", df.shape)
    print("\nColumn names:\n", df.columns)
    
    # Check for duplicates
    print("\nChecking for exact duplicates (all columns):")
    exact_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_duplicates}")
    
    # Check for duplicates based on key columns
    # key_columns = ['start_time', 'stop_time', 'start_station_id', 'end_station_id', 'bike_id']
    # print(f"\nChecking for duplicates based on {key_columns}:")
    # key_duplicates = df.duplicated(subset=key_columns).sum()
    # print(f"Duplicate rows based on key columns: {key_duplicates}")
    
    print("\nDataFrame info:")
    df.info()
    print()
    print()
    print()
    
    # Handle duplicates
    handle_duplicates(
        input_pickle_path="../data/processed/weather/raw_data.pkl",
        output_pickle_path="../data/processed/weather/after_duplicates.pkl",
        subset=None,
        keep='first',
        consider_all_columns=False,
        raise_on_remaining=False
    )
    
    print()
    print()
    print()
    print("*****************After Handling Duplicates************************")
    df = pd.read_pickle("../data/processed/weather/after_duplicates.pkl")
    print("Shape of data:", df.shape)
    print("\nColumn names:\n", df.columns)
    
    # Check for remaining duplicates
    print("\nChecking for remaining exact duplicates:")
    exact_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_duplicates}")
    
    print("\nDataFrame info:")
    df.info()
