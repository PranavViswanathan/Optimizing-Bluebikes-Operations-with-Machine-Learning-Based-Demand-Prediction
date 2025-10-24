# duplicate_data.py

"""
A module for handling duplicate values in datasets with flexible options for
identifying and removing duplicates based on various strategies.
"""

import os
import pickle
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from logger import get_logger
logger = get_logger("duplicate_data")

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
    
    logger.info(f"Initial shape: {df.shape}")
    logger.info(f"Initial memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Determine columns to check for duplicates
    if consider_all_columns:
        check_columns = None  # Check all columns
        logger.info("Checking for duplicates across all columns")
    else:
        if subset is not None:
            _validate_columns_exist(df, subset, "subset")
            check_columns = subset
            logger.info(f"Checking for duplicates based on columns: {check_columns}")
        else:
            # Auto-detect potential key columns (non-numeric columns or ID columns)
            check_columns = _auto_detect_key_columns(df)
            logger.info(f"Auto-detected key columns for duplicate check: {check_columns}")
    
    # Identify duplicates
    duplicates_mask = df.duplicated(subset=check_columns, keep=False)
    num_duplicate_rows = duplicates_mask.sum()
    
    if num_duplicate_rows == 0:
        logger.info("No duplicate rows found!")
        _save_pickle(df, output_pickle_path)
        return output_pickle_path
    
    # Report duplicate statistics
    logger.info(f"\nDuplicate Statistics:")
    logger.info(f"Total duplicate rows: {num_duplicate_rows}")
    logger.info(f"Percentage of duplicates: {(num_duplicate_rows / len(df)) * 100:.2f}%")
    
    if check_columns:
        duplicate_groups = df[duplicates_mask].groupby(check_columns).size()
        logger.info(f"Number of duplicate groups: {len(duplicate_groups)}")
        logger.info(f"Average duplicates per group: {duplicate_groups.mean():.2f}")
        logger.info(f"Max duplicates in a group: {duplicate_groups.max()}")
    
    if report_only:
        logger.info("\nReport only mode - no changes made to data")
        _save_pickle(df, output_pickle_path)
        return output_pickle_path
    
    df = _drop_duplicates(df, check_columns, keep)
    logger.info(f"\nDropped duplicates with keep='{keep}'")

    # Check for remaining duplicates
    remaining_duplicates = df.duplicated(subset=check_columns, keep=False).sum()
    
    if remaining_duplicates > 0:
        message = f"There are {remaining_duplicates} duplicate rows remaining in the dataframe."
        
        if raise_on_remaining:
            logger.error(message)
            raise ValueError(message)
        else:
            logger.warning(message)
    
    logger.info(f"\nFinal shape: {df.shape}")
    logger.info(f"Rows removed: {len(df) - df.shape[0] if not aggregation_rules else num_duplicate_rows}")
    logger.info(f"Final memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Save the data to output pickle
    _save_pickle(df, output_pickle_path)
    return output_pickle_path


def _validate_columns_exist(df: pd.DataFrame, columns: List[str], param_name: str) -> None:
    missing_columns = [col for col in columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"The following columns in '{param_name}' do not exist in the DataFrame: "
            f"{missing_columns}. Available columns: {list(df.columns)}"
        )


def _auto_detect_key_columns(df: pd.DataFrame) -> List[str]:
    key_columns = []
    
    id_keywords = ['id', 'key', 'code', 'identifier']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in id_keywords):
            key_columns.append(col)
    
    time_keywords = ['time', 'date', 'timestamp', 'datetime']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in time_keywords):
            key_columns.append(col)
    
    if not key_columns:
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                unique_ratio = df[col].nunique() / len(df)
                if 0.0001 < unique_ratio < 0.5:
                    key_columns.append(col)
    
    if not key_columns:
        for col in df.columns:
            if df[col].dtype not in ['float32', 'float64']:
                key_columns.append(col)
    
    if not key_columns and len(df.columns) > 0:
        key_columns = [df.columns[0]]
    
    return key_columns[:10]


def _drop_duplicates(df: pd.DataFrame, subset: Optional[List[str]], keep: Union[str, bool]) -> pd.DataFrame:
    if keep not in SUPPORTED_KEEP_OPTIONS:
        raise ValueError(
            f"Invalid 'keep' value: {keep}. "
            f"Supported options: {SUPPORTED_KEEP_OPTIONS}"
        )
    
    initial_rows = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    dropped_rows = initial_rows - len(df)
    
    logger.info(f"Dropped {dropped_rows} duplicate rows")
    
    return df


def _save_pickle(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(df, file)
    logger.info(f"Data saved to {output_path}.")


if __name__ == "__main__":
    logger.info("*****************Before Handling Duplicates************************")
    df = pd.read_pickle("../data/processed/bluebikes/raw_data.pkl")
    logger.info(f"Shape of data: {df.shape}")
    logger.info(f"Column names: {df.columns}")
    
    logger.info("Checking for exact duplicates (all columns):")
    exact_duplicates = df.duplicated().sum()
    logger.info(f"Exact duplicate rows: {exact_duplicates}")
    
    logger.info("DataFrame info:")
    df.info()
    logger.info("")
    
    handle_duplicates(
        input_pickle_path="../data/processed/weather/raw_data.pkl",
        output_pickle_path="../data/processed/weather/after_duplicates.pkl",
        subset=None,
        keep='first',
        consider_all_columns=False,
        raise_on_remaining=False
    )
    
    logger.info("*****************After Handling Duplicates************************")
    df = pd.read_pickle("../data/processed/weather/after_duplicates.pkl")
    logger.info(f"Shape of data: {df.shape}")
    logger.info(f"Column names: {df.columns}")
    
    logger.info("Checking for remaining exact duplicates:")
    exact_duplicates = df.duplicated().sum()
    logger.info(f"Exact duplicate rows: {exact_duplicates}")
    
    logger.info("DataFrame info:")
    df.info()