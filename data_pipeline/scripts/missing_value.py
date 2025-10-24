# missing_value.py

"""
A module for handling missing values in datasets with flexible options for
dropping rows or filling missing values with statistical measures.
"""

import os
import pickle
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from data_pipeline.scripts.logger import get_logger


logger = get_logger("missing_value")

# Determine the absolute path of the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'raw_data.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'after_missing_values.pkl')

# Supported fill strategies
SUPPORTED_STRATEGIES = [
    'mean', 'median', 'mode', 'min', 'max', 'std', 'var', 'sum',
    'forward_fill', 'ffill', 'backward_fill', 'bfill', 'zero'
]


def handle_missing(
    input_pickle_path: str = INPUT_PICKLE_PATH,
    output_pickle_path: str = OUTPUT_PICKLE_PATH,
    drop_columns: Optional[List[str]] = None,
    fill_strategies: Optional[Dict[str, str]] = None,
    raise_on_remaining: bool = True
) -> str:
    """Load DataFrame, handle missing values, and save the cleaned result."""
    if drop_columns is None and fill_strategies is None:
        raise ValueError("At least one of 'drop_columns' or 'fill_strategies' must be provided.")
    
    if not os.path.exists(input_pickle_path):
        raise FileNotFoundError(f"No data found at the specified path: {input_pickle_path}")
    
    with open(input_pickle_path, "rb") as file:
        df = pickle.load(file)
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected a pandas DataFrame, but got {type(df).__name__}")
    
    logger.info(f"Initial missing values: {df.isna().sum().sum()} | Shape: {df.shape}")
    
    if drop_columns:
        _validate_columns_exist(df, drop_columns, "drop_columns")
        initial_rows = len(df)
        df = df.dropna(subset=drop_columns)
        dropped_rows = initial_rows - len(df)
        logger.info(f"Dropped {dropped_rows} rows with missing values in columns: {drop_columns}")
    
    if fill_strategies:
        _validate_columns_exist(df, list(fill_strategies.keys()), "fill_strategies")
        _validate_strategies(fill_strategies)
        for column, strategy in fill_strategies.items():
            df = _fill_column(df, column, strategy)
    
    remaining_missing = df.isna().sum().sum()
    if remaining_missing > 0:
        missing_by_column = df.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]
        message = f"Remaining missing values: {remaining_missing} | Columns: {list(missing_by_column.index)}"
        if raise_on_remaining:
            logger.error(message)
            raise ValueError(message)
        else:
            logger.warning(message)
    
    logger.info(f"Final missing values: {remaining_missing} | Final shape: {df.shape}")
    
    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)
    
    logger.info(f"Data saved to {output_pickle_path}")
    return output_pickle_path


def _validate_columns_exist(df: pd.DataFrame, columns: List[str], param_name: str) -> None:
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The following columns in '{param_name}' do not exist in the DataFrame: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def _validate_strategies(fill_strategies: Dict[str, str]) -> None:
    invalid_strategies = {}
    for column, strategy in fill_strategies.items():
        if strategy.lower() not in SUPPORTED_STRATEGIES:
            invalid_strategies[column] = strategy
    if invalid_strategies:
        raise ValueError(
            f"Invalid fill strategies found: {invalid_strategies}. "
            f"Supported strategies: {SUPPORTED_STRATEGIES}"
        )


def _fill_column(df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
    strategy = strategy.lower()
    missing_count = df[column].isna().sum()
    
    if missing_count == 0:
        logger.info(f"Column '{column}' has no missing values. Skipping.")
        return df
    
    try:
        if strategy == 'mean':
            fill_value = df[column].mean()
            df[column] = df[column].fillna(fill_value)
            logger.info(f"Filled {missing_count} missing in '{column}' with mean={fill_value:.2f}")
        elif strategy == 'median':
            fill_value = df[column].median()
            df[column] = df[column].fillna(fill_value)
            logger.info(f"Filled {missing_count} missing in '{column}' with median={fill_value:.2f}")
        elif strategy == 'mode':
            mode_values = df[column].mode()
            if len(mode_values) == 0:
                raise ValueError(f"Cannot compute mode for column '{column}' (no valid values)")
            fill_value = mode_values[0]
            df[column] = df[column].fillna(fill_value)
            logger.info(f"Filled {missing_count} missing in '{column}' with mode={fill_value}")
        elif strategy == 'min':
            fill_value = df[column].min()
            df[column] = df[column].fillna(fill_value)
            logger.info(f"Filled {missing_count} missing in '{column}' with min={fill_value:.2f}")
        elif strategy == 'max':
            fill_value = df[column].max()
            df[column] = df[column].fillna(fill_value)
            logger.info(f"Filled {missing_count} missing in '{column}' with max={fill_value:.2f}")
        elif strategy == 'std':
            fill_value = df[column].std()
            df[column] = df[column].fillna(fill_value)
            logger.info(f"Filled {missing_count} missing in '{column}' with std={fill_value:.2f}")
        elif strategy == 'var':
            fill_value = df[column].var()
            df[column] = df[column].fillna(fill_value)
            logger.info(f"Filled {missing_count} missing in '{column}' with var={fill_value:.2f}")
        elif strategy == 'sum':
            fill_value = df[column].sum()
            df[column] = df[column].fillna(fill_value)
            logger.info(f"Filled {missing_count} missing in '{column}' with sum={fill_value:.2f}")
        elif strategy in ['forward_fill', 'ffill']:
            df[column] = df[column].fillna(method='ffill')
            logger.info(f"Filled {missing_count} missing in '{column}' using forward fill")
        elif strategy in ['backward_fill', 'bfill']:
            df[column] = df[column].fillna(method='bfill')
            logger.info(f"Filled {missing_count} missing in '{column}' using backward fill")
        elif strategy == 'zero':
            df[column] = df[column].fillna(0)
            logger.info(f"Filled {missing_count} missing in '{column}' with zero")
    except Exception as e:
        logger.error(f"Error applying strategy '{strategy}' to column '{column}': {e}", exc_info=True)
        raise
    return df


if __name__ == "__main__":
    logger.info("***************** Before Filling Missing Values ************************")
    try:
        df = pd.read_pickle("../data/processed/bluebikes/raw_data.pkl")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Missing per column:\n{df.isnull().sum()}")
        logger.info(f"Percentage missing (%):\n{(df.isnull().mean() * 100).round(2)}")

        handle_missing(
            input_pickle_path="../data/processed/colleges/raw_data.pkl",
            output_pickle_path="../data/processed/colleges/raw_data.pkl",
            drop_columns=["end_station_latitude", "end_station_longitude"],
            fill_strategies={"start_station_name": "mode", "end_station_name": "mode"},
            raise_on_remaining=False
        )

        logger.info("***************** After Filling Missing Values ************************")
        df = pd.read_pickle("../data/processed/colleges/raw_data.pkl")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Missing per column:\n{df.isnull().sum()}")
        logger.info(f"Percentage missing (%):\n{(df.isnull().mean() * 100).round(2)}")
    except Exception as e:
        logger.error(f"Error during manual test: {e}", exc_info=True)