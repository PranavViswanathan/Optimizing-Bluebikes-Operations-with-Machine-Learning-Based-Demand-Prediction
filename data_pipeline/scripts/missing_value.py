"""
A module for handling missing values in datasets with flexible options for
dropping rows or filling missing values with statistical measures.
"""

import os
import pickle
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

# Determine the absolute path of the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'raw_data.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'after_missing_values.pkl')

# Supported fill strategies
SUPPORTED_STRATEGIES = ['mean', 'median', 'mode', 'min', 'max', 'std', 'var', 'sum', 
                        'forward_fill', 'ffill', 'backward_fill', 'bfill', 'zero']


def handle_missing(
    input_pickle_path: str = INPUT_PICKLE_PATH,
    output_pickle_path: str = OUTPUT_PICKLE_PATH,
    drop_columns: Optional[List[str]] = None,
    fill_strategies: Optional[Dict[str, str]] = None,
    raise_on_remaining: bool = True
) -> str:
    """
    Load the DataFrame from the input pickle, handle missing values by either
    dropping rows or filling with specified strategies, then save to output pickle.
    
    :param input_pickle_path: Path to the input pickle file.
    :param output_pickle_path: Path to the output pickle file.
    :param drop_columns: List of column names to drop rows with missing values.
    :param fill_strategies: Dictionary mapping column names to fill strategies.
                           Example: {'Age': 'mean', 'Salary': 'median', 'Name': 'mode'}
                           Supported strategies: 'mean', 'median', 'mode', 'min', 'max',
                           'std', 'var', 'sum', 'forward_fill'/'ffill', 
                           'backward_fill'/'bfill', 'zero'
    :param raise_on_remaining: If True, raise ValueError if missing values remain.
                              If False, just print a warning.
    :return: Path to the saved pickle file.
    
    :raises FileNotFoundError: If input pickle file doesn't exist.
    :raises ValueError: If invalid strategies provided or columns don't exist.
    :raises ValueError: If missing values remain and raise_on_remaining is True.
    """
    # Validate inputs
    if drop_columns is None and fill_strategies is None:
        raise ValueError(
            "At least one of 'drop_columns' or 'fill_strategies' must be provided."
        )
    
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
    
    print(f"Initial missing values: {df.isna().sum().sum()}")
    print(f"Initial shape: {df.shape}")
    
    # Handle dropping rows with missing values
    if drop_columns:
        _validate_columns_exist(df, drop_columns, "drop_columns")
        
        initial_rows = len(df)
        df = df.dropna(subset=drop_columns)
        dropped_rows = initial_rows - len(df)
        
        print(f"Dropped {dropped_rows} rows with missing values in columns: {drop_columns}")
    
    # Handle filling missing values with strategies
    if fill_strategies:
        _validate_columns_exist(df, list(fill_strategies.keys()), "fill_strategies")
        _validate_strategies(fill_strategies)
        
        for column, strategy in fill_strategies.items():
            df = _fill_column(df, column, strategy)
    
    # Check if there are any missing values left
    remaining_missing = df.isna().sum().sum()
    
    if remaining_missing > 0:
        missing_by_column = df.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]
        
        message = (
            f"There are {remaining_missing} missing values left in the dataframe.\n"
            f"Missing values by column:\n{missing_by_column}"
        )
        
        if raise_on_remaining:
            print(message)
            raise ValueError(message)
        else:
            print(f"WARNING: {message}")
    
    print(f"Final missing values: {remaining_missing}")
    print(f"Final shape: {df.shape}")
    
    # Save the data to output pickle
    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)
    
    print(f"Data saved to {output_pickle_path}.")
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


def _validate_strategies(fill_strategies: Dict[str, str]) -> None:
    """
    Validate that all fill strategies are supported.
    
    :param fill_strategies: Dictionary of column to strategy mappings.
    :raises ValueError: If any strategy is not supported.
    """
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
    """
    Fill missing values in a column using the specified strategy.
    
    :param df: DataFrame to modify.
    :param column: Column name to fill.
    :param strategy: Fill strategy to use.
    :return: Modified DataFrame.
    :raises ValueError: If strategy cannot be applied to the column.
    """
    strategy = strategy.lower()
    missing_count = df[column].isna().sum()
    
    if missing_count == 0:
        print(f"Column '{column}' has no missing values. Skipping.")
        return df
    
    try:
        if strategy == 'mean':
            fill_value = df[column].mean()
            if pd.isna(fill_value):
                raise ValueError(f"Cannot compute mean for column '{column}' (all values are NaN)")
            df[column] = df[column].fillna(fill_value)
            print(f"Filled {missing_count} missing values in '{column}' with mean: {fill_value:.2f}")
        
        elif strategy == 'median':
            fill_value = df[column].median()
            if pd.isna(fill_value):
                raise ValueError(f"Cannot compute median for column '{column}' (all values are NaN)")
            df[column] = df[column].fillna(fill_value)
            print(f"Filled {missing_count} missing values in '{column}' with median: {fill_value:.2f}")
        
        elif strategy == 'mode':
            mode_values = df[column].mode()
            if len(mode_values) == 0:
                raise ValueError(f"Cannot compute mode for column '{column}' (no valid values)")
            fill_value = mode_values[0]
            df[column] = df[column].fillna(fill_value)
            print(f"Filled {missing_count} missing values in '{column}' with mode: {fill_value}")
        
        elif strategy == 'min':
            fill_value = df[column].min()
            if pd.isna(fill_value):
                raise ValueError(f"Cannot compute min for column '{column}' (all values are NaN)")
            df[column] = df[column].fillna(fill_value)
            print(f"Filled {missing_count} missing values in '{column}' with min: {fill_value:.2f}")
        
        elif strategy == 'max':
            fill_value = df[column].max()
            if pd.isna(fill_value):
                raise ValueError(f"Cannot compute max for column '{column}' (all values are NaN)")
            df[column] = df[column].fillna(fill_value)
            print(f"Filled {missing_count} missing values in '{column}' with max: {fill_value:.2f}")
        
        elif strategy == 'std':
            fill_value = df[column].std()
            if pd.isna(fill_value):
                raise ValueError(f"Cannot compute std for column '{column}' (insufficient data)")
            df[column] = df[column].fillna(fill_value)
            print(f"Filled {missing_count} missing values in '{column}' with std: {fill_value:.2f}")
        
        elif strategy == 'var':
            fill_value = df[column].var()
            if pd.isna(fill_value):
                raise ValueError(f"Cannot compute var for column '{column}' (insufficient data)")
            df[column] = df[column].fillna(fill_value)
            print(f"Filled {missing_count} missing values in '{column}' with var: {fill_value:.2f}")
        
        elif strategy == 'sum':
            fill_value = df[column].sum()
            df[column] = df[column].fillna(fill_value)
            print(f"Filled {missing_count} missing values in '{column}' with sum: {fill_value:.2f}")
        
        elif strategy in ['forward_fill', 'ffill']:
            df[column] = df[column].fillna(method='ffill')
            print(f"Filled {missing_count} missing values in '{column}' using forward fill")
        
        elif strategy in ['backward_fill', 'bfill']:
            df[column] = df[column].fillna(method='bfill')
            print(f"Filled {missing_count} missing values in '{column}' using backward fill")
        
        elif strategy == 'zero':
            df[column] = df[column].fillna(0)
            print(f"Filled {missing_count} missing values in '{column}' with zero")
    
    except TypeError as e:
        raise ValueError(
            f"Cannot apply strategy '{strategy}' to column '{column}'. "
            f"This strategy requires numeric data. Column dtype: {df[column].dtype}. "
            f"Error: {str(e)}"
        )
    except Exception as e:
        raise ValueError(
            f"Error applying strategy '{strategy}' to column '{column}': {str(e)}"
        )
    return df

if __name__=="__main__":
    print("*****************Before Filling Missing Values************************")
    df = pd.read_pickle("../data/processed/bluebikes/raw_data.pkl")
    print("Shape of data:", df.shape)       
    print("\nColumn names:\n", df.columns)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nPercentage of missing values:\n", (df.isnull().mean() * 100).round(2))
    print("\nDataFrame info:\n")
    df.info()
    print()
    print()
    print()

    handle_missing(input_pickle_path="../data/processed/colleges/raw_data.pkl", output_pickle_path="../data/processed/colleges/raw_data.pkl", drop_columns=["end_station_latitude", "end_station_longitude"], fill_strategies={"start_station_name": "mode", "end_station_name": "mode"}, raise_on_remaining=False)

    print()
    print()
    print()
    print("*****************After Filling Missing Values************************")
    df = pd.read_pickle("./data/processed/colleges/raw_data.pkl")
    print("Shape of data:", df.shape)       
    print("\nColumn names:\n", df.columns)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nPercentage of missing values:\n", (df.isnull().mean() * 100).round(2))
    print("\nDataFrame info:\n")
    df.info()
