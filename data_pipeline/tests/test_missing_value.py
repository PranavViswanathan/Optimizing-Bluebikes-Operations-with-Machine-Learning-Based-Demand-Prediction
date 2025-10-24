import os
import pickle
import pytest
import pandas as pd
from data_pipeline.scripts import missing_value


def test_handle_missing_values_drop(tmp_path):
    """Test dropping rows with missing values in specified columns."""
    df = pd.DataFrame({
        "duration": [10, None, 15],
        "start_station_name": ["A", None, "A"],
        "end_station_name": ["X", "Y", None],
    })
    input_pkl = tmp_path / "input.pkl"
    output_pkl = tmp_path / "output.pkl"
    df.to_pickle(input_pkl)

    # Run function
    result_path = missing_value.handle_missing(
        input_pickle_path=str(input_pkl),
        output_pickle_path=str(output_pkl),
        drop_columns=["duration"],
        raise_on_remaining=False
    )

    # Validate results
    assert os.path.exists(result_path)
    cleaned_df = pickle.load(open(result_path, "rb"))
    assert isinstance(cleaned_df, pd.DataFrame)
    # Row with missing duration should be dropped
    assert len(cleaned_df) == 2
    assert cleaned_df["duration"].isna().sum() == 0


def test_handle_missing_values_fill(tmp_path):
    """Test filling missing values using specified strategies."""
    df = pd.DataFrame({
        "duration": [10, None, 20],
        "start_station_name": ["A", None, "A"]
    })
    input_pkl = tmp_path / "input.pkl"
    output_pkl = tmp_path / "output.pkl"
    df.to_pickle(input_pkl)

    # Run function
    result_path = missing_value.handle_missing(
        input_pickle_path=str(input_pkl),
        output_pickle_path=str(output_pkl),
        fill_strategies={"duration": "mean", "start_station_name": "mode"},
        raise_on_remaining=False
    )

    # Validate results
    assert os.path.exists(result_path)
    cleaned_df = pickle.load(open(result_path, "rb"))
    assert isinstance(cleaned_df, pd.DataFrame)
    assert cleaned_df.isna().sum().sum() == 0
    # Mean of [10, 20] = 15 should replace missing duration
    assert cleaned_df["duration"].iloc[1] == 15
    # Mode of ['A', 'A'] = 'A' should replace missing start_station_name
    assert cleaned_df["start_station_name"].iloc[1] == "A"