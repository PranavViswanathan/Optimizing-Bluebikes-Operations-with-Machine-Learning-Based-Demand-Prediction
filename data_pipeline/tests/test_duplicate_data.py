import os
import pickle
import pytest
import pandas as pd
from data_pipeline.scripts import duplicate_data


def test_handle_duplicates_drop_all(tmp_path):
    """Test dropping duplicates using default 'first' keep mode."""
    df = pd.DataFrame({
        "ride_id": [1, 1, 2, 2, 3],
        "duration": [10, 10, 20, 20, 30],
        "station": ["A", "A", "B", "B", "C"]
    })
    input_pkl = tmp_path / "input.pkl"
    output_pkl = tmp_path / "output.pkl"
    df.to_pickle(input_pkl)


    result_path = duplicate_data.handle_duplicates(
        input_pickle_path=str(input_pkl),
        output_pickle_path=str(output_pkl),
        consider_all_columns=True,
        keep="first"
    )


    assert os.path.exists(result_path)
    cleaned_df = pickle.load(open(result_path, "rb"))
    assert isinstance(cleaned_df, pd.DataFrame)
    # Should have dropped exact duplicates (keep first)
    assert len(cleaned_df) == 3
    assert cleaned_df.duplicated().sum() == 0


def test_handle_duplicates_keep_last(tmp_path):
    """Test handling duplicates while keeping the last occurrence."""
    df = pd.DataFrame({
        "ride_id": [1, 1, 2, 2, 3],
        "duration": [10, 15, 20, 25, 30],
    })
    input_pkl = tmp_path / "input.pkl"
    output_pkl = tmp_path / "output.pkl"
    df.to_pickle(input_pkl)

    # Run with keep='last' and check duplicates by ride_id
    result_path = duplicate_data.handle_duplicates(
        input_pickle_path=str(input_pkl),
        output_pickle_path=str(output_pkl),
        consider_all_columns=False,
        subset=["ride_id"],
        keep="last"
    )

    cleaned_df = pickle.load(open(result_path, "rb"))
    assert len(cleaned_df) == 3
    assert cleaned_df.loc[cleaned_df["ride_id"] == 1, "duration"].item() == 15
    assert cleaned_df.loc[cleaned_df["ride_id"] == 2, "duration"].item() == 25


def test_handle_duplicates_no_duplicates(tmp_path):
    """Test case where no duplicates exist."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30]
    })
    input_pkl = tmp_path / "input.pkl"
    output_pkl = tmp_path / "output.pkl"
    df.to_pickle(input_pkl)

    result_path = duplicate_data.handle_duplicates(
        input_pickle_path=str(input_pkl),
        output_pickle_path=str(output_pkl)
    )

    cleaned_df = pickle.load(open(result_path, "rb"))
    assert len(cleaned_df) == 3
    assert cleaned_df.equals(df)


def test_handle_duplicates_invalid_keep(tmp_path):
    """Test invalid keep argument raises ValueError."""
    df = pd.DataFrame({
        "id": [1, 1, 2],
        "value": [10, 10, 20]
    })
    input_pkl = tmp_path / "input.pkl"
    output_pkl = tmp_path / "output.pkl"
    df.to_pickle(input_pkl)

    with pytest.raises(ValueError):
        duplicate_data.handle_duplicates(
            input_pickle_path=str(input_pkl),
            output_pickle_path=str(output_pkl),
            keep="invalid"
        )


def test_handle_duplicates_report_only(tmp_path):
    """Test report-only mode leaves data unchanged."""
    df = pd.DataFrame({
        "ride_id": [1, 1, 2],
        "duration": [10, 10, 20]
    })
    input_pkl = tmp_path / "input.pkl"
    output_pkl = tmp_path / "output.pkl"
    df.to_pickle(input_pkl)

    result_path = duplicate_data.handle_duplicates(
        input_pickle_path=str(input_pkl),
        output_pickle_path=str(output_pkl),
        report_only=True
    )

    cleaned_df = pickle.load(open(result_path, "rb"))
    # Should not drop any rows
    assert len(cleaned_df) == len(df)