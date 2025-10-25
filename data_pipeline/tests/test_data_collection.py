import sys
import types
import pytest
from unittest.mock import patch, MagicMock

# ----------------------------------------------------------------------
# Mock external modules so pytest can import data_collection.py safely
# ----------------------------------------------------------------------
fake_download_data = types.ModuleType("download_data")
fake_download_data.find_zip_links = MagicMock()
fake_download_data.download_zips = MagicMock()

fake_read_zips = types.ModuleType("read_zips")
fake_read_zips.save_year_to_parquet = MagicMock()

fake_boston_colleges = types.ModuleType("BostonColleges")
fake_boston_colleges.BostonCollegesAPI = MagicMock()

fake_noaa = types.ModuleType("NOAA_DataAcq")
fake_noaa.NOAA = MagicMock()

sys.modules["bluebikes_data_helpers"] = types.ModuleType("bluebikes_data_helpers")
sys.modules["bluebikes_data_helpers.download_data"] = fake_download_data
sys.modules["bluebikes_data_helpers.read_zips"] = fake_read_zips
sys.modules["school_noaa_data_collectors"] = types.ModuleType("school_noaa_data_collectors")
sys.modules["school_noaa_data_collectors.BostonColleges"] = fake_boston_colleges
sys.modules["school_noaa_data_collectors.NOAA_DataAcq"] = fake_noaa

# ----------------------------------------------------------------------
# Now safe to import the real module
# ----------------------------------------------------------------------
from data_pipeline.scripts.data_collection import (
    collect_bluebikes_data,
    collect_boston_college_data,
    collect_NOAA_Weather_data,
)

# ----------------------------------------------------------------------
# BlueBikes
# ----------------------------------------------------------------------
@patch("data_pipeline.scripts.data_collection.find_zip_links")
@patch("data_pipeline.scripts.data_collection.download_zips")
@patch("data_pipeline.scripts.data_collection.save_year_to_parquet")
def test_collect_bluebikes_data(mock_save, mock_download, mock_find, tmp_path):
    mock_find.return_value = ["url1.zip"]
    mock_download.return_value = ["file1.zip"]
    mock_save.return_value = tmp_path / "output.parquet"

    collect_bluebikes_data(
        index_url="https://fake.url",
        years=["2024"],
        download_dir=str(tmp_path / "downloads"),
        parquet_dir=str(tmp_path / "parquet"),
        log_path=str(tmp_path / "log.csv"),
    )

    mock_find.assert_called_once()
    mock_download.assert_called_once()
    mock_save.assert_called_once()

# ----------------------------------------------------------------------
# Boston Colleges
# ----------------------------------------------------------------------
@patch("data_pipeline.scripts.data_collection.BostonCollegesAPI")
def test_collect_boston_college_data(mock_api):
    instance = MagicMock()
    mock_api.return_value = instance
    collect_boston_college_data()
    instance.save_to_csv.assert_called_once()

# ----------------------------------------------------------------------
# NOAA Weather
# ----------------------------------------------------------------------
@patch("data_pipeline.scripts.data_collection.NOAA")
def test_collect_NOAA_Weather_data(mock_noaa):
    instance = MagicMock()
    mock_noaa.return_value = instance
    collect_NOAA_Weather_data()
    instance.fetch_training_data_from_api.assert_called_once()
    instance.get_weather_dataframe.assert_called_once()