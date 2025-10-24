import os
import pickle
import pytest
import pandas as pd
from data_pipeline.scripts.data_loader import load_single_file, load_folder, load_data

TEST_DIR = "data_pipeline/tests/tmp"
os.makedirs(TEST_DIR, exist_ok=True)

def test_load_single_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
    df = load_single_file(str(csv_path))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)

def test_load_folder(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    for i in range(2):
        pd.DataFrame({"x": [i], "y": [i + 1]}).to_csv(folder / f"f{i}.csv", index=False)
    df = load_folder(str(folder))
    assert df.shape == (2, 2)

def test_load_data_creates_pickle(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    csv_path = raw_dir / "file.csv"
    pd.DataFrame({"c": [1, 2]}).to_csv(csv_path, index=False)
    pkl_path = load_data(
        pickle_path=str(processed_dir),
        data_paths=[str(raw_dir)],
        dataset_name="testset"
    )
    assert os.path.exists(pkl_path)
    df = pickle.load(open(pkl_path, "rb"))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_unsupported_file_type(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("bad data")
    with pytest.raises(ValueError):
        load_single_file(str(bad_file))