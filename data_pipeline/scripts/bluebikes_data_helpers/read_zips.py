# read_zip_year_to_parquet.py

import os
import re
from pathlib import Path
from zipfile import ZipFile
from typing import Union, List
import pandas as pd
from bluebikes_data_helpers.normalize import (
    _rename_and_coalesce,
    DEFAULT_MAPPING,
    _normalized_mapping,
    _coerce_for_parquet,
)
from bluebikes_data_helpers.record_file import log_file_status

NORM_MAP = _normalized_mapping(DEFAULT_MAPPING)


def read_one_zip_to_df(zip_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read ALL CSV files inside a ZIP and return a single DataFrame.
    - Try UTF-8 with BOM first, then latin1
    - Clean header junk (BOM/mojibake), drop Unnamed/empty
    - Drop all-null junk columns
    - Rename+coalesce BEFORE concatenation
    """
    zip_path = Path(zip_path)
    frames: List[pd.DataFrame] = []

    with ZipFile(zip_path) as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        for name in csv_members:
            try:
                with zf.open(name) as f:
                    df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
            except UnicodeDecodeError:
                with zf.open(name) as f:
                    df = pd.read_csv(f, encoding="latin1", low_memory=False)

            # strong header cleanup (handles ëÀ¼ï, ï»¿, etc.)
            import re, unicodedata
            def _clean_cols(cols):
                out = []
                for c in cols:
                    s = str(c)
                    s = unicodedata.normalize("NFKD", s)        # decompose odd unicode
                    s = s.replace("\ufeff", "")                 # strip BOM if present
                    s = s.encode("ascii", "ignore").decode()    # drop non-ASCII junk
                    s = re.sub(r"\s+", " ", s).strip()          # normalize spaces
                    out.append(s)
                return out

            df.columns = _clean_cols(df.columns)

            
            keep = (df.columns != "") & (~df.columns.str.match(r"^Unnamed", na=False))
            df = df.loc[:, keep]
            df = df.loc[:, df.notna().any(axis=0)] 

            # map + coalesce to canonical schema
            df = _rename_and_coalesce(df, NORM_MAP)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.loc[:, combined.notna().any(axis=0)] 
    return combined


def build_year_df_from_zips(
    zip_dir: Union[str, Path], year: str, log_path: Union[str, Path] = "read_log.csv"
) -> pd.DataFrame:
    """
    Find all ZIPs in `zip_dir` whose filenames contain `year` (e.g., '2015'),
    read & align their CSVs, and return a single combined DataFrame.
    """
    zip_dir = Path(zip_dir)
    zip_paths = sorted(p for p in zip_dir.glob("*.zip") if year in p.name)

    frames: List[pd.DataFrame] = []
    for zp in zip_paths:
        print(f"Reading: {zp.name}")
        try:
            df = read_one_zip_to_df(zp)
            frames.append(df)
            log_file_status(log_path, zp.name, True)
        except Exception as e:
            print(f"Failed to read {zp.name}: {e}")
            log_file_status(log_path, zp.name, False)

    if not frames:
        print(f"No ZIP files found matching year '{year}' in {zip_dir}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Combined shape for {year}: {combined.shape}")
    return combined


def save_year_to_parquet(
    zip_dir: Union[str, Path],
    year: str,
    out_dir: Union[str, Path] = "parquet",
    log_path: Union[str, Path] = "read_log.csv",
) -> str:
    """
    Read all ZIPs for the 'year' from 'zip_dir', and combine into a single Parquet file saved to an 'out_dir'.
    And then delete the ZIPs after successful processing.
    """
    df = build_year_df_from_zips(zip_dir, year, log_path)

    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = out_dir / f"trips_{year}.parquet"
    df = _coerce_for_parquet(df)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"Saved: {out_path} ({len(df):,} rows)")

    deleted = 0
    for zp in Path(zip_dir).glob(f"*{year}*.zip"):
        try:
            zp.unlink()  
            deleted += 1
        except OSError as e:
            print(f"Could not delete {zp.name}: {e}")
    print(f"Deleted {deleted} zip(s) from {zip_dir}")

    return str(out_path)


if __name__ == "__main__":
    save_year_to_parquet(zip_dir="bluebikes_zips", year="2023", out_dir="parquet", log_path="read_log.csv")
