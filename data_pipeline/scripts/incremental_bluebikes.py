# scripts/data_pipeline/incremental_bluebikes.py
"""
Incremental BlueBikes data collection.
Only downloads and processes new ZIP files.
"""

import os
import sys
import re
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from typing import List, Optional
import requests

# Fix import paths
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data_manager import DataManager
from bluebikes_data_helpers.normalize import (
    _rename_and_coalesce,
    DEFAULT_MAPPING,
    _normalized_mapping,
    _coerce_for_parquet,
)
from logger import get_logger

logger = get_logger("incremental_bluebikes")

NORM_MAP = _normalized_mapping(DEFAULT_MAPPING)
INDEX_URL = "https://s3.amazonaws.com/hubway-data/index.html"


def find_available_zips(index_url: str, years: List[str]) -> List[str]:
    """Find all available ZIP URLs for given years."""
    from bluebikes_data_helpers.download_data import find_zip_links
    return find_zip_links(index_url, years)


def download_zip(url: str, temp_dir: Path) -> Optional[Path]:
    """Download a ZIP file to temp directory."""
    filename = url.rsplit("/", 1)[-1]
    local_path = temp_dir / filename
    
    if local_path.exists():
        logger.info(f"Already downloaded: {filename}")
        return local_path
    
    try:
        logger.info(f"Downloading: {filename}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
        return local_path
    except Exception as e:
        logger.error(f"Download failed for {filename}: {e}")
        return None


def read_zip_to_dataframe(zip_path: Path) -> pd.DataFrame:
    """Read all CSVs from a ZIP into a single DataFrame."""
    import unicodedata
    
    def clean_columns(cols):
        out = []
        for c in cols:
            s = str(c)
            s = unicodedata.normalize("NFKD", s)
            s = s.replace("\ufeff", "")
            s = s.encode("ascii", "ignore").decode()
            s = re.sub(r"\s+", " ", s).strip()
            out.append(s)
        return out
    
    frames = []
    
    with ZipFile(zip_path) as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        
        for name in csv_members:
            try:
                with zf.open(name) as f:
                    df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
            except UnicodeDecodeError:
                with zf.open(name) as f:
                    df = pd.read_csv(f, encoding="latin1", low_memory=False)
            
            df.columns = clean_columns(df.columns)
            
            # Drop empty/unnamed columns
            keep = (df.columns != "") & (~df.columns.str.match(r"^Unnamed", na=False))
            df = df.loc[:, keep]
            df = df.loc[:, df.notna().any(axis=0)]
            
            # Normalize column names
            df = _rename_and_coalesce(df, NORM_MAP)
            frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True, sort=False)


def append_to_parquet(new_df: pd.DataFrame, parquet_path: Path) -> int:
    """Append new data to existing parquet or create new one."""
    new_df = _coerce_for_parquet(new_df)
    
    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        logger.info(f"Existing data: {len(existing_df):,} rows")
        
        combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
        
        # Deduplicate by ride_id if available
        if 'ride_id' in combined.columns:
            before = len(combined)
            combined = combined.drop_duplicates(subset=['ride_id'], keep='last')
            dupes = before - len(combined)
            if dupes > 0:
                logger.info(f"Removed {dupes:,} duplicates")
        
        # Sort by time
        if 'start_time' in combined.columns:
            combined = combined.sort_values('start_time').reset_index(drop=True)
    else:
        combined = new_df
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    combined.to_parquet(parquet_path, engine="pyarrow", index=False)
    logger.info(f"Saved {len(combined):,} rows to {parquet_path.name}")
    
    return len(new_df)


def collect_bluebikes_incremental(
    years: List[str] = None,
    data_dir: str = "/opt/airflow/data"
) -> dict:
    """
    Main function: Incrementally collect BlueBikes data.
    
    - Checks which ZIPs have already been processed
    - Downloads only new ZIPs
    - Appends to existing parquet files
    - Deletes processed ZIPs
    
    Returns summary dict.
    """
    if years is None:
        years = ["2024", "2025"]
    
    dm = DataManager(data_dir)
    temp_dir = dm.temp_dir / "bluebikes"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("INCREMENTAL BLUEBIKES COLLECTION")
    logger.info("="*60)
    logger.info(f"Years: {years}")
    logger.info(f"Previously processed ZIPs: {len(dm.metadata['bluebikes']['processed_zips'])}")
    
    summary = {
        "zips_found": 0,
        "zips_skipped": 0,
        "zips_processed": 0,
        "rows_added": 0,
        "errors": []
    }
    
    # Find available ZIPs
    all_urls = find_available_zips(INDEX_URL, years)
    summary["zips_found"] = len(all_urls)
    logger.info(f"Found {len(all_urls)} ZIP files for years {years}")
    
    # Filter to only new ZIPs
    new_urls = []
    for url in all_urls:
        filename = url.rsplit("/", 1)[-1]
        if dm.is_zip_processed(filename):
            logger.info(f"Skipping (already processed): {filename}")
            summary["zips_skipped"] += 1
        else:
            new_urls.append(url)
    
    if not new_urls:
        logger.info("No new ZIP files to process!")
        dm.metadata["bluebikes"]["last_collection"] = pd.Timestamp.now().isoformat()
        dm.save_metadata()
        return summary
    
    logger.info(f"New ZIPs to process: {len(new_urls)}")
    
    # Group by year for parquet updates
    year_data = {year: [] for year in years}
    
    for url in new_urls:
        filename = url.rsplit("/", 1)[-1]
        
        # Download
        zip_path = download_zip(url, temp_dir)
        if zip_path is None:
            summary["errors"].append(f"Download failed: {filename}")
            continue
        
        # Read
        try:
            df = read_zip_to_dataframe(zip_path)
            
            if df.empty:
                logger.warning(f"Empty data in {filename}, marking as processed")
                dm.mark_zip_processed(filename)
                zip_path.unlink()
                continue
            
            logger.info(f"Read {len(df):,} rows from {filename}")
            
            # Determine year from filename
            for year in years:
                if year in filename:
                    year_data[year].append(df)
                    break
            
            # Mark as processed and delete ZIP
            dm.mark_zip_processed(filename)
            zip_path.unlink()
            logger.info(f"Deleted: {filename}")
            
            summary["zips_processed"] += 1
            summary["rows_added"] += len(df)
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            summary["errors"].append(f"Process failed: {filename} - {str(e)}")
    
    # Update parquet files
    for year, dfs in year_data.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True, sort=False)
            parquet_path = dm.get_raw_parquet_path(year)
            append_to_parquet(combined, parquet_path)
            logger.info(f"Updated {year} parquet with {len(combined):,} new rows")
    
    # Save metadata
    dm.metadata["bluebikes"]["last_collection"] = pd.Timestamp.now().isoformat()
    dm.save_metadata()
    
    # Summary
    logger.info("="*60)
    logger.info("COLLECTION COMPLETE")
    logger.info(f"ZIPs processed: {summary['zips_processed']}")
    logger.info(f"ZIPs skipped: {summary['zips_skipped']}")
    logger.info(f"Rows added: {summary['rows_added']:,}")
    if summary["errors"]:
        logger.warning(f"Errors: {summary['errors']}")
    logger.info("="*60)
    
    return summary


if __name__ == "__main__":
    # Test locally
    import json
    result = collect_bluebikes_incremental(
        years=["2024", "2025"],
        data_dir="./test_data"
    )
    print(json.dumps(result, indent=2))