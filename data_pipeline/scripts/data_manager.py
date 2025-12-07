# scripts/data_pipeline/data_manager.py
"""
Central data management for BlueBikes pipeline.
Tracks state, handles incremental updates, manages preprocessing.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import pandas as pd

# Fix import path for logger
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from logger import get_logger

logger = get_logger("data_manager")


class DataManager:
    """
    Manages data state and paths for the pipeline.
    
    Responsibilities:
    - Track which ZIPs have been processed
    - Check if preprocessing is needed
    - Provide consistent paths
    """
    
    def __init__(self, base_dir: str = "/opt/airflow/data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.temp_dir = self.base_dir / "temp"
        self.metadata_path = self.base_dir / "pipeline_metadata.json"
        
        # Ensure directories exist
        for dataset in ["bluebikes", "boston_clg", "NOAA_weather"]:
            (self.raw_dir / dataset).mkdir(parents=True, exist_ok=True)
            (self.processed_dir / dataset).mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load or initialize metadata."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted metadata file, reinitializing")
        
        return self._init_metadata()
    
    def _init_metadata(self) -> dict:
        """Initialize fresh metadata."""
        return {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "last_updated": None,
            "bluebikes": {
                "processed_zips": [],
                "last_collection": None,
                "last_preprocessing": None,
            },
            "NOAA_weather": {
                "last_date": None,
                "last_collection": None,
                "last_preprocessing": None,
            },
            "boston_clg": {
                "last_collection": None,
                "last_preprocessing": None,
            }
        }
    
    def save_metadata(self):
        """Persist metadata to disk."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {self.metadata_path}")
    
    # =========================================================================
    # Path Helpers
    # =========================================================================
    
    def get_raw_parquet_path(self, year: str) -> Path:
        """Get path to BlueBikes raw parquet for a year."""
        return self.raw_dir / "bluebikes" / f"trips_{year}.parquet"
    
    def get_processed_pkl_path(self, dataset: str, stage: str) -> Path:
        """Get path to processed PKL file."""
        return self.processed_dir / dataset / f"{stage}.pkl"
    
    def get_final_pkl_path(self, dataset: str) -> Path:
        """Get path to final processed PKL (after_duplicates.pkl)."""
        return self.processed_dir / dataset / "after_duplicates.pkl"
    
    # =========================================================================
    # Status Checks
    # =========================================================================
    
    def has_raw_data(self, dataset: str) -> bool:
        """Check if raw data exists for a dataset."""
        if dataset == "bluebikes":
            parquets = list((self.raw_dir / "bluebikes").glob("trips_*.parquet"))
            return len(parquets) > 0
        elif dataset == "NOAA_weather":
            csv_path = self.raw_dir / "NOAA_weather" / "boston_daily_weather.csv"
            return csv_path.exists()
        elif dataset == "boston_clg":
            csv_path = self.raw_dir / "boston_clg" / "boston_colleges.csv"
            return csv_path.exists()
        return False
    
    def has_processed_data(self, dataset: str) -> bool:
        """Check if final processed PKL exists."""
        return self.get_final_pkl_path(dataset).exists()
    
    def needs_preprocessing(self, dataset: str) -> Tuple[bool, str]:
        """
        Determine if preprocessing should run.
        
        Returns:
            (needs_preprocessing: bool, reason: str)
        """
        has_raw = self.has_raw_data(dataset)
        has_processed = self.has_processed_data(dataset)
        
        if not has_raw:
            return False, "no_raw_data"
        
        if not has_processed:
            return True, "processed_missing"
        
        # Check if raw is newer than processed
        raw_mtime = self._get_raw_mtime(dataset)
        processed_mtime = self._get_processed_mtime(dataset)
        
        if raw_mtime and processed_mtime and raw_mtime > processed_mtime:
            return True, "raw_updated"
        
        return False, "up_to_date"
    
    def _get_raw_mtime(self, dataset: str) -> Optional[float]:
        """Get modification time of raw data."""
        if dataset == "bluebikes":
            parquets = list((self.raw_dir / "bluebikes").glob("trips_*.parquet"))
            if parquets:
                return max(p.stat().st_mtime for p in parquets)
        else:
            raw_path = self.raw_dir / dataset
            files = list(raw_path.glob("*"))
            if files:
                return max(f.stat().st_mtime for f in files if f.is_file())
        return None
    
    def _get_processed_mtime(self, dataset: str) -> Optional[float]:
        """Get modification time of final processed file."""
        final_path = self.get_final_pkl_path(dataset)
        if final_path.exists():
            return final_path.stat().st_mtime
        return None
    
    # =========================================================================
    # BlueBikes Specific
    # =========================================================================
    
    def is_zip_processed(self, filename: str) -> bool:
        """Check if a ZIP has already been processed."""
        return filename in self.metadata["bluebikes"]["processed_zips"]
    
    def mark_zip_processed(self, filename: str):
        """Mark a ZIP as processed."""
        if filename not in self.metadata["bluebikes"]["processed_zips"]:
            self.metadata["bluebikes"]["processed_zips"].append(filename)
    
    def get_existing_parquet_dates(self, year: str) -> Tuple[Optional[str], Optional[str]]:
        """Get min/max dates from existing parquet."""
        parquet_path = self.get_raw_parquet_path(year)
        
        if not parquet_path.exists():
            return None, None
        
        try:
            df = pd.read_parquet(parquet_path, columns=['start_time'])
            if df.empty:
                return None, None
            
            min_date = df['start_time'].min()
            max_date = df['start_time'].max()
            
            return str(min_date.date()), str(max_date.date())
        except Exception as e:
            logger.warning(f"Could not read dates from {parquet_path}: {e}")
            return None, None
    
    def load_all_bluebikes_parquets(self) -> pd.DataFrame:
        """Load all BlueBikes parquet files into single DataFrame."""
        parquet_dir = self.raw_dir / "bluebikes"
        parquet_files = sorted(parquet_dir.glob("trips_*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
        
        dfs = []
        for pf in parquet_files:
            logger.info(f"Loading {pf.name}")
            dfs.append(pd.read_parquet(pf))
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined):,} total rows from {len(parquet_files)} files")
        
        return combined
    
    # =========================================================================
    # Status Report
    # =========================================================================
    
    def get_status_report(self) -> Dict:
        """Get complete status of all datasets."""
        report = {
            "last_updated": self.metadata.get("last_updated"),
            "datasets": {}
        }
        
        for dataset in ["bluebikes", "NOAA_weather", "boston_clg"]:
            needs_preprocess, reason = self.needs_preprocessing(dataset)
            
            report["datasets"][dataset] = {
                "has_raw_data": self.has_raw_data(dataset),
                "has_processed_data": self.has_processed_data(dataset),
                "needs_preprocessing": needs_preprocess,
                "preprocessing_reason": reason,
            }
            
            if dataset == "bluebikes":
                report["datasets"][dataset]["processed_zips_count"] = len(
                    self.metadata["bluebikes"]["processed_zips"]
                )
        
        return report
    
    def print_status(self):
        """Print human-readable status."""
        report = self.get_status_report()
        
        print("\n" + "="*60)
        print("DATA PIPELINE STATUS")
        print("="*60)
        print(f"Last updated: {report['last_updated'] or 'Never'}")
        
        for dataset, info in report["datasets"].items():
            print(f"\n{dataset}:")
            print(f"  Raw data exists: {info['has_raw_data']}")
            print(f"  Processed data exists: {info['has_processed_data']}")
            print(f"  Needs preprocessing: {info['needs_preprocessing']} ({info['preprocessing_reason']})")
            
            if dataset == "bluebikes":
                print(f"  ZIPs processed: {info['processed_zips_count']}")