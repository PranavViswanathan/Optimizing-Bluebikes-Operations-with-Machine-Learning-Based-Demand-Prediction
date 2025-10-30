# data_collection.py
from pathlib import Path
import argparse
import sys
import os
from typing import List
import pandas as pd
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  

from logger import get_logger
logger = get_logger("data_collection")

try:
    from bluebikes_data_helpers.download_data import find_zip_links, download_zips
    from bluebikes_data_helpers.read_zips import save_year_to_parquet
    from school_noaa_data_collectors.BostonColleges import BostonCollegesAPI
    from school_noaa_data_collectors.NOAA_DataAcq import NOAA
except ImportError as e:
    logger.error(f"[IMPORT ERROR] Could not import helper modules: {e}")
    logger.info("→ Check that your PYTHONPATH includes the project root or use absolute imports.")
    sys.exit(1)


def read_zip_csv(zip_path: Path) -> pd.DataFrame:
    """Read only the main CSV file from a ZIP, ignoring __MACOSX and hidden files."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find the first CSV that's not in __MACOSX or hidden
        csv_files = [f for f in z.namelist() if f.endswith('.csv') and not f.startswith('__MACOSX')]
        if not csv_files:
            raise ValueError(f"No CSV found in {zip_path}")
        target_csv = csv_files[0]
        df = pd.read_csv(z.open(target_csv))
    return df

def collect_bluebikes_data(index_url: str,
                           years: List[str],
                           download_dir: str,
                           parquet_dir: str,
                           log_path: str,
                           max_rows_per_month: int = 50000):
    """
    Collect Bluebikes data for given years.
    Downloads ZIPs, reads each CSV in chunks, samples up to max_rows_per_month, and saves as parquet.
    """
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    parquet_path = Path(parquet_dir)
    parquet_path.mkdir(parents=True, exist_ok=True)

    for year in years:
        logger.info(f"\n=== Year {year} ===")

        # Step 1: Find URLs (existing function)
        try:
            urls = find_zip_links(index_url, [year])
        except Exception as e:
            logger.error(f"[{year}] Could not list ZIPs from index: {e}")
            continue

        if not urls:
            logger.warning(f"[{year}] No ZIP links found at index. Skipping.")
            continue

        # Step 2: Download ZIPs
        try:
            saved = download_zips(urls, out_dir=download_dir)
            if not saved:
                logger.info(f"[{year}] No ZIPs downloaded (maybe already present?).")
            else:
                logger.info(f"[{year}] Downloaded {len(saved)} file(s).")
        except Exception as e:
            logger.error(f"[{year}] Failed to download ZIPs: {e}")
            continue

        # Step 3: Process each ZIP separately
        sampled_dfs = []
        for zip_file in saved:
            zip_file_path = Path(download_dir) / zip_file
            try:
                with zipfile.ZipFile(zip_file_path, "r") as z:
                    # Only process CSV files in ZIP
                    csv_files = [f for f in z.namelist() if f.endswith(".csv") and not f.startswith("__MACOSX/")]
                    if len(csv_files) != 1:
                        logger.warning(f"Skipping {zip_file}: expected 1 CSV, found {csv_files}")
                        continue

                    with z.open(csv_files[0]) as f:
                        df = pd.read_csv(f)
                        # Sample rows
                        if len(df) > max_rows_per_month:
                            df_sampled = df.sample(n=max_rows_per_month, random_state=42)
                        else:
                            df_sampled = df
                        sampled_dfs.append(df_sampled)

            except Exception as e:
                logger.error(f"Failed to process {zip_file}: {e}")
                continue

        # Step 4: Concatenate sampled monthly DataFrames
        if sampled_dfs:
            df_year = pd.concat(sampled_dfs, ignore_index=True)
            # Save as parquet
            out_path = parquet_path / f"bluebikes_{year}.parquet"
            df_year.to_parquet(out_path, index=False)
            logger.info(f"[{year}] Parquet saved to: {out_path}")
        else:
            logger.warning(f"[{year}] No data sampled. Skipping saving parquet.")


def collect_boston_college_data(output_path=None):
    """Collect Boston Colleges data."""
    logger.info("\n=== Collecting Boston Colleges data ===")
    try:
        # Pass output_path to the class constructor OR to save_to_csv
        colleges = BostonCollegesAPI(output_path=output_path) if output_path else BostonCollegesAPI()
        colleges.save_to_csv()
        logger.info("Boston Colleges data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to fetch Boston Colleges data: {e}")
        raise   


def collect_NOAA_Weather_data(output_path=None):
    """Collect NOAA Weather data."""
    logger.info("\n=== Collecting NOAA Weather data ===")
    try:
        # Pass output_path to the class constructor
        noaa = NOAA(output_path=output_path) if output_path else NOAA()
        noaa.fetch_training_data_from_api()
        noaa.get_weather_dataframe()
        logger.info("NOAA Weather data fetched and DataFrame created successfully.")
    except Exception as e:
        logger.error(f"Failed to fetch NOAA data: {e}")
        raise


def parse_args():
    p = argparse.ArgumentParser(description="Download BlueBikes, Boston Colleges, and NOAA datasets.")
    p.add_argument(
        "--index-url",
        default="https://s3.amazonaws.com/hubway-data/index.html",
        help="Index page / bucket listing URL."
    )
    p.add_argument(
        "--years",
        default="2025",
        help="Comma-separated list of years to process."
    )
    p.add_argument(
        "--download-dir",
        default="bluebikes_zips",
        help="Folder to store ZIP files."
    )
    p.add_argument(
        "--parquet-dir",
        default="parquet",
        help="Folder to store Parquet outputs."
    )
    p.add_argument(
        "--log-path",
        default="read_log.csv",
        help="CSV log file for read statuses."
    )
    p.add_argument(
        "--boston-college-output",
        default=None,
        help="Output path for Boston College data."
    )
    p.add_argument(
        "--noaa-output",
        default=None,
        help="Output path for NOAA weather data."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    years = [y.strip() for y in args.years.split(",") if y.strip()]

    logger.info("=== Starting Data Collection Pipeline ===")
    logger.info(f"Years: {years}")

    collect_bluebikes_data(
        index_url=args.index_url,
        years=years,
        download_dir=args.download_dir,
        parquet_dir=args.parquet_dir,
        log_path=args.log_path,
    )

    collect_boston_college_data(output_path=args.boston_college_output)
    collect_NOAA_Weather_data(output_path=args.noaa_output)