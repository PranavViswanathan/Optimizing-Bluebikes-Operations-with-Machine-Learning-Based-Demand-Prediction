#data_collection.py
from pathlib import Path
import argparse

from bluebikes_data_helpers.download_data import find_zip_links, download_zips
from bluebikes_data_helpers.read_zips import save_year_to_parquet

# Added imports
from school_noaa_data_collectors.BostonColleges import BostonCollegesAPI
from school_noaa_data_collectors.NOAA_DataAcq import NOAA

from logger import get_logger
logger = get_logger("data_collection")

def collect_bluebikes_data(index_url: str, years: list[str], download_dir: str, parquet_dir: str, log_path: str):
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    for year in years:
        logger.info(f"\n=== Year {year} ===")

        # 1) Find + download zips for this year
        try:
            urls = find_zip_links(index_url, [year])
        except Exception as e:
            logger.error(f"[{year}] Could not list ZIPs from index: {e}")
            continue

        if not urls:
            logger.warning(f"[{year}] No ZIP links found at index. Skipping.")
            continue

        saved = download_zips(urls, out_dir=download_dir)
        if not saved:
            logger.info(f"[{year}] No ZIPs downloaded (maybe already present?).")
        else:
            logger.info(f"[{year}] Downloaded {len(saved)} file(s).")

        # 2) Build parquet 
        try:
            out_path = save_year_to_parquet(download_dir, year, parquet_dir, log_path)
            logger.info(f"[{year}] Parquet saved to: {out_path}")
        except Exception as e:
            logger.error(f"[{year}] Failed to build/save parquet: {e}")


def collect_boston_college_data():
    """Mandatory: collect Boston Colleges and NOAA weather data."""
    logger.info("\n=== Collecting Boston Colleges data ===")
    try:
        colleges = BostonCollegesAPI()
        colleges.save_to_csv()
        logger.info("Boston Colleges data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to fetch Boston Colleges data: {e}")

        
def collect_NOAA_Weather_data():
    logger.info("\n=== Collecting NOAA Weather data ===")
    try:
        noaa = NOAA()
        noaa.fetch_training_data_from_api()
        noaa.get_weather_dataframe()
        logger.info("NOAA Weather data fetched and DataFrame created successfully.")
    except Exception as e:
        logger.error(f"Failed to fetch NOAA data: {e}")


# def parse_args():
#     p = argparse.ArgumentParser(description="Download BlueBikes/Hubway zips and save yearly Parquet.")
#     p.add_argument("--index-url",
#                    default="https://s3.amazonaws.com/hubway-data/index.html",
#                    help="Index page / bucket listing URL.")
#     p.add_argument("--years",
#                    default="2023,2024,2025",
#                    help="Comma-separated list of years to process (e.g. 2015,2016,2023).")
#     p.add_argument("--download-dir",
#                    default="bluebikes_zips",
#                    help="Folder to store ZIP files.")
#     p.add_argument("--parquet-dir",
#                    default="parquet",
#                    help="Folder to store Parquet outputs.")
#     p.add_argument("--log-path",
#                    default="read_log.csv",
#                    help="CSV log file for read statuses.")
#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     years = [y.strip() for y in args.years.split(",") if y.strip()]
#     collect_bluebikes_data(
#         index_url=args.index_url,
#         years=years,
#         download_dir=args.download_dir,
#         parquet_dir=args.parquet_dir,
#         log_path=args.log_path,
#     )
#     collect_boston_college_data()
#     collect_NOAA_Weather_data()