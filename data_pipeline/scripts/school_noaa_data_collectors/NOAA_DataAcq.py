import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv
import os
import time


class NOAA:
    """
    A class to fetch, update, and manage historical weather data from the NOAA
    (National Oceanic and Atmospheric Administration) Climate Data Online (CDO) API.

    This class provides methods to:
    1. Fetch weather data from NOAA's API for a specific station and time range.
    2. Update or create a CSV file containing daily weather records.
    3. Convert fetched data into a pandas DataFrame and optionally save it as a CSV file.
    """

    def __init__(self):
        """Initializes the NOAA data collector with API credentials and configuration parameters."""
        load_dotenv()
        self.api_token = os.getenv("NOAA_API_KEY")

        if not self.api_token:
            raise ValueError("NOAA_API_KEY not found in environment variables.")

        self.station_id = "GHCND:USW00014739"  # Boston Logan Airport
        self.datatype_ids = ["TMAX", "TMIN", "PRCP"]
        self.start_year = 2015
        self.end_year = 2025
        self.output_file = "data/raw/NOAA/boston_daily_weather_3.csv"

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        self.headers = {"token": self.api_token}
        self.base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
        self.all_data = []

    def fetch_training_data_from_api(self):
        """
        Fetches daily weather data from the NOAA CDO API between start_year and end_year.
        Stores results in self.all_data.
        """
        for year in tqdm(range(self.start_year, self.end_year + 1), desc="Fetching NOAA data by year"):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            params = {
                "datasetid": "GHCND",
                "stationid": self.station_id,
                "startdate": start_date,
                "enddate": end_date,
                "datatypeid": self.datatype_ids,
                "limit": 1000,
                "units": "metric",
                "orderby": "date"
            }

            offset = 1
            while True:
                params["offset"] = offset
                for attempt in range(3):  
                    response = requests.get(self.base_url, headers=self.headers, params=params)
                    if response.status_code == 200:
                        break
                    elif attempt < 2:
                        print(f"[{year}] Retry {attempt + 1}/3 after {2 ** attempt}s (status={response.status_code})")
                        time.sleep(2 ** attempt)
                    else:
                        print(f"[{year}] Failed after 3 retries (status={response.status_code})")
                        return

                data = response.json().get("results", [])
                if not data:
                    break

                self.all_data.extend(data)
                if len(data) < 1000:
                    break

                offset += 1000

    def update_or_create_csv(self, update_existing=True):
        """
        Updates existing NOAA weather CSV or creates a new one if missing.
        """
        start_date = "2015-01-01"
        end_date = "2025-12-31"

        # Case 1: Create new file
        if not update_existing or not os.path.exists(self.output_file):
            print("Creating a new NOAA weather CSV from 2015 to 2025...")
            self.all_data = []
            self.fetch_training_data_from_api()
            self.get_weather_dataframe()
            return

        # Case 2: Update existing file
        print("Updating existing NOAA weather CSV (if new data is available)...")
        existing_df = pd.read_csv(self.output_file)
        existing_df["date"] = pd.to_datetime(existing_df["date"])

        last_recorded_date = existing_df["date"].max()
        print(f"Last recorded date in file: {last_recorded_date.date()}")

        next_start_date = (last_recorded_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if pd.to_datetime(next_start_date) > pd.to_datetime(end_date):
            print("The CSV file is already up to date.")
            return

        print(f"Fetching new data from {next_start_date} to {end_date}...")
        self.all_data = []
        self.fetch_training_data_from_api()

        if not self.all_data:
            print("No new data retrieved from NOAA API.")
            return

        new_df = pd.DataFrame(self.all_data)
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_pivot = new_df.pivot_table(index="date", columns="datatype", values="value").reset_index()

        combined_df = (
            pd.concat([existing_df, new_pivot])
            .drop_duplicates(subset="date")
            .sort_values("date")
        )

        combined_df.to_csv(self.output_file, index=False)
        print(f"Updated NOAA CSV saved to {self.output_file}")

  
    def get_weather_dataframe(self, assign_to_variable=False):
        """
        Converts fetched NOAA API data into a pandas DataFrame and optionally returns it.
        """
        if self.all_data:
            df = pd.DataFrame(self.all_data)
            df["date"] = pd.to_datetime(df["date"])
            df_pivot = df.pivot_table(index="date", columns="datatype", values="value").reset_index()
            df_pivot.to_csv(self.output_file, index=False)
            print(f"Saved Boston weather data to {self.output_file}")

            if assign_to_variable:
                return df_pivot
        else:
            print(" No data retrieved. Run fetch_training_data_from_api() first.")
