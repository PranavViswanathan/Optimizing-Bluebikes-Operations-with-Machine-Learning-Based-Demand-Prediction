import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv
import os

class NOAA:
    """
    A class to fetch, update, and manage historical weather data from the NOAA
    (National Oceanic and Atmospheric Administration) Climate Data Online (CDO) API.

    This class provides methods to:
    1. Fetch weather data from NOAA's API for a specific station and time range.
    2. Update or create a CSV file containing daily weather records.
    3. Convert fetched data into a pandas DataFrame and optionally save it as a CSV file.

    Attributes:
        api_token (str): NOAA API token loaded from environment variables.
        station_id (str): Station ID for the Boston Logan Airport (GHCND:USW00014739).
        datatype_ids (list[str]): List of weather data types to fetch (e.g., TMAX, TMIN, PRCP).
        start_year (int): The starting year for the data fetch range.
        end_year (int): The ending year for the data fetch range.
        output_file (str): Path to the CSV file used for saving or updating weather data.
        headers (dict): Headers for API requests, including the authorization token.
        base_url (str): Base URL of the NOAA CDO API.
        all_data (list): List that stores raw JSON results fetched from the API.
    """

    def __init__(self):
        """Initializes the NOAA data collector with API credentials and configuration parameters."""

        load_dotenv()
        self.api_token = os.getenv("NOAA_API_KEY")

        if not self.api_token:
            raise ValueError("API_KEY not found in environment variables.")

        self.station_id = "GHCND:USW00014739"  # Boston Logan Airport
        self.datatype_ids = ["TMAX", "TMIN", "PRCP"]
        self.start_year = 2015
        self.end_year = 2025
        self.output_file = "data/raw/NOAA/boston_daily_weather_3.csv"

        self.headers = {"token": self.api_token}
        self.base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
        self.all_data = []

    def fetch_training_data_from_api(self):
        """
        Fetches daily weather data from the NOAA CDO API between the configured start and end years.

        This method iterates over each year in the specified range and retrieves data for
        the given station (`self.station_id`) and datatypes (`self.datatype_ids`).
        The fetched data is stored in `self.all_data`.

        Data is requested in batches of 1000 records using pagination via the `offset` parameter.
        The process continues until all data for a given year is retrieved or an API error occurs.

        Raises:
            requests.exceptions.RequestException: If an HTTP request to the API fails.
            KeyError: If the expected data structure is missing in the API response.

        Example:
            >>> noaa = NOAA()
            >>> noaa.fetch_training_data_from_api()
            >>> len(noaa.all_data)
            3650  # Example: number of records fetched

        Notes:
            - This function must be run before generating a CSV or DataFrame.
            - Requires a valid NOAA API key set in the environment variable `NOAA_API_KEY`.
        """
        
        for year in tqdm(range(self.start_year, self.end_year + 1), desc="Years"):
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
                response = requests.get(self.base_url, headers=self.headers, params=params)

                if response.status_code != 200:
                    print(f"Error {response.status_code} for {year}")
                    break

                data = response.json().get("results", [])
                if not data:
                    break

                self.all_data.extend(data)
                if len(data) < 1000:
                    break

                offset += 1000


    def update_or_create_csv(self, update_existing=True):
        """
        Updates an existing weather CSV file with new data or creates a new one.

        Depending on the `update_existing` parameter and file existence, this method:
        - Creates a new CSV file (if `update_existing=False` or file doesn't exist).
        - Updates an existing CSV by appending new records from the last available date onward.

        The new or updated data is fetched directly from the NOAA API and stored in the file
        specified by `self.output_file`.

        Args:
            update_existing (bool): If True, updates the existing file. If False, creates a new one.

        Raises:
            FileNotFoundError: If attempting to update a file that does not exist.
            ValueError: If the existing CSV contains invalid or missing date values.

        Example:
            >>> noaa = NOAA()
            >>> noaa.update_or_create_csv(update_existing=True)
            Updating existing CSV with new NOAA data (if available)
            >>> noaa.update_or_create_csv(update_existing=False)
            Creating a new CSV file from 2015 to 2025.
        """
        start_date = "2015-01-01"
        end_date = "2025-12-31"

        if not update_existing or not os.path.exists(self.output_file):
            print("Creating a new CSV file from 2015 to 2025.")
            self.all_data = []
            self.fetch_data_from_api()
            self.get_weather_dataframe()
            return

        print(" Updating existing CSV with new NOAA data (if available)")
        existing_df["date"] = pd.to_datetime(existing_df["date"])

        last_recorded_date = existing_df["date"].max()
        print(f"Last recorded date in file: {last_recorded_date.date()}")

        next_start_date = (last_recorded_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if pd.to_datetime(next_start_date) > pd.to_datetime(end_date):
            print("The CSV file is already up to date.")
            return

        print(f"Fetching data from {next_start_date} to {end_date}...")
        self.all_data = []
        self.fetch_data_from_api()

        if not self.all_data:
            print("No new data retrieved from NOAA API.")
            return

        new_df = pd.DataFrame(self.all_data)
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_pivot = new_df.pivot_table(index="date", columns="datatype", values="value").reset_index()

        # Merge and save
        combined_df = (
            pd.concat([existing_df, new_pivot])
            .drop_duplicates(subset="date")
            .sort_values("date")
        )

        combined_df.to_csv(self.output_file, index=False)
        print(f" Updated CSV saved to {self.output_file}")

    
    def get_weather_dataframe(self, assign_to_variable=False):
        """
        Converts fetched NOAA API data into a pandas DataFrame and optionally saves it as a CSV file.

        This method reshapes the raw JSON data stored in `self.all_data` into a tabular format
        with columns representing `TMAX`, `TMIN`, and `PRCP`, indexed by date.

        If `assign_to_variable=True`, it returns the resulting DataFrame for further analysis.
        Otherwise, the DataFrame is saved to `self.output_file`.

        Args:
            assign_to_variable (bool): If True, returns the resulting DataFrame instead of just saving it.

        Returns:
            pandas.DataFrame | None: The processed DataFrame if `assign_to_variable=True`, otherwise None.

        Raises:
            ValueError: If no data has been fetched prior to calling this method.

        Example:
            >>> noaa = NOAA()
            >>> noaa.fetch_training_data_from_api()
            >>> df = noaa.get_weather_dataframe(assign_to_variable=True)
            >>> df.head()
        """
        if self.all_data:
            df = pd.DataFrame(self.all_data)
            df["date"] = pd.to_datetime(df["date"])
            df_pivot = df.pivot_table(index="date", columns="datatype", values="value").reset_index()
            df_pivot.to_csv(self.output_file, index=False)
            print(f"Saved Boston weather data to {self.output_file}")
            if self.assign_to_variable:
                return df_pivot
        else:
            print("No data retrieved. Run fetch_data_from_api() first.")

