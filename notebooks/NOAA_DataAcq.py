import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv
import os

class NOAA:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("NOAA_API_KEY")

        if not self.api_token:
            raise ValueError("API_KEY not found in environment variables.")

        self.station_id = "GHCND:USW00014739"  # Boston Logan Airport
        self.datatype_ids = ["TMAX", "TMIN", "PRCP"]
        self.start_year = 2015
        self.end_year = 2025
        self.output_file = "data/raw/NOAA/boston_daily_weather_2.csv"

        self.headers = {"token": self.api_token}
        self.base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
        self.all_data = []

    def fetch_data_from_api(self):
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

    def get_weather_dataframe(self):
        if self.all_data:
            df = pd.DataFrame(self.all_data)
            df["date"] = pd.to_datetime(df["date"])
            df_pivot = df.pivot_table(index="date", columns="datatype", values="value").reset_index()
            df_pivot.to_csv(self.output_file, index=False)
            print(f"Saved Boston weather data to {self.output_file}")
        else:
            print("No data retrieved. Run fetch_data_from_api() first.")
