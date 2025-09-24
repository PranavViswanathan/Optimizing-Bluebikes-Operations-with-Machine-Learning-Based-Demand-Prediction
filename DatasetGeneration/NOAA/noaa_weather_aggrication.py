import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
API_TOKEN = os.getenv("API_KEY")
station_id = "GHCND:USW00014739"  # Boston Logan Airport
datatype_ids = ["TMAX", "TMIN", "PRCP"]
start_year = 2015
end_year = 2025 
output_file = "NOAA/boston_daily_weather.csv"

headers = {"token": API_TOKEN}
base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

all_data = []

for year in tqdm(range(start_year, end_year+1), desc="Years"):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "datatypeid": datatype_ids,
        "limit": 1000,  
        "units": "metric",
        "orderby": "date"
    }

    offset = 1
    while True:
        params["offset"] = offset
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error {response.status_code} for {year}")
            break

        data = response.json().get("results", [])
        if not data:
            break

        all_data.extend(data)
        if len(data) < 1000:
            break

        offset += 1000

if all_data:
    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    df_pivot = df.pivot_table(index="date", columns="datatype", values="value").reset_index()
    df_pivot.to_csv(output_file, index=False)
    print(f"Saved Boston weather to {output_file}")
else:
    print("No data retrieved.")
