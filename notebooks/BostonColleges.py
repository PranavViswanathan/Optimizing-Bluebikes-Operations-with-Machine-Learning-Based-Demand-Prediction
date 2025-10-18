import requests
import pandas as pd
import os

class BostonCollegesAPI:
    """
    A class to interact with the Boston Colleges and Universities dataset from the Boston GIS portal.

    Attributes:
        base_url (str): The base URL for the ArcGIS REST API service.
        layer_id (int): The ID of the layer to query.
        output_file (str): The path to the output CSV file.
    """

    def __init__(self):
        """
        Initializes the BostonCollegesAPI class with necessary configurations.
        """
        self.base_url = "https://gisportal.boston.gov/arcgis/rest/services/Education/OpenData/MapServer"
        self.layer_id = 2  # Colleges/Universities layer
        self.output_file = "data/raw/Boston/boston_colleges.csv"

    def fetch_data(self):
        """
        Fetches data from the ArcGIS REST API and returns it as a pandas DataFrame.

        Returns:
            pd.DataFrame: The data as a DataFrame.
        """
        url = f"{self.base_url}/{self.layer_id}/query"
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "json"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json().get("features", [])
        if not data:
            print("No data found.")
            return pd.DataFrame()

        records = [feature.get("attributes", {}) for feature in data]
        df = pd.DataFrame(records)
        return df

    def save_to_csv(self):
        """
        Fetches data and saves it to a CSV file.
        """
        df = self.fetch_data()
        if not df.empty:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            df.to_csv(self.output_file, index=False)
            print(f"Data saved to {self.output_file}")
        else:
            print("No data to save.")

    def update_csv(self):
        """
        Updates the existing CSV file with the latest data.
        """
        existing_df = pd.read_csv(self.output_file) if os.path.exists(self.output_file) else pd.DataFrame()
        new_df = self.fetch_data()

        if not new_df.empty:
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset="OBJECTID", keep="last")
            combined_df.to_csv(self.output_file, index=False)
            print(f"CSV updated at {self.output_file}")
        else:
            print("No new data to update.")

    def get_dataframe(self):
        """
        Returns the data as a pandas DataFrame.

        Returns:
            pd.DataFrame: The data as a DataFrame.
        """
        return self.fetch_data()

if __name__=="__main__":
    api = BostonCollegesAPI()
    api.save_to_csv()