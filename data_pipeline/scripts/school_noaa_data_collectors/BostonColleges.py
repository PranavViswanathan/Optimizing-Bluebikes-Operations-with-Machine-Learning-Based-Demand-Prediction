import requests
import pandas as pd
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class BostonCollegesAPI:
    """
    A class to interact with the Boston Colleges and Universities dataset from the Boston GIS portal.

    Attributes:
        base_url (str): The base URL for the ArcGIS REST API service.
        layer_id (int): The ID of the layer to query.
        output_file (str): The path to the output CSV file.
    """

    def __init__(self, output_path=None):
        """
        Initializes the BostonCollegesAPI class with necessary configurations.
        
        Args:
            output_path (str, optional): Custom output directory path. If None, uses default PROJECT_DIR path.
        """
        self.base_url = "https://gisportal.boston.gov/arcgis/rest/services/Education/OpenData/MapServer"
        self.layer_id = 2  # Colleges/Universities layer
        
        if output_path:
            self.output_file = os.path.join(output_path, "boston_colleges.csv")
        else:
            self.output_file = os.path.join(PROJECT_DIR, "data_pipeline", "data", "raw", "boston_clg", "boston_colleges.csv")

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

    def save_to_csv(self, output_path=None):
        """
        Fetches data and saves it to a CSV file.
        
        Args:
            output_path (str, optional): Custom output directory path. Overrides instance output_file if provided.
        """
        df = self.fetch_data()
        if not df.empty:
            # If output_path is provided as argument, use it
            if output_path:
                output_file = os.path.join(output_path, "boston_colleges.csv")
            else:
                output_file = self.output_file
                
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
        else:
            print("No data to save.")

    def update_csv(self, output_path=None):
        """
        Updates the existing CSV file with the latest data.
        
        Args:
            output_path (str, optional): Custom output directory path.
        """
        if output_path:
            output_file = os.path.join(output_path, "boston_colleges.csv")
        else:
            output_file = self.output_file
            
        existing_df = pd.read_csv(output_file) if os.path.exists(output_file) else pd.DataFrame()
        new_df = self.fetch_data()

        if not new_df.empty:
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset="OBJECTID", keep="last")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            combined_df.to_csv(output_file, index=False)
            print(f"CSV updated at {output_file}")
        else:
            print("No new data to update.")

    def get_dataframe(self):
        """
        Returns the data as a pandas DataFrame.

        Returns:
            pd.DataFrame: The data as a DataFrame.
        """
        return self.fetch_data()


if __name__ == "__main__":
    api = BostonCollegesAPI()
    api.save_to_csv()