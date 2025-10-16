from BostonColleges import BostonCollegesAPI

# Initialize the API class
api = BostonCollegesAPI()

# Fetch data and save to CSV
api.save_to_csv()  # This will create the folder and file if they don't exist

# Update existing CSV with any new data (avoids duplicates)
api.update_csv()

# Retrieve data as a pandas DataFrame
df = api.get_dataframe()
print(df.head())
