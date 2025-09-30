import os
import zipfile
import pandas as pd
from tqdm import tqdm
zip_folder = "bluebikes_data/zip_files"
output_file = "combined_bluebikes.csv" 

zip_files = sorted([os.path.join(zip_folder, f) for f in os.listdir(zip_folder) if f.lower().endswith(".zip")])
print(f"Found {len(zip_files)} zip files.")


all_dfs = []

for zip_path in tqdm(zip_files, desc="Processing zip files"):
    with zipfile.ZipFile(zip_path, 'r') as z:
        data_files = [name for name in z.namelist() if name.lower().endswith((".csv", ".xls", ".xlsx"))]
        for data_name in data_files:
            with z.open(data_name) as f:
                try:
                    if data_name.lower().endswith(".csv"):
                        df = pd.read_csv(f, low_memory=False)
                    else:
                        df = pd.read_excel(f)
                    df = df.dropna(subset=["starttime", "start station id"])
                    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
                    df = df.dropna(subset=["starttime"])

                    all_dfs.append(df)

                except Exception as e:
                    print(f"Failed to read {data_name} in {zip_path}: {e}")

if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined dataset to {output_file}")
else:
    print("No data found in the zip files.")
