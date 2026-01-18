
import pandas as pd
try:
    df = pd.read_parquet('/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/data_pipeline/data/raw/bluebikes/trips_2024.parquet')
    print(df.columns)
    print(df.head(1).T)
except Exception as e:
    print(e)
