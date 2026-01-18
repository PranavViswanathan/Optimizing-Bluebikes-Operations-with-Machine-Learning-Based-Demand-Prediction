
import pandas as pd
try:
    df = pd.read_parquet('/Users/pranavviswanathan/Northeastern/Education/Fall2025/ML-OPS/Project/data_pipeline/data/raw/bluebikes/trips_2024.parquet')
    print(f"Min date: {df['start_time'].min()}")
    print(f"Max date: {df['start_time'].max()}")
except Exception as e:
    print(e)
