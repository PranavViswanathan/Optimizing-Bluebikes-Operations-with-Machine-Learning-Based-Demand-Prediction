# Data Collection

This document explains the purpose and features of the `reading_data/` folder and how to run its scripts and helper functions.

## What is in `reading_data/`

- `data_collection.py`
  - Orchestrates data collection and storing tasks. Use this to run end-to-end data acquisition.
- `read_log.csv`
  - A CSV log that records read/download events.
- `bluebikes_data_helpers/`
  - Helper package for working with Bluebikes zip files and data.
  - Files:
    - `download_data.py` — functions to download Bluebikes zip files and store them under `bluebikes_zips/`.
    - `normalize.py` — normalizes raw CSVs into a consistent schema.
    - `read_zips.py` — reads the downloaded zip files and generates a dataframe and proceeds to store the processed dataframe as a parquet file 
    - `record_file.py` — records metadata or a log entry after files are processed.
- `bluebikes_zips/`
  - Contains raw zip files (`data/`), and processed parquet in `parquet/`.
- `school_noaa_data_collectors/`
    - `BostonCollege.py` — API to download zoning data and store into csv
    - `NOAA_DataAcq.py` — API to fetch weather data from NOAA website 

## Features

- Read CSVs directly from .zip archives without extracting to disk.
- Normalize multiple CSV formats into a unified schema.
- Convert raw CSVs into parquet for efficient downstream processing.
- Logging of files for read.
- fetch zoning and weather data and store in csv formats 


## Dependencies

- Python 3.8+
- pandas
- pyarrow (optional, for parquet read/write)
- requests (for download utilities)


## How to Run?

Navigate to where `data_collection.py` is and run the below command

```bash
# Run the data collection script which uses helpers
python .\data_collection.py \
--year 2023, 2024, 2025
```

if there is any specific arguements to be passed, they can be done this way

```bash
python data_collection.py \
  --index-url "https://s3.amazonaws.com/hubway-data/index.html" \
  --years 2015,2016,2023 \
  --download-dir bluebikes_zips \
  --parquet-dir parquet \
  --log-path read_log.csv

```



