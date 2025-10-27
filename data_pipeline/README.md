# Data Pipeline 

## Dataset Information 
- Bluebikes Trip Data: This dataset consists of comprehensive, anonymized records for every ride taken on the Bluebikes system in the Greater Boston area, serving as critical urban mobility and time-series data. Each record includes granular details such as the trip's start and end time, its duration, the specific station and location coordinates where the bike was picked up and dropped off, the unique bike ID/ride ID, and the type of user (member or casual rider). This data is essential for analyzing seasonal, daily, and hourly commuting patterns, identifying the most popular routes and stations, and understanding overall usage trends within the bike-sharing network.

- Boston Colleges Data: The Boston Colleges data collection focuses on gathering relevant, structured information about higher education institutions in the Boston area, which typically involves fetching details like school name, location (often including coordinates), enrollment figures, and possibly demographic or geographic attributes for use as fixed, dimensional context. This information is a crucial input for studies where local infrastructure planning or mobility trends (like bike-share usage) need to be correlated with major points of interest and population centers, such as student communities, which significantly influence urban travel demand.

- NOAA Weather Data: The script collects meteorological observations from the National Oceanic and Atmospheric Administration (NOAA) for the Boston area, providing critical environmental variables required to contextualize the collected mobility data. This typically includes time-series data covering weather elements such as temperature, precipitation, and general weather conditions on a daily basis. Merging this data with the Bluebikes trip history allows for a robust analysis of how real-world environmental factors, such as rain or cold temperatures, influence bicycle ridership volume and usage patterns across the city.


## Data Card

Bluebikes Data Card

Zoning Data Card

NOAA Data Card


## Data Sources
The Bluebikes data was pulled from the [Bluebikes System Data Website](https://s3.amazonaws.com/hubway-data/index.html). The Boston School and Colleges data is being queried from [Boston GIS Portal](https://gisportal.boston.gov/arcgis/rest/services/Education/OpenData/MapServer). To pull the NOAA data, we have a NOAA API key that needs to be used to access [NOAA website](https://www.ncei.noaa.gov/cdo-web/api/v2/data).

## Airflow Setup

To be Filled

## DVC Setup
Data versioning via DVC with remote storage on GCS bucket `gs://bluebikes-dvc-storage`.

GCP Project: `bluebikes-project-mlops`
GCS Bucket: `bluebikes-dvc-storage`

### How DVC was setup
Steps used to setup DVC:
```bash
pip install "dvc[gs]" gcsfs

dvc init 
git add .dvc .dvcignore
git commit -m "Init DVC"

dvc remote add -d gcs gs://bluebikes-dvc-storage
dvc remote modify gcs credentialpath <Service Account Key>
git add .dvc/config
git commit -m "Configure GCS remote"

dvc add data/raw/bluebikes data/raw/boston_clg data/raw/NOAA_weather
dvc add data/processed data/temp
git add data/**/*.dvc
git commit -m "track datasets with dvc"
```

Update the YAML file 
Additionally remove any tracking that github may have over the data folder

```bash
dvc repro
git add dvc.yaml dvc.lock
git commit -m "add dvc stage for full pipeline"
dvc push
```

Note: To create a new credential
```bash
gcloud iam service-accounts create <Service Account Name> \
  --description="Service account for DVC access" \
  --display-name="<Service Account Name>"
```

```bash
gcloud projects add-iam-policy-binding bluebikes-project-mlops \
  --member="serviceAccount:<Service Account Name>@bluebikes-project-mlops.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

```bash
gcloud iam service-accounts keys create <Key Name>.json \
  --iam-account=dvc-access@bluebikes-project-mlops.iam.gserviceaccount.com
```

In case of `[WinError3]` do this:
```bash
dvc config cache.dir C:\dvc_cache
set TMP=C:\dvc_tmp
set TEMP=C:\dvc_tmp
dvc repro
```

### How to run after cloning the repository

Navigate to the `/data_pipeline` folder

```bash
set GOOGLE_APPLICATION_CREDENTIALS=<JSON Key>
dvc remote add -d gcs gs://bluebikes-dvc-storage
dvc remote modify gcs credentialpath ../gcp-dvc-shared.json

#in case of path errors (for long names)
dvc config cache.dir C:\dvc_cache
set TMP=C:\dvc_tmp
set TEMP=C:\dvc_tmp
```


```bash
pip install "dvc[gcs]" gcsfs
dvc pull -r gcs
dvc repro

```

```bash
dvc push -r gcs
git add dvc.lock
git commit -m "update data version"
git push origin <branch>
```

### Alternative Run

Run ` python scripts/data_pipeline.py`

Then record changes in the DVC 

```bash
dvc status
dvc commit
dvc push -r gcs
git add dvc.lock
git commit -m "record data after manual run"
```


## Components

Each of the component is modularized. 
They are listed below:

### Data Collection
- `scripts/data_collection.py`
  - Orchestrates data collection and storing tasks. Use this to run end-to-end data acquisition.
- `read_log.csv`
  - A CSV log that records read/download events.
- `scripts/bluebikes_data_helpers/`
  - Helper package for working with Bluebikes zip files and data.
  - Files:
    - `download_data.py` — functions to download Bluebikes zip files and store them under `bluebikes_zips/`.
    - `normalize.py` — normalizes raw CSVs into a consistent schema.
    - `read_zips.py` — reads the downloaded zip files and generates a dataframe and proceeds to store the processed dataframe as a parquet file 
    - `record_file.py` — records metadata or a log entry after files are processed.
- `scripts/school_noaa_data_collectors/`
    - `BostonCollege.py` — API to download zoning data and store into csv
    - `NOAA_DataAcq.py` — API to fetch weather data from NOAA website

### Cleaning of Data
- `scripts/data_loader.py`
    - To be Filled
- `scripts/duplicate_data.py`
    - To be Filled
- `scripts/missing_value.py`
    - To be Filled


## Loggin and Testing 
Logger.py is a linchpin of the process. It uses Python’s built-in logging module and sets up
both console and file output with timestamps. Every module imports the same get logger() function so all
logs go to one place — the logs/ folder — with daily rotating filenames like data pipeline 20251026.log.
This makes debugging way easier since every INFO, WARNING, and ERROR is timestamped and searchable.
Testing. All core scripts are covered by pytest tests under the tests/ directory. test data collection.py
mocks the external APIs so tests run offline. test data loader.py checks that CSVs are loaded correctly and
converted into pickles. test duplicate data.py validates deduplication behaviour under different modes,
while test missing value.py verifies all filling strategies. test logger.py ensures log creation, formatting,
and handler setup work as expected. All tests write to temporary directories so nothing in the actual data
folders gets touched.
Other setup files. The environment is containerised through a Dockerfile and docker-compose.yaml,
while requirements.txt locks dependencies for reproducibility. dvc.yaml and dvc.lock handle version
tracking for data, and small shell scripts (start-airflow.sh and stop-airflow.sh) spin Airflow up and
down for orchestration tests.
In short, the goal was to make the whole system clean, testable, and consistent — one logger, one test
flow, and modular scripts that can run independently or as part of the bigger pipeline

## Anomaly Detection and Alerts


## Folder Structure

```
data_pipeline/
│── README.md   # Data pipeline description, high-level overview, setup instructions, and execution details.
│── .dvc/       # DVC's internal directory. Stores configuration, cache, and state information needed to track data and models.
│── assets/     # Directory for non-code resources like images
│── dags/       # Contains Directed Acyclic Graphs (DAGs), typically for Apache Airflow, defining the workflow and scheduling of the pipeline's tasks.
│── data/       # Stores raw, intermediate, and final datasets. Often separated into subfolders like 'raw', 'processed', 'external', etc.
├── logs/       # Stores execution logs from the pipeline runs, scripts, or Airflow. Essential for debugging and monitoring.
│── scripts/    # Contains Python or shell scripts for specific pipeline steps (e.g., data ingestion, cleaning, transformation, model training).
│── test/       # Contains unit tests and integration tests for the pipeline's scripts and code to ensure correctness.
│── dvc.lock    # Automatically generated file that records the exact versions of data and models being used, ensuring reproducibility.
│── dvc.yaml    # The main DVC configuration file. Defines the pipeline stages (steps) and their dependencies.

```