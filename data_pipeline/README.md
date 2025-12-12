# Data Pipeline 

![alt text](<data_pipeline_dag.jpg>)

## Dataset Information 
- Bluebikes Trip Data: This dataset consists of comprehensive, anonymized records for every ride taken on the Bluebikes system in the Greater Boston area, serving as critical urban mobility and time-series data. Each record includes granular details such as the trip's start and end time, its duration, the specific station and location coordinates where the bike was picked up and dropped off, the unique bike ID/ride ID, and the type of user (member or casual rider). This data is essential for analyzing seasonal, daily, and hourly commuting patterns, identifying the most popular routes and stations, and understanding overall usage trends within the bike-sharing network.

- Boston Colleges Data: The Boston Colleges data collection focuses on gathering relevant, structured information about higher education institutions in the Boston area, which typically involves fetching details like school name, location (often including coordinates), enrollment figures, and possibly demographic or geographic attributes for use as fixed, dimensional context. This information is a crucial input for studies where local infrastructure planning or mobility trends (like bike-share usage) need to be correlated with major points of interest and population centers, such as student communities, which significantly influence urban travel demand.

- NOAA Weather Data: The script collects meteorological observations from the National Oceanic and Atmospheric Administration (NOAA) for the Boston area, providing critical environmental variables required to contextualize the collected mobility data. This typically includes time-series data covering weather elements such as temperature, precipitation, and general weather conditions on a daily basis. Merging this data with the Bluebikes trip history allows for a robust analysis of how real-world environmental factors, such as rain or cold temperatures, influence bicycle ridership volume and usage patterns across the city.

## Data Card

### 1. Bluebikes Trip Data
*Source:* Bluebikes System Data (Boston’s public bike-share program)  
*Description:* This dataset contains detailed trip-level information, including ride identifiers, timestamps, station details, coordinates, and user type.  
*Total Columns:* 13  

*Columns:*
•⁠  ⁠ride_id: Unique identifier for each trip  
•⁠  ⁠rideable_type: Type of bike used (classic/electric)  
•⁠  ⁠start_time: Trip start timestamp (UTC)  
•⁠  ⁠stop_time: Trip end timestamp (UTC)  
•⁠  ⁠start_station_name: Name of the start station  
•⁠  ⁠start_station_id: ID of the start station  
•⁠  ⁠end_station_name: Name of the end station  
•⁠  ⁠end_station_id: ID of the end station  
•⁠  ⁠start_station_latitude: Latitude of start station  
•⁠  ⁠start_station_longitude: Longitude of start station  
•⁠  ⁠end_station_latitude: Latitude of end station  
•⁠  ⁠end_station_longitude: Longitude of end station  
•⁠  ⁠user_type: Type of user (e.g., member or casual)  

*Use Case:*  
Used to analyze ride patterns, durations, user behavior, and geographic distribution of trips across the Bluebikes network.

---

### 2. Boston Colleges Data
*Source:* City of Boston Open Data Portal  
*Description:* This dataset includes information about colleges and universities in Boston, covering their facilities, student numbers, building details, and geographic coordinates.  
*Total Columns:* 28  

*Columns:*
•⁠  ⁠OBJECTID: Unique object identifier  
•⁠  ⁠Match_type: Type of match used to identify the record  
•⁠  ⁠Ref_ID: Reference ID  
•⁠  ⁠ID1: Secondary ID (if available)  
•⁠  ⁠Id: Main school ID  
•⁠  ⁠SchoolId: Unique school identifier  
•⁠  ⁠Name: Name of the college or university  
•⁠  ⁠Address: Street address  
•⁠  ⁠City: City name (typically Boston)  
•⁠  ⁠Zipcode: ZIP code  
•⁠  ⁠Contact: Contact person or department  
•⁠  ⁠PhoneNumbe: Phone number  
•⁠  ⁠YearBuilt: Year the building was constructed  
•⁠  ⁠NumStories: Number of stories in the building  
•⁠  ⁠Cost: Estimated building cost  
•⁠  ⁠NumStudent: Total number of students  
•⁠  ⁠BackupPowe: Indicator of backup power availability (1 = yes, 0 = no)  
•⁠  ⁠ShelterCap: Shelter capacity information  
•⁠  ⁠Latitude: Latitude of the institution  
•⁠  ⁠Longitude: Longitude of the institution  
•⁠  ⁠Comment: Additional comments or notes  
•⁠  ⁠X: X coordinate (projection)  
•⁠  ⁠Y: Y coordinate (projection)  
•⁠  ⁠NumStudent12: Number of students in 2012  
•⁠  ⁠CampusHous: Indicates presence of campus housing  
•⁠  ⁠NumStudents13: Number of students in 2013  
•⁠  ⁠URL: Website of the institution  
•⁠  ⁠Address2013: Address update field (mostly null)  

*Use Case:*  
Used to explore spatial relationships between colleges and Bluebike stations, assess accessibility, and study potential demand from students.

---

### 3. NOAA Weather Data
*Source:* National Oceanic and Atmospheric Administration (NOAA)  
*Description:* Contains daily weather information for Boston, including precipitation and temperature data.  
*Total Columns:* 4  

*Columns:*
•⁠  ⁠date: Observation date (YYYY-MM-DD)  
•⁠  ⁠PRCP: Daily precipitation (inches)  
•⁠  ⁠TMAX: Maximum daily temperature (°F)  
•⁠  ⁠TMIN: Minimum daily temperature (°F)  

*Use Case:*  
Used to study the impact of weather conditions (such as rain or temperature) on Bluebike ride frequency and duration.


## Data Sources
The Bluebikes data was pulled from the [Bluebikes System Data Website](https://s3.amazonaws.com/hubway-data/index.html). The Boston School and Colleges data is being queried from [Boston GIS Portal](https://gisportal.boston.gov/arcgis/rest/services/Education/OpenData/MapServer). To pull the NOAA data, we have a NOAA API key that needs to be used to access [NOAA website](https://www.ncei.noaa.gov/cdo-web/api/v2/data).

## Data Pre-processing 

The initial preprocessing of the data focused on ensuring data quality and consistency for downstream demand prediction modeling. The raw data, obtained from multiple sources and file formats, was consolidated into a single structured DataFrame stored in pickle format for efficient handling.

Missing values were addressed systematically: categorical columns were filled using the mode, while rows with missing values in critical numeric fields were removed. Additionally, duplicate records were identified and removed based on key identifiers to ensure uniqueness. These preprocessing steps provide a clean and reliable dataset, forming the foundation for further feature engineering and integration with auxiliary datasets to enable accurate demand prediction.

## Airflow Setup

This project uses Apache Airflow for orchestrating the Bluebikes data pipeline. The setup runs in Docker containers with the following components:

- **Airflow Webserver**: Web UI for monitoring and managing DAGs
- **Airflow Scheduler**: Executes tasks and manages DAG runs
- **Airflow Worker**: Executes tasks using CeleryExecutor
- **Airflow Triggerer**: Handles deferred tasks
- **PostgreSQL**: Metadata database for Airflow
- **Redis**: Message broker for Celery

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Airflow Ecosystem                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Webserver  │    │   Scheduler  │    │    Worker    │ │
│  │  (Port 8080) │    │  (DAG Exec)  │    │  (CeleryEx)  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │         │
│         └────────────────────┼────────────────────┘         │
│                              │                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  PostgreSQL  │    │    Redis     │    │  Triggerer   │ │
│  │  (Metadata)  │    │   (Broker)   │    │  (Deferred)  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
data_pipeline/
│
├── assets/
│   ├── bluebikes_correlation_pearson.png
│   ├── boston_clg_correlation_pearson.png
│   └── NOAA_weather_correlation_pearson.png
│
├── dags/
│   ├── data_pipeline_dag.py
│   └── test_discord_dag.py
│
├── data/
│   ├── processed/
│   │   ├── bluebikes/
│   │   │   ├── after_duplicates.pkl
│   │   │   ├── after_missing_data.pkl
│   │   │   ├── raw_data.pkl
│   │   │   └── station_id_mapping.pkl
│   │   ├── boston_clg/
│   │   │   ├── after_duplicates.pkl
│   │   │   ├── after_missing_data.pkl
│   │   │   └── raw_data.pkl
│   │   └── NOAA_weather/
│   │       ├── after_missing_data.pkl
│   │       └── raw_data.pkl
│   ├── .gitignore
│   ├── pipeline_metadata.json
│   ├── processed.dvc
│   ├── raw.dvc
│   └── read_log.csv
│
├── scripts/
│   ├── bluebikes_data_helpers/
│   │   ├── __pycache__/
│   │   ├── download_data.py
│   │   ├── normalize.py
│   │   ├── read_zips.py
│   │   ├── record_file.py
│   │   └── __init__.py
│   │
│   ├── logs/
│   │   ├── data_pipeline_20251027.log
│   │   └── data_pipeline_20251113.log
│   │
│   ├── school_noaa_data_collectors/
│   │   ├── __pycache__/
│   │   ├── BostonColleges.py
│   │   ├── NOAA_DataAcq.py
│   │   └── __init__.py
├── tests/
│   ├── test_data_collection.py
│   ├── test_data_loader.py
│   ├── test_duplicate_data.py
│   ├── test_missing_value.py
│   └── __init__.py
├── README.md
```

## Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose)
- At least 4GB RAM allocated to Docker
- Python 3.10+ (for local development)

## Initial Setup

### 1. Environment Variables

Create a `.env` file in the `data_pipeline/` directory:

```bash
cd data_pipeline
cp .env.example .env
```

Edit `.env` with your credentials:

```properties
# .env
NOAA_API_KEY=your_noaa_api_key_here
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_TOKEN
```

**Getting Discord Webhook:**
1. Open Discord → Server Settings → Integrations → Webhooks
2. Click "Create Webhook"
3. Name: `Airflow Alert Bot`
4. Select notification channel
5. Copy webhook URL

**Important:** Never commit `.env` to Git. It's already in `.gitignore`.

### 2. Docker Compose Configuration

The `docker-compose.yaml` includes:

```yaml
x-airflow-common:
  &airflow-common
  image: custom-airflow:latest
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session'
    DISCORD_WEBHOOK_URL: ${DISCORD_WEBHOOK_URL}
    NOAA_API_KEY: ${NOAA_API_KEY}
```

## Starting Airflow

### Method 1: Using Docker Compose (Recommended)

```bash
# Navigate to data_pipeline directory
cd data_pipeline

# Start all services in detached mode
docker compose up -d

# Wait for services to be healthy (~30 seconds)
sleep 30
```

### Method 2: First Time Initialization

If this is your first time running Airflow:

```bash
cd data_pipeline

# Initialize Airflow database
docker compose up airflow-init

# Start all services
docker compose up -d
```

### Verify Services are Running

```bash
# Check container status
docker compose ps

# Expected output:
# NAME                                STATUS              PORTS
# data_pipeline-airflow-scheduler-1   Up (healthy)        8080/tcp
# data_pipeline-airflow-webserver-1   Up (healthy)        0.0.0.0:8080->8080/tcp
# data_pipeline-airflow-worker-1      Up (healthy)        8080/tcp
# data_pipeline-postgres-1            Up (healthy)        5432/tcp
# data_pipeline-redis-1               Up (healthy)        6379/tcp
```

### Access Airflow UI

1. Open browser: `http://localhost:8080`
2. Default credentials:
   - **Username**: `airflow2`
   - **Password**: `airflow2`

## Stopping Airflow

### Graceful Shutdown

```bash
cd data_pipeline

# Stop all containers (preserves data)
docker compose down
```

### Complete Cleanup

```bash
cd data_pipeline

# Stop and remove all volumes (fresh start)
docker compose down -v

# Warning: This deletes all data including:
# - DAG run history
# - Task logs
# - Connection configurations
```

### Stop Specific Service

```bash
cd data_pipeline

# Stop only the scheduler
docker compose stop airflow-scheduler

# Restart it
docker compose start airflow-scheduler
```

## Health Check Script

### Location

```
Project/
├── airflow-health-check.sh    # Health monitoring script
└── data_pipeline/
    └── docker-compose.yaml
```

### Script Features

The `airflow-health-check.sh` script monitors:

- Docker daemon status
- Container health (all 6 services)
- Airflow webserver accessibility
- DAG parsing errors
- Recent scheduler errors

### Usage

```bash
# Run from project root
./airflow-health-check.sh
```

### Sample Output

```
Checking Docker and Airflow environment status
----------------------------------------------------------
Docker daemon is running.

Docker Compose Containers:
NAME                                STATUS                   HEALTH
data_pipeline-airflow-scheduler-1   Up 5 minutes (healthy)   healthy
data_pipeline-airflow-webserver-1   Up 5 minutes (healthy)   healthy
data_pipeline-postgres-1            Up 5 minutes (healthy)   healthy
----------------------------------------------------------

All 6 containers are healthy.

Checking Airflow webserver HTTP response on port 8080...
Airflow Web UI is reachable at http://localhost:8080 (HTTP 302)

----------------------------------------------------------
Checking DAG Health...
----------------------------------------------------------
Scanning for broken DAGs...
No broken DAGs found

Current DAGs:
dag_id                          | filepath                              | paused
================================|=======================================|========
data_pipeline_dag               | dags/data_pipeline_dag.py            | False
test_discord_notifications      | dags/test_discord_dag.py             | True

Recent scheduler errors:
No recent errors in scheduler logs

Health check complete.
```
## Common Operations

### Viewing Logs

```bash
cd data_pipeline

# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-webserver

# View last 100 lines
docker compose logs --tail=100 airflow-scheduler

# Search for errors
docker compose logs airflow-scheduler | grep -i error
```

### Triggering DAGs

```bash
cd data_pipeline

# Trigger main pipeline DAG
docker compose exec airflow-scheduler airflow dags trigger data_pipeline_dag

# Trigger test DAG
docker compose exec airflow-scheduler airflow dags trigger test_discord_notifications

# List all DAGs
docker compose exec airflow-scheduler airflow dags list

# Check for DAG import errors
docker compose exec airflow-scheduler airflow dags list-import-errors
```

### Managing DAGs

```bash
cd data_pipeline

# Pause a DAG
docker compose exec airflow-scheduler airflow dags pause data_pipeline_dag

# Unpause a DAG
docker compose exec airflow-scheduler airflow dags unpause data_pipeline_dag

# Test a DAG (dry run)
docker compose exec airflow-scheduler airflow dags test data_pipeline_dag 2025-10-26
```

### Debugging

```bash
cd data_pipeline

# Enter scheduler container
docker compose exec airflow-scheduler bash

# Check Python imports
docker compose exec airflow-scheduler python -c "from scripts.discord_notifier import send_discord_alert; print('  Import successful')"

# Check environment variables
docker compose exec airflow-scheduler env | grep DISCORD

# Test DAG parsing
docker compose exec airflow-scheduler python /opt/airflow/dags/data_pipeline_dag.py
```

### Restarting Services

```bash
cd data_pipeline

# Restart specific service
docker compose restart airflow-scheduler

# Restart all services
docker compose restart

# Rebuild and restart (after code changes)
docker compose down
docker compose up -d --build
```

## Discord Notifications

### Configuration

Discord notifications are configured via:

1. **Environment variable** in `.env`:
   ```
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   ```

2. **DAG configuration** in `data_pipeline_dag.py`:
   ```python
   from discord_notifier import send_discord_alert, send_dag_success_alert
   
   default_args = {
       'on_failure_callback': send_discord_alert,
   }
   
   with DAG(
       on_success_callback=send_dag_success_alert,
       on_failure_callback=send_discord_alert,
   ) as dag:
   ```

### Notification Types

- **Task Failure**: Sent when any individual task fails
- **DAG Success**: Sent when entire DAG run completes successfully
- **DAG Failure**: Sent when DAG run fails

### Notification Content

Each failure notification includes:

- DAG name
- Failed task name
- Retry attempt (e.g., 1/3)
- Execution date
- Task duration
- Error message
- Direct link to logs

### Testing Notifications

```bash
cd data_pipeline

# Trigger test DAG (includes intentional failure)
docker compose exec airflow-scheduler airflow dags trigger test_discord_notifications

# Check Discord channel for notifications
```

## Troubleshooting

### Issue: Containers Not Starting

**Solution:**
```bash
# Check logs for errors
docker compose logs

# Try clean restart
docker compose down -v
docker compose up -d
```

### Issue: "ModuleNotFoundError" in DAGs

**Solution:**
```bash
# Check Python path setup in scripts
# Each script should have at the top:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
```

### Issue: Discord Notifications Not Working

**Solution:**
```bash
# Verify environment variable is set
docker compose exec airflow-scheduler env | grep DISCORD_WEBHOOK_URL

# Test webhook directly
curl -X POST "$DISCORD_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test message"}'
```

### Issue: DAG Not Appearing in UI

**Solution:**
```bash
# Check for DAG import errors
docker compose exec airflow-scheduler airflow dags list-import-errors

# Check if DAG file is in correct location
docker compose exec airflow-scheduler ls -la /opt/airflow/dags/

# Wait for scheduler to detect (takes ~30 seconds)
```

### Issue: Web UI Not Accessible

**Solution:**
```bash
# Check webserver logs
docker compose logs airflow-webserver

# Check port binding
docker compose ps | grep webserver

# Restart webserver
docker compose restart airflow-webserver
```

## Performance Optimization

### Resource Allocation

Recommended Docker Desktop settings:
- **CPUs**: 4+
- **Memory**: 8GB+
- **Swap**: 2GB+

### Container Resources

In `docker-compose.yaml`, you can limit resources:

```yaml
services:
  airflow-worker:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Backup and Restore

### Backup DAG Runs and Metadata

```bash
cd data_pipeline

# Backup PostgreSQL database
docker compose exec postgres pg_dump -U airflow airflow > airflow_backup.sql

# Backup DAGs and scripts
tar -czf airflow_code_backup.tar.gz dags/ scripts/
```

### Restore from Backup

```bash
cd data_pipeline

# Restore database
docker compose exec -T postgres psql -U airflow airflow < airflow_backup.sql

# Restore code
tar -xzf airflow_code_backup.tar.gz
```
## Quick Reference Commands

```bash
# Start Airflow
cd data_pipeline && docker compose up -d

# Stop Airflow
cd data_pipeline && docker compose down

# Health check
./airflow-health-check.sh

# View logs
cd data_pipeline && docker compose logs -f

# Trigger DAG
cd data_pipeline && docker compose exec airflow-scheduler airflow dags trigger data_pipeline_dag

# Access UI
open http://localhost:8080
```

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
    - This script is responsible for loading the raw data from various sources into a standardized format. It reads the input files, performs initial schema validation, converts data types where required, and saves the processed output as a pickle file for downstream tasks. Also includes logging to track file loading status and handle errors.
- `scripts/duplicate_data.py`
    - This script identifies and handles duplicate records in the given dataset. It supports multiple duplicate-handling strategies including keeping the first/last occurrence and removing all duplicates. It can also auto-detect key columns for duplicate detection when not explicitly defined.
- `scripts/missing_value.py`
    - This script detects and treats missing values across the dataset. It logs a summary of missing data per information and applies configurable strategies such as dropping missing records and filling values using statistical imputation.


## Logging and Testing 
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

### Alerts

The data pipeline implements a comprehensive alerting system using Discord webhooks to provide real-time notifications about pipeline execution status. This ensures immediate visibility into pipeline health and enables rapid response to failures or successful completions.

#### Alert Integration with Airflow

The Discord notification system is integrated directly into the Airflow DAG through callback functions. Two primary alert types are implemented:

- **Failure Alerts**: Triggered when any task or the entire DAG fails
- **Success Alerts**: Triggered when the complete DAG execution finishes successfully

#### Implementation

Alerts are configured at two levels within the DAG:

**Task-Level Alerts** - Individual task failures trigger immediate notifications:
```python
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': send_discord_alert,
}
```

**DAG-Level Alerts** - Overall pipeline status notifications:
```python
with DAG(
    dag_id="data_pipeline_dag",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    on_success_callback=send_dag_success_alert,  
    on_failure_callback=send_discord_alert,      
) as dag:
```

#### Configuration

The Discord webhook URL is managed through environment variables for security and portability:

**Environment Variables (.env file)**:
```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url
NOAA_API_KEY=your_api_key_here
```

**Docker Compose Configuration**:
```yaml
environment:
  AIRFLOW__CORE__EXECUTOR: CeleryExecutor
  DISCORD_WEBHOOK_URL: ${DISCORD_WEBHOOK_URL}
  NOAA_API_KEY: ${NOAA_API_KEY}
```

#### Alert Functionality

The `discord_notifier` module provides two callback functions:

- `send_discord_alert(context)`: Sends detailed failure information including task ID, execution date, error messages, and logs
- `send_dag_success_alert(context)`: Sends confirmation messages when the entire pipeline completes successfully

This dual-level alerting approach allows for:
- Granular monitoring of individual pipeline components (data collection, processing, transformation)
- High-level overview of complete workflow execution status
- Immediate notification of failures enabling rapid troubleshooting
- Confirmation of successful daily pipeline runs


## Dataset Bias Analysis: 

### Why Observed Patterns Are Features, Not Biases
**Key Finding: Natural Demand Patterns**

Observed patterns in the dataset reflect real-world behavior and operational demand rather than problematic biases. Understanding these patterns is critical for accurate predictive modeling.

**Exploratory Analysis and Data Slicing**

Through extensive data slicing and exploratory data analysis (EDA), we investigated temporal, geographic, and user-specific trends to understand the true signals in the dataset. This process ensured that the patterns we observed were meaningful features rather than artifacts or biases.

**Temporal Patterns Reflect Real Demand**

Hourly, daily, and seasonal variations indicate actual user behavior. Peaks and troughs in usage are informative signals for forecasting, helping models learn when demand is high or low. Seasonal changes, such as reduced activity in colder months, are natural and should inform resource allocation rather than be corrected.

**Geographic Concentration Highlights True Hotspots**

Locations with higher activity represent real demand centers, such as transit hubs, workplaces, or educational institutions. These geographic patterns are essential features for predicting demand distribution across the service area.

**User Segmentation Supports Granular Predictions**

Differences in user types or categories reflect genuine behavior. For example, members may follow commute patterns, while casual users follow recreational patterns. Capturing these distinctions improves the model's ability to forecast different demand scenarios.

**Identifying Bias Requiring Mitigation**

While Bluebikes patterns are features, the college dataset exhibits severe geographic concentration and missing institutions, which could mislead demand predictions if used raw. 
Exploratory analysis revealed:
- 40% of colleges concentrated in Fenway/Kenmore, with major Cambridge institutions (MIT, Harvard) missing.
- Student populations skewed, with few large institutions represented.

***Impact on Modeling:***
- Overrepresentation of one area could teach false correlations.
- Underprediction at missing college locations.

***Mitigation Strategy:***
- Use generalized features like near_any_college instead of neighborhood-specific data.
- Engineer robust features such as student density or distance to nearest college.

**Why These Patterns Enhance Prediction**

- Causally meaningful: Peaks, lows, and geographic clusters reflect real-world factors (e.g., work schedules, weather, locations).

- Consistent over time: Repeated daily, weekly, and seasonal patterns indicate stable behavior rather than artifacts.

- Support operational efficiency: Understanding true demand enables better allocation of resources, avoiding over- or under-provisioning.

**Modeling Approach**

- Temporal features for hourly, daily, and seasonal demand trends.

- Location-specific features for geographic distribution.

- External factors (e.g., weather) to modulate seasonal patterns.


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
