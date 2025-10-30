from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from datapipeline import (
    collect_bluebikes_data,
    collect_boston_college_data,
    collect_NOAA_Weather_data,
    DATASETS,
    process_assign_station_ids,
    process_missing,
    process_duplicates
)
from data_loader import load_data
from correlation_matrix import correlation_matrix
from discord_notifier import send_discord_alert, send_dag_success_alert


def load_and_process_dataset(dataset):
    """Load + process a single dataset."""
    try:
        print(f"\nProcessing dataset: {dataset['name']}")
        
        # Load data
        load_data(
            pickle_path=dataset["processed_path"],
            data_paths=[dataset["raw_path"]],
            dataset_name=dataset["name"]
        )

        preprocessing = dataset.get("preprocessing", {})

        # --- Assign station IDs first for Bluebikes ---
        if preprocessing.get("assign_station_ids", False):
            print(f"  -> Assigning station IDs for {dataset['name']}")
            process_assign_station_ids(dataset["processed_path"], dataset["processed_path"])

        # --- Missing values ---
        if "missing_config" in preprocessing:
            print(f"  -> Handling missing values for {dataset['name']}")
            process_missing(
                dataset["processed_path"], 
                dataset["processed_path"], 
                preprocessing["missing_config"],
                dataset.get("name")
            )

        # --- Duplicates ---
        if "duplicates" in preprocessing:
            print(f"  -> Handling duplicates for {dataset['name']}")
            process_duplicates(
                dataset["processed_path"], 
                dataset["processed_path"], 
                preprocessing["duplicates"]
            )

        # --- Correlation Matrix Generation ---
        print(f"  -> Generating correlation matrix for {dataset['name']}")
        if dataset.get("name") == "NOAA_weather":
            pkl_path = os.path.join(dataset["processed_path"], "after_missing_data.pkl")
        else:
            pkl_path = os.path.join(dataset["processed_path"], "after_duplicates.pkl")
        
        correlation_matrix(
            pkl_path=pkl_path,
            dataset_name=dataset["name"],
            method='pearson'
        )

        print(f"{dataset['name']} processed successfully.")

    except Exception as e:
        print(f"FAILED: {dataset['name']} - {e}")
        raise e


def collect_boston_college_wrapper():
    """Wrapper to collect boston college data to Airflow path."""
    output_path = "/opt/airflow/working_data/raw/boston_clg"
    collect_boston_college_data(output_path=output_path)


def collect_noaa_weather_wrapper():
    """Wrapper to collect NOAA weather data to Airflow path."""
    output_path = "/opt/airflow/working_data/raw/NOAA_weather"
    collect_NOAA_Weather_data(output_path=output_path)


def process_bluebikes_wrapper():
    """Wrapper to process bluebikes dataset."""
    dataset = next(d for d in DATASETS if d['name'] == 'bluebikes')
    load_and_process_dataset(dataset)


def process_boston_college_wrapper():
    """Wrapper to process boston_clg dataset."""
    dataset = next(d for d in DATASETS if d['name'] == 'boston_clg')
    load_and_process_dataset(dataset)


def process_noaa_weather_wrapper():
    """Wrapper to process NOAA_weather dataset."""
    dataset = next(d for d in DATASETS if d['name'] == 'NOAA_weather')
    load_and_process_dataset(dataset)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': send_discord_alert,
}

with DAG(
    dag_id="data_pipeline_dag",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    description='Data collection and processing pipeline with Discord notifications',
    tags=['bluebikes', 'data-pipeline', 'production'],
    on_success_callback=send_dag_success_alert,  
    on_failure_callback=send_discord_alert,      
) as dag:

    # Collection tasks
    collect_bluebikes = PythonOperator(
        task_id="collect_bluebikes",
        python_callable=collect_bluebikes_data,
        op_kwargs={
            "index_url": "https://s3.amazonaws.com/hubway-data/index.html",
            "years": ["2025"],
            "download_dir": "/opt/airflow/working_data/temp/bluebikes",
            "parquet_dir": "/opt/airflow/working_data/raw/bluebikes",
            "log_path": "/opt/airflow/working_data/read_log.csv",
        },
    )

    collect_boston_college = PythonOperator(
        task_id="collect_boston_college",
        python_callable=collect_boston_college_wrapper,
    )

    collect_noaa_weather = PythonOperator(
        task_id="collect_noaa_weather",
        python_callable=collect_noaa_weather_wrapper,
    )
    
    # Processing tasks
    process_bluebikes = PythonOperator(
        task_id="process_bluebikes",
        python_callable=process_bluebikes_wrapper,
    )

    process_boston_colleges = PythonOperator(
        task_id="process_boston_colleges",
        python_callable=process_boston_college_wrapper,
    )

    process_NOAA_weather = PythonOperator(
        task_id="process_NOAA_weather",
        python_callable=process_noaa_weather_wrapper,
    )
    
    # Dependencies
    collect_bluebikes >> process_bluebikes
    collect_boston_college >> process_boston_colleges
    collect_noaa_weather >> process_NOAA_weather