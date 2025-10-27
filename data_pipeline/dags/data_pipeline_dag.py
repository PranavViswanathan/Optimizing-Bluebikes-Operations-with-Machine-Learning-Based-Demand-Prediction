from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from datapipeline import (
    collect_bluebikes_data,
    collect_boston_college_data,
    collect_NOAA_Weather_data,
    DATASETS
)
from discord_notifier import send_discord_alert, send_dag_success_alert
from correlation_matrix import correlation_matrix
from assign_station_ids import assign_station_ids  # ✅ Import for Bluebikes IDs

def load_and_process_dataset(dataset):
    """Load + process a single dataset."""
    from data_loader import load_data
    from missing_value import handle_missing
    from duplicate_data import handle_duplicates

    try:
        print(f"\nProcessing dataset: {dataset['name']}")
        load_data(
            pickle_path=dataset["processed_path"],
            data_paths=[dataset["raw_path"]],
            dataset_name=dataset["name"]
        )

        preprocessing = dataset.get("preprocessing", {})

        if "missing_config" in preprocessing:
            print(f"  -> Handling missing values for {dataset['name']}")
            handle_missing(
                input_pickle_path=os.path.join(dataset["processed_path"], "raw_data.pkl"),
                output_pickle_path=os.path.join(dataset["processed_path"], "raw_data.pkl"),
                **preprocessing["missing_config"]
            )

        if "duplicates" in preprocessing:
            print(f"  -> Handling duplicates for {dataset['name']}")
            handle_duplicates(
                input_pickle_path=os.path.join(dataset["processed_path"], "raw_data.pkl"),
                output_pickle_path=os.path.join(dataset["processed_path"], "raw_data.pkl"),
                **preprocessing["duplicates"]
            )

        print(f"{dataset['name']} processed successfully.")

    except Exception as e:
        print(f"FAILED: {dataset['name']} - {e}")
        raise e


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
    description='Bluebikes data collection and processing pipeline with Discord notifications',
    tags=['bluebikes', 'data-pipeline', 'production'],
    on_success_callback=send_dag_success_alert,  
    on_failure_callback=send_discord_alert,      
) as dag:

    # ------------------- Data Collection -------------------
    t1 = PythonOperator(
        task_id="collect_bluebikes",
        python_callable=collect_bluebikes_data,
        op_kwargs={
            "index_url": "https://s3.amazonaws.com/hubway-data/index.html",
            "years": ["2023", "2024", "2025"],
            "download_dir": "/opt/airflow/data/bluebikes_zips",
            "parquet_dir": "/opt/airflow/data/parquet",
            "log_path": "/opt/airflow/data/read_log.csv",
        },
    )

    t2 = PythonOperator(
        task_id="collect_boston_college",
        python_callable=collect_boston_college_data,
    )

    t3 = PythonOperator(
        task_id="collect_noaa_weather",
        python_callable=collect_NOAA_Weather_data,
    )

    # ------------------- Assign Station IDs for Bluebikes -------------------
    assign_ids = PythonOperator(
        task_id="assign_station_ids",
        python_callable=lambda: assign_station_ids(
            input_pickle_path=os.path.join(DATASETS[0]["processed_path"], "raw_data.pkl"),
            output_pickle_path=os.path.join(DATASETS[0]["processed_path"], "raw_data.pkl")
        ),
    )

    # ------------------- Data Processing -------------------
    process_bluebikes = PythonOperator(
        task_id="process_bluebikes",
        python_callable=lambda: load_and_process_dataset(DATASETS[0]),
    )

    process_boston_colleges = PythonOperator(
        task_id="process_boston_colleges",
        python_callable=lambda: load_and_process_dataset(DATASETS[1]),
    )

    process_NOAA_weather = PythonOperator(
        task_id="process_NOAA_weather",
        python_callable=lambda: load_and_process_dataset(DATASETS[2]),
    )

    # ------------------- Correlation Matrix -------------------
    correlate_bluebikes = PythonOperator(
        task_id="correlate_bluebikes",
        python_callable=lambda: correlation_matrix(
            pkl_path=os.path.join(DATASETS[0]["processed_path"], "raw_data.pkl"),
            dataset_name="bluebikes",
            method="pearson"
        ),
    )

    correlate_boston_colleges = PythonOperator(
        task_id="correlate_boston_colleges",
        python_callable=lambda: correlation_matrix(
            pkl_path=os.path.join(DATASETS[1]["processed_path"], "raw_data.pkl"),
            dataset_name="boston_clg",
            method="pearson"
        ),
    )

    correlate_noaa_weather = PythonOperator(
        task_id="correlate_noaa_weather",
        python_callable=lambda: correlation_matrix(
            pkl_path=os.path.join(DATASETS[2]["processed_path"], "raw_data.pkl"),
            dataset_name="NOAA_weather",
            method="pearson"
        ),
    )

    # ------------------- DAG Dependencies -------------------
    t1 >> assign_ids >> process_bluebikes >> correlate_bluebikes
    t2 >> process_boston_colleges >> correlate_boston_colleges
    t3 >> process_NOAA_weather >> correlate_noaa_weather
