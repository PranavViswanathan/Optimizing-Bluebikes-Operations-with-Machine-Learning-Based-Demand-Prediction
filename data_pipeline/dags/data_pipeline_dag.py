from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))


from datapipeline import (
    collect_bluebikes_data,
    collect_boston_college_data,
    collect_NOAA_Weather_data,
    DATASETS
)


def load_and_process_dataset(dataset):
    """Load + process a single dataset."""
    from data_loader import load_data
    from missing_value import handle_missing
    from duplicate_data import handle_duplicates

    try:
        print(f"\nProcessing dataset: {dataset['name']}")

        # Load raw → processed
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



with DAG(
    dag_id="data_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:


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

    process_bluebikes = PythonOperator(
        task_id="process_bluebikes",
        python_callable=lambda: print("Processing BlueBikes data..."),
    )

    process_boston_colleges = PythonOperator(
        task_id="process_boston_colleges",
        python_callable=lambda: print("Processing Boston Colleges data..."),
    )

    process_NOAA_weather = PythonOperator(
        task_id="process_NOAA_weather",
        python_callable=lambda: print("Processing NOAA weather data..."),
    )


    t1 >> process_bluebikes
    t2 >> process_boston_colleges
    t3 >> process_NOAA_weather
