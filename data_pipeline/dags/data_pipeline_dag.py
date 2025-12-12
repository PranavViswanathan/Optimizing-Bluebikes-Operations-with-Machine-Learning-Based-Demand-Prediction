# dags/data_pipeline_dag.py
"""
Updated Data Pipeline DAG with incremental collection and smart preprocessing.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys
import os

from scripts.data_pipeline.data_manager import DataManager
from scripts.data_pipeline.incremental_bluebikes import collect_bluebikes_incremental
from scripts.data_pipeline.data_pipeline import (
    collect_boston_college_data,
    collect_NOAA_Weather_data,
    DATASETS,
    process_assign_station_ids,
    process_missing,
    process_duplicates
)
from scripts.data_pipeline.data_loader import load_data
from scripts.data_pipeline.correlation_matrix import correlation_matrix
from scripts.data_pipeline.discord_notifier import send_discord_alert, send_dag_success_alert


import logging

logging.basicConfig(level=logging.INFO, format='[DATA_PIPELINE] %(message)s')
log = logging.getLogger("data_pipeline")



def check_pipeline_status(**context):
    """
    First task: Check status of all datasets.
    Determines what work needs to be done.
    """
    dm = DataManager("/opt/airflow/data")
    dm.print_status()
    
    status = dm.get_status_report()
    context['task_instance'].xcom_push(key='pipeline_status', value=status)
    
    return status


def collect_bluebikes_task(**context):
    """Collect BlueBikes data incrementally."""
    result = collect_bluebikes_incremental(
        years=["2024", "2025"],
        data_dir="/opt/airflow/data"
    )
    
    context['task_instance'].xcom_push(key='collection_result', value=result)
    return result


def collect_noaa_wrapper(**context):
    """Collect NOAA weather data."""
    output_path = "/opt/airflow/data/raw/NOAA_weather"
    collect_NOAA_Weather_data(output_path=output_path)


def collect_boston_wrapper(**context):
    """Collect Boston colleges data."""
    output_path = "/opt/airflow/data/raw/boston_clg"
    collect_boston_college_data(output_path=output_path)


def preprocess_bluebikes(**context):
    """
    Preprocess BlueBikes data.
    Only runs if raw data exists and preprocessing is needed.
    """
    dm = DataManager("/opt/airflow/data")
    
    needs_preprocess, reason = dm.needs_preprocessing("bluebikes")
    
    if not needs_preprocess:
        print(f"BlueBikes preprocessing not needed: {reason}")
        return {"skipped": True, "reason": reason}
    
    print(f"BlueBikes preprocessing needed: {reason}")
    
    dataset_config = next(d for d in DATASETS if d['name'] == 'bluebikes')
    processed_path = dataset_config["processed_path"]
    preprocessing = dataset_config.get("preprocessing", {})
    
    # Step 1: Load all parquets into raw_data.pkl
    print("Loading raw parquet files...")
    df = dm.load_all_bluebikes_parquets()
    
    raw_pkl_path = dm.get_processed_pkl_path("bluebikes", "raw_data")
    raw_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(raw_pkl_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"Saved {len(df):,} rows to raw_data.pkl")
    
    # Step 2: Assign station IDs
    if preprocessing.get("assign_station_ids", False):
        print("Assigning station IDs...")
        process_assign_station_ids(processed_path, processed_path)
    
    # Step 3: Handle missing values
    if "missing_config" in preprocessing:
        print("Handling missing values...")
        process_missing(
            processed_path, 
            processed_path, 
            preprocessing["missing_config"], 
            "bluebikes"
        )
    
    # Step 4: Handle duplicates
    if "duplicates" in preprocessing:
        print("Handling duplicates...")
        process_duplicates(
            processed_path, 
            processed_path, 
            preprocessing["duplicates"]
        )
    
    # Step 5: Generate correlation matrix
    print("Generating correlation matrix...")
    pkl_path = os.path.join(processed_path, "after_duplicates.pkl")
    correlation_matrix(
        pkl_path=pkl_path,
        dataset_name="bluebikes",
        method='pearson'
    )
    
    # Update metadata
    dm.metadata["bluebikes"]["last_preprocessing"] = datetime.now().isoformat()
    dm.save_metadata()
    
    print("BlueBikes preprocessing complete!")
    return {"skipped": False, "reason": reason}


def preprocess_noaa(**context):
    """Preprocess NOAA weather data."""
    dm = DataManager("/opt/airflow/data")
    
    needs_preprocess, reason = dm.needs_preprocessing("NOAA_weather")
    
    if not needs_preprocess:
        print(f"NOAA preprocessing not needed: {reason}")
        return {"skipped": True, "reason": reason}
    
    print(f"NOAA preprocessing needed: {reason}")
    
    dataset_config = next(d for d in DATASETS if d['name'] == 'NOAA_weather')
    processed_path = dataset_config["processed_path"]
    preprocessing = dataset_config.get("preprocessing", {})
    
    # Load raw data
    load_data(
        pickle_path=processed_path,
        data_paths=[dataset_config["raw_path"]],
        dataset_name="NOAA_weather"
    )
    
    # Handle missing values
    if "missing_config" in preprocessing:
        print("Handling missing values...")
        process_missing(
            processed_path, 
            processed_path, 
            preprocessing["missing_config"], 
            "NOAA_weather"
        )
    
    # Handle duplicates
    if "duplicates" in preprocessing:
        print("Handling duplicates...")
        process_duplicates(
            processed_path, 
            processed_path, 
            preprocessing["duplicates"]
        )
    
    # Generate correlation matrix
    pkl_path = os.path.join(processed_path, "after_missing_data.pkl")
    correlation_matrix(
        pkl_path=pkl_path,
        dataset_name="NOAA_weather",
        method='pearson'
    )
    
    # Update metadata
    dm.metadata["NOAA_weather"]["last_preprocessing"] = datetime.now().isoformat()
    dm.save_metadata()
    
    print("NOAA preprocessing complete!")
    return {"skipped": False, "reason": reason}


def preprocess_boston(**context):
    """Preprocess Boston colleges data."""
    dm = DataManager("/opt/airflow/data")
    
    needs_preprocess, reason = dm.needs_preprocessing("boston_clg")
    
    if not needs_preprocess:
        print(f"Boston colleges preprocessing not needed: {reason}")
        return {"skipped": True, "reason": reason}
    
    print(f"Boston colleges preprocessing needed: {reason}")
    
    dataset_config = next(d for d in DATASETS if d['name'] == 'boston_clg')
    processed_path = dataset_config["processed_path"]
    preprocessing = dataset_config.get("preprocessing", {})
    
    # Load raw data
    load_data(
        pickle_path=processed_path,
        data_paths=[dataset_config["raw_path"]],
        dataset_name="boston_clg"
    )
    
    # Handle missing values
    if "missing_config" in preprocessing:
        process_missing(
            processed_path, 
            processed_path, 
            preprocessing["missing_config"], 
            "boston_clg"
        )
    
    # Handle duplicates
    if "duplicates" in preprocessing:
        process_duplicates(
            processed_path, 
            processed_path, 
            preprocessing["duplicates"]
        )
    
    # Generate correlation matrix
    pkl_path = os.path.join(processed_path, "after_duplicates.pkl")
    correlation_matrix(
        pkl_path=pkl_path,
        dataset_name="boston_clg",
        method='pearson'
    )
    
    # Update metadata
    dm.metadata["boston_clg"]["last_preprocessing"] = datetime.now().isoformat()
    dm.save_metadata()
    
    print("Boston colleges preprocessing complete!")
    return {"skipped": False, "reason": reason}


def final_status_check(**context):
    """Final task: Verify all data is ready."""
    dm = DataManager("/opt/airflow/data")
    
    print("\n" + "="*60)
    print("FINAL DATA STATUS")
    print("="*60)
    
    all_ready = True
    
    for dataset in ["bluebikes", "NOAA_weather"]:
        has_data = dm.has_processed_data(dataset)
        status = "  Ready" if has_data else "✗ Missing"
        print(f"{dataset}: {status}")
        
        if not has_data:
            all_ready = False
    
    if all_ready:
        print("\nAll datasets ready for model training!")
    else:
        print("\nWARNING: Some datasets missing!")
    
    return all_ready

def upload_data_to_gcs(**context):
    from google.cloud import storage
    from pathlib import Path
    import json
    
    LARGE_FILE_THRESHOLD = 50 * 1024 * 1024
    MAX_FILE_SIZE = 500 * 1024 * 1024
    UPLOAD_TIMEOUT = 600
    
    PRIORITY_FILES = {"after_duplicates.pkl", "after_missing_data.pkl", "raw_data.pkl"}
    PRIORITY_PATTERNS = ["after_duplicates.pkl", "_metadata.json"]
    
    EXCLUDE_PATTERNS = {'.dvc', '.dvcignore', 'dvc.lock', 'dvc.yaml', 'temp', 'tmp', '__pycache__', '.cache', '.gitignore', '.git'}
    EXCLUDE_EXTENSIONS = {'.dvc', '.pyc', '.pyo'}
    
    def is_priority(path):
        name = path.name
        return name in PRIORITY_FILES or any(p in name for p in PRIORITY_PATTERNS)
    
    def should_exclude(path):
        if any(part in EXCLUDE_PATTERNS or part.startswith('.') for part in path.parts):
            return True
        return path.suffix in EXCLUDE_EXTENSIONS
    
    def upload_large(bucket, blob_path, file_path, size):
        blob = bucket.blob(blob_path)
        blob.chunk_size = 10 * 1024 * 1024
        try:
            blob.upload_from_filename(str(file_path), timeout=UPLOAD_TIMEOUT)
            return True
        except Exception as e:
            log.error(f"Failed resumable upload {file_path.name}: {e}")
            return False
    
    bucket_name = os.environ.get("GCS_MODEL_BUCKET")
    prefix = os.environ.get("GCS_DATA_PREFIX", "data")
    
    if not bucket_name:
        log.warning("GCS_MODEL_BUCKET not set, skipping upload")
        return {'uploaded': False, 'reason': 'GCS_MODEL_BUCKET not set'}
    
    data_dir = Path("/opt/airflow/data")
    if not data_dir.exists():
        return {'uploaded': False, 'reason': 'Data directory not found'}
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    uploaded, skipped, failed = [], [], []
    total_bytes = 0
    
    all_files = [f for f in data_dir.rglob('*') if f.is_file() and not should_exclude(f)]
    log.info(f"Found {len(all_files)} files to upload to gs://{bucket_name}/{prefix}/")
    
    for idx, file_path in enumerate(all_files, 1):
        relative_path = file_path.relative_to(data_dir)
        blob_path = f"{prefix}/{relative_path}"
        size = file_path.stat().st_size
        priority = is_priority(file_path)
        
        if MAX_FILE_SIZE and size > MAX_FILE_SIZE and not priority:
            skipped.append({'path': str(relative_path), 'reason': 'too_large', 'size_mb': size / (1024*1024)})
            continue
        
        try:
            if size > LARGE_FILE_THRESHOLD:
                if not upload_large(bucket, blob_path, file_path, size):
                    failed.append({'path': str(relative_path), 'priority': priority})
                    continue
            else:
                bucket.blob(blob_path).upload_from_filename(str(file_path), timeout=120)
            
            uploaded.append({'path': str(relative_path), 'size': size, 'priority': priority})
            total_bytes += size
            
            if idx % 20 == 0:
                log.info(f"Progress: {idx}/{len(all_files)}")
        except Exception as e:
            log.error(f"Failed {relative_path}: {e}")
            failed.append({'path': str(relative_path), 'priority': priority})
    
    manifest = {
        'upload_date': context['ds'],
        'timestamp': datetime.now().isoformat(),
        'run_id': context['run_id'],
        'files_uploaded': len(uploaded),
        'files_skipped': len(skipped),
        'files_failed': len(failed),
        'total_mb': round(total_bytes / (1024 * 1024), 2),
    }
    
    manifest_path = f"{prefix}/manifests/data_manifest_{context['ds_nodash']}.json"
    bucket.blob(manifest_path).upload_from_string(json.dumps(manifest, indent=2), content_type='application/json')
    bucket.blob(f"{prefix}/manifests/latest_manifest.json").upload_from_string(json.dumps(manifest, indent=2), content_type='application/json')
    
    log.info(f"Upload complete: {len(uploaded)} uploaded, {len(skipped)} skipped, {len(failed)} failed, {total_bytes/(1024*1024):.1f} MB total")
    
    priority_failed = [f for f in failed if f.get('priority')]
    if priority_failed:
        log.error(f"PRIORITY FILES FAILED: {[f['path'] for f in priority_failed]}")
    
    return {'uploaded': True, 'files_count': len(uploaded), 'failed_count': len(failed), 'total_mb': total_bytes/(1024*1024)}

def should_run_drift_monitoring(**context):
    """
    Decide whether to trigger drift monitoring based on whether
    new BlueBikes data was added in this run.
    """
    ti = context["task_instance"]
    result = ti.xcom_pull(
        task_ids="collect_bluebikes",
        key="collection_result"
    ) or {}

    rows_added = result.get("rows_added", 0)
    zips_processed = result.get("zips_processed", 0)

    # If nothing new, skip drift
    if (rows_added or 0) > 0 or (zips_processed or 0) > 0:
        print(f"New data detected: rows_added={rows_added}, zips_processed={zips_processed}")
        return "trigger_drift_monitoring"
    else:
        print("No new BlueBikes data; skipping drift monitoring.")
        return "skip_drift_monitoring"



# DAG Definition
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
    description='Incremental data collection, preprocessing, and GCS backup pipeline',
    tags=['bluebikes', 'data-pipeline', 'production', 'gcs'],
    on_success_callback=send_dag_success_alert,
    on_failure_callback=send_discord_alert,
) as dag:
    
    # Task 1: Check current status
    check_status = PythonOperator(
        task_id="check_status",
        python_callable=check_pipeline_status,
    )
    
    # Task 2: Collection tasks (parallel)
    collect_bluebikes = PythonOperator(
        task_id="collect_bluebikes",
        python_callable=collect_bluebikes_task,
    )
    
    collect_noaa = PythonOperator(
        task_id="collect_noaa",
        python_callable=collect_noaa_wrapper,
    )
    
    collect_boston = PythonOperator(
        task_id="collect_boston",
        python_callable=collect_boston_wrapper,
    )
    
    # Task 3: Preprocessing tasks
    preprocess_bb = PythonOperator(
        task_id="preprocess_bluebikes",
        python_callable=preprocess_bluebikes,
    )
    
    preprocess_noaa_task = PythonOperator(
        task_id="preprocess_noaa",
        python_callable=preprocess_noaa,
    )
    
    preprocess_boston_task = PythonOperator(
        task_id="preprocess_boston",
        python_callable=preprocess_boston,
    )
    
    # Task 4: Final verification
    final_check = PythonOperator(
        task_id="final_status_check",
        python_callable=final_status_check,
    )
    
    upload_to_gcs = PythonOperator(
        task_id="upload_data_to_gcs",
        python_callable=upload_data_to_gcs,
    )

    branch_drift = BranchPythonOperator(
        task_id="branch_drift_monitoring",
        python_callable=should_run_drift_monitoring,
    )

    # If data changed: trigger the drift monitoring DAG
    trigger_drift = TriggerDagRunOperator(
        task_id="trigger_drift_monitoring",
        trigger_dag_id="drift_monitoring_dag",  
        reset_dag_run=True,
        wait_for_completion=False,
    )

    # If no data change: do nothing (no-op)
    skip_drift = EmptyOperator(
        task_id="skip_drift_monitoring"
    )

    # Dependencies
    check_status >> [collect_bluebikes, collect_noaa, collect_boston]
    
    collect_bluebikes >> preprocess_bb
    collect_noaa >> preprocess_noaa_task
    collect_boston >> preprocess_boston_task
    
    [preprocess_bb, preprocess_noaa_task, preprocess_boston_task] >> final_check

    final_check >> upload_to_gcs
    upload_to_gcs >> branch_drift
    branch_drift >> [trigger_drift, skip_drift]


# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime, timedelta
# import sys
# import os

# # sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

# from scripts.data_pipeline.data_pipeline import (
#     collect_bluebikes_data,
#     collect_boston_college_data,
#     collect_NOAA_Weather_data,
#     DATASETS,
#     process_assign_station_ids,
#     process_missing,
#     process_duplicates
# )
# from scripts.data_pipeline.data_loader import load_data
# from scripts.data_pipeline.correlation_matrix import correlation_matrix
# from scripts.data_pipeline.discord_notifier import send_discord_alert, send_dag_success_alert


# def load_and_process_dataset(dataset):
#     """Load + process a single dataset."""
#     try:
#         print(f"\nProcessing dataset: {dataset['name']}")
        
#         # Load data
#         load_data(
#             pickle_path=dataset["processed_path"],
#             data_paths=[dataset["raw_path"]],
#             dataset_name=dataset["name"]
#         )

#         preprocessing = dataset.get("preprocessing", {})

#         # --- Assign station IDs first for Bluebikes ---
#         if preprocessing.get("assign_station_ids", False):
#             print(f"  -> Assigning station IDs for {dataset['name']}")
#             process_assign_station_ids(dataset["processed_path"], dataset["processed_path"])

#         # --- Missing values ---
#         if "missing_config" in preprocessing:
#             print(f"  -> Handling missing values for {dataset['name']}")
#             process_missing(
#                 dataset["processed_path"], 
#                 dataset["processed_path"], 
#                 preprocessing["missing_config"],
#                 dataset.get("name")
#             )

#         # --- Duplicates ---
#         if "duplicates" in preprocessing:
#             print(f"  -> Handling duplicates for {dataset['name']}")
#             process_duplicates(
#                 dataset["processed_path"], 
#                 dataset["processed_path"], 
#                 preprocessing["duplicates"]
#             )

#         # --- Correlation Matrix Generation ---
#         print(f"  -> Generating correlation matrix for {dataset['name']}")
#         if dataset.get("name") == "NOAA_weather":
#             pkl_path = os.path.join(dataset["processed_path"], "after_missing_data.pkl")
#         else:
#             pkl_path = os.path.join(dataset["processed_path"], "after_duplicates.pkl")
        
#         correlation_matrix(
#             pkl_path=pkl_path,
#             dataset_name=dataset["name"],
#             method='pearson'
#         )

#         print(f"{dataset['name']} processed successfully.")

#     except Exception as e:
#         print(f"FAILED: {dataset['name']} - {e}")
#         raise e


# def collect_boston_college_wrapper():
#     """Wrapper to collect boston college data to Airflow path."""
#     output_path = "/opt/airflow/data/raw/boston_clg"
#     collect_boston_college_data(output_path=output_path)


# def collect_noaa_weather_wrapper():
#     """Wrapper to collect NOAA weather data to Airflow path."""
#     output_path = "/opt/airflow/data/raw/NOAA_weather"
#     collect_NOAA_Weather_data(output_path=output_path)


# def process_bluebikes_wrapper():
#     """Wrapper to process bluebikes dataset."""
#     dataset = next(d for d in DATASETS if d['name'] == 'bluebikes')
#     load_and_process_dataset(dataset)


# def process_boston_college_wrapper():
#     """Wrapper to process boston_clg dataset."""
#     dataset = next(d for d in DATASETS if d['name'] == 'boston_clg')
#     load_and_process_dataset(dataset)


# def process_noaa_weather_wrapper():
#     """Wrapper to process NOAA_weather dataset."""
#     dataset = next(d for d in DATASETS if d['name'] == 'NOAA_weather')
#     load_and_process_dataset(dataset)


# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 2,
#     'retry_delay': timedelta(minutes=5),
#     'on_failure_callback': send_discord_alert,
# }

# with DAG(
#     dag_id="data_pipeline_dag",
#     default_args=default_args,
#     start_date=datetime(2025, 1, 1),
#     schedule_interval="@daily",
#     catchup=False,
#     description='Data collection and processing pipeline with Discord notifications',
#     tags=['bluebikes', 'data-pipeline', 'production'],
#     on_success_callback=send_dag_success_alert,  
#     on_failure_callback=send_discord_alert,      
# ) as dag:

#     # Collection tasks
#     collect_bluebikes = PythonOperator(
#         task_id="collect_bluebikes",
#         python_callable=collect_bluebikes_data,
#         op_kwargs={
#             "index_url": "https://s3.amazonaws.com/hubway-data/index.html",
#             "years": ["2024", "2025"],
#             "download_dir": "/opt/airflow/data/temp/bluebikes",
#             "parquet_dir": "/opt/airflow/data/raw/bluebikes",
#             "log_path": "/opt/airflow/data/read_log.csv",
#         },
#     )

#     collect_boston_college = PythonOperator(
#         task_id="collect_boston_college",
#         python_callable=collect_boston_college_wrapper,
#     )

#     collect_noaa_weather = PythonOperator(
#         task_id="collect_noaa_weather",
#         python_callable=collect_noaa_weather_wrapper,
#     )
    
#     # Processing tasks
#     process_bluebikes = PythonOperator(
#         task_id="process_bluebikes",
#         python_callable=process_bluebikes_wrapper,
#     )

#     process_boston_colleges = PythonOperator(
#         task_id="process_boston_colleges",
#         python_callable=process_boston_college_wrapper,
#     )

#     process_NOAA_weather = PythonOperator(
#         task_id="process_NOAA_weather",
#         python_callable=process_noaa_weather_wrapper,
#     )
    
#     # Dependencies
#     collect_bluebikes >> process_bluebikes
#     collect_boston_college >> process_boston_colleges
#     collect_noaa_weather >> process_NOAA_weather