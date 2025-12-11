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
        status = "✓ Ready" if has_data else "✗ Missing"
        print(f"{dataset}: {status}")
        
        if not has_data:
            all_ready = False
    
    if all_ready:
        print("\nAll datasets ready for model training!")
    else:
        print("\nWARNING: Some datasets missing!")
    
    return all_ready

def upload_data_to_gcs(**context):
    """
    Upload processed data folder to GCS bucket.
    
    This function uploads the entire data folder (raw and processed)
    to Google Cloud Storage, excluding:
    - DVC files (.dvc, .dvcignore)
    - Temp folders
    - __pycache__ directories
    - .gitignore files
    
    IMPORTANT: Large files (>50MB) use resumable uploads to handle
    network timeouts gracefully.
    """
    from google.cloud import storage
    from pathlib import Path
    import json
    
    log.info("="*60)
    log.info("UPLOADING DATA TO GOOGLE CLOUD STORAGE")
    log.info("="*60)
    
    # ===========================================
    # CONFIGURATION - Adjust these as needed
    # ===========================================
    
    # Files larger than this will use resumable upload (in bytes)
    LARGE_FILE_THRESHOLD = 50 * 1024 * 1024  # 50 MB
    
    # Maximum file size to upload (skip files larger than this)
    # Set to None to upload all files regardless of size
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB (set to None to disable)
    
    # Timeout for uploads (in seconds)
    UPLOAD_TIMEOUT = 600  # 10 minutes for large files
    
    # Get GCS configuration from environment
    bucket_name = os.environ.get("GCS_MODEL_BUCKET")
    prefix = os.environ.get("GCS_DATA_PREFIX", "data")
    
    if not bucket_name:
        log.warning("GCS_MODEL_BUCKET not set, skipping data upload")
        return {
            'uploaded': False, 
            'reason': 'GCS_MODEL_BUCKET environment variable not set'
        }
    
    # Define the local data directory
    data_dir = Path("/opt/airflow/data")
    
    if not data_dir.exists():
        log.error(f"Data directory not found: {data_dir}")
        return {'uploaded': False, 'reason': 'Data directory not found'}
    
    # Define patterns/folders to exclude
    EXCLUDE_PATTERNS = {
        # DVC related files
        '.dvc',
        '.dvcignore',
        'dvc.lock',
        'dvc.yaml',
        # Temp and cache
        'temp',
        'tmp',
        '__pycache__',
        '.cache',
        # Git related
        '.gitignore',
        '.git',
    }
    
    EXCLUDE_EXTENSIONS = {
        '.dvc',
        '.pyc',
        '.pyo',
    }
    
    def should_exclude(file_path: Path) -> bool:
        """Check if a file or directory should be excluded from upload."""
        for part in file_path.parts:
            if part in EXCLUDE_PATTERNS:
                return True
            if part.startswith('.') and part not in ['.']:
                return True
        
        if file_path.suffix in EXCLUDE_EXTENSIONS:
            return True
        
        return False
    
    def upload_large_file_resumable(bucket, blob_path, file_path, file_size):
        """
        Upload large files using resumable upload.
        
        Resumable uploads can handle network interruptions and timeouts
        by uploading in chunks and resuming from where they left off.
        """
        blob = bucket.blob(blob_path)
        
        # Configure for large file upload
        blob.chunk_size = 10 * 1024 * 1024  # 10 MB chunks
        
        log.info(f"  Starting resumable upload: {file_path.name} ({file_size / (1024*1024):.1f} MB)")
        
        try:
            # Use resumable upload with timeout
            blob.upload_from_filename(
                str(file_path),
                timeout=UPLOAD_TIMEOUT,
            )
            log.info(f"  Completed: {file_path.name}")
            return True
            
        except Exception as e:
            log.error(f"  Failed resumable upload for {file_path.name}: {e}")
            return False
    
    try:
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Track upload statistics
        uploaded_files = []
        skipped_files = []
        failed_files = []
        total_bytes = 0
        
        execution_date = context['ds_nodash']
        
        log.info(f"Scanning data directory: {data_dir}")
        log.info(f"Target bucket: gs://{bucket_name}/{prefix}/")
        log.info(f"Large file threshold: {LARGE_FILE_THRESHOLD / (1024*1024):.0f} MB")
        if MAX_FILE_SIZE:
            log.info(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.0f} MB")
        log.info(f"Excluding patterns: {EXCLUDE_PATTERNS}")
        
        # Collect all files first to show progress
        all_files = []
        for file_path in data_dir.rglob('*'):
            if file_path.is_dir():
                continue
            if should_exclude(file_path):
                skipped_files.append({'path': str(file_path), 'reason': 'excluded_pattern'})
                continue
            all_files.append(file_path)
        
        log.info(f"Found {len(all_files)} files to upload")
        
        # Upload files
        for idx, file_path in enumerate(all_files, 1):
            relative_path = file_path.relative_to(data_dir)
            blob_path = f"{prefix}/{relative_path}"
            file_size = file_path.stat().st_size
            
            # Check if file is too large
            if MAX_FILE_SIZE and file_size > MAX_FILE_SIZE:
                log.warning(f"  Skipping (too large): {relative_path} ({file_size / (1024*1024):.1f} MB)")
                skipped_files.append({
                    'path': str(relative_path),
                    'reason': 'exceeds_max_size',
                    'size_mb': round(file_size / (1024*1024), 2)
                })
                continue
            
            try:
                # Use resumable upload for large files
                if file_size > LARGE_FILE_THRESHOLD:
                    success = upload_large_file_resumable(bucket, blob_path, file_path, file_size)
                    if not success:
                        failed_files.append({
                            'path': str(relative_path),
                            'size_mb': round(file_size / (1024*1024), 2)
                        })
                        continue
                else:
                    # Regular upload for smaller files
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(str(file_path), timeout=120)
                
                uploaded_files.append({
                    'local_path': str(relative_path),
                    'gcs_path': blob_path,
                    'size_bytes': file_size
                })
                total_bytes += file_size
                
                # Progress logging every 10 files or for large files
                if idx % 10 == 0 or file_size > LARGE_FILE_THRESHOLD:
                    log.info(f"  Progress: {idx}/{len(all_files)} files uploaded")
                    
            except Exception as e:
                log.error(f"  Failed to upload {relative_path}: {e}")
                failed_files.append({
                    'path': str(relative_path),
                    'error': str(e),
                    'size_mb': round(file_size / (1024*1024), 2)
                })
        
        # Create and upload manifest
        manifest = {
            'upload_date': context['ds'],
            'upload_timestamp': datetime.now().isoformat(),
            'execution_date': execution_date,
            'airflow_run_id': context['run_id'],
            'total_files_uploaded': len(uploaded_files),
            'total_files_skipped': len(skipped_files),
            'total_files_failed': len(failed_files),
            'total_bytes': total_bytes,
            'total_mb': round(total_bytes / (1024 * 1024), 2),
            'uploaded_files': [f['local_path'] for f in uploaded_files],
            'skipped_files': skipped_files,
            'failed_files': failed_files,
        }
        
        manifest_blob_path = f"{prefix}/manifests/data_manifest_{execution_date}.json"
        manifest_blob = bucket.blob(manifest_blob_path)
        manifest_blob.upload_from_string(
            json.dumps(manifest, indent=2),
            content_type='application/json'
        )
        
        # Update latest manifest
        latest_manifest_path = f"{prefix}/manifests/latest_manifest.json"
        latest_blob = bucket.blob(latest_manifest_path)
        latest_blob.upload_from_string(
            json.dumps(manifest, indent=2),
            content_type='application/json'
        )
        
        # Log summary
        log.info("="*60)
        log.info("UPLOAD COMPLETE")
        log.info("="*60)
        log.info(f"Files uploaded: {len(uploaded_files)}")
        log.info(f"Files skipped: {len(skipped_files)}")
        log.info(f"Files failed: {len(failed_files)}")
        log.info(f"Total size uploaded: {total_bytes / (1024*1024):.2f} MB")
        log.info(f"Manifest: gs://{bucket_name}/{manifest_blob_path}")
        
        if failed_files:
            log.warning("Failed files:")
            for f in failed_files:
                log.warning(f"  - {f['path']} ({f.get('size_mb', '?')} MB)")
        
        # Push results to XCom
        context['task_instance'].xcom_push(key='gcs_upload_result', value={
            'uploaded': True,
            'bucket': bucket_name,
            'prefix': prefix,
            'files_count': len(uploaded_files),
            'failed_count': len(failed_files),
            'total_mb': round(total_bytes / (1024 * 1024), 2),
            'manifest_path': manifest_blob_path
        })
        
        return {
            'uploaded': True,
            'bucket': bucket_name,
            'prefix': prefix,
            'files_count': len(uploaded_files),
            'skipped_count': len(skipped_files),
            'failed_count': len(failed_files),
            'total_bytes': total_bytes,
            'manifest_path': manifest_blob_path
        }
        
    except Exception as e:
        log.error(f"Failed to upload data to GCS: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'uploaded': False,
            'error': str(e)
        }

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
        trigger_dag_id="drift_monitoring_dag",  # change if your drift DAG has a different dag_id
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