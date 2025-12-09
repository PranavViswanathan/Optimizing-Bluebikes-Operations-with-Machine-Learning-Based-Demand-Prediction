"""
Drift Monitoring DAG for BlueBikes Model Pipeline

Updated to match new data splits:
- Training: Jan 2024 - Sep 2025 (baseline reference)
- Test: Oct 2025 - Nov 2025
- Production: Dec 2025+ (data for drift monitoring)

For demo purposes (before Dec 2025 data exists):
- Can compare Test data against Training baseline
- Can inject artificial drift to demonstrate detection
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import sys
import os
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='[MONITORING] %(message)s')
log = logging.getLogger("drift_monitoring")

from scripts.data_pipeline.discord_notifier import send_discord_alert, send_dag_success_alert


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_prerequisites(**context):
    """Check if all prerequisites for monitoring exist."""
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline/monitoring')
    
    from monitoring_config import (
        get_baseline_path,
        PRODUCTION_MODEL_PATH,
        PRODUCTION_METADATA_PATH,
        PROCESSED_DATA_PATH,
    )
    
    log.info("="*60)
    log.info("CHECKING MONITORING PREREQUISITES")
    log.info("="*60)
    
    issues = []
    
    baseline_path = get_baseline_path()
    if baseline_path.exists():
        log.info(f"✓ Baseline found: {baseline_path}")
    else:
        issues.append(f"Baseline not found at {baseline_path}")
        log.warning(f"✗ Baseline missing: {baseline_path}")
    
    if PRODUCTION_MODEL_PATH.exists():
        log.info(f"✓ Production model found: {PRODUCTION_MODEL_PATH}")
    else:
        issues.append(f"Production model not found at {PRODUCTION_MODEL_PATH}")
        log.warning(f"✗ Model missing: {PRODUCTION_MODEL_PATH}")
    
    if PRODUCTION_METADATA_PATH.exists():
        log.info(f"✓ Model metadata found: {PRODUCTION_METADATA_PATH}")
        with open(PRODUCTION_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        log.info(f"  Model version: {metadata.get('version', 'N/A')}")
    else:
        issues.append(f"Model metadata not found at {PRODUCTION_METADATA_PATH}")
    
    if PROCESSED_DATA_PATH.exists():
        log.info(f"✓ Processed data found: {PROCESSED_DATA_PATH}")
    else:
        issues.append(f"Processed data not found at {PROCESSED_DATA_PATH}")
    
    context['task_instance'].xcom_push(key='prerequisites_ok', value=len(issues) == 0)
    context['task_instance'].xcom_push(key='issues', value=issues)
    
    if issues:
        log.error(f"Prerequisites check failed with {len(issues)} issue(s)")
        raise ValueError(f"Prerequisites not met: {issues}")
    
    return True


def load_current_data(**context):
    """
    Load current data for drift analysis.
    
    Logic:
    1. If genuinely new data exists (after test period) → use that
    2. If demo mode → use test data as "production" simulation
    3. If inject_drift=True → artificially modify data for demo
    """
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline')
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline/monitoring')

    import pandas as pd
    import numpy as np
    from feature_generation import load_and_prepare_data
    from monitoring_config import get_config, DATA_SPLITS
    
    log.info("="*60)
    log.info("LOADING DATA FOR DRIFT ANALYSIS")
    log.info("="*60)
    
    config = get_config()
    
    # Load all data
    X, y, feature_columns = load_and_prepare_data()
    X["date"] = pd.to_datetime(X["date"])
    
    log.info(f"Total data points: {len(X)}")
    log.info(f"Date range in data: {X['date'].min()} to {X['date'].max()}")
    
    # Get date boundaries from config
    test_end = config.data_splits.get_test_end()
    production_start = config.data_splits.get_production_start()
    test_start = config.data_splits.get_test_start()
    
    log.info(f"\nData Split Configuration:")
    log.info(f"  Training: {config.data_splits.train_start} to {config.data_splits.train_end}")
    log.info(f"  Test: {config.data_splits.test_start} to {config.data_splits.test_end}")
    log.info(f"  Production starts: {config.data_splits.production_start}")
    
    # Check if we have genuine production data (after test period)
    max_date = X["date"].max()
    has_production_data = max_date > test_end
    
    # Check DAG conf for demo/injection settings
    dag_conf = context.get('dag_run').conf or {}
    demo_mode = dag_conf.get('demo_mode', False)
    inject_drift = dag_conf.get('inject_drift', False)
    drift_type = dag_conf.get('drift_type', 'all')
    
    if has_production_data and not demo_mode:
        # =====================================================
        # REAL PRODUCTION MODE: Use data after test period
        # =====================================================
        log.info("\n>>> PRODUCTION MODE: Using real post-test data")
        
        # Get the latest month of production data
        production_mask = X["date"] > test_end
        X_production = X.loc[production_mask].copy()
        
        if len(X_production) == 0:
            log.warning("No production data found, falling back to demo mode")
            demo_mode = True
        else:
            # Use latest month
            latest_month = X_production["date"].max().to_period('M')
            X_production["month_period"] = X_production["date"].dt.to_period('M')
            current_mask = X_production["month_period"] == latest_month
            
            X_current = X_production.loc[current_mask].drop(columns=["month_period"]).copy()
            y_current = y.loc[X_current.index].copy()
            
            log.info(f"Production data month: {latest_month}")
            log.info(f"Samples: {len(X_current)}")
    
    if not has_production_data or demo_mode:
        # =====================================================
        # DEMO MODE: Use test data as simulated "production"
        # =====================================================
        log.info("\n>>> DEMO MODE: Using test data as simulated production")
        log.info("(No real production data available yet)")
        
        test_mask = (X["date"] >= test_start) & (X["date"] <= test_end)
        X_current = X.loc[test_mask].copy()
        y_current = y.loc[test_mask].copy()
        
        log.info(f"Using test period: {test_start.date()} to {test_end.date()}")
        log.info(f"Samples: {len(X_current)}")
    
    # =====================================================
    # DRIFT INJECTION (for demonstration)
    # =====================================================
    if inject_drift:
        log.info("\n" + "="*60)
        log.info(f"INJECTING ARTIFICIAL DRIFT: {drift_type}")
        log.info("="*60)
        
        if drift_type in ["temperature", "all"]:
            if 'temp_avg' in X_current.columns:
                original_mean = X_current['temp_avg'].mean()
                X_current['temp_avg'] = X_current['temp_avg'] + 10
                log.info(f"  ✓ Temperature drift: +10°C (mean: {original_mean:.1f} → {X_current['temp_avg'].mean():.1f})")
        
        if drift_type in ["demand", "all"]:
            if 'rides_last_hour' in X_current.columns:
                X_current['rides_last_hour'] = X_current['rides_last_hour'] * 1.8
                log.info("  ✓ Demand drift: +80%")
            if 'rides_rolling_3h' in X_current.columns:
                X_current['rides_rolling_3h'] = X_current['rides_rolling_3h'] * 1.8
        
        if drift_type in ["temporal", "all"]:
            if 'is_weekend' in X_current.columns:
                flip_mask = (X_current['is_weekend'] == 0) & (np.random.random(len(X_current)) < 0.4)
                X_current.loc[flip_mask, 'is_weekend'] = 1
                log.info("  ✓ Temporal drift: weekend pattern shift")
        
        if drift_type in ["hour", "all"]:
            if 'hour' in X_current.columns:
                X_current['hour'] = (X_current['hour'] + 3) % 24
                # Also update cyclical features
                X_current['hour_sin'] = np.sin(2 * np.pi * X_current['hour'] / 24)
                X_current['hour_cos'] = np.cos(2 * np.pi * X_current['hour'] / 24)
                log.info("  ✓ Hour drift: +3 hour shift")
        
        log.info("Drift injection complete!")
    
    # Validate we have enough data
    min_samples = 100
    if len(X_current) < min_samples:
        log.warning(f"Only {len(X_current)} samples (minimum recommended: {min_samples})")
    
    # Save to temp file
    temp_path = '/tmp/current_data_for_monitoring.pkl'
    X_current.to_pickle(temp_path)
    
    # Push metadata to XCom
    context['task_instance'].xcom_push(key='current_data_path', value=temp_path)
    context['task_instance'].xcom_push(key='n_samples', value=len(X_current))
    context['task_instance'].xcom_push(key='demo_mode', value=demo_mode or not has_production_data)
    context['task_instance'].xcom_push(key='drift_injected', value=inject_drift)
    context['task_instance'].xcom_push(key='date_range', value=f"{X_current['date'].min()} to {X_current['date'].max()}")
    
    log.info(f"\nData prepared: {len(X_current)} samples")
    log.info(f"Saved to: {temp_path}")
    
    return len(X_current)


def run_drift_detection(**context):
    """Run Evidently AI drift detection."""
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline')
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline/monitoring')

    import pandas as pd
    from drift_detector import EvidentlyDriftDetector
    from monitoring_config import get_config
    
    log.info("="*60)
    log.info("RUNNING EVIDENTLY AI DRIFT DETECTION")
    log.info("="*60)
    
    ti = context['task_instance']
    current_data_path = ti.xcom_pull(task_ids='load_current_data', key='current_data_path')
    demo_mode = ti.xcom_pull(task_ids='load_current_data', key='demo_mode')
    drift_injected = ti.xcom_pull(task_ids='load_current_data', key='drift_injected')
    date_range = ti.xcom_pull(task_ids='load_current_data', key='date_range')
    
    current_data = pd.read_pickle(current_data_path)
    log.info(f"Loaded {len(current_data)} samples for analysis")
    log.info(f"Date range: {date_range}")
    log.info(f"Demo mode: {demo_mode}")
    log.info(f"Drift injected: {drift_injected}")
    
    # Initialize detector - loads baseline (training data reference)
    detector = EvidentlyDriftDetector()
    detector.load_reference_data()  # Loads training data baseline
    detector.load_model()
    
    # Log what we're comparing
    if hasattr(detector, 'reference_data') and detector.reference_data is not None:
        log.info(f"Reference (baseline): {len(detector.reference_data)} samples from training data")
    
    # Run full monitoring suite
    report = detector.run_full_monitoring(
        current_data=current_data,
        current_actuals=None,
        generate_html=True,
        save_reports=True
    )
    
    # Add context to report
    report['context'] = {
        'demo_mode': demo_mode,
        'drift_injected': drift_injected,
        'current_data_range': date_range,
        'baseline_source': 'training_data'
    }
    
    # Push results to XCom
    ti.xcom_push(key='drift_report', value={
        'overall_status': report.get('overall_status'),
        'recommended_action': report.get('recommended_action'),
        'alerts': report.get('alerts', []),
        'data_drift_detected': report.get('data_drift', {}).get('dataset_drift', False),
        'n_drifted_features': report.get('data_drift', {}).get('n_drifted_features', 0),
        'drift_share': report.get('data_drift', {}).get('drift_share', 0),
        'prediction_drift_detected': report.get('prediction_drift', {}).get('drift_detected', False),
        'html_report_path': report.get('data_drift', {}).get('html_report_path', ''),
        'demo_mode': demo_mode,
        'drift_injected': drift_injected,
    })
    
    log.info("="*60)
    log.info("DRIFT DETECTION COMPLETE")
    log.info("="*60)
    log.info(f"Overall Status: {report.get('overall_status')}")
    log.info(f"Data Drift Detected: {report.get('data_drift', {}).get('dataset_drift', False)}")
    log.info(f"Features Drifted: {report.get('data_drift', {}).get('n_drifted_features', 0)}")
    log.info(f"Recommended Action: {report.get('recommended_action')}")
    
    if demo_mode:
        log.info("\n⚠️  DEMO MODE: Results based on test data vs training baseline")
    if drift_injected:
        log.info("⚠️  DRIFT INJECTED: Artificial drift was added for demonstration")
    
    return report.get('overall_status')


def evaluate_drift_action(**context):
    """Decide what action to take based on drift detection results."""
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    log.info("="*60)
    log.info("EVALUATING DRIFT DETECTION RESULTS")
    log.info("="*60)
    
    overall_status = drift_report.get('overall_status', 'UNKNOWN')
    recommended_action = drift_report.get('recommended_action', 'none')
    demo_mode = drift_report.get('demo_mode', False)
    drift_injected = drift_report.get('drift_injected', False)
    
    log.info(f"Status: {overall_status}")
    log.info(f"Recommended: {recommended_action}")
    log.info(f"Demo Mode: {demo_mode}")
    
    if drift_report.get('alerts'):
        log.info("Alerts:")
        for alert in drift_report['alerts']:
            log.info(f"  ⚠ {alert}")
    
    # In demo mode with injected drift, still trigger alerts but maybe skip retraining
    if demo_mode and drift_injected:
        log.info("\n⚠️  Demo mode with injected drift - will send alerts but skip actual retraining")
        if overall_status == 'CRITICAL':
            return 'send_critical_alert'
        elif overall_status == 'WARNING':
            return 'send_warning_alert'
        else:
            return 'log_healthy_status'
    
    # Normal operation
    if overall_status == 'CRITICAL' or recommended_action == 'retrain':
        log.info("→ Branching to: send_critical_alert + trigger_retraining")
        return 'send_critical_alert'
    elif overall_status == 'WARNING':
        log.info("→ Branching to: send_warning_alert")
        return 'send_warning_alert'
    else:
        log.info("→ Branching to: log_healthy_status")
        return 'log_healthy_status'


def send_critical_alert(**context):
    """Send critical alert via Discord."""
    import requests
    
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    
    if not webhook_url:
        log.warning("DISCORD_WEBHOOK_URL not set, skipping alert")
        return
    
    alerts = drift_report.get('alerts', [])
    n_drifted = drift_report.get('n_drifted_features', 0)
    drift_share = drift_report.get('drift_share', 0)
    demo_mode = drift_report.get('demo_mode', False)
    
    title = "🚨 CRITICAL: Model Drift Detected"
    if demo_mode:
        title += " (DEMO)"
    
    message = {
        "embeds": [{
            "title": title,
            "description": "Significant drift detected in BlueBikes demand prediction model.",
            "color": 15158332,
            "fields": [
                {"name": "Status", "value": drift_report.get('overall_status', 'CRITICAL'), "inline": True},
                {"name": "Drifted Features", "value": f"{n_drifted} ({drift_share:.1%})", "inline": True},
                {"name": "Action", "value": "Retraining triggered" if not demo_mode else "Demo - no retraining", "inline": True},
                {"name": "Alerts", "value": "\n".join(alerts[:5]) if alerts else "See report", "inline": False}
            ],
            "footer": {"text": f"Drift Monitoring | {context['ds']}"},
            "timestamp": datetime.utcnow().isoformat()
        }]
    }
    
    try:
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
        log.info("Critical alert sent to Discord")
    except Exception as e:
        log.error(f"Failed to send Discord alert: {e}")


def send_warning_alert(**context):
    """Send warning alert via Discord."""
    import requests
    
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    
    if not webhook_url:
        log.warning("DISCORD_WEBHOOK_URL not set, skipping alert")
        return
    
    n_drifted = drift_report.get('n_drifted_features', 0)
    demo_mode = drift_report.get('demo_mode', False)
    
    title = "⚠️ WARNING: Minor Drift Detected"
    if demo_mode:
        title += " (DEMO)"
    
    message = {
        "embeds": [{
            "title": title,
            "description": "Some drift detected in BlueBikes model. Monitoring closely.",
            "color": 16776960,
            "fields": [
                {"name": "Drifted Features", "value": str(n_drifted), "inline": True},
                {"name": "Action", "value": "Continued monitoring", "inline": True},
            ],
            "footer": {"text": f"Drift Monitoring | {context['ds']}"}
        }]
    }
    
    try:
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
        log.info("Warning alert sent to Discord")
    except Exception as e:
        log.error(f"Failed to send Discord alert: {e}")


def log_healthy_status(**context):
    """Log healthy status when no significant drift detected."""
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    log.info("="*60)
    log.info("MODEL STATUS: HEALTHY")
    log.info("="*60)
    log.info("No significant drift detected.")
    log.info(f"Data drift: {drift_report.get('data_drift_detected', False)}")
    log.info(f"Prediction drift: {drift_report.get('prediction_drift_detected', False)}")
    
    if drift_report.get('demo_mode'):
        log.info("(Demo mode - comparing test data vs training baseline)")
    
    return "healthy"


def check_retraining_cooldown(**context):
    """Check if we're within the retraining cooldown period."""
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline/monitoring')

    from monitoring_config import get_config, LOGS_DIR
    from datetime import datetime, timedelta
    
    # Check if this is demo mode - skip retraining in demo
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    if drift_report.get('demo_mode') and drift_report.get('drift_injected'):
        log.info("Demo mode with injected drift - skipping actual retraining")
        context['task_instance'].xcom_push(key='skip_retraining', value=True)
        return False
    
    config = get_config()
    cooldown_hours = config.retraining.retraining_cooldown_hours
    max_per_week = config.retraining.max_retrains_per_week
    
    log_path = LOGS_DIR / "retraining_history.json"
    
    if not log_path.exists():
        log.info("No retraining history found - OK to retrain")
        return True
    
    with open(log_path, 'r') as f:
        history = json.load(f)
    
    last_retrain = history.get('last_retraining')
    if last_retrain:
        last_time = datetime.fromisoformat(last_retrain)
        cooldown_end = last_time + timedelta(hours=cooldown_hours)
        
        if datetime.now() < cooldown_end:
            log.warning(f"Within cooldown period. Next retraining after {cooldown_end}")
            return False
    
    week_ago = datetime.now() - timedelta(days=7)
    recent_retrains = [
        r for r in history.get('retraining_runs', [])
        if datetime.fromisoformat(r['timestamp']) > week_ago
    ]
    
    if len(recent_retrains) >= max_per_week:
        log.warning(f"Weekly limit reached ({max_per_week} retrains)")
        return False
    
    return True


def log_retraining_triggered(**context):
    """Log that retraining was triggered."""
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline/monitoring')
    
    from monitoring_config import LOGS_DIR
    
    ti = context['task_instance']
    
    # Check if we should skip
    if ti.xcom_pull(task_ids='check_retraining_cooldown', key='skip_retraining'):
        log.info("Skipping retraining log (demo mode)")
        return
    
    log_path = LOGS_DIR / "retraining_history.json"
    
    if log_path.exists():
        with open(log_path, 'r') as f:
            history = json.load(f)
    else:
        history = {'retraining_runs': []}
    
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    history['last_retraining'] = datetime.now().isoformat()
    history['retraining_runs'].append({
        'timestamp': datetime.now().isoformat(),
        'trigger': 'drift_detected',
        'drift_status': drift_report.get('overall_status'),
        'n_drifted_features': drift_report.get('n_drifted_features'),
        'airflow_run_id': context['run_id'],
    })
    
    history['retraining_runs'] = history['retraining_runs'][-50:]
    
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    log.info(f"Retraining triggered and logged")


def cleanup_temp_files(**context):
    """Clean up temporary files."""
    import os
    
    temp_files = ['/tmp/current_data_for_monitoring.pkl']
    
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
            log.info(f"Removed: {f}")


def upload_drift_reports_to_gcs(**context):
    """
    Upload drift monitoring reports (HTML and JSON) to GCS bucket.
    This preserves monitoring history for auditing and analysis.
    """
    import os
    from google.cloud import storage
    from datetime import datetime
    
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report') or {}
    
    bucket_name = os.environ.get("GCS_MODEL_BUCKET")
    prefix = os.environ.get("GCS_MONITORING_PREFIX", "monitoring/drift_reports")
    
    if not bucket_name:
        log.warning("GCS_MODEL_BUCKET not set, skipping upload")
        return {'uploaded': False, 'reason': 'GCS_MODEL_BUCKET not set'}
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    report_date = context['ds_nodash']
    uploaded_files = []
    
    # Define report directories (from monitoring_config.py)
    html_reports_dir = "/opt/airflow/scripts/model_pipeline/monitoring/reports/html"
    json_reports_dir = "/opt/airflow/scripts/model_pipeline/monitoring/reports/json"
    
    # Upload HTML reports
    if os.path.exists(html_reports_dir):
        for filename in os.listdir(html_reports_dir):
            if filename.endswith('.html') and report_date in filename:
                local_path = os.path.join(html_reports_dir, filename)
                blob_path = f"{prefix}/html/{filename}"
                
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                uploaded_files.append(blob_path)
                log.info(f"Uploaded HTML report: {blob_path}")
    
    # Upload JSON reports
    if os.path.exists(json_reports_dir):
        for filename in os.listdir(json_reports_dir):
            if filename.endswith('.json') and report_date in filename:
                local_path = os.path.join(json_reports_dir, filename)
                blob_path = f"{prefix}/json/{filename}"
                
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                uploaded_files.append(blob_path)
                log.info(f"Uploaded JSON report: {blob_path}")
    
    # Also upload a summary with metadata
    summary = {
        'report_date': report_date,
        'execution_date': context['ds'],
        'run_id': context['run_id'],
        'overall_status': drift_report.get('overall_status', 'UNKNOWN'),
        'data_drift_detected': drift_report.get('data_drift_detected', False),
        'n_drifted_features': drift_report.get('n_drifted_features', 0),
        'drift_share': drift_report.get('drift_share', 0),
        'recommended_action': drift_report.get('recommended_action', 'none'),
        'alerts': drift_report.get('alerts', []),
        'timestamp': datetime.utcnow().isoformat(),
    }
    
    summary_blob_path = f"{prefix}/summaries/drift_summary_{report_date}.json"
    summary_blob = bucket.blob(summary_blob_path)
    summary_blob.upload_from_string(
        json.dumps(summary, indent=2),
        content_type='application/json'
    )
    uploaded_files.append(summary_blob_path)
    log.info(f"Uploaded summary: {summary_blob_path}")
    
    log.info(f"Total files uploaded to GCS: {len(uploaded_files)}")
    
    return {
        'uploaded': True,
        'bucket': bucket_name,
        'files_count': len(uploaded_files),
        'files': uploaded_files,
    }

# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id='drift_monitoring_dag',
    default_args=default_args,
    description='Drift monitoring with Evidently AI - compares production data vs training baseline',
    schedule_interval=None,  # Triggered by data pipeline or manually
    catchup=False,
    max_active_runs=1,
    tags=['monitoring', 'drift-detection', 'evidently', 'bluebikes'],
    on_failure_callback=send_discord_alert,
) as dag:
    
    # Task definitions
    check_prereqs = PythonOperator(
        task_id='check_prerequisites',
        python_callable=check_prerequisites,
    )
    
    load_data = PythonOperator(
        task_id='load_current_data',
        python_callable=load_current_data,
    )
    
    detect_drift = PythonOperator(
        task_id='run_drift_detection',
        python_callable=run_drift_detection,
    )
    
    evaluate_action = BranchPythonOperator(
        task_id='evaluate_drift_action',
        python_callable=evaluate_drift_action,
    )
    
    # Branch A: Critical
    critical_alert = PythonOperator(
        task_id='send_critical_alert',
        python_callable=send_critical_alert,
    )
    
    check_cooldown = PythonOperator(
        task_id='check_retraining_cooldown',
        python_callable=check_retraining_cooldown,
    )
    
    log_retrain = PythonOperator(
        task_id='log_retraining_triggered',
        python_callable=log_retraining_triggered,
    )
    
    trigger_retrain = TriggerDagRunOperator(
        task_id='trigger_retraining',
        trigger_dag_id='bluebikes_integrated_bias_training',
        wait_for_completion=False,
        conf={'triggered_by': 'drift_monitoring', 'reason': 'drift_detected'},
    )
    
    # Branch B: Warning
    warning_alert = PythonOperator(
        task_id='send_warning_alert',
        python_callable=send_warning_alert,
    )
    
    # Branch C: Healthy
    log_healthy = PythonOperator(
        task_id='log_healthy_status',
        python_callable=log_healthy_status,
    )
    
    # Cleanup and upload
    cleanup = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup_temp_files,
        trigger_rule='none_failed_min_one_success',
    )
    
    upload_reports = PythonOperator(
    task_id='upload_drift_reports_to_gcs',
    python_callable=upload_drift_reports_to_gcs,
    trigger_rule='none_failed_min_one_success',  # Run even if some branches skipped
)
    
    # End markers
    end_critical = EmptyOperator(task_id='end_critical_path', trigger_rule='none_failed_min_one_success')
    end_warning = EmptyOperator(task_id='end_warning_path')
    end_healthy = EmptyOperator(task_id='end_healthy_path')
    
    # Main flow
    check_prereqs >> load_data >> detect_drift >> evaluate_action

    # Branch A: Critical path
    evaluate_action >> critical_alert >> check_cooldown >> log_retrain >> trigger_retrain >> end_critical

    # Branch B: Warning path
    evaluate_action >> warning_alert >> end_warning

    # Branch C: Healthy path
    evaluate_action >> log_healthy >> end_healthy

    # Upload reports after drift detection (regardless of which branch)
    detect_drift >> upload_reports

    # Cleanup after all branches AND upload
    [end_critical, end_warning, end_healthy, upload_reports] >> cleanup