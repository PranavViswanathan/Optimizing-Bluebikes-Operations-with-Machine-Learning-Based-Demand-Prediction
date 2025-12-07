"""
Drift Monitoring DAG for BlueBikes Model Pipeline
Runs daily to detect data drift, prediction drift, and trigger retraining if needed.

Uses Evidently AI for drift detection as per deployment guidelines.
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

logging.basicConfig(level=logging.INFO, format='[MONITORING] %(message)s')
log = logging.getLogger("drift_monitoring")

# Import Discord notifier from your existing code
from scripts.data_pipeline.discord_notifier import send_discord_alert, send_dag_success_alert

# =============================================================================
# DEFAULT ARGS
# =============================================================================

default_args = {
    'owner': 'Nikhil',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def check_prerequisites(**context):
    """
    Check if all prerequisites for monitoring exist:
    - Baseline data
    - Production model
    - Recent data to analyze
    """
    # sys.path.insert(0, '/opt/airflow/model_pipeline/monitoring')
    # sys.path.insert(0, '/opt/airflow/scripts/model_pipeline')
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
    
    # Check baseline
    baseline_path = get_baseline_path()
    if baseline_path.exists():
        log.info(f"✓ Baseline found: {baseline_path}")
    else:
        issues.append(f"Baseline not found at {baseline_path}")
        log.warning(f"✗ Baseline missing: {baseline_path}")
    
    # Check production model
    if PRODUCTION_MODEL_PATH.exists():
        log.info(f"✓ Production model found: {PRODUCTION_MODEL_PATH}")
    else:
        issues.append(f"Production model not found at {PRODUCTION_MODEL_PATH}")
        log.warning(f"✗ Model missing: {PRODUCTION_MODEL_PATH}")
    
    # Check metadata
    if PRODUCTION_METADATA_PATH.exists():
        log.info(f"✓ Model metadata found: {PRODUCTION_METADATA_PATH}")
        with open(PRODUCTION_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        log.info(f"  Model version: {metadata.get('version', 'N/A')}")
        log.info(f"  Model type: {metadata.get('model_type', 'N/A')}")
    else:
        issues.append(f"Model metadata not found at {PRODUCTION_METADATA_PATH}")
        log.warning(f"✗ Metadata missing: {PRODUCTION_METADATA_PATH}")
    
    # Check processed data
    if PROCESSED_DATA_PATH.exists():
        log.info(f"✓ Processed data found: {PROCESSED_DATA_PATH}")
    else:
        issues.append(f"Processed data not found at {PROCESSED_DATA_PATH}")
        log.warning(f"✗ Data missing: {PROCESSED_DATA_PATH}")
    
    # Push results to XCom
    context['task_instance'].xcom_push(key='prerequisites_ok', value=len(issues) == 0)
    context['task_instance'].xcom_push(key='issues', value=issues)
    
    if issues:
        log.error(f"Prerequisites check failed with {len(issues)} issue(s)")
        for issue in issues:
            log.error(f"  - {issue}")
        raise ValueError(f"Prerequisites not met: {issues}")
    
    log.info("All prerequisites satisfied!")
    return True


def inject_artificial_drift(X: pd.DataFrame, drift_type: str = "all") -> pd.DataFrame:
    """
    Inject artificial drift into data for testing Evidently AI detection.
    
    Args:
        X: Original DataFrame
        drift_type: Type of drift to inject
            - "temperature": Shift temperature distribution
            - "demand": Change demand patterns  
            - "temporal": Shift hour/day distributions
            - "all": Apply all drift types
    """
    X_drifted = X.copy()
    
    if drift_type in ["temperature", "all"]:
        # Simulate warmer weather (climate shift)
        if 'temp_avg' in X_drifted.columns:
            X_drifted['temp_avg'] = X_drifted['temp_avg'] + 8  # Add 8 degrees
            print("Injected temperature drift: +8 degrees")
    
    if drift_type in ["demand", "all"]:
        # Simulate increased demand (popularity growth)
        if 'rides_last_hour' in X_drifted.columns:
            X_drifted['rides_last_hour'] = X_drifted['rides_last_hour'] * 1.5  # 50% increase
            print("Injected demand drift: +50%")
        if 'rides_rolling_3h' in X_drifted.columns:
            X_drifted['rides_rolling_3h'] = X_drifted['rides_rolling_3h'] * 1.5
    
    if drift_type in ["temporal", "all"]:
        # Simulate shift in usage patterns (more weekend usage)
        if 'is_weekend' in X_drifted.columns:
            # Randomly flip some weekday to weekend
            flip_mask = (X_drifted['is_weekend'] == 0) & (np.random.random(len(X_drifted)) < 0.3)
            X_drifted.loc[flip_mask, 'is_weekend'] = 1
            print("Injected temporal drift: shifted weekend patterns")
    
    if drift_type in ["hour", "all"]:
        # Simulate shift in peak hours
        if 'hour' in X_drifted.columns:
            X_drifted['hour'] = (X_drifted['hour'] + 2) % 24  # Shift all hours by 2
            print("Injected hour drift: shifted by +2 hours")
    
    return X_drifted




def load_current_data(**context):
    """
    Load the most recent data for drift analysis.
    Optionally inject artificial drift for testing.
    """
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline')
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline/monitoring')

    import pandas as pd
    import numpy as np
    from feature_generation import load_and_prepare_data
    from monitoring_config import get_config
    
    log.info("="*60)
    log.info("LOADING CURRENT DATA FOR ANALYSIS")
    log.info("="*60)
    
    config = get_config()
    
    # Load and prepare data
    X, y, feature_columns = load_and_prepare_data()
    X["date"] = pd.to_datetime(X["date"])
    
    log.info(f"Total data points: {len(X)}")
    log.info(f"Date range in data: {X['date'].min()} to {X['date'].max()}")
    
    # Use last 7 days of AVAILABLE data
    max_date = X["date"].max()
    analysis_window = config.schedule.analysis_window_hours
    cutoff_date = max_date - pd.Timedelta(hours=analysis_window * 7)
    
    log.info(f"Using data from {cutoff_date} to {max_date}")
    
    recent_mask = (X["date"] >= cutoff_date) & (X["date"] <= max_date)
    X_recent = X.loc[recent_mask].copy()
    y_recent = y.loc[recent_mask].copy()
    
    log.info(f"Recent data points: {len(X_recent)}")
    
    # =========================================================
    # DRIFT INJECTION FOR TESTING (set via DAG conf or env var)
    # =========================================================
    inject_drift = context['dag_run'].conf.get('inject_drift', False)
    drift_type = context['dag_run'].conf.get('drift_type', 'all')
    
    if inject_drift:
        log.info("="*60)
        log.info(f"INJECTING ARTIFICIAL DRIFT: {drift_type}")
        log.info("="*60)
        
        if drift_type in ["temperature", "all"]:
            if 'temp_avg' in X_recent.columns:
                X_recent['temp_avg'] = X_recent['temp_avg'] + 8
                log.info("  ✓ Temperature drift: +8 degrees")
        
        if drift_type in ["demand", "all"]:
            if 'rides_last_hour' in X_recent.columns:
                X_recent['rides_last_hour'] = X_recent['rides_last_hour'] * 1.5
                log.info("  ✓ Demand drift: +50%")
            if 'rides_rolling_3h' in X_recent.columns:
                X_recent['rides_rolling_3h'] = X_recent['rides_rolling_3h'] * 1.5
        
        if drift_type in ["temporal", "all"]:
            if 'is_weekend' in X_recent.columns:
                flip_mask = (X_recent['is_weekend'] == 0) & (np.random.random(len(X_recent)) < 0.3)
                X_recent.loc[flip_mask, 'is_weekend'] = 1
                log.info("  ✓ Temporal drift: weekend pattern shift")
        
        if drift_type in ["hour", "all"]:
            if 'hour' in X_recent.columns:
                X_recent['hour'] = (X_recent['hour'] + 2) % 24
                log.info("  ✓ Hour drift: +2 hour shift")
        
        log.info("Drift injection complete!")
    
    # Save to temp file
    temp_path = '/tmp/current_data_for_monitoring.pkl'
    X_recent.to_pickle(temp_path)
    
    context['task_instance'].xcom_push(key='current_data_path', value=temp_path)
    context['task_instance'].xcom_push(key='n_samples', value=len(X_recent))
    context['task_instance'].xcom_push(key='drift_injected', value=inject_drift)
    
    return len(X_recent)


def run_drift_detection(**context):
    """
    Run Evidently AI drift detection on current data.
    """
    # sys.path.insert(0, '/opt/airflow/model_pipeline/monitoring')
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline')
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline/monitoring')

    import pandas as pd
    from drift_detector import EvidentlyDriftDetector
    from monitoring_config import get_config
    
    log.info("="*60)
    log.info("RUNNING EVIDENTLY AI DRIFT DETECTION")
    log.info("="*60)
    
    # Load current data from temp file
    ti = context['task_instance']
    current_data_path = ti.xcom_pull(task_ids='load_current_data', key='current_data_path')
    
    current_data = pd.read_pickle(current_data_path)
    log.info(f"Loaded {len(current_data)} samples for analysis")
    
    # Initialize detector and run analysis
    detector = EvidentlyDriftDetector()
    detector.load_reference_data()  # Loads from default baseline path
    detector.load_model()
    
    # Run full monitoring suite
    report = detector.run_full_monitoring(
        current_data=current_data,
        current_actuals=None,  # Ground truth not available yet
        generate_html=True,
        save_reports=True
    )
    
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
    })
    
    log.info("="*60)
    log.info("DRIFT DETECTION COMPLETE")
    log.info("="*60)
    log.info(f"Overall Status: {report.get('overall_status')}")
    log.info(f"Recommended Action: {report.get('recommended_action')}")
    
    return report.get('overall_status')


def evaluate_drift_action(**context):
    """
    Decide what action to take based on drift detection results.
    Returns the task_id to branch to.
    """
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    log.info("="*60)
    log.info("EVALUATING DRIFT DETECTION RESULTS")
    log.info("="*60)
    
    overall_status = drift_report.get('overall_status', 'UNKNOWN')
    recommended_action = drift_report.get('recommended_action', 'none')
    
    log.info(f"Status: {overall_status}")
    log.info(f"Recommended: {recommended_action}")
    
    if drift_report.get('alerts'):
        log.info("Alerts:")
        for alert in drift_report['alerts']:
            log.info(f"  ⚠ {alert}")
    
    # Determine branch
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
    """
    Send critical alert via Discord when major drift detected.
    """
    import requests
    
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    
    if not webhook_url:
        log.warning("DISCORD_WEBHOOK_URL not set, skipping alert")
        return
    
    # Build alert message
    alerts = drift_report.get('alerts', [])
    n_drifted = drift_report.get('n_drifted_features', 0)
    drift_share = drift_report.get('drift_share', 0)
    
    message = {
        "embeds": [{
            "title": "🚨 CRITICAL: Model Drift Detected",
            "description": "Significant drift detected in BlueBikes demand prediction model.",
            "color": 15158332,  # Red
            "fields": [
                {
                    "name": "Status",
                    "value": drift_report.get('overall_status', 'CRITICAL'),
                    "inline": True
                },
                {
                    "name": "Drifted Features",
                    "value": f"{n_drifted} ({drift_share:.1%})",
                    "inline": True
                },
                {
                    "name": "Action",
                    "value": "Automatic retraining triggered",
                    "inline": True
                },
                {
                    "name": "Alerts",
                    "value": "\n".join(alerts[:5]) if alerts else "No specific alerts",
                    "inline": False
                }
            ],
            "footer": {
                "text": f"Airflow DAG: drift_monitoring | {context['ds']}"
            },
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
    """
    Send warning alert via Discord when minor drift detected.
    """
    import requests
    
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    
    if not webhook_url:
        log.warning("DISCORD_WEBHOOK_URL not set, skipping alert")
        return
    
    n_drifted = drift_report.get('n_drifted_features', 0)
    alerts = drift_report.get('alerts', [])
    
    message = {
        "embeds": [{
            "title": "⚠️ WARNING: Minor Drift Detected",
            "description": "Some drift detected in BlueBikes model. Monitoring closely.",
            "color": 16776960,  # Yellow
            "fields": [
                {
                    "name": "Drifted Features",
                    "value": str(n_drifted),
                    "inline": True
                },
                {
                    "name": "Action",
                    "value": "Continued monitoring",
                    "inline": True
                },
                {
                    "name": "Details",
                    "value": "\n".join(alerts[:3]) if alerts else "Minor distribution shifts",
                    "inline": False
                }
            ],
            "footer": {
                "text": f"Airflow DAG: drift_monitoring | {context['ds']}"
            }
        }]
    }
    
    try:
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
        log.info("Warning alert sent to Discord")
    except Exception as e:
        log.error(f"Failed to send Discord alert: {e}")


def log_healthy_status(**context):
    """
    Log healthy status when no significant drift detected.
    """
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    log.info("="*60)
    log.info("MODEL STATUS: HEALTHY")
    log.info("="*60)
    log.info("No significant drift detected.")
    log.info(f"Data drift: {drift_report.get('data_drift_detected', False)}")
    log.info(f"Prediction drift: {drift_report.get('prediction_drift_detected', False)}")
    log.info("No action required.")
    
    return "healthy"


def check_retraining_cooldown(**context):
    """
    Check if we're within the retraining cooldown period.
    Prevents too frequent retraining.
    """
    # sys.path.insert(0, '/opt/airflow/model_pipeline/monitoring')
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline/monitoring')

    from monitoring_config import get_config, LOGS_DIR
    from datetime import datetime, timedelta
    
    config = get_config()
    cooldown_hours = config.retraining.retraining_cooldown_hours
    max_per_week = config.retraining.max_retrains_per_week
    
    log_path = LOGS_DIR / "retraining_history.json"
    
    if not log_path.exists():
        log.info("No retraining history found - OK to retrain")
        return True
    
    with open(log_path, 'r') as f:
        history = json.load(f)
    
    # Check cooldown
    last_retrain = history.get('last_retraining')
    if last_retrain:
        last_time = datetime.fromisoformat(last_retrain)
        cooldown_end = last_time + timedelta(hours=cooldown_hours)
        
        if datetime.now() < cooldown_end:
            log.warning(f"Within cooldown period. Next retraining allowed after {cooldown_end}")
            context['task_instance'].xcom_push(key='cooldown_active', value=True)
            return False
    
    # Check weekly limit
    week_ago = datetime.now() - timedelta(days=7)
    recent_retrains = [
        r for r in history.get('retraining_runs', [])
        if datetime.fromisoformat(r['timestamp']) > week_ago
    ]
    
    if len(recent_retrains) >= max_per_week:
        log.warning(f"Weekly limit reached ({max_per_week} retrains). Skipping.")
        context['task_instance'].xcom_push(key='weekly_limit_reached', value=True)
        return False
    
    log.info("Retraining cooldown check passed")
    return True


def log_retraining_triggered(**context):
    """
    Log that retraining was triggered.
    """
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline/monitoring')
    
    from monitoring_config import LOGS_DIR
    
    log_path = LOGS_DIR / "retraining_history.json"
    
    # Load or create history
    if log_path.exists():
        with open(log_path, 'r') as f:
            history = json.load(f)
    else:
        history = {'retraining_runs': []}
    
    # Add new entry
    ti = context['task_instance']
    drift_report = ti.xcom_pull(task_ids='run_drift_detection', key='drift_report')
    
    history['last_retraining'] = datetime.now().isoformat()
    history['retraining_runs'].append({
        'timestamp': datetime.now().isoformat(),
        'trigger': 'drift_detected',
        'drift_status': drift_report.get('overall_status'),
        'n_drifted_features': drift_report.get('n_drifted_features'),
        'airflow_run_id': context['run_id'],
    })
    
    # Keep only last 50 entries
    history['retraining_runs'] = history['retraining_runs'][-50:]
    
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    log.info(f"Retraining triggered and logged to {log_path}")


def cleanup_temp_files(**context):
    """
    Clean up temporary files created during monitoring.
    """
    import os
    import glob
    
    temp_files = [
        '/tmp/current_data_for_monitoring.pkl',
    ]
    
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
            log.info(f"Removed temp file: {f}")


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id='drift_monitoring_dag',
    default_args=default_args,
    description='Daily drift monitoring with Evidently AI',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    catchup=False,
    max_active_runs=1,
    tags=['monitoring', 'drift-detection', 'evidently', 'bluebikes'],
    on_failure_callback=send_discord_alert,
) as dag:
    
    # Task 1: Check prerequisites
    check_prereqs = PythonOperator(
        task_id='check_prerequisites',
        python_callable=check_prerequisites,
    )
    
    # Task 2: Load current data
    load_data = PythonOperator(
        task_id='load_current_data',
        python_callable=load_current_data,
    )
    
    # Task 3: Run drift detection
    detect_drift = PythonOperator(
        task_id='run_drift_detection',
        python_callable=run_drift_detection,
    )
    
    # Task 4: Evaluate and branch
    evaluate_action = BranchPythonOperator(
        task_id='evaluate_drift_action',
        python_callable=evaluate_drift_action,
    )
    
    # Branch A: Critical - Send alert and trigger retraining
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
        trigger_dag_id='bluebikes_integrated_bias_training',  # Your training DAG
        wait_for_completion=False,
        conf={'triggered_by': 'drift_monitoring', 'reason': 'drift_detected'},
    )
    
    # Branch B: Warning - Just alert
    warning_alert = PythonOperator(
        task_id='send_warning_alert',
        python_callable=send_warning_alert,
    )
    
    # Branch C: Healthy - Just log
    log_healthy = PythonOperator(
        task_id='log_healthy_status',
        python_callable=log_healthy_status,
    )
    
    # Final: Cleanup
    cleanup = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup_temp_files,
        trigger_rule='none_failed_min_one_success',
    )
    
    # End markers for each branch
    end_critical = EmptyOperator(
        task_id='end_critical_path',
        trigger_rule='none_failed_min_one_success',
    )
    
    end_warning = EmptyOperator(
        task_id='end_warning_path',
    )
    
    end_healthy = EmptyOperator(
        task_id='end_healthy_path',
    )
    
    # =============================================================================
    # TASK DEPENDENCIES
    # =============================================================================
    
    # Main flow
    check_prereqs >> load_data >> detect_drift >> evaluate_action
    
    # Branch A: Critical path
    evaluate_action >> critical_alert >> check_cooldown >> log_retrain >> trigger_retrain >> end_critical
    
    # Branch B: Warning path
    evaluate_action >> warning_alert >> end_warning
    
    # Branch C: Healthy path
    evaluate_action >> log_healthy >> end_healthy
    
    # Cleanup after all branches
    [end_critical, end_warning, end_healthy] >> cleanup