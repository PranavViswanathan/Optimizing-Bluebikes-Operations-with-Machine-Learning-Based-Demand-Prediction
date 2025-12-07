from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import sys
import os
import logging
logging.basicConfig(level=logging.INFO, format='[PIPELINE] %(message)s')
log = logging.getLogger("pipeline")


from scripts.data_pipeline.discord_notifier import send_discord_alert, send_dag_success_alert

default_args = {
    'owner': 'Nikhil',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'bluebikes_integrated_bias_training',
    default_args=default_args,
    description='Integrated training pipeline with bias detection and mitigation',
    schedule_interval='@weekly',
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'training', 'bias-detection', 'bluebikes', 'production'],
    on_success_callback=send_dag_success_alert,
    on_failure_callback=send_discord_alert,
)

def run_integrated_pipeline(**context):
    import subprocess
    import json
    log.info(" Starting Integrated BlueBikes Pipeline with Bias Detection ")
    log.info(f"Execution Date: {context['ds']}")
    log.info(f"Run ID: {context['run_id']}")
    
    date_str = context['ds_nodash']
    
    # Key fix: Write results file IMMEDIATELY after getting results,
    # and handle the case where comparison_report might have serialization issues
    training_script = f"""
import sys
import os
import traceback
import logging
import json
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("integrated_pipeline")

sys.path.insert(0, '/opt/airflow/scripts/model_pipeline')
sys.path.append('/opt/airflow/scripts/model_pipeline')
os.chdir('/opt/airflow/scripts/model_pipeline')

date_str = '{date_str}'
results_file = '/tmp/integrated_results_' + date_str + '.json'

def safe_float(val, default=0.0):
    '''Safely convert any numeric type to float'''
    if val is None:
        return default
    try:
        if isinstance(val, (np.floating, np.integer)):
            return float(val)
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    '''Safely convert any numeric type to int'''
    if val is None:
        return default
    try:
        if isinstance(val, (np.floating, np.integer)):
            return int(val)
        return int(val)
    except (ValueError, TypeError):
        return default

try:
    from integrated_training_pipeline import IntegratedBlueBikesTrainer

    trainer = IntegratedBlueBikesTrainer(
        experiment_name='bluebikes_bias_integrated_' + date_str
    )

    log.info("Starting integrated pipeline...")
    results = trainer.run_complete_pipeline(
        models_to_train=['xgboost', 'lightgbm', 'randomforest'],
        tune=False
    )

    if results is None:
        raise Exception("Pipeline returned None")
    
    baseline_metrics = results.get('baseline_metrics', {{}})
    mitigated_metrics = results.get('mitigated_metrics', {{}})
    comparison = results.get('comparison_report', {{}})
    
    if not baseline_metrics or not mitigated_metrics:
        raise Exception("Missing metrics in results")
    
    # Build pipeline_results with SAFE type conversion
    # This handles numpy types that can cause JSON serialization issues
    improvement = comparison.get('improvement', {{}}) if comparison else {{}}
    
    pipeline_results = {{
        'best_model': str(trainer.best_model_name) if trainer.best_model_name else 'unknown',
        'baseline_test_r2': safe_float(baseline_metrics.get('test_r2')),
        'baseline_test_mae': safe_float(baseline_metrics.get('test_mae')),
        'baseline_test_rmse': safe_float(baseline_metrics.get('test_rmse')),
        'mitigated_test_r2': safe_float(mitigated_metrics.get('test_r2')),
        'mitigated_test_mae': safe_float(mitigated_metrics.get('test_mae')),
        'mitigated_test_rmse': safe_float(mitigated_metrics.get('test_rmse')),
        'baseline_bias_issues': safe_int(comparison.get('baseline_bias_issues') if comparison else 0),
        'mitigated_bias_issues': safe_int(comparison.get('mitigated_bias_issues') if comparison else 0),
        'r2_improvement': safe_float(improvement.get('r2_improvement')),
        'mae_improvement': safe_float(improvement.get('mae_improvement')),
        'bias_issues_reduction': safe_int(improvement.get('bias_issues_reduction')),
        'baseline_model_path': str(trainer.best_model_path) if trainer.best_model_path else '',
        'mitigated_model_path': str(trainer.mitigated_model_path) if trainer.mitigated_model_path else ''
    }}
    
    # Save results IMMEDIATELY - before any logging that might fail
    log.info(f"Writing results to {{results_file}}")
    with open(results_file, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    log.info(f"Results file written successfully")
    
    # Now do the summary logging (if this fails, results are already saved)
    log.info("="*80)
    log.info(" Pipeline Complete ")
    log.info("="*80)
    log.info(f"Best Model: {{pipeline_results['best_model']}}")
    log.info(f"Baseline R2: {{pipeline_results['baseline_test_r2']:.4f}}")
    log.info(f"Mitigated R2: {{pipeline_results['mitigated_test_r2']:.4f}}")
    
    # Explicit successful exit
    sys.exit(0)

except Exception as e:
    log.error("="*80)
    log.error("PIPELINE ERROR")
    log.error("="*80)
    log.error(f"Error type: {{type(e).__name__}}")
    log.error(f"Error message: {{str(e)}}")
    log.error("Full traceback:")
    traceback.print_exc()
    
    # Check if we have partial results from trainer
    try:
        if 'trainer' in dir() and trainer is not None:
            # Try to salvage what we can
            partial_results = {{
                'best_model': str(getattr(trainer, 'best_model_name', 'unknown')),
                'baseline_model_path': str(getattr(trainer, 'best_model_path', '')),
                'mitigated_model_path': str(getattr(trainer, 'mitigated_model_path', '')),
                'error': str(e),
                'partial': True
            }}
            
            # Try to get metrics if available
            if hasattr(trainer, 'baseline_bias_report') and trainer.baseline_bias_report:
                overall = trainer.baseline_bias_report.get('overall_performance', {{}})
                partial_results['baseline_test_r2'] = safe_float(overall.get('r2'))
                partial_results['baseline_test_mae'] = safe_float(overall.get('mae'))
                partial_results['baseline_test_rmse'] = safe_float(overall.get('rmse'))
            
            if hasattr(trainer, 'final_bias_report') and trainer.final_bias_report:
                overall = trainer.final_bias_report.get('overall_performance', {{}})
                partial_results['mitigated_test_r2'] = safe_float(overall.get('r2'))
                partial_results['mitigated_test_mae'] = safe_float(overall.get('mae'))
                partial_results['mitigated_test_rmse'] = safe_float(overall.get('rmse'))
            
            with open(results_file, 'w') as f:
                json.dump(partial_results, f, indent=2)
            log.info(f"Partial results saved to: {{results_file}}")
    except Exception as save_err:
        log.error(f"Could not save partial results: {{save_err}}")
    
    error_info = {{
        'error': str(e),
        'error_type': type(e).__name__,
        'traceback': traceback.format_exc()
    }}
    
    try:
        error_file = '/tmp/integrated_error_' + date_str + '.json'
        with open(error_file, 'w') as f:
            json.dump(error_info, f, indent=2)
        log.error(f"Error info saved to: {{error_file}}")
    except:
        pass
    
    sys.exit(1)
"""
    
    script_path = f'/tmp/integrated_script_{date_str}.py'
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=False,
            cwd='/opt/airflow/scripts/model_pipeline/'
        )
        
        log.info("="*80)
        log.info("SUBPROCESS OUTPUT")
        log.info("="*80)
        log.info("STDOUT:")
        log.info(result.stdout)
        if result.stderr:
            log.info("\nSTDERR:")
            log.info(result.stderr)
        log.info("="*80)
        
        # IMPORTANT: Check for results file FIRST, regardless of return code
        # The pipeline might have succeeded but exited with wrong code
        results_file = f'/tmp/integrated_results_{date_str}.json'
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Check if this is a partial result due to error
            if results.get('partial'):
                log.warning("Results file contains partial data due to error")
                if result.returncode != 0:
                    raise Exception(f"Pipeline failed with partial results: {results.get('error', 'Unknown error')}")
            
            for key, value in results.items():
                if key != 'partial' and key != 'error':
                    context['task_instance'].xcom_push(key=key, value=value)
            
            log.info("="*80)
            log.info(" Results Summary ")
            log.info("="*80)
            log.info(f"Model: {results.get('best_model', 'N/A')}")
            log.info(f"Mitigated R²: {results.get('mitigated_test_r2', 'N/A')}")
            log.info(f"Mitigated MAE: {results.get('mitigated_test_mae', 'N/A')}")
            log.info(f"Bias Reduction: {results.get('bias_issues_reduction', 'N/A')} issues")
            
            return results
        else:
            # No results file found
            log.error(f"Results file not found at: {results_file}")
            log.info("Checking /tmp directory:")
            import glob
            tmp_files = glob.glob('/tmp/integrated_*')
            for f in tmp_files:
                log.info(f"  Found: {f}")
            
            # Check for error file
            error_file = f'/tmp/integrated_error_{date_str}.json'
            if os.path.exists(error_file):
                with open(error_file, 'r') as f:
                    error_info = json.load(f)
                raise Exception(f"Pipeline failed: {error_info.get('error', 'Unknown error')}")
            
            raise Exception(f"Results file not found and subprocess exited with code {result.returncode}")
            
    except subprocess.CalledProcessError as e:
        log.error(f"Error running integrated pipeline: {e}")
        log.error(f"STDOUT: {e.stdout}")
        log.error(f"STDERR: {e.stderr}")
        raise
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)



def validate_mitigated_model(**context):
    """Validate the bias-mitigated model"""
    ti = context['task_instance']
    
    mitigated_r2 = ti.xcom_pull(task_ids='run_integrated_pipeline', key='mitigated_test_r2')
    mitigated_mae = ti.xcom_pull(task_ids='run_integrated_pipeline', key='mitigated_test_mae')
    baseline_r2 = ti.xcom_pull(task_ids='run_integrated_pipeline', key='baseline_test_r2')
    bias_reduction = ti.xcom_pull(task_ids='run_integrated_pipeline', key='bias_issues_reduction')
    best_model = ti.xcom_pull(task_ids='run_integrated_pipeline', key='best_model')
    
    MIN_R2 = 0.70
    MAX_MAE = 110
    MIN_BIAS_REDUCTION = 0  
    
    log.info("="*60)
    log.info("MITIGATED MODEL VALIDATION")
    log.info("="*60)
    log.info(f"Model: {best_model}")
    log.info(f"Baseline R²: {baseline_r2:.4f}")
    log.info(f"Mitigated R²: {mitigated_r2:.4f} (threshold: >{MIN_R2})")
    log.info(f"Mitigated MAE: {mitigated_mae:.2f} (threshold: <{MAX_MAE})")
    log.info(f"Bias Issues Reduction: {bias_reduction} (threshold: >={MIN_BIAS_REDUCTION})")
    
    validation_passed = True
    reasons = []
    
    if mitigated_r2 < MIN_R2:
        validation_passed = False
        reasons.append(f"R² {mitigated_r2:.4f} below threshold {MIN_R2}")
    
    if mitigated_mae > MAX_MAE:
        validation_passed = False
        reasons.append(f"MAE {mitigated_mae:.2f} above threshold {MAX_MAE}")
    
    if bias_reduction < MIN_BIAS_REDUCTION:
        validation_passed = False
        reasons.append(f"Bias increased by {abs(bias_reduction)} issues")
    
    if validation_passed:
        log.info("Mitigated model passed validation!")
        return True
    else:
        log.info("Mitigated model failed validation:")
        for reason in reasons:
            log.info(f"  - {reason}")
        raise Exception("Mitigated model did not meet validation thresholds")

def promote_mitigated_model(**context):
    """Promote bias-mitigated model to production"""
    import json
    import shutil
    from datetime import datetime
    
    log.info("="*60)
    log.info("MODEL PROMOTION DECISION (Bias-Mitigated)")
    log.info("="*60)
    
    ti = context['task_instance']
    model_name = ti.xcom_pull(task_ids='run_integrated_pipeline', key='best_model')
    mitigated_r2 = ti.xcom_pull(task_ids='run_integrated_pipeline', key='mitigated_test_r2')
    mitigated_mae = ti.xcom_pull(task_ids='run_integrated_pipeline', key='mitigated_test_mae')
    mitigated_rmse = ti.xcom_pull(task_ids='run_integrated_pipeline', key='mitigated_test_rmse')
    r2_improvement = ti.xcom_pull(task_ids='run_integrated_pipeline', key='r2_improvement')
    mae_improvement = ti.xcom_pull(task_ids='run_integrated_pipeline', key='mae_improvement')
    bias_reduction = ti.xcom_pull(task_ids='run_integrated_pipeline', key='bias_issues_reduction')
    baseline_bias_issues = ti.xcom_pull(task_ids='run_integrated_pipeline', key='baseline_bias_issues')
    mitigated_bias_issues = ti.xcom_pull(task_ids='run_integrated_pipeline', key='mitigated_bias_issues')
    
    version_dir = "/opt/airflow/models/versions"
    production_link = "/opt/airflow/models/production/current_model.pkl"
    metadata_file = "/opt/airflow/models/model_versions.json"
    
    os.makedirs(version_dir, exist_ok=True)
    os.makedirs(os.path.dirname(production_link), exist_ok=True)
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                version_history = json.load(f)
        else:
            version_history = {
                "versions": [],
                "current_production": None
            }
        
        current_prod_metrics = None
        current_prod_version = None
        if version_history["current_production"]:
            for v in version_history["versions"]:
                if v["version"] == version_history["current_production"]:
                    current_prod_metrics = {
                        'test_r2': v['metrics']['test_r2'],
                        'test_mae': v['metrics']['test_mae']
                    }
                    current_prod_version = v["version"]
                    break
        
        should_promote = False
        promotion_reason = ""
        
        if current_prod_metrics is None:
            should_promote = True
            promotion_reason = "First bias-mitigated production deployment"
            log.info("No existing production model - deploying first version")
        else:
            r2_delta = mitigated_r2 - current_prod_metrics['test_r2']
            mae_delta = current_prod_metrics['test_mae'] - mitigated_mae
            
            log.info(f"\nCurrent Production Model (v{current_prod_version}):")
            log.info(f"  R²: {current_prod_metrics['test_r2']:.4f}")
            log.info(f"  MAE: {current_prod_metrics['test_mae']:.2f}")
            log.info(f"\nNew Bias-Mitigated Model:")
            log.info(f"  R²: {mitigated_r2:.4f} (Δ: {r2_delta:+.4f})")
            log.info(f"  MAE: {mitigated_mae:.2f} (Δ: {mae_delta:+.2f})")
            log.info(f"  Bias Improvement: {baseline_bias_issues} -> {mitigated_bias_issues} issues")
            log.info(f"  vs Baseline: R² {r2_improvement:+.4f}, MAE {mae_improvement:+.2f}")
            
            if bias_reduction > 0:
                should_promote = True
                promotion_reason = f"Bias reduced by {bias_reduction} issues"
                if r2_delta > 0:
                    promotion_reason += f", R² improved by {r2_delta:.4f}"
            elif r2_delta > 0.01 and mae_delta > 0:
                should_promote = True
                promotion_reason = f"Better performance: R² +{r2_delta:.4f}, MAE {mae_delta:+.2f}"
            elif r2_delta > 0.02:
                should_promote = True
                promotion_reason = f"Significant R² improvement: +{r2_delta:.4f}"
            else:
                promotion_reason = "Performance/bias improvement below thresholds"
        
        new_version = len(version_history["versions"]) + 1
        
        temp_model_path = f'/opt/airflow/artifacts/model_pipeline/models/mitigated_model_{model_name}.pkl'
        versioned_model_path = f"{version_dir}/model_v{new_version}_{context['ds_nodash']}_bias_mitigated.pkl"
        
        if os.path.exists(temp_model_path):
            shutil.copy(temp_model_path, versioned_model_path)
            log.info(f"\nSaved bias-mitigated model as version {new_version}")
        else:
            raise Exception(f"Model file not found at {temp_model_path}")
        
        if should_promote:
            log.info(f"\n✓ PROMOTING BIAS-MITIGATED MODEL TO PRODUCTION")
            log.info(f"  Reason: {promotion_reason}")
            
            if os.path.exists(production_link):
                os.remove(production_link)
            shutil.copy(versioned_model_path, production_link)
            log.info(f"  Updated production model link")
            
            if version_history["current_production"]:
                for v in version_history["versions"]:
                    if v["version"] == version_history["current_production"]:
                        v["status"] = "archived"
                        v["archived_date"] = context['ds']
                        log.info(f"  Archived previous version {v['version']}")
            
            version_entry = {
                "version": new_version,
                "model_type": model_name,
                "created_date": context['ds'],
                "promoted_date": context['ds'],
                "promotion_reason": promotion_reason,
                "file_path": versioned_model_path,
                "metrics": {
                    "test_r2": mitigated_r2,
                    "test_mae": mitigated_mae,
                    "test_rmse": mitigated_rmse,
                    "baseline_r2": ti.xcom_pull(task_ids='run_integrated_pipeline', key='baseline_test_r2'),
                    "baseline_mae": ti.xcom_pull(task_ids='run_integrated_pipeline', key='baseline_test_mae'),
                    "r2_improvement": r2_improvement,
                    "mae_improvement": mae_improvement
                },
                "bias_metrics": {
                    "baseline_issues": baseline_bias_issues,
                    "mitigated_issues": mitigated_bias_issues,
                    "issues_reduced": bias_reduction
                },
                "status": "production",
                "bias_mitigated": True
            }
            
            version_history["versions"].append(version_entry)
            version_history["current_production"] = new_version
            
            context['task_instance'].xcom_push(key='model_promoted', value=True)
            context['task_instance'].xcom_push(key='production_version', value=new_version)
            
        else:
            log.info(f"\n✗ NOT PROMOTING MODEL")
            log.info(f"  Reason: {promotion_reason}")
            
            version_entry = {
                "version": new_version,
                "model_type": model_name,
                "created_date": context['ds'],
                "promotion_reason": promotion_reason,
                "file_path": versioned_model_path,
                "metrics": {
                    "test_r2": mitigated_r2,
                    "test_mae": mitigated_mae,
                    "test_rmse": mitigated_rmse,
                    "r2_improvement": r2_improvement,
                    "mae_improvement": mae_improvement
                },
                "bias_metrics": {
                    "baseline_issues": baseline_bias_issues,
                    "mitigated_issues": mitigated_bias_issues,
                    "issues_reduced": bias_reduction
                },
                "status": "staging",
                "bias_mitigated": True
            }
            
            version_history["versions"].append(version_entry)
            log.info(f"  Model saved as version {new_version} in staging")
            
            context['task_instance'].xcom_push(key='model_promoted', value=False)
            context['task_instance'].xcom_push(key='production_version', value=new_version)
        
        with open(metadata_file, 'w') as f:
            json.dump(version_history, f, indent=2)
        log.info(f"\nVersion history updated: {metadata_file}")
        
        return {'promoted': should_promote, 'reason': promotion_reason, 'version': new_version}
        
    except Exception as e:
        log.info(f"Error in model promotion: {e}")
        import traceback
        traceback.print_exc()
        raise

def push_model_to_github(**context):
    """Push the current production model + metadata to GitHub Release"""
    import os
    import subprocess

    ti = context['task_instance']
    production_version = ti.xcom_pull(task_ids='promote_mitigated_model', key='production_version')
    model_promoted = ti.xcom_pull(task_ids='promote_mitigated_model', key='model_promoted')

    if not model_promoted:
        log.info("Model was not promoted to production; skipping GitHub upload.")
        return {'uploaded': False, 'reason': 'Model not promoted'}

    model_path = "/opt/airflow/models/production/current_model.pkl"
    metadata_path = "/opt/airflow/models/production/current_metadata.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    github_repo = os.environ.get("GITHUB_REPO")
    github_token = os.environ.get("GITHUB_TOKEN")

    if not github_repo or not github_token:
        raise RuntimeError("GITHUB_REPO or GITHUB_TOKEN not set in environment")

    date_str = context['ds_nodash']
    tag = f"model-v{production_version}-{date_str}"

    log.info("=" * 60)
    log.info("PUSHING MODEL ARTIFACT TO GITHUB")
    log.info("=" * 60)
    log.info(f"Repo: {github_repo}")
    log.info(f"Tag: {tag}")

    env = os.environ.copy()
    env["GITHUB_TOKEN"] = github_token

    # Check if release already exists
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", github_repo],
        env=env,
        capture_output=True
    )
    
    if result.returncode == 0:
        log.info(f"Release {tag} already exists - deleting to recreate with fresh artifacts")
        subprocess.check_call(
            ["gh", "release", "delete", tag, "--repo", github_repo, "--yes"],
            env=env
        )

    # Create release
    subprocess.check_call(
        [
            "gh", "release", "create", tag,
            "--repo", github_repo,
            "--title", f"BlueBikes bias-mitigated model {tag}",
            "--notes", f"Automated Airflow deployment on {context['ds']} (version {production_version})",
        ],
        env=env,
    )

    # Upload artifacts
    subprocess.check_call(
        [
            "gh", "release", "upload", tag,
            model_path, metadata_path,
            "--repo", github_repo,
        ],
        env=env,
    )

    log.info("Model and metadata successfully uploaded to GitHub Release.")
    return {'uploaded': True, 'tag': tag}


def push_model_to_gcs(**context):
    """Upload the current production model + metadata to a GCS bucket."""
    import os
    from google.cloud import storage

    ti = context['task_instance']
    model_promoted = ti.xcom_pull(
        task_ids='promote_mitigated_model',
        key='model_promoted'
    )

    if not model_promoted:
        log.info("Model was not promoted to production; skipping GCS upload.")
        return {'uploaded': False, 'reason': 'Model not promoted'}

    model_path = "/opt/airflow/models/production/current_model.pkl"
    metadata_path = "/opt/airflow/models/production/current_metadata.json"

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Production model or metadata not found: {model_path}, {metadata_path}"
        )

    bucket_name = os.environ.get("GCS_MODEL_BUCKET")
    prefix = os.environ.get("GCS_MODEL_PREFIX", "models/production")

    if not bucket_name:
        raise RuntimeError("GCS_MODEL_BUCKET environment variable is not set")

    # Create client using the service account key pointed to by GOOGLE_APPLICATION_CREDENTIALS
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Use timestamp or production version as part of the path if you like
    production_version = ti.xcom_pull(
        task_ids='promote_mitigated_model',
        key='production_version'
    ) or "unknown_version"

    model_blob = bucket.blob(f"{prefix}/{production_version}/current_model.pkl")
    metadata_blob = bucket.blob(f"{prefix}/{production_version}/current_metadata.json")

    log.info(f"Uploading model to gs://{bucket_name}/{model_blob.name}")
    model_blob.upload_from_filename(model_path)

    log.info(f"Uploading metadata to gs://{bucket_name}/{metadata_blob.name}")
    metadata_blob.upload_from_filename(metadata_path)

    log.info("Model and metadata successfully uploaded to GCS.")
    return {
        'uploaded': True,
        'bucket': bucket_name,
        'model_blob': model_blob.name,
        'metadata_blob': metadata_blob.name
    }

def deploy_mitigated_model(**context):
    """Deploy bias-mitigated model"""
    import json
    from datetime import datetime
    
    log.info("BIAS-MITIGATED MODEL DEPLOYMENT")
    
    ti = context['task_instance']
    model_promoted = ti.xcom_pull(task_ids='promote_mitigated_model', key='model_promoted')
    
    if not model_promoted:
        log.info("Model was not promoted to production, skipping deployment")
        return {'deployed': False, 'reason': 'Model not promoted'}
    
    best_model = ti.xcom_pull(task_ids='run_integrated_pipeline', key='best_model')
    production_version = ti.xcom_pull(task_ids='promote_mitigated_model', key='production_version')
    
    metadata = {
        'model_type': best_model,
        'version': production_version,
        'deployed_date': context['ds'],
        'deployment_timestamp': datetime.now().isoformat(),
        'bias_mitigated': True,
        'metrics': {
            'baseline_r2': ti.xcom_pull(task_ids='run_integrated_pipeline', key='baseline_test_r2'),
            'baseline_mae': ti.xcom_pull(task_ids='run_integrated_pipeline', key='baseline_test_mae'),
            'mitigated_r2': ti.xcom_pull(task_ids='run_integrated_pipeline', key='mitigated_test_r2'),
            'mitigated_mae': ti.xcom_pull(task_ids='run_integrated_pipeline', key='mitigated_test_mae'),
            'mitigated_rmse': ti.xcom_pull(task_ids='run_integrated_pipeline', key='mitigated_test_rmse'),
            'r2_improvement': ti.xcom_pull(task_ids='run_integrated_pipeline', key='r2_improvement'),
            'mae_improvement': ti.xcom_pull(task_ids='run_integrated_pipeline', key='mae_improvement')
        },
        'bias_metrics': {
            'baseline_issues': ti.xcom_pull(task_ids='run_integrated_pipeline', key='baseline_bias_issues'),
            'mitigated_issues': ti.xcom_pull(task_ids='run_integrated_pipeline', key='mitigated_bias_issues'),
            'issues_reduced': ti.xcom_pull(task_ids='run_integrated_pipeline', key='bias_issues_reduction')
        },
        'airflow_run_id': context['run_id'],
        'model_path': '/opt/airflow/models/production/current_model.pkl'
    }
    
    metadata_path = '/opt/airflow/models/production/current_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log.info(f"\nBias-mitigated model deployment completed")
    log.info(f"  Model Type: {best_model}")
    log.info(f"  Version: {production_version}")
    log.info(f"  Mitigated R²: {metadata['metrics']['mitigated_r2']:.4f}")
    log.info(f"  Mitigated MAE: {metadata['metrics']['mitigated_mae']:.2f}")
    log.info(f"  Bias Issues: {metadata['bias_metrics']['baseline_issues']} -> {metadata['bias_metrics']['mitigated_issues']}")
    log.info(f"  Model Path: {metadata['model_path']}")
    
    with open('/opt/airflow/models/production/CURRENT_VERSION.txt', 'w') as f:
        f.write(f"Version: {production_version}\n")
        f.write(f"Model: {best_model}\n")
        f.write(f"Deployed: {context['ds']}\n")
        f.write(f"Bias Mitigated: Yes\n")
        f.write(f"Mitigated R²: {metadata['metrics']['mitigated_r2']:.4f}\n")
        f.write(f"Mitigated MAE: {metadata['metrics']['mitigated_mae']:.2f}\n")
        f.write(f"Bias Issues Reduced: {metadata['bias_metrics']['issues_reduced']}\n")
    
    return {'deployed': True, 'metadata': metadata}



def generate_monitoring_baseline(**context):
    """
    Generate Evidently AI baseline after model deployment.
    This enables drift monitoring for the newly deployed model.
    """
    import sys
    # sys.path.insert(0, '/opt/airflow/model_pipeline/monitoring')
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline/monitoring')
    
    log.info("="*60)
    log.info("GENERATING MONITORING BASELINE")
    log.info("="*60)
    
    ti = context['task_instance']
    
    # Check if model was promoted
    model_promoted = ti.xcom_pull(task_ids='promote_mitigated_model', key='model_promoted')
    
    if not model_promoted:
        log.info("Model was not promoted, skipping baseline generation")
        return {'generated': False, 'reason': 'Model not promoted'}
    
    try:
        from baseline_stats import generate_baseline_from_training
        
        # Generate baseline from the newly deployed model
        baseline_path = generate_baseline_from_training()
        
        log.info(f"✓ Monitoring baseline generated: {baseline_path}")
        
        return {
            'generated': True,
            'baseline_path': str(baseline_path),
        }
        
    except Exception as e:
        log.error(f"Failed to generate monitoring baseline: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the DAG - baseline can be generated manually
        return {'generated': False, 'error': str(e)}

def cleanup_temp_files(**context):
    """Cleanup temporary files"""
    import glob
    
    log.info("="*60)
    log.info("CLEANUP")
    log.info("="*60)
    
    patterns = [
        f'/tmp/integrated_script_{context["ds_nodash"]}.py',
        f'/tmp/integrated_results_{context["ds_nodash"]}.json'
    ]
    
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                log.info(f"Removed {file}")
            except Exception as e:
                log.info(f"Could not remove {file}: {e}")

with dag:
    start = DummyOperator(task_id='start')
    
    run_pipeline = PythonOperator(
        task_id='run_integrated_pipeline',
        python_callable=run_integrated_pipeline,
        provide_context=True
    )
    
    validate = PythonOperator(
        task_id='validate_mitigated_model',
        python_callable=validate_mitigated_model,
        provide_context=True
    )
    
    promote = PythonOperator(
        task_id='promote_mitigated_model',
        python_callable=promote_mitigated_model,
        provide_context=True
    )
    
    deploy = PythonOperator(
        task_id='deploy_mitigated_model',
        python_callable=deploy_mitigated_model,
        provide_context=True
    )
    
    push_to_github = PythonOperator(
        task_id='push_model_to_github',
        python_callable=push_model_to_github,
        provide_context=True
    )

    generate_baseline = PythonOperator(
        task_id='generate_monitoring_baseline',
        python_callable=generate_monitoring_baseline,
        provide_context=True,
    )

    cleanup = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup_temp_files,
        provide_context=True,
        trigger_rule='none_failed_min_one_success'
    )
    
    end = DummyOperator(
        task_id='end',
        trigger_rule='all_success'
    )
    
    failure_cleanup = PythonOperator(
        task_id='failure_cleanup',
        python_callable=cleanup_temp_files,
        provide_context=True,
        trigger_rule='one_failed'
    )
    
    start >> run_pipeline >> validate >> promote >> deploy >> [push_to_github, generate_baseline] >> cleanup >> end
    [run_pipeline, validate, promote, deploy] >> failure_cleanup