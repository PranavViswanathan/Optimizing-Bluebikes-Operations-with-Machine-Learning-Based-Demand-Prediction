from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import sys
import os

sys.path.insert(0, '/opt/airflow/model_pipeline/mlflow')
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from discord_notifier import send_discord_alert, send_dag_success_alert

default_args = {
    'owner': 'Pranav',
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
    print(" Starting Integrated BlueBikes Pipeline with Bias Detection ")
    print(f"Execution Date: {context['ds']}")
    print(f"Run ID: {context['run_id']}")
    
    date_str = context['ds_nodash']
    
    training_script = f"""
import sys
import os
import traceback

# Add paths
sys.path.insert(0, '/opt/airflow/model_pipeline/mlflow')
sys.path.append('/opt/airflow/model_pipeline/mlflow')
os.chdir('/opt/airflow/model_pipeline/mlflow')

# Define date variable for use in script
date_str = '{date_str}'

try:
    # Import from integrated_training_pipeline.py
    from integrated_training_pipeline import IntegratedBlueBikesTrainer
    import json

    # Initialize integrated trainer
    trainer = IntegratedBlueBikesTrainer(
        experiment_name='bluebikes_bias_integrated_' + date_str
    )

    # Run complete pipeline
    print("Starting integrated pipeline...")
    results = trainer.run_complete_pipeline(
        models_to_train=['xgboost', 'lightgbm', 'randomforest'],
        tune=False
    )

    if results:
        # Extract metrics
        baseline_metrics = results['baseline_metrics']
        mitigated_metrics = results['mitigated_metrics']
        comparison = results['comparison_report']
        
        # Prepare output for Airflow
        pipeline_results = {{
            'best_model': trainer.best_model_name,
            'baseline_test_r2': float(baseline_metrics['test_r2']),
            'baseline_test_mae': float(baseline_metrics['test_mae']),
            'baseline_test_rmse': float(baseline_metrics['test_rmse']),
            'mitigated_test_r2': float(mitigated_metrics['test_r2']),
            'mitigated_test_mae': float(mitigated_metrics['test_mae']),
            'mitigated_test_rmse': float(mitigated_metrics['test_rmse']),
            'baseline_bias_issues': comparison['baseline_bias_issues'],
            'mitigated_bias_issues': comparison['mitigated_bias_issues'],
            'r2_improvement': comparison['improvement']['r2_improvement'],
            'mae_improvement': comparison['improvement']['mae_improvement'],
            'bias_issues_reduction': comparison['improvement']['bias_issues_reduction'],
            'baseline_model_path': trainer.best_model_path,
            'mitigated_model_path': trainer.mitigated_model_path
        }}
        
        # Save results
        results_file = '/tmp/integrated_results_' + date_str + '.json'
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        print("="*80)
        print(" Pipeline Complete ")
        print("="*80)
        print(f"Best Model: {{trainer.best_model_name}}")
        print(f"Baseline R²: {{baseline_metrics['test_r2']:.4f}}")
        print(f"Mitigated R²: {{mitigated_metrics['test_r2']:.4f}}")
        print(f"Bias Issues: {{comparison['baseline_bias_issues']}} -> {{comparison['mitigated_bias_issues']}}")
        print(f"Results saved to: {{results_file}}")
    else:
        raise Exception("Pipeline failed to complete - results is None")

except Exception as e:
    print("="*80)
    print("PIPELINE ERROR")
    print("="*80)
    print(f"Error type: {{type(e).__name__}}")
    print(f"Error message: {{str(e)}}")
    print("\\nFull traceback:")
    traceback.print_exc()
    
    # Try to save partial results if possible
    print("\\nAttempting to save error information...")
    error_info = {{
        'error': str(e),
        'error_type': type(e).__name__,
        'traceback': traceback.format_exc()
    }}
    
    try:
        error_file = '/tmp/integrated_error_' + date_str + '.json'
        with open(error_file, 'w') as f:
            json.dump(error_info, f, indent=2)
        print(f"Error info saved to: {{error_file}}")
    except:
        print("Could not save error info")
    
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
            cwd='/opt/airflow/model_pipeline/mlflow'
        )
        
        print("="*80)
        print("SUBPROCESS OUTPUT")
        print("="*80)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        print("="*80)
        
        if result.returncode != 0:
            print(f"\nSubprocess exited with code {result.returncode}")
            results_file = f'/tmp/integrated_results_{date_str}.json'
            if os.path.exists(results_file):
                print("Found partial results file, attempting to use it...")
            else:
                print("No results file found")
                raise Exception(f"Subprocess failed with exit code {result.returncode}")
        
        results_file = f'/tmp/integrated_results_{date_str}.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            for key, value in results.items():
                context['task_instance'].xcom_push(key=key, value=value)
            
            print("="*80)
            print(" Results Summary ")
            print("="*80)
            print(f"Model: {results['best_model']}")
            print(f"Mitigated R²: {results['mitigated_test_r2']:.4f}")
            print(f"Mitigated MAE: {results['mitigated_test_mae']:.2f}")
            print(f"Bias Reduction: {results['bias_issues_reduction']} issues")
            
            return results
        else:
            print("\nResults file not found at:", results_file)
            print("\nChecking /tmp directory:")
            import glob
            tmp_files = glob.glob('/tmp/integrated_*')
            for f in tmp_files:
                print(f"  Found: {f}")
            
            raise Exception("Results file not found - pipeline may have crashed during execution")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running integrated pipeline: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
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
    
    print("="*60)
    print("MITIGATED MODEL VALIDATION")
    print("="*60)
    print(f"Model: {best_model}")
    print(f"Baseline R²: {baseline_r2:.4f}")
    print(f"Mitigated R²: {mitigated_r2:.4f} (threshold: >{MIN_R2})")
    print(f"Mitigated MAE: {mitigated_mae:.2f} (threshold: <{MAX_MAE})")
    print(f"Bias Issues Reduction: {bias_reduction} (threshold: >={MIN_BIAS_REDUCTION})")
    
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
        print("Mitigated model passed validation!")
        return True
    else:
        print("Mitigated model failed validation:")
        for reason in reasons:
            print(f"  - {reason}")
        raise Exception("Mitigated model did not meet validation thresholds")

def promote_mitigated_model(**context):
    """Promote bias-mitigated model to production"""
    import json
    import shutil
    from datetime import datetime
    
    print("="*60)
    print("MODEL PROMOTION DECISION (Bias-Mitigated)")
    print("="*60)
    
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
            print("No existing production model - deploying first version")
        else:
            r2_delta = mitigated_r2 - current_prod_metrics['test_r2']
            mae_delta = current_prod_metrics['test_mae'] - mitigated_mae
            
            print(f"\nCurrent Production Model (v{current_prod_version}):")
            print(f"  R²: {current_prod_metrics['test_r2']:.4f}")
            print(f"  MAE: {current_prod_metrics['test_mae']:.2f}")
            print(f"\nNew Bias-Mitigated Model:")
            print(f"  R²: {mitigated_r2:.4f} (Δ: {r2_delta:+.4f})")
            print(f"  MAE: {mitigated_mae:.2f} (Δ: {mae_delta:+.2f})")
            print(f"  Bias Improvement: {baseline_bias_issues} -> {mitigated_bias_issues} issues")
            print(f"  vs Baseline: R² {r2_improvement:+.4f}, MAE {mae_improvement:+.2f}")
            
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
        
        temp_model_path = f'/opt/airflow/model_pipeline/mlflow/mitigated_model_{model_name}.pkl'
        versioned_model_path = f"{version_dir}/model_v{new_version}_{context['ds_nodash']}_bias_mitigated.pkl"
        
        if os.path.exists(temp_model_path):
            shutil.copy(temp_model_path, versioned_model_path)
            print(f"\nSaved bias-mitigated model as version {new_version}")
        else:
            raise Exception(f"Model file not found at {temp_model_path}")
        
        if should_promote:
            print(f"\n✓ PROMOTING BIAS-MITIGATED MODEL TO PRODUCTION")
            print(f"  Reason: {promotion_reason}")
            
            if os.path.exists(production_link):
                os.remove(production_link)
            shutil.copy(versioned_model_path, production_link)
            print(f"  Updated production model link")
            
            if version_history["current_production"]:
                for v in version_history["versions"]:
                    if v["version"] == version_history["current_production"]:
                        v["status"] = "archived"
                        v["archived_date"] = context['ds']
                        print(f"  Archived previous version {v['version']}")
            
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
            print(f"\n✗ NOT PROMOTING MODEL")
            print(f"  Reason: {promotion_reason}")
            
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
            print(f"  Model saved as version {new_version} in staging")
            
            context['task_instance'].xcom_push(key='model_promoted', value=False)
            context['task_instance'].xcom_push(key='production_version', value=new_version)
        
        with open(metadata_file, 'w') as f:
            json.dump(version_history, f, indent=2)
        print(f"\nVersion history updated: {metadata_file}")
        
        return {'promoted': should_promote, 'reason': promotion_reason, 'version': new_version}
        
    except Exception as e:
        print(f"Error in model promotion: {e}")
        import traceback
        traceback.print_exc()
        raise

def deploy_mitigated_model(**context):
    """Deploy bias-mitigated model"""
    import json
    from datetime import datetime
    
    print("BIAS-MITIGATED MODEL DEPLOYMENT")
    
    ti = context['task_instance']
    model_promoted = ti.xcom_pull(task_ids='promote_mitigated_model', key='model_promoted')
    
    if not model_promoted:
        print("Model was not promoted to production, skipping deployment")
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
    
    print(f"\nBias-mitigated model deployment completed")
    print(f"  Model Type: {best_model}")
    print(f"  Version: {production_version}")
    print(f"  Mitigated R²: {metadata['metrics']['mitigated_r2']:.4f}")
    print(f"  Mitigated MAE: {metadata['metrics']['mitigated_mae']:.2f}")
    print(f"  Bias Issues: {metadata['bias_metrics']['baseline_issues']} -> {metadata['bias_metrics']['mitigated_issues']}")
    print(f"  Model Path: {metadata['model_path']}")
    
    with open('/opt/airflow/models/production/CURRENT_VERSION.txt', 'w') as f:
        f.write(f"Version: {production_version}\n")
        f.write(f"Model: {best_model}\n")
        f.write(f"Deployed: {context['ds']}\n")
        f.write(f"Bias Mitigated: Yes\n")
        f.write(f"Mitigated R²: {metadata['metrics']['mitigated_r2']:.4f}\n")
        f.write(f"Mitigated MAE: {metadata['metrics']['mitigated_mae']:.2f}\n")
        f.write(f"Bias Issues Reduced: {metadata['bias_metrics']['issues_reduced']}\n")
    
    return {'deployed': True, 'metadata': metadata}

def cleanup_temp_files(**context):
    """Cleanup temporary files"""
    import glob
    
    print("="*60)
    print("CLEANUP")
    print("="*60)
    
    patterns = [
        f'/tmp/integrated_script_{context["ds_nodash"]}.py',
        f'/tmp/integrated_results_{context["ds_nodash"]}.json'
    ]
    
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Removed {file}")
            except Exception as e:
                print(f"Could not remove {file}: {e}")

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
    
    start >> run_pipeline >> validate >> promote >> deploy >> cleanup >> end
    [run_pipeline, validate, promote, deploy] >> failure_cleanup