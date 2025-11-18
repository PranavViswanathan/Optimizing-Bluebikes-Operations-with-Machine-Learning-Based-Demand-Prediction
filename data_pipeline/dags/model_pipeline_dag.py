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
    'bluebikes_model_training',
    default_args=default_args,
    description='Train, version, and deploy BlueBikes demand prediction models',
    schedule_interval='@weekly',
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'training', 'bluebikes', 'production'],
    on_success_callback=send_dag_success_alert,  
    on_failure_callback=send_discord_alert, 
)

def run_training_pipeline(**context):
    import subprocess
    import json
    
    print(" Starting BlueBikes Model Training Pipeline ")
    print(f"Execution Date: {context['ds']}")
    print(f"Run ID: {context['run_id']}")
    training_script = """
import sys
import os
sys.path.append('/opt/airflow/model_pipeline/mlflow')
os.chdir('/opt/airflow/model_pipeline/mlflow')

from exp_tracking import BlueBikesModelTrainer
import joblib

# Initialize trainer
trainer = BlueBikesModelTrainer(experiment_name='bluebikes_airflow_{date}')

# Load and prepare data - Returns 6 values
X_train, X_test, X_val, y_train, y_test, y_val = trainer.load_and_prepare_data()

# Train models - Pass all 6 datasets
models_to_train = ['xgboost', 'lightgbm', 'randomforest']
results = trainer.train_all_models(
    X_train, X_test, X_val, y_train, y_test, y_val,
    models_to_train=models_to_train,
    tune=False
)

if results:
    comparison_df = trainer.compare_models(results)
    best_model_name, best_model, best_metrics, best_run_id = trainer.select_best_model(
        results, 
        metric='test_r2'
    )
    
    # Save outputs
    comparison_df.to_csv('/tmp/model_comparison_{date}.csv', index=False)
    joblib.dump(best_model, '/tmp/best_model_{date}.pkl')
    
    # Note: Removed register_model call - using custom versioning instead
    
    # Save results for Airflow
    import json
    results_dict = {{
        'best_model': best_model_name,
        'test_r2': float(best_metrics['test_r2']),
        'test_mae': float(best_metrics['test_mae']),
        'test_rmse': float(best_metrics['test_rmse']),
        'run_id': best_run_id
    }}
    
    with open('/tmp/training_results_{date}.json', 'w') as f:
        json.dump(results_dict, f)
    
    print(f"Best Model: {{best_model_name}}")
    print(f"Test R²: {{best_metrics['test_r2']:.4f}}")
else:
    raise Exception("No models trained successfully")
""".format(date=context['ds_nodash'])
    
    script_path = f'/tmp/training_script_{context["ds_nodash"]}.py'
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True,
            cwd='/opt/airflow/model_pipeline/mlflow'
        )
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        results_file = f'/tmp/training_results_{context["ds_nodash"]}.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            for key, value in results.items():
                context['task_instance'].xcom_push(key=key, value=value)
            
            print("="*80)
            print(f" Pipeline Complete! Best Model: {results['best_model'].upper()} ")
            print(f" Test R²: {results['test_r2']:.4f}")
            print("="*80)
            
            return results
        else:
            raise Exception("Results file not found")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running training script: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

def validate_model_performance(**context):
    ti = context['task_instance']
    test_r2 = ti.xcom_pull(task_ids='run_training', key='test_r2')
    test_mae = ti.xcom_pull(task_ids='run_training', key='test_mae')
    best_model = ti.xcom_pull(task_ids='run_training', key='best_model')
    
    if test_r2 is None:
        results_file = f'/tmp/training_results_{context["ds_nodash"]}.json'
        if os.path.exists(results_file):
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            test_r2 = results.get('test_r2')
            test_mae = results.get('test_mae')
            best_model = results.get('best_model')
    
    MIN_R2 = 0.70
    MAX_MAE = 110
    
    print("="*60)
    print("MODEL VALIDATION")
    print("="*60)
    
    if test_r2 and test_mae:
        print(f"Model: {best_model}")
        print(f"R² Score: {test_r2:.4f} (threshold: >{MIN_R2})")
        print(f"MAE: {test_mae:.2f} (threshold: <{MAX_MAE})")
        
        if test_r2 >= MIN_R2 and test_mae <= MAX_MAE:
            print("✓ Model passed validation!")
            return True
        else:
            print("✗ Model failed validation")
            raise Exception("Model did not meet performance thresholds")
    else:
        print("⚠ No metrics found for validation")
        return False

def promote_model_to_production(**context):
    import json
    import shutil
    from datetime import datetime
    print("MODEL PROMOTION DECISION (Custom Versioning)")
    
    ti = context['task_instance']
    new_model_name = ti.xcom_pull(task_ids='run_training', key='best_model')
    new_test_r2 = ti.xcom_pull(task_ids='run_training', key='test_r2')
    new_test_mae = ti.xcom_pull(task_ids='run_training', key='test_mae')
    new_test_rmse = ti.xcom_pull(task_ids='run_training', key='test_rmse')
    new_run_id = ti.xcom_pull(task_ids='run_training', key='run_id')
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
            promotion_reason = "First production deployment"
            print("No existing production model - will deploy first version")
        else:
            r2_improvement = new_test_r2 - current_prod_metrics['test_r2']
            mae_improvement = current_prod_metrics['test_mae'] - new_test_mae
            
            print(f"\nCurrent Production Model (v{current_prod_version}):")
            print(f"  R²: {current_prod_metrics['test_r2']:.4f}")
            print(f"  MAE: {current_prod_metrics['test_mae']:.2f}")
            print(f"\nNew Model:")
            print(f"  R²: {new_test_r2:.4f} (Δ: {r2_improvement:+.4f})")
            print(f"  MAE: {new_test_mae:.2f} (Δ: {mae_improvement:+.2f})")
            
            if r2_improvement > 0.01 and mae_improvement > 0:
                should_promote = True
                promotion_reason = f"Better performance: R² +{r2_improvement:.4f}, MAE -{mae_improvement:.2f}"
            elif r2_improvement > 0.02:
                should_promote = True
                promotion_reason = f"Significant R² improvement: +{r2_improvement:.4f}"
            elif mae_improvement > 5:
                should_promote = True
                promotion_reason = f"Significant MAE improvement: -{mae_improvement:.2f}"
            else:
                promotion_reason = "Performance improvement below thresholds"
        new_version = len(version_history["versions"]) + 1
        
        temp_model_path = f'/tmp/best_model_{context["ds_nodash"]}.pkl'
        versioned_model_path = f"{version_dir}/model_v{new_version}_{context['ds_nodash']}.pkl"
        
        if os.path.exists(temp_model_path):
            shutil.copy(temp_model_path, versioned_model_path)
            print(f"\nSaved model as version {new_version}")
        else:
            raise Exception(f"Model file not found at {temp_model_path}")
        
        if should_promote:
            print(f"\nPROMOTING MODEL TO PRODUCTION")
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
                "model_type": new_model_name,
                "created_date": context['ds'],
                "promoted_date": context['ds'],
                "promotion_reason": promotion_reason,
                "mlflow_run_id": new_run_id,
                "file_path": versioned_model_path,
                "metrics": {
                    "test_r2": new_test_r2,
                    "test_mae": new_test_mae,
                    "test_rmse": new_test_rmse
                },
                "status": "production"
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
                "model_type": new_model_name,
                "created_date": context['ds'],
                "promotion_reason": promotion_reason,
                "mlflow_run_id": new_run_id,
                "file_path": versioned_model_path,
                "metrics": {
                    "test_r2": new_test_r2,
                    "test_mae": new_test_mae,
                    "test_rmse": new_test_rmse
                },
                "status": "staging"
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

def deploy_model(**context):
    import json
    from datetime import datetime
    print("MODEL DEPLOYMENT")
    
    ti = context['task_instance']
    model_promoted = ti.xcom_pull(task_ids='promote_model', key='model_promoted')
    
    if not model_promoted:
        print("Model was not promoted to production, skipping deployment")
        return {'deployed': False, 'reason': 'Model not promoted'}
    
    best_model_name = ti.xcom_pull(task_ids='run_training', key='best_model')
    production_version = ti.xcom_pull(task_ids='promote_model', key='production_version')
    
    metadata = {
        'model_type': best_model_name,
        'version': production_version,
        'deployed_date': context['ds'],
        'deployment_timestamp': datetime.now().isoformat(),
        'test_r2': ti.xcom_pull(task_ids='run_training', key='test_r2'),
        'test_mae': ti.xcom_pull(task_ids='run_training', key='test_mae'),
        'test_rmse': ti.xcom_pull(task_ids='run_training', key='test_rmse'),
        'mlflow_run_id': ti.xcom_pull(task_ids='run_training', key='run_id'),
        'airflow_run_id': context['run_id'],
        'model_path': '/opt/airflow/models/production/current_model.pkl'
    }
    metadata_path = '/opt/airflow/models/production/current_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel deployment completed")
    print(f"  Model Type: {best_model_name}")
    print(f"  Version: {production_version}")
    print(f"  Performance: R²={metadata['test_r2']:.4f}, MAE={metadata['test_mae']:.2f}")
    print(f"  Model Path: {metadata['model_path']}")
    print(f"  Metadata: {metadata_path}")
    with open('/opt/airflow/models/production/CURRENT_VERSION.txt', 'w') as f:
        f.write(f"Version: {production_version}\n")
        f.write(f"Model: {best_model_name}\n")
        f.write(f"Deployed: {context['ds']}\n")
        f.write(f"R²: {metadata['test_r2']:.4f}\n")
        f.write(f"MAE: {metadata['test_mae']:.2f}\n")
    
    return {'deployed': True, 'metadata': metadata}

def cleanup_temp_files(**context):
    import glob
    
    print("="*60)
    print("CLEANUP")
    print("="*60)
    
    patterns = [
        f'/tmp/training_script_{context["ds_nodash"]}.py',
        f'/tmp/training_results_{context["ds_nodash"]}.json',
        f'/tmp/model_comparison_{context["ds_nodash"]}.csv',
        f'/tmp/best_model_{context["ds_nodash"]}.pkl'
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
    
    run_training = PythonOperator(
        task_id='run_training',
        python_callable=run_training_pipeline,  
        provide_context=True
    )
    
    validate = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model_performance,
        provide_context=True
    )
    
    promote = PythonOperator(
        task_id='promote_model',
        python_callable=promote_model_to_production,
        provide_context=True
    )
    
    deploy = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
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
    
    start >> run_training >> validate >> promote >> deploy >> cleanup >> end
    [run_training, validate, promote, deploy] >> failure_cleanup