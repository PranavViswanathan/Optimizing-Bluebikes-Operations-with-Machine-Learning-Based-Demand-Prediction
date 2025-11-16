from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
import sys
import os
from pathlib import Path
sys.path.insert(0, '/opt/airflow/plugins/mlflow')

from exp_tracking import BlueBikesModelTrainer
import joblib
default_args = {
    'owner': 'data-team',
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
    description='Train and compare BlueBikes demand prediction models',
    schedule_interval='@weekly',
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'training', 'bluebikes'],
)

def run_training_pipeline(**context):
    import subprocess
    import json
    
    print("="*80)
    print(" Starting BlueBikes Model Training Pipeline ")
    print("="*80)
    print(f"Execution Date: {context['ds']}")
    print(f"Run ID: {context['run_id']}")
    training_script = """
import sys
import os
sys.path.append('/opt/airflow/model_pipeline/mlflow')
os.chdir('/opt/airflow/model_pipeline/mlflow')

# Now import and run
from exp_tracking import BlueBikesModelTrainer
import joblib

# Initialize trainer
trainer = BlueBikesModelTrainer(experiment_name='bluebikes_airflow_{date}')

# Load and prepare data
X_train, X_test, y_train, y_test = trainer.load_and_prepare_data()

# Train models
models_to_train = ['xgboost', 'lightgbm']
results = trainer.train_all_models(
    X_train, X_test, y_train, y_test,
    models_to_train=models_to_train
)

if results:
    # Compare models
    comparison_df = trainer.compare_models(results)
    
    # Select best model
    best_model_name, best_model, best_metrics, best_run_id = trainer.select_best_model(
        results, 
        metric='test_r2'
    )
    
    # Save outputs
    comparison_df.to_csv('/tmp/model_comparison_{date}.csv', index=False)
    joblib.dump(best_model, '/tmp/best_model_{date}.pkl')
    
    # Register model
    trainer.register_model(best_model_name, best_run_id, best_model_name)
    
    # Save results for Airflow
    import json
    results_dict = {{
        'best_model': best_model_name,
        'test_r2': float(best_metrics['test_r2']),
        'test_mae': float(best_metrics['test_mae']),
        'test_rmse': float(best_metrics['test_rmse'])
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
        
        # Read the results
        results_file = f'/tmp/training_results_{context["ds_nodash"]}.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Push to XCom
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
        # Cleanup
        if os.path.exists(script_path):
            os.remove(script_path)

def run_training_simple(**context):
    import subprocess
    
    print("="*80)
    print(" Running Model Training Script ")
    print("="*80)
    cmd = [
        sys.executable,
        '/opt/airflow/model_pipeline/mlflow/exp_tracking.py'
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd='/opt/airflow/model_pipeline/mlflow',
        env={**os.environ, 'PYTHONPATH': '/opt/airflow/model_pipeline/mlflow'}
    )
    
    print("Output:", result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    if result.returncode != 0:
        raise Exception(f"Training script failed with return code {result.returncode}")
    return {'status': 'success', 'return_code': result.returncode}

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
    MAX_MAE = 100
    
    print("="*60)
    print("MODEL VALIDATION")
    print("="*60)
    
    if test_r2 and test_mae:
        print(f"Model: {best_model}")
        print(f"R² Score: {test_r2:.4f} (threshold: >{MIN_R2})")
        print(f"MAE: {test_mae:.2f} (threshold: <{MAX_MAE})")
        
        if test_r2 >= MIN_R2 and test_mae <= MAX_MAE:
            print("Model passed validation!")
            return True
        else:
            print("Model failed validation")
            raise Exception("Model did not meet performance thresholds")
    else:
        print("No metrics found for validation")
        return False

def cleanup_temp_files(**context):
    import glob
    
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
                print(f" Removed {file}")
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
    cleanup = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup_temp_files,
        provide_context=True,
        trigger_rule='all_done'
    )
    end = DummyOperator(
        task_id='end',
        trigger_rule='all_done'
    )
    start >> run_training >> validate >> cleanup >> end