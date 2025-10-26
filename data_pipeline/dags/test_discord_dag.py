from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from discord_notifier import send_discord_alert, send_dag_success_alert

def test_success():
    print("This task will succeed and trigger success notification")
    return "Success!"

def test_failure():
    print("This task will fail and trigger failure notification")
    raise Exception("TEST FAILURE - Discord notification test!")

with DAG(
    dag_id='test_discord_notifications',
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  
    catchup=False,
    default_args={
        'on_failure_callback': send_discord_alert,
    },
    on_success_callback=send_dag_success_alert,
    tags=['test', 'discord']
) as dag:
    
    success_task = PythonOperator(
        task_id='test_success_notification',
        python_callable=test_success,
    )
    
    failure_task = PythonOperator(
        task_id='test_failure_notification',
        python_callable=test_failure,
    )
    
    success_task >> failure_task 