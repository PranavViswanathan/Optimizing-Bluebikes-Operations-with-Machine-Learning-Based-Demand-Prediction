# scripts/discord_notifier.py

import sys
from pathlib import Path
import requests
import os
from datetime import datetime

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from logger import get_logger

logger = get_logger("discord_notifier")

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')


def send_discord_alert(context):
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not set. Skipping Discord notification.")
        return
    task_instance = context.get('task_instance')
    dag_id = context.get('dag').dag_id
    task_id = task_instance.task_id
    execution_date = context.get('execution_date')
    exception = context.get('exception')
    log_url = task_instance.log_url

    embed = {
        "title": "Airflow Task Failure",
        "color": 15158332,  
        "fields": [
            {
                "name": "DAG",
                "value": f"`{dag_id}`",
                "inline": True
            },
            {
                "name": "Task",
                "value": f"`{task_id}`",
                "inline": True
            },
            {
                "name": "Execution Date",
                "value": str(execution_date),
                "inline": False
            },
            {
                "name": "Error",
                "value": f"```{str(exception)[:1000]}```",
                "inline": False
            },
            {
                "name": "Log URL",
                "value": f"[View Logs]({log_url})" if log_url else "N/A",
                "inline": False
            }
        ],
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {
            "text": "Bluebikes Data Pipeline"
        }
    }
    
    payload = {
        "username": "Airflow Alert Bot",
        "embeds": [embed]
    }
    
    try:
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"Discord notification sent for {dag_id}.{task_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Discord notification: {e}")


def send_dag_success_alert(context):
    if not DISCORD_WEBHOOK_URL:
        return
    
    dag_id = context.get('dag').dag_id
    execution_date = context.get('execution_date')
    
    embed = {
        "title": "Airflow DAG Success",
        "color": 3066993, 
        "fields": [
            {
                "name": "DAG",
                "value": f"`{dag_id}`",
                "inline": True
            },
            {
                "name": "Execution Date",
                "value": str(execution_date),
                "inline": True
            }
        ],
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {
            "text": "Bluebikes Data Pipeline"
        }
    }
    
    payload = {
        "username": "Airflow Alert Bot",
        "embeds": [embed]
    }
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Discord success notification sent for {dag_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Discord notification: {e}")