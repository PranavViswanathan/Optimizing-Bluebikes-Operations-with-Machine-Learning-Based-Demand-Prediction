#!/bin/bash
set -e

cd "$(dirname "$0")/data_pipeline" || exit 1

echo "Stopping Airflow containers..."
docker compose down

echo " Removing unused resources..."
docker system prune -f

echo  "All containers stopped and cleaned up!"
