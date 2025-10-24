

set -e  
cd "$(dirname "$0")/data_pipeline" || exit 1

echo "Starting Airflow containers..."
docker compose up -d --build

echo "Airflow environment started successfully!"
echo "Access Airflow UI at: http://localhost:8080"
