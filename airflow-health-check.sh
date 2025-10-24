#!/bin/bash
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/data_pipeline" && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yaml"

echo "Checking Docker and Airflow environment status..."
echo "----------------------------------------------------------"

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is NOT running. Please start Docker."
  exit 1
else
  echo "Docker daemon is running."
fi

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "docker-compose.yaml not found in $PROJECT_DIR"
  exit 1
fi

echo ""
echo "Docker Compose Containers:"
docker compose -f "$COMPOSE_FILE" ps --format "table {{.Names}}\t{{.Status}}\t{{.Health}}" || true
echo "----------------------------------------------------------"
echo ""

healthy_containers=$(docker compose -f "$COMPOSE_FILE" ps --format "{{.Health}}" | grep -c "healthy" || true)
total_containers=$(docker compose -f "$COMPOSE_FILE" ps --format "{{.Names}}" | wc -l | tr -d ' ')

if [ "$healthy_containers" -eq "$total_containers" ] && [ "$total_containers" -gt 0 ]; then
  echo "All $total_containers containers are healthy."
else
  echo "$healthy_containers / $total_containers containers are healthy."
  echo "Use 'docker compose -f $COMPOSE_FILE ps' or 'docker compose -f $COMPOSE_FILE logs <service>' for details."
fi

echo ""
docker compose -f "$COMPOSE_FILE" ps --format "{{.Names}} : {{.Status}}" || true

echo ""
echo "Waiting 10 seconds for Airflow Webserver to fully start..."
sleep 10
echo ""

WEB_PORT=8080  
if curl -s -I "http://localhost:$WEB_PORT" | grep -q "200 OK"; then
  echo "Airflow Web UI is reachable at http://localhost:$WEB_PORT"
else
  echo "Airflow Web UI not reachable on port $WEB_PORT."
  echo "Try running: docker compose -f $COMPOSE_FILE logs airflow-webserver"
fi

echo ""
echo "Health check complete."
echo "----------------------------------------------------------"
