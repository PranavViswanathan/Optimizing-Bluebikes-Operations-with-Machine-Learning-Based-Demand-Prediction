#!/bin/bash
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/data_pipeline" && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yaml"

echo "Checking Docker and Airflow environment status"
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
echo "Waiting 10 seconds for Airflow Webserver to fully start"
sleep 10
echo ""

WEB_PORT=$(docker compose -f "$COMPOSE_FILE" port airflow-webserver 8080 | cut -d: -f2)

if [ -z "$WEB_PORT" ]; then
  echo "Could not detect host port for airflow-webserver. Using 8080 by default."
  WEB_PORT=8080
fi

echo "Checking Airflow webserver HTTP response on port $WEB_PORT..."
HTTP_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$WEB_PORT" || echo "000")

if [ "$HTTP_RESPONSE" = "200" ] || [ "$HTTP_RESPONSE" = "302" ] || [ "$HTTP_RESPONSE" = "301" ]; then
  echo "Airflow Web UI is reachable at http://localhost:$WEB_PORT (HTTP $HTTP_RESPONSE)"
else
  echo "Airflow Web UI not reachable on port $WEB_PORT (HTTP $HTTP_RESPONSE)."
  echo "Try running: docker compose -f $COMPOSE_FILE logs airflow-webserver"
fi

echo ""
echo "Health check complete."
echo "----------------------------------------------------------"