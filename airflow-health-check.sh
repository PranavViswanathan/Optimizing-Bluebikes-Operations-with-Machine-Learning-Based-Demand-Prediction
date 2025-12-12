#!/bin/bash
set -e
COMPOSE_FILE="docker-compose.yaml"

echo "Checking Docker and Airflow environment status"
echo "----------------------------------------------------------"

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is NOT running. Please start Docker."
  exit 1
else
  echo "  Docker daemon is running."
fi

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "✗ docker-compose.yaml not found in $PROJECT_DIR"
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
  echo "  All $total_containers containers are healthy."
else
  echo "⚠ $healthy_containers / $total_containers containers are healthy."
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
  echo "  Airflow Web UI is reachable at http://localhost:$WEB_PORT (HTTP $HTTP_RESPONSE)"
else
  echo "✗ Airflow Web UI not reachable on port $WEB_PORT (HTTP $HTTP_RESPONSE)."
  echo "Try running: docker compose -f $COMPOSE_FILE logs airflow-webserver"
fi

echo ""
echo "----------------------------------------------------------"
echo "Checking DAG Health..."
echo "----------------------------------------------------------"

echo "Scanning for broken DAGs..."
BROKEN_DAGS=$(docker compose -f "$COMPOSE_FILE" exec -T airflow-scheduler airflow dags list-import-errors 2>/dev/null || echo "")

if [ -z "$BROKEN_DAGS" ] || echo "$BROKEN_DAGS" | grep -q "No data found"; then
  echo "  No broken DAGs found"
else
  echo "✗ BROKEN DAGs detected:"
  echo "$BROKEN_DAGS"
  echo ""
  echo "To see detailed error logs, run:"
  echo "  docker compose -f $COMPOSE_FILE logs airflow-scheduler | grep -A 20 'Broken DAG'"
fi

echo ""

echo "Current DAGs:"
docker compose -f "$COMPOSE_FILE" exec -T airflow-scheduler \
  airflow dags list --output table 2>/dev/null || echo "Could not retrieve DAG list"

echo ""

echo "Recent scheduler errors (last 50 lines):"
docker compose -f "$COMPOSE_FILE" logs --tail=50 airflow-scheduler 2>/dev/null | \
  grep -E "(ERROR|CRITICAL|Broken DAG|ModuleNotFoundError|ImportError)" || echo "  No recent errors in scheduler logs"

echo ""
echo "Health check complete."
echo "----------------------------------------------------------"
echo ""
echo "Quick Commands:"
echo "  View DAG errors:     docker compose -f $COMPOSE_FILE exec airflow-scheduler airflow dags list-import-errors"
echo "  View scheduler logs: docker compose -f $COMPOSE_FILE logs -f airflow-scheduler"
echo "  View webserver logs: docker compose -f $COMPOSE_FILE logs -f airflow-webserver"
echo "  Test DAG parse:      docker compose -f $COMPOSE_FILE exec airflow-scheduler python /opt/airflow/dags/data_pipeline_dag.py"