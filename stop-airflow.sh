#!/bin/bash
# ============================================================
# BlueBikes MLOps - Stop Airflow Services
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}"
echo "============================================================"
echo "   Stopping BlueBikes MLOps Pipeline"
echo "============================================================"
echo -e "${NC}"

# Check for flags
REMOVE_VOLUMES=false
if [ "$1" == "--clean" ] || [ "$1" == "-c" ]; then
    REMOVE_VOLUMES=true
    echo -e "${RED}Warning: This will remove all data volumes!${NC}"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Stop services
echo -e "${YELLOW}Stopping services...${NC}"

if [ "$REMOVE_VOLUMES" = true ]; then
    docker compose down -v
    echo -e "${RED}Volumes removed.${NC}"
else
    docker compose down
fi

echo -e "\n${GREEN}"
echo "============================================================"
echo "   Services Stopped Successfully"
echo "============================================================"
echo -e "${NC}"
echo ""
echo "To restart: ./start-airflow.sh"
echo ""