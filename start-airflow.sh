#!/bin/bash
# ============================================================
# BlueBikes MLOps - Start Airflow Services
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "============================================================"
echo "   Starting BlueBikes MLOps Pipeline"
echo "============================================================"
echo -e "${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found!${NC}"
    echo "Please run ./setup.sh first or copy .env.example to .env"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}Error: Docker is not running!${NC}"
    echo "Please start Docker and try again."
    exit 1
fi

# Start services
echo -e "${YELLOW}Starting services...${NC}"
docker compose up -d

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check service health
echo -e "\n${YELLOW}Checking service status...${NC}"
docker compose ps

echo -e "\n${GREEN}"
echo "============================================================"
echo "   Services Started Successfully! 🚀"
echo "============================================================"
echo -e "${NC}"
echo ""
echo "Access points:"
echo "  • Airflow UI:  ${BLUE}http://localhost:8080${NC}"
echo "  • Username:    airflow (or as configured in .env)"
echo "  • Password:    airflow (or as configured in .env)"
echo ""
echo "Useful commands:"
echo "  • View logs:   docker compose logs -f"
echo "  • Stop:        ./stop-airflow.sh"
echo "  • Status:      docker compose ps"
echo ""