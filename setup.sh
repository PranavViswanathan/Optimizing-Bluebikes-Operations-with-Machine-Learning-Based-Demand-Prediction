#!/bin/bash
# ============================================================
# BlueBikes MLOps Project - Setup Script
# ============================================================
# This script helps you set up the project from scratch
# Run with: chmod +x setup.sh && ./setup.sh
# ============================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo "   BlueBikes MLOps Project - Setup Wizard"
echo "============================================================"
echo -e "${NC}"

# ------------------------------------------------------------
# Step 1: Check Prerequisites
# ------------------------------------------------------------
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "  ${GREEN} ${NC} $1 is installed"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 is NOT installed"
        return 1
    fi
}

MISSING_DEPS=0

check_command "docker" || MISSING_DEPS=1
check_command "docker-compose" || check_command "docker compose" || MISSING_DEPS=1
check_command "git" || MISSING_DEPS=1
check_command "python3" || MISSING_DEPS=1

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "\n${RED}Please install missing dependencies before continuing.${NC}"
    echo "  - Docker: https://docs.docker.com/get-docker/"
    echo "  - Git: https://git-scm.com/downloads"
    echo "  - Python 3.10+: https://www.python.org/downloads/"
    exit 1
fi

echo -e "${GREEN}All prerequisites installed!${NC}\n"

# ------------------------------------------------------------
# Step 2: Environment Configuration
# ------------------------------------------------------------
echo -e "${YELLOW}Step 2: Setting up environment configuration...${NC}"

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "  ${GREEN} ${NC} Created .env from .env.example"
        echo -e "  ${YELLOW}!${NC} Please edit .env and add your API keys"
    else
        echo -e "  ${RED}✗${NC} .env.example not found!"
        exit 1
    fi
else
    echo -e "  ${GREEN} ${NC} .env already exists"
fi

# ------------------------------------------------------------
# Step 3: Create Required Directories
# ------------------------------------------------------------
echo -e "\n${YELLOW}Step 3: Creating required directories...${NC}"

DIRS=(
    "keys"
    "data_pipeline/data/raw/bluebikes"
    "data_pipeline/data/raw/NOAA_weather"
    "data_pipeline/data/raw/boston_clg"
    "data_pipeline/data/processed/bluebikes"
    "data_pipeline/data/processed/NOAA_weather"
    "data_pipeline/data/processed/boston_clg"
    "data_pipeline/logs"
    "model_pipeline/models/production"
    "model_pipeline/models/versions"
    "model_pipeline/mlruns"
    "model_pipeline/artifacts"
    "model_pipeline/monitoring/baselines"
    "model_pipeline/monitoring/reports/html"
    "model_pipeline/monitoring/reports/json"
    "model_pipeline/monitoring/logs"
    "model_pipeline/monitoring/predictions"
)

for dir in "${DIRS[@]}"; do
    mkdir -p "$dir"
    echo -e "  ${GREEN} ${NC} Created $dir"
done

# ------------------------------------------------------------
# Step 4: Check for GCS Service Account Key
# ------------------------------------------------------------
echo -e "\n${YELLOW}Step 4: Checking GCS service account key...${NC}"

if [ -f "keys/gcs_service_account.json" ]; then
    echo -e "  ${GREEN} ${NC} GCS service account key found"
else
    echo -e "  ${YELLOW}!${NC} GCS service account key NOT found"
    echo ""
    echo "  To use Google Cloud features, you need to:"
    echo "  1. Go to GCP Console: https://console.cloud.google.com"
    echo "  2. Create a service account with these roles:"
    echo "     - Storage Admin"
    echo "     - Storage Object Admin"
    echo "  3. Download the JSON key file"
    echo "  4. Save it as: keys/gcs_service_account.json"
    echo ""
    echo "  For now, the project will run in LOCAL MODE (no cloud features)"
    
    # Create a placeholder file
    echo '{"note": "Replace this with your actual GCS service account key"}' > keys/gcs_service_account.json
fi

# ------------------------------------------------------------
# Step 5: Set Airflow UID
# ------------------------------------------------------------
echo -e "\n${YELLOW}Step 5: Setting Airflow user permissions...${NC}"

# Get current user ID
CURRENT_UID=$(id -u)

# Update .env with correct UID
if grep -q "AIRFLOW_UID=" .env; then
    sed -i.bak "s/AIRFLOW_UID=.*/AIRFLOW_UID=$CURRENT_UID/" .env
else
    echo "AIRFLOW_UID=$CURRENT_UID" >> .env
fi
echo -e "  ${GREEN} ${NC} Set AIRFLOW_UID=$CURRENT_UID"

# ------------------------------------------------------------
# Step 6: Build Docker Images
# ------------------------------------------------------------
echo -e "\n${YELLOW}Step 6: Building Docker images...${NC}"
echo "  This may take several minutes on first run..."

docker compose build --no-cache

echo -e "  ${GREEN} ${NC} Docker images built successfully"

# ------------------------------------------------------------
# Step 7: Initialize Airflow Database
# ------------------------------------------------------------
echo -e "\n${YELLOW}Step 7: Initializing Airflow...${NC}"

docker compose up airflow-init

echo -e "  ${GREEN} ${NC} Airflow initialized"

# ------------------------------------------------------------
# Complete!
# ------------------------------------------------------------
echo -e "\n${GREEN}"
echo "============================================================"
echo "   Setup Complete! 🎉"
echo "============================================================"
echo -e "${NC}"
echo "Next steps:"
echo ""
echo "  1. Edit .env file with your API keys:"
echo "     ${BLUE}nano .env${NC}"
echo ""
echo "  2. Start the services:"
echo "     ${BLUE}./start-airflow.sh${NC}"
echo "     or"
echo "     ${BLUE}docker compose up -d${NC}"
echo ""
echo "  3. Access Airflow UI:"
echo "     ${BLUE}http://localhost:8080${NC}"
echo "     Username: airflow (or what you set in .env)"
echo "     Password: airflow (or what you set in .env)"
echo ""
echo "  4. Trigger the data pipeline DAG to collect data"
echo ""
echo "For more info, see README.md"
echo ""