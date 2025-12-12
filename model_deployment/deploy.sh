#!/bin/bash

# deploy.sh - Deploy Bluebikes prediction service to Cloud Run

# Configuration
PROJECT_ID="mlops-480416"
REGION="us-central1"
SERVICE_NAME="bluebikes-prediction"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
GCS_BUCKET="mlruns234"
MODEL_PATH="models/production/current_model.pkl"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Starting deployment of Bluebikes prediction service...${NC}"

# Step 1: Build Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:latest .

if [ $? -ne 0 ]; then
    echo -e "${RED}Docker build failed${NC}"
    exit 1
fi

# Step 2: Push to Container Registry
echo -e "${GREEN}Pushing to Container Registry...${NC}"
docker push ${IMAGE_NAME}:latest

if [ $? -ne 0 ]; then
    echo -e "${RED}Docker push failed${NC}"
    exit 1
fi

# Step 3: Deploy to Cloud Run
echo -e "${GREEN}Deploying to Cloud Run...${NC}"
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10 \
    --timeout 300 \
    --set-env-vars GCS_BUCKET=${GCS_BUCKET},MODEL_PATH=${MODEL_PATH} \
    --project ${PROJECT_ID}

if [ $? -ne 0 ]; then
    echo -e "${RED}Deployment failed${NC}"
    exit 1
fi

# Step 4: Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)' \
    --project ${PROJECT_ID})

echo -e "${GREEN}  Deployment successful!${NC}"
echo -e "${GREEN}Service URL: ${SERVICE_URL}${NC}"

# Step 5: Test the deployment
echo -e "${YELLOW}Testing deployment...${NC}"

# Test health endpoint
echo -e "Testing health endpoint..."
curl -s ${SERVICE_URL}/health | python -m json.tool

# Test prediction endpoint
echo -e "\nTesting prediction endpoint..."
curl -X POST ${SERVICE_URL}/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [14, 2, 6, 75.0, 60.0, 0, 0]}' \
    -s | python -m json.tool

echo -e "${GREEN}Deployment complete and tested!${NC}"