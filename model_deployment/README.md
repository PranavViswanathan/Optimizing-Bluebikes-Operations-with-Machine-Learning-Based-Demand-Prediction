# Model Deployment

## Overview
This section covers the deployment architecture for the Bluebikes demand prediction model and web application. This system uses Google Cloud Platform services to provide scalable, serverless model serving with automatic updates.

## Architecture Components

### 1. Model Serving (Cloud Run)
- **Service**: Containerized Flask API deployed on Google Cloud Run
- **URL**: `https://bluebikes-prediction-202855070348.us-central1.run.app`
- **Auto-scaling**: 1-10 instances based on traffic
- **Model Storage**: Google Cloud Storage 

### 2. Web Application
- **Frontend**: React application hosted on Google Cloud Storage with CDN
- **Backend**: Node.js API gateway on Cloud Run 
- **Real-time Data**: Integration with Bluebikes GBFS API

## Workflow

The Bluebikes demand prediction system implements an end-to-end machine learning operations (MLOps) workflow that integrates model training, deployment, and serving. This deployment architecture enables continuous model improvement while maintaining high availability for real-time predictions.

### System Overview

The deployment workflow orchestrates the journey from trained models to production predictions, implementing industry best practices for scalable machine learning systems. The architecture prioritizes operational efficiency, enabling data scientists to focus on model improvement while the infrastructure handles deployment complexities automatically.

### Model Training to Deployment Pipeline

The workflow begins when Apache Airflow completes model training based on the latest Bluebikes usage data. Upon successful training and validation, the model artifacts are persisted to Google Cloud Storage, establishing a centralized repository for model versioning. This storage pattern creates an immutable audit trail of all model versions while enabling rapid rollback capabilities if needed.

The serving infrastructure dynamically loads models from storage rather than embedding them in container images. This design decision significantly reduces deployment time and enables hot-swapping of models without service interruption.

### Containerization and Service Deployment

The serving layer utilizes Docker containers to encapsulate the prediction service and its dependencies. These containers are built once and reused across multiple model versions, as the model artifacts are loaded at runtime rather than build time.

### Model Serving Architecture

The prediction service exposes RESTful API endpoints that handle inference requests. When a prediction request arrives, the service performs several operations in sequence: feature validation, inference execution, and result post-processing. 

The serving layer implements a stateless design where each request is independent, enabling horizontal scaling across multiple instances. This pattern ensures that the system can handle traffic spikes during peak usage periods, such as morning and evening commutes when bike demand predictions are most critical.

### Web Application Integration

The user-facing application consists of multiple layers working in concert. The React frontend, distributed globally through a CDN, provides interactive visualizations of bike availability and demand predictions. This static deployment pattern ensures low latency for users regardless of geographic location.

Behind the frontend, a Node.js backend service acts as an orchestration layer, aggregating data from multiple sources. It retrieves real-time bike availability from the Bluebikes GBFS API, requests predictions from the model serving layer, and implements business logic for features like the rebalancing recommendations. This separation of concerns allows each component to scale independently based on its specific resource requirements.

## Model Deployment Files
```
model_deployment/
├── app.py                 # Flask API for model 
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── deploy.sh            # Deployment scripts
└── redeploy.sh
```

## API Endpoints

### Model Service Endpoints
- `GET /health` - Service health check
- `POST /predict` - Single prediction (requires 48 features)
- `POST /batch_predict` - Batch predictions
- `POST /reload` - Reload model from GCS (called by Airflow)
- `GET /metrics` - Service metrics

## Model Updates

When Airflow trains a new model:

1. Model is saved to the GCS bucket
2. Airflow calls the reload endpoint:
```bash
curl -X POST https://bluebikes-prediction-202855070348.us-central1.run.app/reload \
  -H "Content-Type: application/json" \
  -d '{}'
```
3. Service loads new model without downtime

Model Service Deployment Guide (Cloud Run + Container Registry)

This section describes how to deploy and redeploy the Model Prediction Service using Docker, a container registry, and Cloud Run. The process is fully automated through shell scripts.

## Initial Deployment (build → push → deploy → test)

The deploy.sh script performs the full model deployment workflow:

```bash
./deploy.sh
```

### Environment Variables Required

These environment variables must be set before running the script:

| Variable     | Description                                   |
|--------------|-----------------------------------------------|
| PROJECT_ID   | The Google Cloud project ID                   |
| REGION       | Deployment region for Cloud Run               |
| SERVICE_NAME | Name of the model service                     |
| IMAGE_NAME   | Container image path                          |
| GCS_BUCKET   | Bucket where the model artifacts are stored   |
| MODEL_PATH   | Path within the bucket to the model file      |


These are already configured inside the script—you only need to edit them when changing environments.

### Script: deploy.sh

This script:

- Builds the Docker image
- Pushes it
- Deploys it
- Tests the service

No manual steps are required once the script is configured.

# Manual Deployment (Without Using the Shell Script)

You can deploy the model service manually by following the steps below.

### Prerequisites

- Google Cloud SDK installed and configured
- Docker installed on your local machine
- A GCP project with Cloud Run API enabled
- A trained model stored in Google Cloud Storage

## Step 1: Build the Docker Image

Build the container image locally:
```bash
docker build -t gcr.io//:latest .
```

**Example:**
```bash
docker build -t gcr.io/bluebikes-mlops/bluebikes-prediction:latest .
```

## Step 2: Push the Image to Google Container Registry

Authenticate Docker with GCP and push the built image:
```bash
# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker

# Push the image to GCR
docker push gcr.io//:latest
```

## Step 3: Deploy the Service to Cloud Run

Deploy the pushed image to Cloud Run with the required settings:
```bash
gcloud run deploy  \
  --image gcr.io//:latest \
  --platform managed \
  --region  \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --timeout 300 \
  --set-env-vars GCS_BUCKET=,MODEL_PATH= \
  --project 
```


### Redeployment (fast rebuild + redeploy)

For iterative development, use:

```bash
./redeploy.sh
```

### Testing

Test the prediction endpoint with sample data:
```bash
curl -X POST /predict \
  -H "Content-Type: application/json" \
  -d '{"features": []}'
```

**Example:**
```bash
curl -X POST https://bluebikes-prediction-xxxxx-uc.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      8, 1, 6, 2025, 9, 0.951, 0.309, 0.782, 0.623, 0.0, 1.0,
      1, 0, 0, 0, 0, 0, 1, 0,
      25, 15, 0.0, 10, 20.0, 0, 0, 0, 0,
      250, 240, 245, 235, 3200,
      10.2, 2.8, 9.5, 1.5, 0.7, 1.3, 0.85,
      0, 0, 0, 0, 0, 0, 0, 0
    ]
  }'
```

Expected response:
```json
{
  "prediction": 25.5,
  "model_version": "v1.0",
  "timestamp": "2024-12-12T10:31:00Z"
}
```

## Monitoring

- **Logs**: `gcloud run services logs read bluebikes-prediction --region us-central1`
- **Metrics**: Available in GCP Console under Cloud Run services
- **Health Monitoring**: Automated health checks every 30 seconds

## Cost Optimization

- Cloud Run scales to zero when not in use
- Frontend served from GCS (~$1-5/month)
- Model updates don't require container rebuilds
- Estimated monthly cost: ~$50-100 depending on traffic

