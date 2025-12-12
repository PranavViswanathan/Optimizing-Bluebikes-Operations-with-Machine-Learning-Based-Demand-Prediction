# Model Deployment

## Overview
This section covers the deployment architecture for the Bluebikes demand prediction model and web application. This system uses Google Cloud Platform services to provide scalable, serverless model serving with automatic updates.

## Watch Project Deployment Here: [Watch the project video](https://youtu.be/wjhKFdzRk6o)

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

## Monitoring

- **Logs**: `gcloud run services logs read bluebikes-prediction --region us-central1`
- **Metrics**: Available in GCP Console under Cloud Run services
- **Health Monitoring**: Automated health checks every 30 seconds

## Cost Optimization

- Cloud Run scales to zero when not in use
- Frontend served from GCS (~$1-5/month)
- Model updates don't require container rebuilds
- Estimated monthly cost: ~$50-100 depending on traffic

