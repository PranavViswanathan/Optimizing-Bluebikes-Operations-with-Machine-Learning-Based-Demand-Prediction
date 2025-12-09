# Deploying Bluebikes to Google Cloud Run

This guide explains how to deploy the full stack (Frontend, Node Backend, ML Service, Historical Service) to Google Cloud Run.

## Prerequisites

1.  **Google Cloud SDK**: Ensure `gcloud` is installed and authenticated (`gcloud auth login`).
2.  **GCP Project**: You need an active Google Cloud Project with billing enabled.

## Quick Start

1.  Open a terminal in `bluebikes-ui/`.
2.  Make the deploy script executable:
    ```bash
    chmod +x deploy-gcp.sh
    ```
3.  Run the script:
    ```bash
    ./deploy-gcp.sh
    ```
4.  Follow the prompts (enter Project ID and Region).

## What the Script Does

1.  **Enables APIs**: `run.googleapis.com`, `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`.
2.  **Creates Artifact Registry**: A repo named `bluebikes-repo` to store Docker images.
3.  **Deploys ML Service**: Python XGBoost service.
4.  **Deploys Historical Service**: Python service serving processed Parquet data.
5.  **Deploys Backend**: Node.js API gateway that proxies requests to ML and Historical services.
6.  **Deploys Frontend**: React App served via Nginx, configured to talk to the Backend.

## Manual Deployment Steps (If script fails)

### 1. ML Service
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/bluebikes-ml backend/
gcloud run deploy bluebikes-ml --image gcr.io/PROJECT_ID/bluebikes-ml --platform managed
```

### 2. Historical Service
(Run from project root)
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/bluebikes-historical --file bluebikes-ui/backend/Dockerfile.historical .
gcloud run deploy bluebikes-historical --image gcr.io/PROJECT_ID/bluebikes-historical --platform managed 
```

### 3. Backend
**Important**: Obtain the URLs of the ML and Historical services first.
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/bluebikes-backend --file backend/Dockerfile.node backend/
gcloud run deploy bluebikes-backend --image gcr.io/PROJECT_ID/bluebikes-backend --set-env-vars ML_SERVICE_URL=...,HISTORICAL_SERVICE_URL=...
```

### 4. Frontend
```bash
# Update cloudbuild.yaml or build manually with ARG
docker build --build-arg REACT_APP_API_URL=BACKEND_URL -t gcr.io/PROJECT_ID/bluebikes-frontend frontend/
docker push gcr.io/PROJECT_ID/bluebikes-frontend
gcloud run deploy bluebikes-frontend --image gcr.io/PROJECT_ID/bluebikes-frontend
```

## Troubleshooting

-   **Backend Connection Error**: Ensure the Node Backend has the correct `ML_SERVICE_URL` and `HISTORICAL_SERVICE_URL` environment variables set.
-   **Frontend API Error**: Ensure the Frontend was built with the correct `REACT_APP_API_URL`. Check the network tab in browser.
-   **CORS Issues**: The Node backend handles CORS, but if you hit services directly, you might need to check headers.
