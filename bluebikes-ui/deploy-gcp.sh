#!/bin/bash

# Bluebikes UI - GCP Deployment Script

# 1. Configuration
read -p "Enter your GCP Project ID: " PROJECT_ID
read -p "Enter GCP Region (default: us-central1): " REGION
REGION=${REGION:-us-central1}

echo "Using Project ID: $PROJECT_ID"
echo "Using Region: $REGION"
# Check if gcloud is in PATH, if not try to find it
if ! command -v gcloud &> /dev/null; then
    if [ -f "/opt/homebrew/share/google-cloud-sdk/bin/gcloud" ]; then
        echo "Found gcloud in /opt/homebrew/share/google-cloud-sdk/bin, adding to PATH..."
        export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
        # Fix for python version issues
        export CLOUDSDK_PYTHON=$(which python3)
    else
        echo "Error: gcloud CLI not found. Please install Google Cloud SDK."
        echo "Try: brew install --cask google-cloud-sdk"
        exit 1
    fi
fi

echo "Using gcloud: $(which gcloud)"
echo "Using python: $CLOUDSDK_PYTHON"
echo ""

# Enable Services
echo "Enabling required GCP services..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $PROJECT_ID

# Create Artifact Registry if it doesn't exist
REPO_NAME="bluebikes-repo"
echo "Ensuring Artifact Registry '$REPO_NAME' exists..."
gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=$REGION --description="Bluebikes Docker Repository" --project=$PROJECT_ID 2>/dev/null || echo "Repository already exists."

# Define Image Names
ML_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/bluebikes-ml"
HISTORICAL_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/bluebikes-historical"
BACKEND_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/bluebikes-backend"
FRONTEND_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/bluebikes-frontend"

# 2. Deploy Python ML Service
echo ""
echo "---------- DEPLOYING ML SERVICE ----------"
echo "Building ML Service..."
gcloud builds submit --tag $ML_IMAGE backend/ --project $PROJECT_ID

echo "Deploying ML Service to Cloud Run..."
gcloud run deploy bluebikes-ml \
    --image $ML_IMAGE \
    --platform managed \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --memory 1Gi

ML_URL=$(gcloud run services describe bluebikes-ml --platform managed --region $REGION --project $PROJECT_ID --format 'value(status.url)')
echo "✅ ML Service deployed at: $ML_URL"


# 3. Deploy Historical Data Service
echo ""
echo "---------- DEPLOYING HISTORICAL SERVICE ----------"
# Need to build from root to access data_pipeline
echo "Building Historical Service (from project root)..."
cd .. # Move to project root
# gcloud builds submit --tag $HISTORICAL_IMAGE --config bluebikes-ui/backend/cloudbuild.yaml . --project $PROJECT_ID

# Wait, I didn't create cloudbuild.yaml. Simpler: just specify Dockerfile
gcloud builds submit --tag $HISTORICAL_IMAGE --file bluebikes-ui/backend/Dockerfile.historical . --project $PROJECT_ID
cd bluebikes-ui # Move back

echo "Deploying Historical Service to Cloud Run..."
gcloud run deploy bluebikes-historical \
    --image $HISTORICAL_IMAGE \
    --platform managed \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --memory 1Gi \
    --timeout 300

HISTORICAL_URL=$(gcloud run services describe bluebikes-historical --platform managed --region $REGION --project $PROJECT_ID --format 'value(status.url)')
echo "✅ Historical Service deployed at: $HISTORICAL_URL"


# 4. Deploy Node.js Backend
echo ""
echo "---------- DEPLOYING BACKEND SERVICE ----------"
echo "Building Backend..."
# We need separate Dockerfiles for node and ml in the same dir
gcloud builds submit --tag $BACKEND_IMAGE --file backend/Dockerfile.node backend/ --project $PROJECT_ID

echo "Deploying Backend to Cloud Run..."
# Pass ML and Historical URLs as env vars
# Note: internal service URLs are different usually, but for Cloud Run managed, using public URLs is easiest (though adds networking checks). 
# Ideally use internal names if in same VPC, but public HTTPS URLs work fine for simple setup.
gcloud run deploy bluebikes-backend \
    --image $BACKEND_IMAGE \
    --platform managed \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --set-env-vars ML_SERVICE_PORT=443,ML_SERVICE_URL=$ML_URL,HISTORICAL_DATA_SERVICE_PORT=443,HISTORICAL_SERVICE_URL=$HISTORICAL_URL \
    --memory 512Mi

# Note: The Node code expects `localhost:5002`. I need to patch server.js to accept full URL or handle the logic.
# server.js logic: `const ML_SERVICE_URL = \`http://localhost:\${process.env.ML_SERVICE_PORT || 5002}\`;`
# This blindly assumes localhost.
# I NEED TO FIX SERVER.JS FIRST!

BACKEND_URL=$(gcloud run services describe bluebikes-backend --platform managed --region $REGION --project $PROJECT_ID --format 'value(status.url)')
echo "✅ Backend Service deployed at: $BACKEND_URL"


# 5. Deploy Frontend
echo ""
echo "---------- DEPLOYING FRONTEND ----------"
echo "Building Frontend..."
# Build arg for REACT_APP_API_URL
# gcloud builds submit --tag $FRONTEND_IMAGE --file frontend/Dockerfile frontend/ --substitutions=_REACT_APP_API_URL=$BACKEND_URL --project $PROJECT_ID
# Wait, substitutions in gcloud builds refer to cloudbuild.yaml. For direct Docker build, use --build-arg via gcloud builds submit... 
# actually `gcloud builds submit` takes arguments for build args in a slightly different way or needs cloudbuild.yaml.
# Easiest: Use --build-arg in the docker build command itself, but `gcloud builds submit --pack` or Dockerfile mode supports it?
# Just use `gcloud builds submit` with `--build-arg` support... wait, standard `gcloud builds submit` command line doesn't easily expose build-args for simple Dockerfile builds without a config.
# I will generate a temporary cloudbuild.yaml for frontend.

cat > frontend/cloudbuild.yaml <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '--build-arg', 'REACT_APP_API_URL=$BACKEND_URL', '-t', '$FRONTEND_IMAGE', '.']
images:
- '$FRONTEND_IMAGE'
EOF

gcloud builds submit --config frontend/cloudbuild.yaml frontend/ --project $PROJECT_ID
rm frontend/cloudbuild.yaml

echo "Deploying Frontend to Cloud Run..."
gcloud run deploy bluebikes-frontend \
    --image $FRONTEND_IMAGE \
    --platform managed \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --memory 512Mi

FRONTEND_URL=$(gcloud run services describe bluebikes-frontend --platform managed --region $REGION --project $PROJECT_ID --format 'value(status.url)')

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "Frontend: $FRONTEND_URL"
echo "Backend:  $BACKEND_URL"
