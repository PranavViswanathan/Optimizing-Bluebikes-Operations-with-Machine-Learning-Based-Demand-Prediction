#!/bin/bash
echo "Redeploying with latest changes"

# Rebuild and push
docker build -t gcr.io/mlops-480416/bluebikes-prediction:latest . && \
docker push gcr.io/mlops-480416/bluebikes-prediction:latest && \

# Deploy
gcloud run deploy bluebikes-prediction \
  --image gcr.io/mlops-480416/bluebikes-prediction:latest \
  --region us-central1 \
  --project mlops-480416

echo "Deployment complete!"