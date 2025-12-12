#!/bin/bash

# === CONFIG ===
BUCKET_NAME="bluebikes-frontend"   
URL_MAP_NAME="bluebikes-frontend-map"   

echo "Building React app for production..."
npm run build

if [ $? -ne 0 ]; then
  echo "Build failed. Fix errors before deploying."
  exit 1
fi

echo "Deploying to bucket: gs://$BUCKET_NAME ..."

# Try gsutil first, fallback to gcloud storage if it fails
gsutil -m rsync -r build gs://$BUCKET_NAME 2>/dev/null
if [ $? -ne 0 ]; then
  echo "gsutil failed (likely Python version issue), using gcloud storage instead..."
  gcloud storage cp -r build/* gs://$BUCKET_NAME/
  
  if [ $? -ne 0 ]; then
    echo "Upload failed!"
    exit 1
  fi
fi

echo "Invalidating CDN cache for $URL_MAP_NAME ..."
gcloud compute url-maps invalidate-cdn-cache $URL_MAP_NAME --path "/*"

echo "Deployment complete! Changes are live (after CDN refresh)."