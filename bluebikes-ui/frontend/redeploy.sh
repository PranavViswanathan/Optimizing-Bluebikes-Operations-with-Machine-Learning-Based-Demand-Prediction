#!/bin/bash

# === CONFIG ===
BUCKET_NAME="bluebikes-frontend"   
URL_MAP_NAME="bluebikes-url-map"   

echo "Building React app for production"
npm run build

if [ $? -ne 0 ]; then
  echo "Build failed. Fix errors before deploying."
  exit 1
fi

echo "Deploying to bucket: gs://$BUCKET_NAME ..."
gsutil -m rsync -r build gs://$BUCKET_NAME

echo "Invalidating CDN cache for $URL_MAP_NAME ..."
gcloud compute url-maps invalidate-cdn-cache $URL_MAP_NAME --path "/*"

echo "Deployment complete,changes are live (after CDN refresh)."
