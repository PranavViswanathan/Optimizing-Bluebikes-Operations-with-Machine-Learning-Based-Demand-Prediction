#!/bin/bash

# Bluebikes UI - Start Script
# This script starts all required services for the Bluebikes application

echo "Starting Bluebikes UI Services."
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if .env exists
if [ ! -f "backend/.env" ]; then
    echo "Warning: backend/.env not found. Copying from .env.example..."
    cp backend/.env.example backend/.env
    echo "Created backend/.env - please review configuration"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start Node.js Backend
echo "Starting Node.js Backend (port 5001)..."
cd backend
node server.js > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../logs/backend.pid
cd ..
echo "Backend started (PID: $BACKEND_PID)"

# Wait for backend to initialize
sleep 2

# Start Python ML Service
echo "Starting Python ML Service (port 5002)..."
cd backend
python ml-service.py > ../logs/ml-service.log 2>&1 &
ML_PID=$!
echo $ML_PID > ../logs/ml-service.pid
cd ..
echo " ML Service started (PID: $ML_PID)"

# Wait for ML service to initialize
sleep 2

# Start React Frontend
echo "Starting React Frontend (port 3000)..."
cd frontend
npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../logs/frontend.pid
cd ..
echo "Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "All services started successfully!"
echo ""
echo "Service Status:"
echo "   Backend:  http://localhost:5001  (PID: $BACKEND_PID)"
echo "   ML API:   http://localhost:5002  (PID: $ML_PID)"
echo "   Frontend: http://localhost:3000  (PID: $FRONTEND_PID)"
echo ""
echo "Logs are available in the logs/ directory"
echo "To stop all services, run: ./stop.sh"
echo ""
echo " Waiting for services to fully initialize..."
sleep 8
echo ""
echo " Opening browser in 3 seconds..."
sleep 3

# Open browser (macOS)
open http://localhost:3000

echo ""
echo "✨ Bluebikes UI is ready!"
