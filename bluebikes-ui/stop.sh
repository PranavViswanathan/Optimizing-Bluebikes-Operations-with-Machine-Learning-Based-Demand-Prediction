#!/bin/bash

# Bluebikes UI - Stop Script
# This script stops all running services

echo "Stopping Bluebikes UI Services."
echo ""

# Change to script directory
cd "$(dirname "$0")"

STOPPED_COUNT=0

# Stop Backend
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo "   Stopping Backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        STOPPED_COUNT=$((STOPPED_COUNT + 1))
    fi
    rm -f logs/backend.pid
fi

# Stop ML Service
if [ -f "logs/ml-service.pid" ]; then
    ML_PID=$(cat logs/ml-service.pid)
    if kill -0 $ML_PID 2>/dev/null; then
        echo "   Stopping ML Service (PID: $ML_PID)..."
        kill $ML_PID
        STOPPED_COUNT=$((STOPPED_COUNT + 1))
    fi
    rm -f logs/ml-service.pid
fi

# Stop Historical Data Service
# if [ -f "logs/historical-service.pid" ]; then
#     HIST_PID=$(cat logs/historical-service.pid)
#     if kill -0 $HIST_PID 2>/dev/null; then
#         echo "   Stopping Historical Service (PID: $HIST_PID)..."
#         kill $HIST_PID
#         STOPPED_COUNT=$((STOPPED_COUNT + 1))
#     fi
#     rm -f logs/historical-service.pid
# fi

# Stop Frontend
if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "   Stopping Frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        STOPPED_COUNT=$((STOPPED_COUNT + 1))
    fi
    rm -f logs/frontend.pid
fi

# Also kill any lingering processes by name
pkill -f "node server.js" 2>/dev/null && echo "   Cleaned up lingering backend process"
pkill -f "python ml-service.py" 2>/dev/null && echo "   Cleaned up lingering ML service"
# pkill -f "python historical-data-service.py" 2>/dev/null && echo "   Cleaned up lingering Historical service"
pkill -f "react-scripts start" 2>/dev/null && echo "   Cleaned up lingering frontend process"

echo ""
if [ $STOPPED_COUNT -gt 0 ]; then
    echo "Stopped $STOPPED_COUNT service(s)"
else
    echo "ℹ No running services found"
fi
echo ""
echo "Cleaning up log files..."
rm -f logs/*.log

echo "All services stopped and cleaned up"
