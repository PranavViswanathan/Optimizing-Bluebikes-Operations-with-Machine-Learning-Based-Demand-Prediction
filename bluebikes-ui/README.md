# Bluebikes Station Map UI

A full-stack web application for visualizing Bluebikes stations with real-time availability data and ML-based demand predictions.

## Features

- **Interactive Map**: View all ~590 Bluebikes stations on an interactive Leaflet map with color-coded availability
- **Real-time Data**: Live bike and dock availability from Bluebikes GBFS API
- **ML Predictions**: XGBoost-powered demand forecasting for each station
- **AI Rebalancing**: Intelligent bike redistribution recommendations using ML predictions
- **List View**: Searchable and sortable table of all stations
- **Detailed View**: Comprehensive analytics for individual stations
- **Premium UI**: Modern dark theme with glassmorphism and smooth animations
- **Historical Trends**: Visualize past utilization patterns (7 days, 30 days, 12 weeks)
- **Route Planning**: Find nearest drop-off stations with turn-by-turn directions
- **Deployed Model Support**: Call external ML APIs (AWS SageMaker, Azure ML, etc.)

## Tech Stack

### Backend
- **Node.js + Express**: API gateway for GBFS data
- **Python + Flask**: ML prediction service
- **XGBoost**: Trained model for demand forecasting
- **Node-Cache**: Response caching for performance

### Frontend
- **React**: Component-based UI
- **React Router**: Client-side routing
- **Leaflet**: Interactive maps
- **Material UI**: Component library
- **Recharts**: Data visualization (for future enhancements)

## Prerequisites

- **Node.js**: 18.0 or higher
- **Python**: 3.11 or higher
- **npm**: Comes with Node.js

## Installation

### 1. Clone and Navigate

```bash
cd bluebikes-ui
```

### 2. Install Backend Dependencies

```bash
cd backend
npm install
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd ../frontend
npm install
```

## Quick Start

The easiest way to run the application is using the provided scripts:

```bash
# Start all services and open browser
./start.sh

# Stop all services
./stop.sh
```

That's it! The `start.sh` script will:
- Start the backend server (port 5001)
- Start the ML prediction service (port 5002)
- Start the React frontend (port 3000)
- Automatically open your browser to http://localhost:3000

Logs are saved to the `logs/` directory for debugging.

---

## Running the Application (Manual)

If you prefer to run services manually in separate terminals:

You need to run **3 services** in separate terminals:

### Terminal 1: Node.js Backend

```bash
cd backend
node server.js
```

The API server will run on `http://localhost:5001`

### Terminal 2: Python ML Service

```bash
cd backend
python ml-service.py
```

The ML service will run on `http://localhost:5002`

### Terminal 3: React Frontend

```bash
cd frontend
npm start
```

The React app will open at `http://localhost:3000`

## Usage

1. **Map View** (default): 
   - Hover over station markers to see availability and predictions
   - Markers are color-coded: 🟢 Green (5+ bikes), 🟡 Yellow (1-5 bikes), 🔴 Red (empty)
   - Click markers for detailed popups

2. **List View**:
   - Click "List View" in navigation
   - Search stations by name
   - Sort by name, bikes available, docks, or capacity
   - Click "View" to see station details

3. **Detail View**:
   - Click any station to see comprehensive information
   - Real-time availability
   - ML demand prediction
   - Visual availability bars

4. **Rebalancing View**:
   - Click "Rebalancing" in navigation to access bike rebalancing recommendations
   - AI-powered system identifies stations requiring rebalancing
   - Shows optimal bike transfer routes between stations
   - Includes distance, quantity, and priority information

## Rebalancing Algorithm

The rebalancing feature uses a **surplus-based algorithm** that combines real-time availability data with ML demand predictions to generate intelligent bike redistribution recommendations.

### How It Works

#### 1. **Identify Recipients (Stations Needing Bikes)**

The algorithm identifies stations that will likely run out of bikes soon:

```
Deficit = Predicted Demand - Current Available Bikes
```

A station becomes a **recipient** if:
- `Deficit ≥ 3 bikes` (configurable threshold)
- It has available docks to receive bikes
- Current bikes are insufficient to meet predicted hourly demand

**Example**: 
- Station has 2 bikes available
- ML predicts 8 bikes will be needed in the next hour
- Deficit = 8 - 2 = 6 bikes → **Needs rebalancing**

#### 2. **Identify Donors (Stations With Surplus Bikes)**

The algorithm finds stations with excess bikes that can spare them:

```
Surplus = Current Available Bikes - (Predicted Demand + Safety Buffer)
```

A station becomes a **donor** if:
- `Surplus ≥ 8 bikes` (configurable threshold)
- It has more bikes than predicted demand + buffer
- Bikes can be safely removed without hurting availability

**Example**:
- Station has 25 bikes available
- ML predicts 5 bikes will be needed in the next hour
- Safety buffer = 5 bikes
- Surplus = 25 - (5 + 5) = 15 bikes → **Can donate bikes**

#### 3. **Match Recipients with Nearby Donors**

For each recipient station:
1. **Find nearby donors** within 1 km radius (configurable)
2. **Calculate distance** using Haversine formula
3. **Sort by proximity** (closer donors preferred)
4. **Determine transfer quantity**:
   ```
   Bikes to Move = min(Deficit, Donor Surplus, Available Docks)
   ```
5. **Generate recommendation** with route details

#### 4. **Prioritization**

Recommendations are sorted by urgency:
- Higher deficit = Higher priority
- Closer matches = Better efficiency
- Larger transfers = More impact

### Configuration Parameters

You can tune the algorithm in `frontend/src/components/RebalancingView.jsx`:

```javascript
const deficitThreshold = 3;   // Min bikes needed to flag recipient
const donorMinSurplus = 8;    // Min surplus to qualify as donor
const safetyBuffer = 5;       // Buffer kept at donor stations
const maxDistance = 1.0;      // Max distance for matches (km)
```

### Example Scenario

**Recipient Station** (Downtown Crossing):
- Current bikes: 1
- Predicted demand: 7 bikes/hour
- Deficit: 6 bikes ⚠️

**Donor Station** (Park Street - 0.3km away):
- Current bikes: 22
- Predicted demand: 3 bikes/hour
- Surplus: 14 bikes ✅

**Recommendation**:
> Move **6 bikes** from Park St → Downtown Crossing  
> Distance: 0.3 km | Priority: High

### ML Integration

The rebalancing algorithm relies on **accurate demand predictions** from your ML model:

- **With Real Model**: Uses your trained XGBoost model for precise hourly demand forecasts
- **With Mock Data**: Uses simplified time-based heuristics (for testing/demo)

For best results, deploy your trained model following the [Deployed Model Guide](deployed_model_guide.md).

### Benefits

✅ **Proactive**: Prevents stockouts before they occur  
✅ **Efficient**: Matches based on proximity and capacity  
✅ **Data-Driven**: Uses ML predictions instead of static rules  
✅ **Scalable**: Analyzes all 590+ stations in real-time  
✅ **Actionable**: Provides specific transfer quantities and routes

## API Endpoints

### Backend (Port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stations` | GET | All station information |
| `/api/stations/status` | GET | Real-time status for all stations |
| `/api/stations/:id/status` | GET | Status for specific station |
| `/api/predict` | POST | ML prediction for demand |
| `/health` | GET | Health check |

### ML Service (Port 5001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Generate demand prediction |
| `/predict` | POST | Generate demand prediction |
| `/health` | GET | Health check |

### Historical Data Service (Port 5003)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/historical/:id/:range` | GET | Historical data (hourly, daily, weekly) |
| `/health` | GET | Health check |

## ML Model Integration

### Using a Trained Model

1. Copy your trained model file to `backend/models/best_model.pkl`

2. The model should be a scikit-learn compatible model (XGBoost, LightGBM, etc.) saved with `joblib` or `pickle`

3. The ML service will automatically load it on startup

### Mock Predictions

If no model is found, the service will return mock predictions based on simple heuristics (time of day, day of week). A warning will appear in the UI.

## Project Structure

```
bluebikes-ui/
├── backend/
│   ├── server.js           # Express API server
│   ├── ml-service.py       # Python ML prediction service
│   ├── package.json        # Node.js dependencies
│   ├── requirements.txt    # Python dependencies
│   ├── .env                # Environment variables
│   └── models/             # Trained ML models
├── frontend/
│   ├── public/
│   │   └── index.html      # HTML template
│   ├── src/
│   │   ├── App.jsx         # Main app component
│   │   ├── index.js        # Entry point
│   │   ├── components/     # React components
│   │   │   ├── MapView.jsx
│   │   │   ├── StationList.jsx
│   │   │   ├── StationDetail.jsx
│   │   │   └── StationPopup.jsx
│   │   ├── context/
│   │   │   └── StationContext.jsx  # State management
│   │   └── styles/
│   │       └── App.css     # Premium styling
│   └── package.json        # Frontend dependencies
└── README.md
```

## Configuration

### Backend Environment Variables

Edit `backend/.env`:

```env
# Bluebikes GBFS API
GBFS_BASE_URL=https://gbfs.lyft.com/gbfs/1.1/bos/en

# Backend Server
PORT=5001
NODE_ENV=development

# Local ML Service (default)
ML_SERVICE_PORT=5002
# Local ML Service (default)
ML_SERVICE_PORT=5002
MODEL_PATH=./models/best_model.pkl

# Historical Data Service
HISTORICAL_DATA_SERVICE_PORT=5003
HISTORICAL_DATA_PATH=../../data_pipeline/data/raw/bluebikes

# External Deployed ML API (optional)
USE_EXTERNAL_ML_API=false
# EXTERNAL_ML_API_URL=https://your-ml-api.example.com
# ML_API_KEY=your-api-key-here
```

**Using a Deployed Model?** See the [Deployed Model Guide](deployed_model_guide.md) for instructions on connecting to AWS SageMaker, Azure ML, or other hosted ML services.

## Development

### Backend Development

```bash
cd backend
npm run dev  # Uses nodemon for auto-reload
```

### Frontend Development

```bash
cd frontend
npm start  # Auto-reloads on file changes
```

## Data Sources

- **Bluebikes GBFS API**: https://gbfs.lyft.com/gbfs/1.1/bos/en/gbfs.json
- **Station Information**: Real-time from GBFS `station_information.json`
- **Station Status**: Real-time from GBFS `station_status.json`

## Performance

- **Caching**: GBFS responses cached for 60 seconds
- **Lazy Loading**: Station status fetched on-demand
- **Debouncing**: Search input debounced for smooth UX

## Troubleshooting

### Backend won't start
- Ensure Node.js 18+ is installed: `node --version`
- Check port 5000 is available
- Install dependencies: `npm install`

### ML Service won't start
- Ensure Python 3.11+ is installed: `python --version`
- Install dependencies: `pip install -r requirements.txt`
- Check port 5001 is available

### Frontend shows "Failed to fetch stations"
- Ensure backend is running on port 5000
- Check browser console for CORS errors
- Verify GBFS API is accessible

### Predictions show "Using mock data"
- Place trained model at `backend/models/best_model.pkl`
- Ensure model is compatible with joblib/pickle
- Check ML service logs for loading errors

## Future Enhancements

- [x] AI-powered bike rebalancing recommendations
- [x] External ML API support for deployed models
- [x] Historical trend charts
- [x] Route planning between stations
- [ ] 24-hour demand forecasting with time series
- [ ] Weather integration with live data
- [ ] Mobile responsiveness improvements
- [ ] Dark/Light theme toggle
- [ ] Export rebalancing routes to CSV/JSON

## License

MIT

## Author

Built for MLOps Final Project - Bluebikes Demand Prediction
