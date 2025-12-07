# Bluebikes Station Map UI

A full-stack web application for visualizing Bluebikes stations with real-time availability data and ML-based demand predictions.

## Features

- **Interactive Map**: View all ~450 Bluebikes stations on an interactive Leaflet map
- **Real-time Data**: Live bike and dock availability from Bluebikes GBFS API
- **ML Predictions**: XGBoost-powered demand forecasting for each station
- **List View**: Searchable and sortable table of all stations
- **Detailed View**: Comprehensive analytics for individual stations
- **Premium UI**: Modern dark theme with glassmorphism and smooth animations

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

## Running the Application

You need to run **3 services** in separate terminals:

### Terminal 1: Node.js Backend

```bash
cd backend
npm start
```

The API server will run on `http://localhost:5000`

### Terminal 2: Python ML Service

```bash
cd backend
python ml-service.py
```

The ML service will run on `http://localhost:5001`

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
GBFS_BASE_URL=https://gbfs.lyft.com/gbfs/1.1/bos/en
MODEL_PATH=./models/best_model.pkl
PORT=5000
ML_SERVICE_PORT=5001
NODE_ENV=development
```

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

- [ ] Historical trend charts
- [ ] 24-hour demand forecasting
- [ ] Weather integration with live data
- [ ] User location detection
- [ ] Route planning between stations
- [ ] Mobile responsiveness improvements
- [ ] Dark/Light theme toggle

## License

MIT

## Author

Built for MLOps Final Project - Bluebikes Demand Prediction
