const express = require('express');
const cors = require('cors');
const axios = require('axios');
const NodeCache = require('node-cache');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;
const GBFS_BASE_URL = process.env.GBFS_BASE_URL || 'https://gbfs.lyft.com/gbfs/1.1/bos/en';

// ML Service Configuration
// Always point to local ML service for feature engineering
// The local service will handle forwarding to external model if needed
const ML_SERVICE_URL = `http://localhost:${process.env.ML_SERVICE_PORT || 5002}`;

// Historical Data Service Configuration
const HISTORICAL_SERVICE_URL = `http://localhost:${process.env.HISTORICAL_DATA_SERVICE_PORT || 5003}`;

// Initialize cache with 60-second TTL for real-time data
const cache = new NodeCache({ stdTTL: 60, checkperiod: 120 });

// Middleware
app.use(cors());
app.use(express.json());

// Logging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// ========== GBFS API PROXY ENDPOINTS ==========

/**
 * GET /api/stations
 * Returns all station information with id, name, lat, long, capacity
 */
app.get('/api/stations', async (req, res) => {
  try {
    const cacheKey = 'stations_info';
    const cached = cache.get(cacheKey);
    
    if (cached) {
      console.log('Returning cached station information');
      return res.json(cached);
    }

    const response = await axios.get(`${GBFS_BASE_URL}/station_information.json`);
    const stations = response.data.data.stations;
    
    // Cache station info for longer (stations don't change often)
    cache.set(cacheKey, stations, 3600); // 1 hour TTL
    
    res.json(stations);
  } catch (error) {
    console.error('Error fetching station information:', error.message);
    res.status(500).json({ 
      error: 'Failed to fetch station information',
      message: error.message 
    });
  }
});

/**
 * GET /api/stations/status
 * Returns real-time status for all stations (bikes available, docks available)
 */
app.get('/api/stations/status', async (req, res) => {
  try {
    const cacheKey = 'stations_status';
    const cached = cache.get(cacheKey);
    
    if (cached) {
      console.log('Returning cached station status');
      return res.json(cached);
    }

    const response = await axios.get(`${GBFS_BASE_URL}/station_status.json`);
    const status = response.data.data.stations;
    
    // Cache status for 60 seconds (real-time data)
    cache.set(cacheKey, status, 60);
    
    res.json(status);
  } catch (error) {
    console.error('Error fetching station status:', error.message);
    res.status(500).json({ 
      error: 'Failed to fetch station status',
      message: error.message 
    });
  }
});

/**
 * GET /api/stations/:id/status
 * Returns real-time status for a specific station
 */
app.get('/api/stations/:id/status', async (req, res) => {
  try {
    const stationId = req.params.id;
    const cacheKey = `station_status_${stationId}`;
    const cached = cache.get(cacheKey);
    
    if (cached) {
      console.log(`Returning cached status for station ${stationId}`);
      return res.json(cached);
    }

    // Fetch all statuses and filter for the requested station
    const response = await axios.get(`${GBFS_BASE_URL}/station_status.json`);
    const stations = response.data.data.stations;
    const stationStatus = stations.find(s => s.station_id === stationId);
    
    if (!stationStatus) {
      return res.status(404).json({ error: 'Station not found' });
    }
    
    cache.set(cacheKey, stationStatus, 60);
    res.json(stationStatus);
  } catch (error) {
    console.error(`Error fetching status for station ${req.params.id}:`, error.message);
    res.status(500).json({ 
      error: 'Failed to fetch station status',
      message: error.message 
    });
  }
});

// ========== ML PREDICTION ENDPOINT ==========

/**
 * POST /api/predict
 * Request ML prediction for bike demand
 * Body: { station_id, datetime, temperature, precipitation }
 */
// Replace your existing /api/predict endpoint in server.js with this:

// In server.js, replace the /api/predict endpoint with this updated version:

app.post('/api/predict', async (req, res) => {
  try {
    const { station_id, datetime, temperature, precipitation } = req.body;
    
    if (!station_id) {
      return res.status(400).json({ error: 'station_id is required' });
    }

    // If using external ML API (Cloud Run)
    if (USE_EXTERNAL_ML_API && EXTERNAL_ML_API_URL) {
      // Generate 48 features for Cloud Run
      const features = generateFeaturesForCloudRun(station_id, datetime, temperature, precipitation);
      
      try {
        const response = await axios.post(
          `${EXTERNAL_ML_API_URL}/predict`,
          { features: features },
          { 
            timeout: 10000,
            headers: { 'Content-Type': 'application/json' }
          }
        );
        
        // Format for frontend
        const formattedResponse = {
          station_id: station_id,
          datetime: datetime || new Date().toISOString(),
          predicted_demand: Math.max(0, Math.round(response.data.prediction)),
          model_version: response.data.model_version || 'cloud_run',
          confidence: 'high'
        };
        
        return res.json(formattedResponse);
      } catch (cloudRunError) {
        console.error('Cloud Run error:', cloudRunError.message);
        // Fall through to mock prediction
      }
    }
    
    // Fallback mock prediction
    const mockDemand = getMockPrediction(station_id, datetime);
    res.json({
      station_id: station_id,
      datetime: datetime || new Date().toISOString(),
      predicted_demand: mockDemand,
      model_version: 'mock',
      confidence: 'low'
    });
    
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: 'Prediction failed' });
  }
});

// Add these helper functions to your server.js:

function generateFeaturesForCloudRun(stationId, datetime, temperature = 15, precipitation = 0) {
  const dt = datetime ? new Date(datetime) : new Date();
  
  const hour = dt.getHours();
  const dayOfWeek = dt.getDay();
  const month = dt.getMonth() + 1;
  const year = dt.getFullYear();
  const day = dt.getDate();
  
  // Generate all 48 features
  return [
    hour, dayOfWeek, month, year, day,
    Math.sin(2 * Math.PI * hour / 24),
    Math.cos(2 * Math.PI * hour / 24),
    Math.sin(2 * Math.PI * dayOfWeek / 7),
    Math.cos(2 * Math.PI * dayOfWeek / 7),
    Math.sin(2 * Math.PI * month / 12),
    Math.cos(2 * Math.PI * month / 12),
    hour >= 7 && hour <= 9 ? 1 : 0, // morning rush
    hour >= 17 && hour <= 19 ? 1 : 0, // evening rush
    hour >= 22 || hour <= 5 ? 1 : 0, // night
    hour >= 11 && hour <= 14 ? 1 : 0, // midday
    dayOfWeek === 0 || dayOfWeek === 6 ? 1 : 0, // weekend
    0, 0, 0, // interaction features
    temperature + 2, temperature - 2, precipitation, 4, temperature,
    precipitation > 0.1 ? 1 : 0, 
    precipitation > 0.5 ? 1 : 0,
    temperature < 5 ? 1 : 0,
    temperature > 25 ? 1 : 0,
    100, 95, 98, 280, 2200, // historical (mock)
    12.5, 3.2, 11.0, 1.8, 0.9, 1.5, 0.7, // statistical (mock)
    0, 0, 0, 0, 0, 0, 0, 0 // padding
  ];
}

function getMockPrediction(stationId, datetime) {
  const dt = datetime ? new Date(datetime) : new Date();
  const hour = dt.getHours();
  const dayOfWeek = dt.getDay();
  
  // More realistic mock predictions for rebalancing
  const stationHash = parseInt(stationId) || stationId.charCodeAt(0);
  const stationFactor = (stationHash % 10) / 10;
  
  if (dayOfWeek > 0 && dayOfWeek < 6) { // Weekday
    if (hour >= 7 && hour <= 9) return Math.round(20 + stationFactor * 30); // 20-50
    if (hour >= 17 && hour <= 19) return Math.round(25 + stationFactor * 35); // 25-60
    if (hour >= 11 && hour <= 14) return Math.round(15 + stationFactor * 20); // 15-35
    return Math.round(5 + stationFactor * 15); // 5-20
  } else { // Weekend
    if (hour >= 10 && hour <= 16) return Math.round(18 + stationFactor * 25); // 18-43
    return Math.round(5 + stationFactor * 10); // 5-15
  }
}

// ========== HISTORICAL DATA ENDPOINT ==========

/**
 * GET /api/historical/:stationId/:timeRange
 * Proxy requests to historical data service
 * timeRange: hourly | daily | weekly
 */
app.get('/api/historical/:stationId/:timeRange', async (req, res) => {
  try {
    const { stationId, timeRange } = req.params;
    
    // Validate time range
    if (!['hourly', 'daily', 'weekly'].includes(timeRange)) {
      return res.status(400).json({ error: 'Invalid time range. Use: hourly, daily, or weekly' });
    }
    
    // Forward request to historical data service
    const response = await axios.get(
      `${HISTORICAL_SERVICE_URL}/api/historical/${stationId}/${timeRange}`,
      { timeout: 30000 } // 30 second timeout for data processing
    );
    
    res.json(response.data);
  } catch (error) {
    console.error('Error calling historical data service:', error.message);
    
    // If historical service is unavailable, return a graceful error
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ 
        error: 'Historical data service is currently unavailable',
        data: []
      });
    }
    
    res.status(500).json({ 
      error: 'Failed to get historical data',
      message: error.message 
    });
  }
});

// ========== HEALTH CHECK ==========

app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    cache_keys: cache.keys().length
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Bluebikes Backend Server running on port ${PORT}`);
  console.log(`GBFS API: ${GBFS_BASE_URL}`);
  console.log(`ML Service: ${ML_SERVICE_URL}`);
  console.log(`Historical Data Service: ${HISTORICAL_SERVICE_URL}`);
  console.log(`\nAvailable endpoints:`);
  console.log(`  GET  /api/stations`);
  console.log(`  GET  /api/stations/status`);
  console.log(`  GET  /api/stations/:id/status`);
  console.log(`  POST /api/predict`);
  console.log(`  GET  /api/historical/:stationId/:timeRange`);
  console.log(`  GET  /health`);
});
