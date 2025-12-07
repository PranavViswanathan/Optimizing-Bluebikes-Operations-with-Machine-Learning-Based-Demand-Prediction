const express = require('express');
const cors = require('cors');
const axios = require('axios');
const NodeCache = require('node-cache');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;
const GBFS_BASE_URL = process.env.GBFS_BASE_URL || 'https://gbfs.lyft.com/gbfs/1.1/bos/en';

// ML Service Configuration
// Set USE_EXTERNAL_ML_API=true to use a deployed model API
const USE_EXTERNAL_ML_API = process.env.USE_EXTERNAL_ML_API === 'true';
const EXTERNAL_ML_API_URL = process.env.EXTERNAL_ML_API_URL;
const ML_SERVICE_URL = USE_EXTERNAL_ML_API && EXTERNAL_ML_API_URL 
  ? EXTERNAL_ML_API_URL 
  : `http://localhost:${process.env.ML_SERVICE_PORT || 5002}`;

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
app.post('/api/predict', async (req, res) => {
  try {
    const { station_id, datetime, temperature, precipitation } = req.body;
    
    if (!station_id) {
      return res.status(400).json({ error: 'station_id is required' });
    }

    // Forward request to ML service (local or external)
    const mlRequestPayload = {
      station_id,
      datetime: datetime || new Date().toISOString(),
      temperature: temperature || 15,
      precipitation: precipitation || 0
    };

    const response = await axios.post(`${ML_SERVICE_URL}/predict`, mlRequestPayload, {
      timeout: 10000, // 10 second timeout for external APIs
      headers: {
        'Content-Type': 'application/json',
        // Add authorization header if using external API with auth
        ...(process.env.ML_API_KEY && { 'Authorization': `Bearer ${process.env.ML_API_KEY}` })
      }
    });

    res.json(response.data);
  } catch (error) {
    console.error('Error calling ML service:', error.message);
    
    // If ML service is unavailable, return a graceful error
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ 
        error: 'ML prediction service is currently unavailable',
        predicted_demand: null
      });
    }
    
    res.status(500).json({ 
      error: 'Failed to get prediction',
      message: error.message 
    });
  }
});

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
  console.log(`🚀 Bluebikes Backend Server running on port ${PORT}`);
  console.log(`📍 GBFS API: ${GBFS_BASE_URL}`);
  console.log(`🤖 ML Service: ${ML_SERVICE_URL}`);
  console.log(`   ML Mode: ${USE_EXTERNAL_ML_API ? '☁️  External API' : '🏠 Local Service'}`);
  console.log(`📊 Historical Data Service: ${HISTORICAL_SERVICE_URL}`);
  console.log(`\nAvailable endpoints:`);
  console.log(`  GET  /api/stations`);
  console.log(`  GET  /api/stations/status`);
  console.log(`  GET  /api/stations/:id/status`);
  console.log(`  POST /api/predict`);
  console.log(`  GET  /api/historical/:stationId/:timeRange`);
  console.log(`  GET  /health`);
});
