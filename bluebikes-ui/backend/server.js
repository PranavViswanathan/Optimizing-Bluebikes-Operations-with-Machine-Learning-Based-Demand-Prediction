const express = require('express');
const cors = require('cors');

const corsOptions = {
  origin: [
    'http://localhost:3000',
    'http://localhost:5173',
    'http://34.110.183.151',                    
    'https://storage.googleapis.com',           
    /\.run\.app$/,                              
    /\.googleapis\.com$/                        
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
};

const axios = require('axios');
const NodeCache = require('node-cache');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;
const GBFS_BASE_URL = process.env.GBFS_BASE_URL || 'https://gbfs.lyft.com/gbfs/1.1/bos/en';
const EXTERNAL_ML_API_URL = process.env.EXTERNAL_ML_API_URL || 'https://bluebikes-prediction-202855070348.us-central1.run.app';
const HISTORICAL_SERVICE_URL = `http://localhost:${process.env.HISTORICAL_DATA_SERVICE_PORT || 5003}`;

// Initialize cache
const cache = new NodeCache({ stdTTL: 60, checkperiod: 120 });

// ========== REAL TRAINING DATA ==========

// Historical patterns from your actual training data
const WEEKDAY_PATTERNS = {
  0: 90, 1: 51, 2: 26, 3: 14, 4: 21, 5: 86,
  6: 254, 7: 616, 8: 1088, 9: 699, 10: 479, 11: 501,
  12: 582, 13: 593, 14: 636, 15: 784, 16: 1101,
  17: 1459, 18: 1141, 19: 824, 20: 575, 21: 434,
  22: 323, 23: 202
};

const WEEKEND_PATTERNS = {
  0: 234, 1: 209, 2: 123, 3: 36, 4: 23, 5: 35,
  6: 78, 7: 144, 8: 268, 9: 451, 10: 611, 11: 721,
  12: 821, 13: 860, 14: 883, 15: 891, 16: 877,
  17: 834, 18: 765, 19: 614, 20: 456, 21: 358,
  22: 302, 23: 229
};

// Station shares by name (top 100 for performance)
const STATION_SHARES_BY_NAME = {
  "MIT at Mass Ave / Amherst St": 0.017595,
  "Central Square at Mass Ave / Essex St": 0.013825,
  "Harvard Square at Mass Ave/ Dunster": 0.01213,
  "MIT Vassar St": 0.010887,
  "MIT Pacific St at Purrington St": 0.009769,
  "Charles Circle - Charles St at Cambridge St": 0.009595,
  "Ames St at Main St": 0.008707,
  "Christian Science Plaza - Massachusetts Ave at Westland Ave": 0.008487,
  "Boylston St at Fairfield St": 0.008182,
  "Mass Ave/Lafayette Square": 0.008176,
  "Beacon St at Massachusetts Ave": 0.008035,
  "South Station - 700 Atlantic Ave": 0.00788,
  "Commonwealth Ave at Agganis Way": 0.007487,
  "MIT Stata Center at Vassar St / Main St": 0.007287,
  "Forsyth St at Huntington Ave": 0.007137,
  "Mass Ave at Albany St": 0.006972,
  "Ruggles T Stop - Columbus Ave at Melnea Cass Blvd": 0.00687,
  "Landmark Center - Brookline Ave at Park Dr": 0.006465,
  "Boylston St at Jersey St": 0.0064,
  "Massachusetts Ave at Boylston St.": 0.006397,
  "Central Sq Post Office / Cambridge City Hall at Mass Ave / Pleasant St": 0.006286,
  "Chinatown T Stop": 0.006163,
  "Kendall T": 0.00613,
  "Cambridge St at Joy St": 0.006056,
  "Beacon St at Charles St": 0.005959,
  "Boylston St at Massachusetts Ave": 0.00589,
  "Cross St at Hanover St": 0.005843,
  "955 Mass Ave": 0.005834,
  "Lower Cambridgeport at Magazine St / Riverside Rd": 0.005594,
  "Inman Square at Springfield St.": 0.005576,
  "One Kendall Square at Hampshire St / Portland St": 0.005551,
  "Back Bay T Stop - Dartmouth St at Stuart St": 0.005503,
  "MIT Carleton St at Amherst St": 0.005434,
  "Harvard University River Houses at DeWolfe St / Cowperthwaite St": 0.005387,
  "Longwood Ave at Binney St": 0.005359,
  "MIT Hayward St at Amherst St": 0.005269,
  "Mugar Way at Beacon St": 0.005227,
  "Sennott Park Broadway at Norfolk Street": 0.005211,
  "Deerfield St at Commonwealth Ave": 0.005179,
  "Newbury St at Hereford St": 0.005145,
  "Government Center - Cambridge St at Court St": 0.005071,
  "Copley Square - Dartmouth St at Boylston St": 0.005051,
  "Lewis Wharf at Atlantic Ave": 0.005014,
  "Boylston St at Exeter St": 0.004929,
  "Boylston St at Arlington St": 0.004908
};

// Middleware
app.use(cors(corsOptions));
app.use(express.json());
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// ========== GBFS API ENDPOINTS ==========

app.get('/api/stations', async (req, res) => {
  try {
    const cacheKey = 'stations_info';
    const cached = cache.get(cacheKey);
    
    if (cached) {
      return res.json(cached);
    }

    const response = await axios.get(`${GBFS_BASE_URL}/station_information.json`);
    const stations = response.data.data.stations;
    cache.set(cacheKey, stations, 3600);
    res.json(stations);
  } catch (error) {
    console.error('Error fetching stations:', error.message);
    res.status(500).json({ error: 'Failed to fetch stations' });
  }
});

app.get('/api/stations/status', async (req, res) => {
  try {
    const cacheKey = 'stations_status';
    const cached = cache.get(cacheKey);
    
    if (cached) {
      return res.json(cached);
    }

    const response = await axios.get(`${GBFS_BASE_URL}/station_status.json`);
    const status = response.data.data.stations;
    cache.set(cacheKey, status, 60);
    res.json(status);
  } catch (error) {
    console.error('Error fetching station status:', error.message);
    res.status(500).json({ error: 'Failed to fetch station status' });
  }
});

app.get('/api/stations/:id/status', async (req, res) => {
  try {
    const stationId = req.params.id;
    const cacheKey = `station_status_${stationId}`;
    const cached = cache.get(cacheKey);
    
    if (cached) {
      return res.json(cached);
    }

    const response = await axios.get(`${GBFS_BASE_URL}/station_status.json`);
    const stations = response.data.data.stations;
    const stationStatus = stations.find(s => s.station_id === stationId);
    
    if (!stationStatus) {
      return res.status(404).json({ error: 'Station not found' });
    }
    
    cache.set(cacheKey, stationStatus, 60);
    res.json(stationStatus);
  } catch (error) {
    console.error(`Error fetching station ${req.params.id}:`, error.message);
    res.status(500).json({ error: 'Failed to fetch station status' });
  }
});

// ========== ML PREDICTION ENDPOINT ==========

app.post('/api/predict', async (req, res) => {
  try {
    const { station_id, datetime, temperature = 15, precipitation = 0 } = req.body;
    
    if (!station_id) {
      return res.status(400).json({ error: 'station_id is required' });
    }

    console.log('===========================================');
    console.log(' PREDICTION REQUEST');
    console.log('Station:', station_id);
    console.log('DateTime:', datetime || 'now');
    console.log('Temperature:', temperature, 'C');
    console.log('Precipitation:', precipitation, 'mm');

    const systemFeatures = generateSystemWideFeaturesForCloudRun(datetime, temperature, precipitation);
    
    try {
      console.log(' Calling Cloud Run ML Service...');
      
      const response = await axios.post(
        `${EXTERNAL_ML_API_URL}/predict`,
        { features: systemFeatures },
        { 
          timeout: 10000,
          headers: { 'Content-Type': 'application/json' }
        }
      );
      
      let systemWidePrediction = response.data.prediction;
      console.log(' Raw ML prediction:', systemWidePrediction.toFixed(1));
      
      // CLAMP negative predictions to 0 - this is valid behavior
      // You can't have negative bike rides!
      let predictionNote = 'Prediction based on trained ML model';
      let confidence = 'high';
      
      if (systemWidePrediction < 0) {
        console.log(' Clamping negative prediction to 0 (low demand period)');
        systemWidePrediction = 0;
        predictionNote = 'Low demand period (late night)';
        confidence = 'medium';
      }
      
      // Get station share BY NAME
      const stationShare = await getStationShareByName(station_id);
      const stationPrediction = Math.max(0, Math.round(systemWidePrediction * stationShare));
      
      console.log(' System prediction (clamped):', systemWidePrediction.toFixed(1));
      console.log(' Station share:', (stationShare * 100).toFixed(2) + '%');
      console.log(' Final prediction:', stationPrediction);
      console.log('===========================================\n');
      
      return res.json({
        station_id: station_id,
        datetime: datetime || new Date().toISOString(),
        predicted_demand: stationPrediction,
        system_wide_prediction: Math.round(Math.max(0, systemWidePrediction)),
        station_share: stationShare,
        model_version: response.data.model_version || 'production',
        confidence: confidence,
        note: predictionNote
      });
      
    } catch (cloudRunError) {
      console.error(' Cloud Run Error:', cloudRunError.message);
      
      const mockDemand = getMockPrediction(station_id, datetime, temperature, precipitation);
      return res.json({
        station_id: station_id,
        datetime: datetime || new Date().toISOString(),
        predicted_demand: mockDemand,
        model_version: 'mock_fallback',
        confidence: 'medium',
        error: cloudRunError.message
      });
    }
    
  } catch (error) {
    console.error(' Error:', error);
    res.status(500).json({ error: 'Prediction failed', details: error.message });
  }
});

// ========== HELPER FUNCTIONS ==========

function generateSystemWideFeaturesForCloudRun(datetime, temperature = 15, precipitation = 0) {
  let hour, dayOfWeek, month, year, day;
  
  if (datetime) {
    // Check if datetime is UTC (ends with 'Z')
    if (datetime.endsWith('Z')) {
      // UTC timestamp - convert to Boston time
      const utcDate = new Date(datetime);
      
      // Boston is UTC-5 (EST) in winter (Nov-Mar), UTC-4 (EDT) in summer
      const BOSTON_OFFSET_HOURS = -5; // December = EST
      const bostonMs = utcDate.getTime() + (BOSTON_OFFSET_HOURS * 60 * 60 * 1000);
      const bostonDate = new Date(bostonMs);
      
      hour = bostonDate.getUTCHours();
      dayOfWeek = bostonDate.getUTCDay();
      month = bostonDate.getUTCMonth() + 1;
      year = bostonDate.getUTCFullYear();
      day = bostonDate.getUTCDate();
      
      console.log(` Input (UTC): ${datetime}`);
      console.log(` Converted to Boston: hour=${hour}, day=${day}`);
    } else {
      // No 'Z' suffix - parse as local time directly from string
      // This avoids timezone interpretation issues
      const match = datetime.match(/(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
      
      if (match) {
        year = parseInt(match[1]);
        month = parseInt(match[2]);
        day = parseInt(match[3]);
        hour = parseInt(match[4]);
        
        // Calculate day of week from the date
        const tempDate = new Date(year, month - 1, day);
        dayOfWeek = tempDate.getDay();
        
        console.log(` Parsed local time: ${year}-${month}-${day} hour=${hour}`);
      } else {
        // Fallback - shouldn't happen with proper input
        console.warn(` Could not parse datetime: ${datetime}, using current time`);
        const now = new Date();
        hour = now.getUTCHours();
        dayOfWeek = now.getUTCDay();
        month = now.getUTCMonth() + 1;
        year = now.getUTCFullYear();
        day = now.getUTCDate();
      }
    }
  } else {
    // No datetime provided - use current Boston time
    const now = new Date();
    const BOSTON_OFFSET_HOURS = -5;
    const bostonMs = now.getTime() + (BOSTON_OFFSET_HOURS * 60 * 60 * 1000);
    const bostonDate = new Date(bostonMs);
    
    hour = bostonDate.getUTCHours();
    dayOfWeek = bostonDate.getUTCDay();
    month = bostonDate.getUTCMonth() + 1;
    year = bostonDate.getUTCFullYear();
    day = bostonDate.getUTCDate();
    
    console.log(` No datetime provided, using Boston time: hour=${hour}`);
  }
  
  // Use REAL historical patterns from training data
  const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
  const baseRides = isWeekend ? WEEKEND_PATTERNS[hour] : WEEKDAY_PATTERNS[hour];
  
  // Calculate related historical features
  const rides_last_hour = baseRides;
  const rides_same_hour_yesterday = Math.round(baseRides * 0.95);
  const rides_same_hour_last_week = Math.round(baseRides * 0.98);
  const rides_rolling_3h = Math.round(baseRides * 1.05);
  const rides_rolling_24h = Math.round(baseRides * 8);
  
  console.log(` Hour ${hour} (${isWeekend ? 'weekend' : 'weekday'}): baseRides=${baseRides}`);
  
  // Convert to NOAA units (Fahrenheit and inches)
  const temp_fahrenheit = (temperature * 9/5) + 32;
  const temp_max = temp_fahrenheit + 4;
  const temp_min = temp_fahrenheit - 4;
  const temp_range = 8;
  const temp_avg = temp_fahrenheit;
  const precip_inches = precipitation / 25.4;
  
  console.log(` Temperature: ${temperature}C = ${temp_fahrenheit.toFixed(1)}F`);
  
  const features = [
    // 1-5: Temporal
    hour, dayOfWeek, month, year, day,
    
    // 6-11: Cyclical encodings
    Math.sin(2 * Math.PI * hour / 24),
    Math.cos(2 * Math.PI * hour / 24),
    Math.sin(2 * Math.PI * dayOfWeek / 7),
    Math.cos(2 * Math.PI * dayOfWeek / 7),
    Math.sin(2 * Math.PI * month / 12),
    Math.cos(2 * Math.PI * month / 12),
    
    // 12-16: Time of day flags
    hour >= 7 && hour <= 9 ? 1 : 0,
    hour >= 17 && hour <= 19 ? 1 : 0,
    hour >= 22 || hour <= 5 ? 1 : 0,
    hour >= 11 && hour <= 14 ? 1 : 0,
    isWeekend ? 1 : 0,
    
    // 17-19: Interaction features
    isWeekend * (hour >= 22 || hour <= 5 ? 1 : 0),
    (!isWeekend ? 1 : 0) * (hour >= 7 && hour <= 9 ? 1 : 0),
    (!isWeekend ? 1 : 0) * (hour >= 17 && hour <= 19 ? 1 : 0),
    
    // 20-24: Weather (Fahrenheit and inches)
    temp_max,
    temp_min,
    precip_inches,
    temp_range,
    temp_avg,
    
    // 25-28: Weather flags
    precip_inches > 0.01 ? 1 : 0,
    precip_inches > 0.1 ? 1 : 0,
    temp_fahrenheit < 50 ? 1 : 0,
    temp_fahrenheit > 77 ? 1 : 0,
    
    // 29-33: Historical patterns
    rides_last_hour,
    rides_same_hour_yesterday,
    rides_same_hour_last_week,
    rides_rolling_3h,
    rides_rolling_24h,
    
    // 34-40: Trip statistics
    15.5, 8.2, 12.0, 3.8, 1.9, 2.5, 0.65,
    
    // 41-48: Bias mitigation features
    hour === 8 ? 1 : 0,
    (hour === 17 || hour === 18) ? 1 : 0,
    (hour === 8 || hour === 17 || hour === 18) ? 1.0 :
      (hour === 7 || hour === 9 || hour === 16 || hour === 19) ? 0.5 : 0.0,
    rides_last_hour > 800 ? 1 : 0,
    rides_last_hour < 200 ? 1 : 0,
    Math.abs(rides_last_hour - rides_rolling_3h),
    ((hour === 8 ? 1 : 0) + 
     ((hour === 17 || hour === 18) ? 1 : 0) +
     ((!isWeekend ? 1 : 0) * (hour >= 7 && hour <= 9 ? 1 : 0)) +
     ((!isWeekend ? 1 : 0) * (hour >= 17 && hour <= 19 ? 1 : 0))) > 0 ? 1 : 0,
    hour >= 0 && hour < 6 ? 0 :
      hour >= 6 && hour < 10 ? 1 :
      hour >= 10 && hour < 14 ? 2 :
      hour >= 14 && hour < 18 ? 3 : 4
  ];
  
  console.log(`Generated ${features.length} features`);
  
  return features;
}

async function getStationShareByName(stationId) {
  try {
    // Fetch station info from GBFS to get name
    const cacheKey = 'stations_info';
    let stations = cache.get(cacheKey);
    
    if (!stations) {
      const response = await axios.get(`${GBFS_BASE_URL}/station_information.json`);
      stations = response.data.data.stations;
      cache.set(cacheKey, stations, 3600);
    }
    
    const station = stations.find(s => s.station_id === stationId);
    
    if (station && station.name) {
      const stationName = station.name.trim();
      
      // Try exact match first
      if (STATION_SHARES_BY_NAME[stationName]) {
        const share = STATION_SHARES_BY_NAME[stationName];
        console.log(` Found share for "${stationName}": ${(share * 100).toFixed(2)}%`);
        return share;
      }
      
      // Try case-insensitive match
      const lowerName = stationName.toLowerCase();
      for (const [name, share] of Object.entries(STATION_SHARES_BY_NAME)) {
        if (name.toLowerCase() === lowerName) {
          console.log(` Found share (case-insensitive) for "${name}": ${(share * 100).toFixed(2)}%`);
          return share;
        }
      }
      
      console.warn(`Station "${stationName}" not in training data, using average`);
    } else {
      console.warn(` Station ${stationId} not found in GBFS`);
    }
  } catch (err) {
    console.warn('Could not fetch station info:', err.message);
  }
  
  // Fallback to average share
  return 0.005;
}

function getMockPrediction(stationId, datetime, temperature = 15, precipitation = 0) {
  const dt = datetime ? new Date(datetime) : new Date();
  const hour = dt.getHours();
  const dayOfWeek = dt.getDay();
  const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
  
  // Use REAL hourly patterns
  const systemBase = isWeekend ? WEEKEND_PATTERNS[hour] : WEEKDAY_PATTERNS[hour];
  
  // Apply weather adjustments
  let weatherMultiplier = 1.0;
  if (temperature < 5) weatherMultiplier *= 0.6;
  else if (temperature < 10) weatherMultiplier *= 0.8;
  if (temperature > 30) weatherMultiplier *= 0.85;
  if (precipitation > 0.5) weatherMultiplier *= 0.5;
  else if (precipitation > 0.1) weatherMultiplier *= 0.7;
  
  const systemPrediction = Math.round(systemBase * weatherMultiplier);
  const stationPrediction = Math.round(systemPrediction * 0.005); // Average 0.5%
  
  return Math.max(0, stationPrediction);
}



app.get('/api/historical/:stationId/:timeRange', async (req, res) => {
  try {
    const { stationId, timeRange } = req.params;
    
    if (!['hourly', 'daily', 'weekly'].includes(timeRange)) {
      return res.status(400).json({ error: 'Invalid time range' });
    }
    
    const response = await axios.get(
      `${HISTORICAL_SERVICE_URL}/api/historical/${stationId}/${timeRange}`,
      { timeout: 30000 }
    );
    
    res.json(response.data);
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ 
        error: 'Historical service unavailable',
        data: []
      });
    }
    
    res.status(500).json({ error: 'Failed to get historical data' });
  }
});



app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    cache_keys: cache.keys().length,
    ml_service: EXTERNAL_ML_API_URL
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Bluebikes Backend Server running on port ${PORT}`);
  console.log(`GBFS API: ${GBFS_BASE_URL}`);
  console.log(`ML Service: ${EXTERNAL_ML_API_URL}`);
  console.log(`Historical Service: ${HISTORICAL_SERVICE_URL}`);
  console.log(`\n✅ Loaded ${Object.keys(WEEKDAY_PATTERNS).length} hourly patterns`);
  console.log(`✅ Loaded ${Object.keys(STATION_SHARES_BY_NAME).length} station shares`);
  console.log(`\nAvailable endpoints:`);
  console.log(`  GET  /api/stations`);
  console.log(`  GET  /api/stations/status`);
  console.log(`  GET  /api/stations/:id/status`);
  console.log(`  POST /api/predict`);
  console.log(`  GET  /api/historical/:stationId/:timeRange`);
  console.log(`  GET  /health`);
});