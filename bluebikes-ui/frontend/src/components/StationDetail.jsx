import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useStations } from '../context/StationContext';

const StationDetail = () => {
    const { stationId } = useParams();
    const navigate = useNavigate();
    const { getStationInfo, fetchStationStatus, stationStatus, getPrediction } = useStations();
    const [station, setStation] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadData = async () => {
            setLoading(true);

            // Get station info
            const stationInfo = getStationInfo(stationId);
            setStation(stationInfo);

            // Fetch current status
            if (!stationStatus[stationId]) {
                await fetchStationStatus(stationId);
            }

            // Get prediction
            const pred = await getPrediction(stationId);
            setPrediction(pred);

            setLoading(false);
        };

        loadData();
    }, [stationId]);

    if (loading) {
        return (
            <div className="loading-container">
                <div className="spinner"></div>
                <p>Loading station details...</p>
            </div>
        );
    }

    if (!station) {
        return (
            <div className="error-container">
                <h2>Station Not Found</h2>
                <p>No station found with ID: {stationId}</p>
                <button className="back-btn" onClick={() => navigate(-1)}>
                    ← Go Back
                </button>
            </div>
        );
    }

    const status = stationStatus[stationId];
    const bikesAvailable = status?.num_bikes_available || 0;
    const docksAvailable = status?.num_docks_available || 0;
    const isActive = status?.is_renting && status?.is_returning;

    return (
        <div className="station-detail-view">
            <button className="back-btn" onClick={() => navigate(-1)}>
                ← Back
            </button>

            <div className="detail-header">
                <h1>{station.name}</h1>
                <span className={`status-badge large ${isActive ? 'active' : 'inactive'}`}>
                    {status ? (isActive ? 'Active' : 'Inactive') : 'Unknown'}
                </span>
            </div>

            <div className="detail-grid">
                {/* Station Info Card */}
                <div className="detail-card">
                    <h3>📍 Station Information</h3>
                    <div className="detail-item">
                        <span className="label">Station ID:</span>
                        <span className="value">{station.station_id}</span>
                    </div>
                    <div className="detail-item">
                        <span className="label">Capacity:</span>
                        <span className="value">{station.capacity || 'N/A'} docks</span>
                    </div>
                    <div className="detail-item">
                        <span className="label">Location:</span>
                        <span className="value">
                            {station.lat.toFixed(4)}°N, {station.lon.toFixed(4)}°W
                        </span>
                    </div>
                    <div className="detail-item">
                        <span className="label">Address:</span>
                        <span className="value">{station.address || 'Not available'}</span>
                    </div>
                </div>

                {/* Real-time Status Card */}
                <div className="detail-card highlight">
                    <h3>⚡ Real-time Status</h3>
                    {status ? (
                        <>
                            <div className="detail-item large">
                                <span className="label">🚴 Available Bikes:</span>
                                <span className={`value bikes-value ${bikesAvailable === 0 ? 'empty' : bikesAvailable <= 5 ? 'low' : 'good'}`}>
                                    {bikesAvailable}
                                </span>
                            </div>
                            <div className="detail-item large">
                                <span className="label">🅿️ Available Docks:</span>
                                <span className="value">{docksAvailable}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">Is Renting:</span>
                                <span className="value">{status.is_renting ? '✅ Yes' : '❌ No'}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">Is Returning:</span>
                                <span className="value">{status.is_returning ? '✅ Yes' : '❌ No'}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">Last Reported:</span>
                                <span className="value">
                                    {status.last_reported
                                        ? new Date(status.last_reported * 1000).toLocaleString()
                                        : 'Unknown'}
                                </span>
                            </div>
                        </>
                    ) : (
                        <p className="no-data">Status data unavailable</p>
                    )}
                </div>

                {/* ML Prediction Card */}
                <div className="detail-card prediction-card">
                    <h3>🔮 ML Demand Prediction</h3>
                    {prediction && prediction.predicted_demand !== null ? (
                        <>
                            <div className="prediction-display">
                                <div className="prediction-main">
                                    <span className="prediction-number">{prediction.predicted_demand}</span>
                                    <span className="prediction-unit">rides/hour</span>
                                </div>
                                <p className="prediction-subtitle">Predicted demand for next hour</p>
                            </div>

                            <div className="detail-item">
                                <span className="label">Model:</span>
                                <span className="value">{prediction.model_version || 'XGBoost v1'}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">Confidence:</span>
                                <span className={`value confidence-${prediction.confidence}`}>
                                    {prediction.confidence || 'N/A'}
                                </span>
                            </div>
                            {prediction.model_status === 'mock' && (
                                <div className="warning-box">
                                    <p>⚠️ Using mock predictions (ML model not loaded)</p>
                                    <p className="small-text">Place trained model at backend/models/best_model.pkl</p>
                                </div>
                            )}
                            <div className="detail-item">
                                <span className="label">Prediction Time:</span>
                                <span className="value">
                                    {prediction.datetime
                                        ? new Date(prediction.datetime).toLocaleString()
                                        : 'Now'}
                                </span>
                            </div>
                        </>
                    ) : (
                        <div className="no-data">
                            <p>Prediction unavailable</p>
                            <p className="small-text">ML service may be offline</p>
                        </div>
                    )}
                </div>

                {/* Availability Indicator */}
                <div className="detail-card availability-indicator">
                    <h3>📊 Availability Status</h3>
                    <div className="availability-visualization">
                        {status && (
                            <>
                                <div className="availability-bar">
                                    <div
                                        className="bikes-bar"
                                        style={{ width: `${(bikesAvailable / (station.capacity || 1)) * 100}%` }}
                                    >
                                        <span>{bikesAvailable} bikes</span>
                                    </div>
                                </div>
                                <div className="availability-bar">
                                    <div
                                        className="docks-bar"
                                        style={{ width: `${(docksAvailable / (station.capacity || 1)) * 100}%` }}
                                    >
                                        <span>{docksAvailable} docks</span>
                                    </div>
                                </div>
                                <div className="capacity-info">
                                    Total Capacity: {station.capacity || 'N/A'} docks
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StationDetail;
