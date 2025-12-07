import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStations } from '../context/StationContext';

const StationPopup = ({ station }) => {
    const { stationStatus, getPrediction } = useStations();
    const [prediction, setPrediction] = useState(null);
    const [loadingPrediction, setLoadingPrediction] = useState(false);
    const navigate = useNavigate();

    const status = stationStatus[station.station_id];

    useEffect(() => {
        // Fetch prediction when popup opens
        const fetchPrediction = async () => {
            setLoadingPrediction(true);
            const pred = await getPrediction(station.station_id);
            setPrediction(pred);
            setLoadingPrediction(false);
        };

        fetchPrediction();
    }, [station.station_id]);

    const handleViewDetails = () => {
        navigate(`/stations/${station.station_id}`);
    };

    return (
        <div className="station-popup">
            <h3 className="popup-title">{station.name}</h3>

            <div className="popup-section">
                <p className="popup-label">Station ID:</p>
                <p className="popup-value">{station.station_id}</p>
            </div>

            {status ? (
                <>
                    <div className="popup-section">
                        <p className="popup-label">🚴 Available Bikes:</p>
                        <p className="popup-value bikes-available">{status.num_bikes_available || 0}</p>
                    </div>

                    <div className="popup-section">
                        <p className="popup-label">🅿️ Available Docks:</p>
                        <p className="popup-value">{status.num_docks_available || 0}</p>
                    </div>

                    <div className="popup-section">
                        <p className="popup-label">📊 Status:</p>
                        <p className="popup-value">
                            <span className={`status-badge ${status.is_renting && status.is_returning ? 'active' : 'inactive'}`}>
                                {status.is_renting && status.is_returning ? 'Active' : 'Inactive'}
                            </span>
                        </p>
                    </div>
                </>
            ) : (
                <div className="popup-loading">
                    <div className="mini-spinner"></div>
                    <p>Loading status...</p>
                </div>
            )}

            <div className="popup-section prediction-section">
                <p className="popup-label">🔮 Predicted Demand (Next Hour):</p>
                {loadingPrediction ? (
                    <div className="popup-loading">
                        <div className="mini-spinner"></div>
                    </div>
                ) : prediction && prediction.predicted_demand !== null ? (
                    <div>
                        <p className="popup-value prediction-value">
                            {prediction.predicted_demand} rides
                        </p>
                        {prediction.model_status === 'mock' && (
                            <p className="prediction-note">⚠️ Using mock data (model not loaded)</p>
                        )}
                    </div>
                ) : (
                    <p className="popup-value">Unavailable</p>
                )}
            </div>

            <button className="view-details-btn" onClick={handleViewDetails}>
                View Full Details →
            </button>
        </div>
    );
};

export default StationPopup;
