import React, { useState, useEffect } from 'react';
import { useStations } from '../context/StationContext';

const RebalancingView = () => {
    const { stations, stationStatus, getPrediction, fetchAllStationStatus } = useStations();
    const [predictions, setPredictions] = useState({});
    const [loadingPredictions, setLoadingPredictions] = useState(true);
    const [searchRadius, setSearchRadius] = useState(1.5); // miles
    const [recommendations, setRecommendations] = useState([]);

    // Fetch all predictions on mount
    useEffect(() => {
        const fetchPredictions = async () => {
            setLoadingPredictions(true);
            const predMap = {};

            // Fetch predictions for all stations
            for (let i = 0; i < stations.length; i++) {
                const station = stations[i];
                const pred = await getPrediction(station.station_id);
                if (pred) {
                    predMap[station.station_id] = pred;
                }
            }

            setPredictions(predMap);
            setLoadingPredictions(false);
        };

        if (stations.length > 0) {
            fetchAllStationStatus();
            fetchPredictions();
        }
    }, [stations]);

    // Calculate distance between two coordinates (Haversine formula)
    const calculateDistance = (lat1, lon1, lat2, lon2) => {
        const R = 3958.8; // Radius of the Earth in miles
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a =
            Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    };

    // Calculate rebalancing recommendations
    useEffect(() => {
        if (Object.keys(predictions).length === 0 || Object.keys(stationStatus).length === 0) {
            return;
        }

        const recs = [];

        // Analyze each station
        stations.forEach(station => {
            const status = stationStatus[station.station_id];
            const prediction = predictions[station.station_id];

            if (!status || !prediction) return;

            const bikesAvailable = status.num_bikes_available || 0;
            const predictedDemand = prediction.predicted_demand || 0;
            const capacity = station.capacity || 20;

            // Station needs bikes: low inventory relative to predicted demand
            const deficit = predictedDemand - bikesAvailable;
            const needsBikes = bikesAvailable <= 5 && deficit > 0;

            if (needsBikes) {
                // Find nearby stations with excess bikes
                const nearbyDonors = stations
                    .filter(donor => {
                        if (donor.station_id === station.station_id) return false;

                        const donorStatus = stationStatus[donor.station_id];
                        const donorPrediction = predictions[donor.station_id];

                        if (!donorStatus || !donorPrediction) return false;

                        const donorBikes = donorStatus.num_bikes_available || 0;
                        const donorDemand = donorPrediction.predicted_demand || 0;

                        // Donor has excess: bikes available exceed predicted demand with buffer
                        // Must have at least 3 bikes surplus after meeting predicted demand
                        const surplus = donorBikes - donorDemand;
                        const hasExcess = surplus >= 3;

                        if (!hasExcess) return false;

                        // Check distance
                        const distance = calculateDistance(
                            station.lat, station.lon,
                            donor.lat, donor.lon
                        );

                        return distance <= searchRadius;
                    })
                    .map(donor => {
                        const distance = calculateDistance(
                            station.lat, station.lon,
                            donor.lat, donor.lon
                        );
                        const donorStatus = stationStatus[donor.station_id];
                        return {
                            ...donor,
                            distance,
                            availableBikes: donorStatus.num_bikes_available
                        };
                    })
                    .sort((a, b) => a.distance - b.distance);

                if (nearbyDonors.length > 0) {
                    const bestDonor = nearbyDonors[0];
                    const donorPrediction = predictions[bestDonor.station_id];
                    const donorSurplus = bestDonor.availableBikes - (donorPrediction?.predicted_demand || 0);
                    const bikesNeeded = Math.min(
                        capacity - bikesAvailable,        // Space available at recipient
                        Math.ceil(predictedDemand * 1.5), // 1.5x predicted demand
                        donorSurplus - 2,                 // Leave 2 bikes buffer at donor
                        10                                // Max 10 bikes per rebalancing
                    );

                    recs.push({
                        recipient: station,
                        donor: bestDonor,
                        bikesNeeded,
                        currentBikes: bikesAvailable,
                        predictedDemand,
                        donorBikes: bestDonor.availableBikes,
                        distance: bestDonor.distance,
                        priority: predictedDemand - bikesAvailable // Higher = more urgent
                    });
                }
            }
        });

        // Sort by priority (most urgent first)
        recs.sort((a, b) => b.priority - a.priority);

        setRecommendations(recs);
    }, [predictions, stationStatus, stations, searchRadius]);

    if (loadingPredictions) {
        return (
            <div className="loading-container">
                <div className="spinner"></div>
                <p>Analyzing station demand and availability...</p>
            </div>
        );
    }

    return (
        <div className="rebalancing-view">
            <div className="rebalancing-header">
                <h2>🔄 Bike Rebalancing Recommendations</h2>
                <p className="rebalancing-subtitle">
                    AI-powered suggestions to optimize bike distribution based on predicted demand
                </p>
            </div>

            <div className="rebalancing-controls">
                <div className="control-group">
                    <label htmlFor="radius">Search Radius:</label>
                    <select
                        id="radius"
                        value={searchRadius}
                        onChange={(e) => setSearchRadius(parseFloat(e.target.value))}
                        className="radius-select"
                    >
                        <option value="0.5">0.5 miles</option>
                        <option value="1.0">1.0 mile</option>
                        <option value="1.5">1.5 miles</option>
                        <option value="2.0">2.0 miles</option>
                        <option value="3.0">3.0 miles</option>
                    </select>
                </div>
                <div className="stats-summary">
                    <span className="stat-badge">
                        {recommendations.length} Recommendations
                    </span>
                    <span className="stat-badge">
                        {recommendations.reduce((sum, r) => sum + r.bikesNeeded, 0)} Total Bikes to Move
                    </span>
                </div>
            </div>

            {recommendations.length === 0 ? (
                <div className="no-recommendations">
                    <h3> All Stations Well Balanced!</h3>
                    <p>No rebalancing needed at this time. All stations have adequate bikes for predicted demand.</p>
                </div>
            ) : (
                <div className="recommendations-list">
                    {recommendations.map((rec, index) => (
                        <div key={index} className="recommendation-card">
                            <div className="rec-header">
                                <span className="rec-number">#{index + 1}</span>
                                <span className={`priority-badge priority-${rec.priority > 5 ? 'high' : rec.priority > 2 ? 'medium' : 'low'}`}>
                                    {rec.priority > 5 ? 'High Priority' : rec.priority > 2 ? 'Medium Priority' : 'Low Priority'}
                                </span>
                            </div>

                            <div className="rec-content">
                                <div className="station-info donor">
                                    <h4>🚴 Source (Surplus Inventory)</h4>
                                    <h3>{rec.donor.name}</h3>
                                    <div className="station-stats">
                                        <div className="stat">
                                            <span className="label">Available Bikes:</span>
                                            <span className="value success">{rec.donorBikes}</span>
                                        </div>
                                        <div className="stat">
                                            <span className="label">After Transfer:</span>
                                            <span className="value">{rec.donorBikes - rec.bikesNeeded}</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="rec-action">
                                    <div className="arrow-container">
                                        <div className="bikes-to-move">
                                            Move {rec.bikesNeeded} bikes →
                                        </div>
                                        <div className="distance-info">
                                            {rec.distance.toFixed(2)} mi
                                        </div>
                                    </div>
                                </div>

                                <div className="station-info recipient">
                                    <h4>📍 Destination (Low inventory)</h4>
                                    <h3>{rec.recipient.name}</h3>
                                    <div className="station-stats">
                                        <div className="stat">
                                            <span className="label">Current Bikes:</span>
                                            <span className="value danger">{rec.currentBikes}</span>
                                        </div>
                                        <div className="stat">
                                            <span className="label">Predicted Demand:</span>
                                            <span className="value">{rec.predictedDemand} rides</span>
                                        </div>
                                        <div className="stat">
                                            <span className="label">Capacity:</span>
                                            <span className="value">{rec.recipient.capacity}</span>
                                        </div>
                                    </div>
                                </div>

                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default RebalancingView;
