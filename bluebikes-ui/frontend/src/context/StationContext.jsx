import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const StationContext = createContext();

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://bluebikes-backend-202855070348.us-central1.run.app/api';

export const useStations = () => {
    const context = useContext(StationContext);
    if (!context) {
        throw new Error('useStations must be used within StationProvider');
    }
    return context;
};

export const StationProvider = ({ children }) => {
    const [stations, setStations] = useState([]);
    const [stationStatus, setStationStatus] = useState({});
    const [selectedStation, setSelectedStation] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Fetch station information on mount
    useEffect(() => {
        fetchStations();
    }, []);

    const fetchStations = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await axios.get(`${API_BASE_URL}/stations`);
            setStations(response.data);
        } catch (err) {
            console.error('Error fetching stations:', err);
            setError('Failed to load stations. Please check if the backend is running.');
        } finally {
            setLoading(false);
        }
    };

    const fetchStationStatus = async (stationId) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/stations/${stationId}/status`);
            setStationStatus(prev => ({
                ...prev,
                [stationId]: response.data
            }));
            return response.data;
        } catch (err) {
            console.error(`Error fetching status for station ${stationId}:`, err);
            return null;
        }
    };

    const fetchAllStationStatus = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/stations/status`);
            const statusMap = {};
            response.data.forEach(status => {
                statusMap[status.station_id] = status;
            });
            setStationStatus(statusMap);
            return statusMap;
        } catch (err) {
            console.error('Error fetching all station statuses:', err);
            return {};
        }
    };

    const getPrediction = async (stationId, datetime = null) => {
        try {
            const response = await axios.post(`${API_BASE_URL}/predict`, {
                station_id: stationId,
                datetime: datetime || new Date().toISOString(),
                temperature: 15, // Default values
                precipitation: 0
            });
            return response.data;
        } catch (err) {
            console.error(`Error getting prediction for station ${stationId}:`, err);
            return { predicted_demand: null, error: 'Prediction unavailable' };
        }
    };

    const getStationInfo = (stationId) => {
        return stations.find(s => s.station_id === stationId);
    };

    const value = {
        stations,
        stationStatus,
        selectedStation,
        setSelectedStation,
        loading,
        error,
        fetchStations,
        fetchStationStatus,
        fetchAllStationStatus,
        getPrediction,
        getStationInfo
    };

    return (
        <StationContext.Provider value={value}>
            {children}
        </StationContext.Provider>
    );
};
