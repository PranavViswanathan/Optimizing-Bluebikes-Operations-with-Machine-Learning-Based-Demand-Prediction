import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import { useStations } from '../context/StationContext';
import StationPopup from './StationPopup';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default marker icon issue in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Custom marker icons based on availability
const createCustomIcon = (availability) => {
    let color;
    if (availability === null || availability === undefined) {
        color = '#808080'; // Gray for unknown
    } else if (availability === 0) {
        color = '#ef4444'; // Red for no bikes
    } else if (availability <= 5) {
        color = '#f59e0b'; // Orange/Yellow for low availability
    } else {
        color = '#10b981'; // Green for good availability
    }

    return L.divIcon({
        className: 'custom-marker',
        html: `
      <div style="
        background-color: ${color};
        width: 24px;
        height: 24px;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
      "></div>
    `,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
    });
};

// Component to update map bounds when stations load
const MapUpdater = ({ stations }) => {
    const map = useMap();

    useEffect(() => {
        if (stations && stations.length > 0) {
            const bounds = stations.map(s => [s.lat, s.lon]);
            if (bounds.length > 0) {
                map.fitBounds(bounds, { padding: [50, 50], maxZoom: 14 });
            }
        }
    }, [stations, map]);

    return null;
};

const MapView = () => {
    const { stations, stationStatus, fetchStationStatus, fetchAllStationStatus, loading, error } = useStations();
    const [hoveredStation, setHoveredStation] = useState(null);
    const [filter, setFilter] = useState('all'); // 'all', 'available', 'low', 'empty'

    // Fetch all station statuses on mount
    useEffect(() => {
        if (stations.length > 0) {
            fetchAllStationStatus();
        }
    }, [stations]);

    // Filter stations based on selected filter
    const getFilteredStations = () => {
        if (filter === 'all') return stations;

        return stations.filter(station => {
            const status = stationStatus[station.station_id];
            if (!status) return false;

            const bikesAvailable = status.num_bikes_available || 0;

            switch (filter) {
                case 'available':
                    return bikesAvailable > 5;
                case 'low':
                    return bikesAvailable >= 1 && bikesAvailable <= 5;
                case 'empty':
                    return bikesAvailable === 0;
                default:
                    return true;
            }
        });
    };

    const filteredStations = getFilteredStations();

    if (loading) {
        return (
            <div className="loading-container">
                <div className="spinner"></div>
                <p>Loading Bluebikes stations...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="error-container">
                <h2>⚠️ Error</h2>
                <p>{error}</p>
                <p className="error-hint">
                    Make sure the backend server is running on <code>http://localhost:5000</code>
                </p>
            </div>
        );
    }

    if (stations.length === 0) {
        return (
            <div className="error-container">
                <h2>No Stations Found</h2>
                <p>Unable to load station data</p>
            </div>
        );
    }

    // Boston coordinates as default center
    const bostonCenter = [42.3601, -71.0589];

    return (
        <div className="map-view">
            <div className="map-info">
                <div className="info-card">
                    <span className="info-label">
                        {filter === 'all' ? 'Total Stations:' : 'Filtered Stations:'}
                    </span>
                    <span className="info-value">{filteredStations.length}</span>
                </div>
                <div className="legend">
                    <button
                        className={`legend-item filter-btn ${filter === 'all' ? 'active' : ''}`}
                        onClick={() => setFilter('all')}
                    >
                        <span className="legend-dot all-dot"></span>
                        All ({stations.length})
                    </button>
                    <button
                        className={`legend-item filter-btn ${filter === 'available' ? 'active' : ''}`}
                        onClick={() => setFilter('available')}
                    >
                        <span className="legend-dot" style={{ backgroundColor: '#10b981' }}></span>
                        Available (5+)
                    </button>
                    <button
                        className={`legend-item filter-btn ${filter === 'low' ? 'active' : ''}`}
                        onClick={() => setFilter('low')}
                    >
                        <span className="legend-dot" style={{ backgroundColor: '#f59e0b' }}></span>
                        Low (1-5)
                    </button>
                    <button
                        className={`legend-item filter-btn ${filter === 'empty' ? 'active' : ''}`}
                        onClick={() => setFilter('empty')}
                    >
                        <span className="legend-dot" style={{ backgroundColor: '#ef4444' }}></span>
                        Empty (0)
                    </button>
                </div>
            </div>

            <MapContainer
                center={bostonCenter}
                zoom={13}
                style={{ height: 'calc(100vh - 250px)', width: '100%' }}
                className="leaflet-map"
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                <MapUpdater stations={stations} />

                {filteredStations.map((station) => {
                    const status = stationStatus[station.station_id];
                    const bikesAvailable = status?.num_bikes_available;

                    return (
                        <Marker
                            key={station.station_id}
                            position={[station.lat, station.lon]}
                            icon={createCustomIcon(bikesAvailable)}
                            eventHandlers={{
                                mouseover: () => {
                                    setHoveredStation(station.station_id);
                                    if (!stationStatus[station.station_id]) {
                                        fetchStationStatus(station.station_id);
                                    }
                                },
                                mouseout: () => setHoveredStation(null)
                            }}
                        >
                            <Popup>
                                <StationPopup station={station} />
                            </Popup>
                        </Marker>
                    );
                })}
            </MapContainer>
        </div>
    );
};

export default MapView;
