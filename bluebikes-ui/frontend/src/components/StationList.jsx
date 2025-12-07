import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStations } from '../context/StationContext';

const StationList = () => {
    const { stations, stationStatus, fetchAllStationStatus, loading, error } = useStations();
    const [searchTerm, setSearchTerm] = useState('');
    const [sortConfig, setSortConfig] = useState({ key: 'name', direction: 'asc' });
    const navigate = useNavigate();

    useEffect(() => {
        if (stations.length > 0 && Object.keys(stationStatus).length === 0) {
            fetchAllStationStatus();
        }
    }, [stations]);

    const handleSort = (key) => {
        setSortConfig(prev => ({
            key,
            direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
        }));
    };

    const filteredStations = stations.filter(station =>
        station.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const sortedStations = [...filteredStations].sort((a, b) => {
        const aStatus = stationStatus[a.station_id];
        const bStatus = stationStatus[b.station_id];

        let aValue, bValue;

        switch (sortConfig.key) {
            case 'name':
                aValue = a.name;
                bValue = b.name;
                break;
            case 'bikes':
                aValue = aStatus?.num_bikes_available || 0;
                bValue = bStatus?.num_bikes_available || 0;
                break;
            case 'docks':
                aValue = aStatus?.num_docks_available || 0;
                bValue = bStatus?.num_docks_available || 0;
                break;
            case 'capacity':
                aValue = a.capacity || 0;
                bValue = b.capacity || 0;
                break;
            default:
                return 0;
        }

        if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
        if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
    });

    if (loading) {
        return (
            <div className="loading-container">
                <div className="spinner"></div>
                <p>Loading stations...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="error-container">
                <h2>⚠️ Error</h2>
                <p>{error}</p>
            </div>
        );
    }

    return (
        <div className="station-list-view">
            <div className="list-header">
                <h2>All Bluebikes Stations ({stations.length})</h2>
                <input
                    type="text"
                    placeholder="🔍 Search stations..."
                    className="search-input"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                />
            </div>

            <div className="table-container">
                <table className="station-table">
                    <thead>
                        <tr>
                            <th onClick={() => handleSort('name')} className="sortable">
                                Station Name {sortConfig.key === 'name' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                            </th>
                            <th>ID</th>
                            <th onClick={() => handleSort('bikes')} className="sortable">
                                Bikes {sortConfig.key === 'bikes' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                            </th>
                            <th onClick={() => handleSort('docks')} className="sortable">
                                Docks {sortConfig.key === 'docks' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                            </th>
                            <th onClick={() => handleSort('capacity')} className="sortable">
                                Capacity {sortConfig.key === 'capacity' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                            </th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sortedStations.map(station => {
                            const status = stationStatus[station.station_id];
                            const bikesAvailable = status?.num_bikes_available || 0;
                            const docksAvailable = status?.num_docks_available || 0;
                            const isActive = status?.is_renting && status?.is_returning;

                            return (
                                <tr key={station.station_id} className="station-row">
                                    <td className="station-name">{station.name}</td>
                                    <td className="station-id">{station.station_id}</td>
                                    <td>
                                        <span className={`availability-badge ${bikesAvailable === 0 ? 'empty' : bikesAvailable <= 5 ? 'low' : 'good'}`}>
                                            {status ? bikesAvailable : '—'}
                                        </span>
                                    </td>
                                    <td>
                                        <span className="availability-value">
                                            {status ? docksAvailable : '—'}
                                        </span>
                                    </td>
                                    <td>{station.capacity || '—'}</td>
                                    <td>
                                        <span className={`status-badge ${isActive ? 'active' : 'inactive'}`}>
                                            {status ? (isActive ? 'Active' : 'Inactive') : 'Unknown'}
                                        </span>
                                    </td>
                                    <td>
                                        <button
                                            className="view-btn"
                                            onClick={() => navigate(`/stations/${station.station_id}`)}
                                        >
                                            View
                                        </button>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {sortedStations.length === 0 && searchTerm && (
                <div className="no-results">
                    <p>No stations found matching "{searchTerm}"</p>
                </div>
            )}
        </div>
    );
};

export default StationList;
