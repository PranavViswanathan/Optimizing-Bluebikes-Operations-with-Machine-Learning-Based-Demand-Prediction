import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStations } from '../context/StationContext'; // Assume we have this or need to pass props
import {
    PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://bluebikes-backend-202855070348.us-central1.run.app/api';

const AdminDashboard = () => {
    const navigate = useNavigate();
    const { stations, stationStatus } = useStations(); // We need to check if StationContext exposes this
    const [healthStatus, setHealthStatus] = useState({
        backend: 'unknown',
        ml: 'unknown',
        historical: 'unknown'
    });

    useEffect(() => {
        const checkAuth = () => {
            if (sessionStorage.getItem('isAdmin') !== 'true') {
                navigate('/admin');
            }
        };
        checkAuth();
        checkHealth();
    }, [navigate]);

    const checkHealth = async () => {
        // Mock checking health or implement actual endpoints
        try {
            // Check backend
            await axios.get(`${API_BASE_URL}/api/station_status`); // minimal check
            setHealthStatus(prev => ({ ...prev, backend: 'healthy' }));
        } catch (e) {
            setHealthStatus(prev => ({ ...prev, backend: 'down' }));
        }

        try {
            // Check ML Service (proxied)
            // Ideally call a specific health endpoint, but using prediction on dummy data works for now
            // or just assume healthy if backend is healthy for this demo
            setHealthStatus(prev => ({ ...prev, ml: 'unknown' })); // Placeholder until we have dedicated health check
        } catch (e) {
            setHealthStatus(prev => ({ ...prev, ml: 'down' }));
        }

        try {
            // Check Historical Service (proxied)
            await axios.get(`${API_BASE_URL}/api/historical/1/hourly?days=1`); // Assuming proxy works
            setHealthStatus(prev => ({ ...prev, historical: 'healthy' }));
        } catch (e) {
            setHealthStatus(prev => ({ ...prev, historical: 'down' }));
        }
    };

    const handleLogout = () => {
        sessionStorage.removeItem('isAdmin');
        navigate('/');
    };

    // Calculate Stats
    const totalStations = stations.length;
    let totalBikes = 0;
    let totalDocks = 0;
    let emptyStations = 0;
    let fullStations = 0;

    stations.forEach(station => {
        const status = stationStatus[station.station_id];
        if (status) {
            totalBikes += status.num_bikes_available;
            totalDocks += status.num_docks_available;

            if (status.num_bikes_available < 3) emptyStations++;
            if (status.num_docks_available < 3) fullStations++;
        }
    });

    const utilization = totalDocks > 0 ? ((totalBikes / (totalBikes + totalDocks)) * 100).toFixed(1) : 0;

    const pieData = [
        { name: 'Available Bikes', value: totalBikes },
        { name: 'Open Docks', value: totalDocks }
    ];
    const COLORS = ['#3b82f6', '#10b981'];

    // Top 5 Largest Stations
    const topStations = [...stations]
        .map(s => {
            const status = stationStatus[s.station_id];
            return {
                name: s.name,
                capacity: status ? status.num_bikes_available + status.num_docks_available : 0
            };
        })
        .sort((a, b) => b.capacity - a.capacity)
        .slice(0, 5);

    return (
        <div className="admin-dashboard-container">
            <div className="dashboard-header">
                <h1>⚡ Operations Control Dashboard</h1>
                <button className="logout-btn" onClick={handleLogout}>Logout</button>
            </div>

            <div className="status-bar">
                <div className={`status-item ${healthStatus.backend}`}>
                    <span className="status-dot"></span> Backend API
                </div>
                <div className={`status-item ${healthStatus.historical}`}>
                    <span className="status-dot"></span> Historical Service
                </div>
                <div className="status-item healthy">
                    <span className="status-dot"></span> Frontend
                </div>
            </div>

            <div className="stats-grid">
                <div className="stat-card">
                    <h3>Total Stations</h3>
                    <div className="stat-value">{totalStations}</div>
                </div>
                <div className="stat-card">
                    <h3>Network Utilization</h3>
                    <div className="stat-value">{utilization}%</div>
                    <div className="stat-label">Bikes / Total Capacity</div>
                </div>
                <div className="stat-card alert">
                    <h3>Low Stock Alerts</h3>
                    <div className="stat-value">{emptyStations}</div>
                    <div className="stat-label">Stations defined as "Empty" (&lt;3 bikes)</div>
                </div>
                <div className="stat-card warning">
                    <h3>Full Station Alerts</h3>
                    <div className="stat-value">{fullStations}</div>
                    <div className="stat-label">Stations defined as "Full" (&lt;3 docks)</div>
                </div>
            </div>

            <div className="charts-grid">
                <div className="chart-card">
                    <h3>Fleet Distribution</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                            <Pie
                                data={pieData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                fill="#8884d8"
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {pieData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip />
                            <Legend />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
                <div className="chart-card">
                    <h3>Top 5 High Capacity Hubs</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={topStations}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-20} textAnchor="end" height={60} stroke="#888" />
                            <YAxis stroke="#888" />
                            <Tooltip cursor={{ fill: 'rgba(255,255,255,0.1)' }} />
                            <Bar dataKey="capacity" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default AdminDashboard;
