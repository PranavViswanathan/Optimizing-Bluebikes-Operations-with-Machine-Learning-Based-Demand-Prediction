import React, { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://bluebikes-backend-202855070348.us-central1.run.app/api';

const HistoricalChart = ({ stationId }) => {
    const [timeRange, setTimeRange] = useState('daily'); // 'hourly', 'daily', 'weekly'
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchHistoricalData = async () => {
            setLoading(true);
            setError(null);

            try {
                const response = await axios.get(
                    `${API_BASE_URL}/api/historical/${stationId}/${timeRange}`
                );

                // Transform data for Recharts
                const transformedData = response.data.data.map(item => ({
                    ...item,
                    time: formatTime(item.time, timeRange)
                }));

                setData(transformedData);
            } catch (err) {
                console.error('Error fetching historical data:', err);
                setError(err.response?.data?.message || 'Failed to load historical data');
            } finally {
                setLoading(false);
            }
        };

        if (stationId) {
            fetchHistoricalData();
        }
    }, [stationId, timeRange]);

    const formatTime = (timestamp, range) => {
        const date = new Date(timestamp);

        switch (range) {
            case 'hourly':
                return date.toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    hour: 'numeric',
                    hour12: true
                });
            case 'daily':
                return date.toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric'
                });
            case 'weekly':
                return `Week of ${date.toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric'
                })}`;
            default:
                return timestamp;
        }
    };

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="historical-chart-tooltip">
                    <p className="tooltip-label">{label}</p>
                    <p className="tooltip-pickups">
                        <span className="tooltip-dot pickups-dot"></span>
                        Pickups: {payload[0].value}
                    </p>
                    <p className="tooltip-dropoffs">
                        <span className="tooltip-dot dropoffs-dot"></span>
                        Drop-offs: {payload[1].value}
                    </p>
                    <p className="tooltip-total">
                        Total: {payload[0].value + payload[1].value}
                    </p>
                </div>
            );
        }
        return null;
    };

    const renderContent = () => {
        if (loading) {
            return (
                <div className="historical-chart-loading" style={{ height: '300px' }}>
                    <div className="mini-spinner"></div>
                    <p>Loading historical data...</p>
                </div>
            );
        }

        if (error) {
            return (
                <div className="historical-chart-error" style={{ height: '300px' }}>
                    <p> {error}</p>
                    <p className="error-hint">Make sure the historical data service is running</p>
                </div>
            );
        }

        if (data.length === 0) {
            return (
                <div className="historical-chart-empty" style={{ height: '300px' }}>
                    <p> No historical data available for this station</p>
                </div>
            );
        }

        const totalRides = data.reduce((sum, item) => sum + item.pickups + item.dropoffs, 0);

        if (totalRides === 0) {
            return (
                <div className="historical-chart-empty" style={{ height: '300px' }}>
                    <p> Data not collected for this timeframe</p>
                </div>
            );
        }

        return (
            <ResponsiveContainer width="100%" height={300}>
                <LineChart
                    data={data}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                    <XAxis
                        dataKey="time"
                        stroke="#888"
                        tick={{ fill: '#888', fontSize: 12 }}
                        angle={-45}
                        textAnchor="end"
                        height={80}
                    />
                    <YAxis
                        stroke="#888"
                        tick={{ fill: '#888', fontSize: 12 }}
                        label={{ value: 'Rides', angle: -90, position: 'insideLeft', fill: '#888' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ paddingTop: '10px' }} iconType="line" />
                    <Line
                        type="monotone"
                        dataKey="pickups"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={{ fill: '#3b82f6', r: 3 }}
                        activeDot={{ r: 5 }}
                        name="Pickups"
                    />
                    <Line
                        type="monotone"
                        dataKey="dropoffs"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={{ fill: '#10b981', r: 3 }}
                        activeDot={{ r: 5 }}
                        name="Drop-offs"
                    />
                </LineChart>
            </ResponsiveContainer>
        );
    };

    return (
        <div className="historical-chart-container">
            <div className="chart-header">
                <h4> Historical Trends</h4>
                <div className="time-range-tabs">
                    <button
                        className={`time-tab ${timeRange === 'hourly' ? 'active' : ''}`}
                        onClick={() => setTimeRange('hourly')}
                    >
                        7 Days
                    </button>
                    <button
                        className={`time-tab ${timeRange === 'daily' ? 'active' : ''}`}
                        onClick={() => setTimeRange('daily')}
                    >
                        30 Days
                    </button>
                    <button
                        className={`time-tab ${timeRange === 'weekly' ? 'active' : ''}`}
                        onClick={() => setTimeRange('weekly')}
                    >
                        12 Weeks
                    </button>
                </div>
            </div>

            {renderContent()}

            <div className="chart-footer">
                <p className="chart-info">
                    {timeRange === 'hourly' && ' Hourly data for the past 7 days'}
                    {timeRange === 'daily' && ' Daily data for the past 30 days'}
                    {timeRange === 'weekly' && ' Weekly data for the past 12 weeks'}
                </p>
            </div>
        </div>
    );
};

export default HistoricalChart;
