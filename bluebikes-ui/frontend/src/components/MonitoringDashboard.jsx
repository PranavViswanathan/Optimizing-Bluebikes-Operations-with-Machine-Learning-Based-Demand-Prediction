/**
 * Monitoring Dashboard Component for BlueBikes UI
 * 
 * Add to your React frontend:
 * 1. Save as src/components/MonitoringDashboard.jsx
 * 2. Add route in App.jsx: <Route path="/monitoring" element={<MonitoringDashboard />} />
 * 3. Add navigation link
 */

import React, { useState, useEffect, useCallback } from 'react';
import '../styles/MonitoringDashboard.css';

// Cloud Run API base URL
// const API_BASE = process.env.REACT_APP_API_URL || 'https://bluebikes-prediction-202855070348.us-central1.run.app';
const API_BASE = process.env.REACT_APP_MONITORING_URL || 'https://bluebikes-prediction-202855070348.us-central1.run.app';

// Status badge component
const StatusBadge = ({ status }) => {
  const statusConfig = {
    HEALTHY: { className: 'status-healthy', icon: '✓', label: 'Healthy' },
    WARNING: { className: 'status-warning', icon: '⚠', label: 'Warning' },
    CRITICAL: { className: 'status-critical', icon: '✕', label: 'Critical' },
    NO_DATA: { className: 'status-nodata', icon: '?', label: 'No Data' },
    ERROR: { className: 'status-error', icon: '!', label: 'Error' }
  };
  
  const config = statusConfig[status] || statusConfig.NO_DATA;
  
  return (
    <span className={`status-badge ${config.className}`}>
      <span className="status-icon">{config.icon}</span>
      {config.label}
    </span>
  );
};

// Metric card component
const MetricCard = ({ title, value, subtitle, highlight }) => (
  <div className={`metric-card ${highlight ? 'highlight' : ''}`}>
    <div className="metric-title">{title}</div>
    <div className="metric-value">{value}</div>
    {subtitle && <div className="metric-subtitle">{subtitle}</div>}
  </div>
);

// Progress bar for drift visualization
const DriftBar = ({ value, threshold = 0.3 }) => {
  const percentage = Math.min(value * 100, 100);
  const isHigh = value >= threshold;
  
  return (
    <div className="drift-bar-container">
      <div 
        className={`drift-bar-fill ${isHigh ? 'high' : 'normal'}`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
};

// Main Dashboard Component
const MonitoringDashboard = () => {
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [features, setFeatures] = useState([]);
  const [baseline, setBaseline] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch monitoring status
  const fetchStatus = useCallback(async () => {
  try {
    const timestamp = Date.now();
    const response = await fetch(
      `${API_BASE}/monitoring/api/status?_t=${timestamp}`,
      {
        cache: 'no-store',  // Tell browser not to cache
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      }
    );
    if (!response.ok) throw new Error('Failed to fetch status');
    const data = await response.json();
    setStatus(data);
    return data;
  } catch (err) {
    console.error('Error fetching status:', err);
    setError(err.message);
    return null;
  }
}, []);

  // Fetch history
  const fetchHistory = useCallback(async () => {
  try {
    const timestamp = Date.now();
    const response = await fetch(
      `${API_BASE}/monitoring/api/history?limit=14&_t=${timestamp}`,
      { cache: 'no-store' }
    );
    if (!response.ok) throw new Error('Failed to fetch history');
    const data = await response.json();
    setHistory(data);
  } catch (err) {
    console.error('Error fetching history:', err);
  }
}, []);

  // Fetch feature drift details
  const fetchFeatures = useCallback(async (date) => {
  try {
    const timestamp = Date.now();
    const response = await fetch(
      `${API_BASE}/monitoring/api/features/${date}?_t=${timestamp}`,
      { cache: 'no-store' }
    );
    if (!response.ok) throw new Error('Failed to fetch features');
    const data = await response.json();
    setFeatures(data);
  } catch (err) {
    console.error('Error fetching features:', err);
  }
}, []);

  // Fetch baseline info
  const fetchBaseline = useCallback(async () => {
    try {
      const timestamp = Date.now();
      const response = await fetch(
        `${API_BASE}/monitoring/api/baseline?_t=${timestamp}`,
        { cache: 'no-store' }
      );
      if (!response.ok) throw new Error('Failed to fetch baseline');
      const data = await response.json();
      setBaseline(data);
    } catch (err) {
      console.error('Error fetching baseline:', err);
    }
  }, []);
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      const statusData = await fetchStatus();
      await fetchHistory();
      await fetchBaseline();
      
      if (statusData?.reportDate) {
        await fetchFeatures(statusData.reportDate);
      }
      
      setLoading(false);
    };
    
    loadData();
  }, [fetchStatus, fetchHistory, fetchBaseline, fetchFeatures]);

  // Refresh handler
  const handleRefresh = async () => {
    setLoading(true);
    const statusData = await fetchStatus();
    await fetchHistory();
    if (statusData?.reportDate) {
      await fetchFeatures(statusData.reportDate);
    }
    setLoading(false);
  };

  // Format date
  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    if (dateStr.length === 8) {
      return `${dateStr.slice(0,4)}-${dateStr.slice(4,6)}-${dateStr.slice(6,8)}`;
    }
    try {
      return new Date(dateStr).toLocaleString();
    } catch {
      return dateStr;
    }
  };

  if (loading && !status) {
    return (
      <div className="monitoring-dashboard">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading monitoring data...</p>
        </div>
      </div>
    );
  }

  if (error && !status) {
    return (
      <div className="monitoring-dashboard">
        <div className="error-container">
          <h2>Error Loading Dashboard</h2>
          <p>{error}</p>
          <button onClick={handleRefresh}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="monitoring-dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <h1>Model Monitoring</h1>
          <p className="subtitle">BlueBikes Demand Prediction - Drift Detection</p>
        </div>
        <div className="header-right">
          <span className="last-updated">
            Updated: {formatDate(status?.lastUpdated)}
          </span>
          <button 
            className="refresh-btn" 
            onClick={handleRefresh}
            disabled={loading}
          >
            {loading ? '↻' : '⟳'} Refresh
          </button>
        </div>
      </header>

      {/* Status Banner */}
      <div className={`status-banner status-${status?.status?.toLowerCase()}`}>
        <div className="status-content">
          <StatusBadge status={status?.status} />
          <span className="status-message">
            {status?.status === 'HEALTHY' && 'All systems operating normally'}
            {status?.status === 'WARNING' && 'Minor drift detected - monitoring closely'}
            {status?.status === 'CRITICAL' && 'Significant drift detected - action required'}
            {status?.status === 'NO_DATA' && 'No monitoring data available'}
          </span>
        </div>
        <div className="status-action">
          Action: <strong>{status?.recommendedAction || 'none'}</strong>
        </div>
      </div>

      {/* Demo Mode Notice */}
      {status?.context?.demo_mode && (
        <div className="demo-notice">
           Demo Mode: Comparing test data against training baseline
          {status?.context?.drift_injected && ' (Artificial drift injected)'}
        </div>
      )}

      {/* Tabs */}
      <div className="tabs">
        {['overview', 'features', 'history', 'baseline'].map(tab => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'overview' && (
          <div className="overview-tab">
            {/* Metrics Grid */}
            <div className="metrics-grid">
              <MetricCard
                title="Data Drift Share"
                value={`${((status?.dataDrift?.driftShare || 0) * 100).toFixed(1)}%`}
                subtitle={`${status?.dataDrift?.driftedFeatures || 0}/${status?.dataDrift?.totalFeatures || 0} features`}
                highlight={status?.dataDrift?.detected}
              />
              <MetricCard
                title="Prediction Shift"
                value={`${(status?.predictionDrift?.meanShiftPct || 0).toFixed(1)}%`}
                subtitle={`Severity: ${status?.predictionDrift?.severity || 'none'}`}
                highlight={status?.predictionDrift?.detected}
              />
              <MetricCard
                title="Model R²"
                value={status?.performance?.currentR2?.toFixed(3) || 'N/A'}
                subtitle="Current performance"
              />
              <MetricCard
                title="Model MAE"
                value={status?.performance?.currentMAE?.toFixed(1) || 'N/A'}
                subtitle="Mean Absolute Error"
              />
            </div>

            {/* Alerts */}
            {status?.alerts?.length > 0 && (
              <div className="alerts-section">
                <h3>Active Alerts</h3>
                <ul className="alerts-list">
                  {status.alerts.map((alert, i) => (
                    <li key={i} className="alert-item">
                      <span className="alert-icon">⚠</span>
                      {alert}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Top Drifted Features */}
            <div className="drifted-features-section">
              <h3>Top Drifted Features</h3>
              {features.filter(f => f.driftDetected).length > 0 ? (
                <div className="feature-list">
                  {features.filter(f => f.driftDetected).slice(0, 5).map(feature => (
                    <div key={feature.name} className="feature-row">
                      <span className="feature-name">{feature.name}</span>
                      <DriftBar value={feature.driftScore} />
                      <span className="feature-score">
                        {(feature.driftScore * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="no-drift">No significant feature drift detected</p>
              )}
            </div>

            {/* View Full Report Button */}
            <div className="report-link">
              <a 
                href={`${API_BASE}/monitoring/report/${status?.reportDate}`}
                target="_blank"
                rel="noopener noreferrer"
                className="btn-primary"
              >
                 View Full Evidently Report
              </a>
            </div>
          </div>
        )}

        {activeTab === 'features' && (
          <div className="features-tab">
            <h3>Feature Drift Analysis</h3>
            <table className="features-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Status</th>
                  <th>Drift Score</th>
                  <th>Test</th>
                </tr>
              </thead>
              <tbody>
                {features.map(f => (
                  <tr key={f.name} className={f.driftDetected ? 'drifted' : ''}>
                    <td className="feature-name">{f.name}</td>
                    <td>
                      <span className={`drift-status ${f.driftDetected ? 'detected' : 'stable'}`}>
                        {f.driftDetected ? '✕ Drifted' : '✓ Stable'}
                      </span>
                    </td>
                    <td>
                      <div className="score-cell">
                        <DriftBar value={f.driftScore} />
                        <span>{(f.driftScore * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="stat-test">{f.statTest}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="history-tab">
            <h3>Monitoring History</h3>
            <table className="history-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Status</th>
                  <th>Drift Share</th>
                  <th>Features</th>
                  <th>Alerts</th>
                  <th>Report</th>
                </tr>
              </thead>
              <tbody>
                {history.map(h => (
                  <tr key={h.date}>
                    <td>{formatDate(h.date)}</td>
                    <td><StatusBadge status={h.status} /></td>
                    <td>
                      <div className="score-cell">
                        <DriftBar value={h.driftShare} />
                        <span>{((h.driftShare || 0) * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>{h.driftedFeatures}</td>
                    <td>{h.alertCount}</td>
                    <td>
                      <a 
                        href={`${API_BASE}/monitoring/report/${h.date}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        View →
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'baseline' && (
          <div className="baseline-tab">
            <h3>Baseline Information</h3>
            {baseline?.status === 'NO_BASELINE' ? (
              <p>No baseline found. Run model training to generate baseline.</p>
            ) : (
              <div className="baseline-info">
                <div className="info-grid">
                  <div className="info-item">
                    <label>Version</label>
                    <value>{baseline?.version || 'N/A'}</value>
                  </div>
                  <div className="info-item">
                    <label>Model</label>
                    <value>{baseline?.modelName || 'N/A'}</value>
                  </div>
                  <div className="info-item">
                    <label>Created</label>
                    <value>{formatDate(baseline?.createdAt)}</value>
                  </div>
                  <div className="info-item">
                    <label>Source</label>
                    <value>{baseline?.baselineSource || 'N/A'}</value>
                  </div>
                  <div className="info-item">
                    <label>Reference Samples</label>
                    <value>{baseline?.referenceSamples?.toLocaleString() || 'N/A'}</value>
                  </div>
                  <div className="info-item">
                    <label>Features</label>
                    <value>{baseline?.features || 'N/A'}</value>
                  </div>
                </div>
                
                {baseline?.dataSplits && (
                  <div className="data-splits">
                    <h4>Data Splits</h4>
                    <p>Training: {baseline.dataSplits.train_start} to {baseline.dataSplits.train_end}</p>
                    <p>Test: {baseline.dataSplits.test_start} to {baseline.dataSplits.test_end}</p>
                  </div>
                )}
                
                {baseline?.performanceBaseline && (
                  <div className="performance-baseline">
                    <h4>Performance Baseline</h4>
                    <div className="metrics-row">
                      <span>R²: {baseline.performanceBaseline.test_r2?.toFixed(4) || 'N/A'}</span>
                      <span>MAE: {baseline.performanceBaseline.test_mae?.toFixed(2) || 'N/A'}</span>
                      <span>RMSE: {baseline.performanceBaseline.test_rmse?.toFixed(2) || 'N/A'}</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MonitoringDashboard;
