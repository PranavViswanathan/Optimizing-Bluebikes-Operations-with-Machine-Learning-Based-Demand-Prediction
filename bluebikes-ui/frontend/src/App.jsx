import React from 'react';
import { HashRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import { StationProvider } from './context/StationContext';
import MapView from './components/MapView';
import StationList from './components/StationList';
import StationDetail from './components/StationDetail';
import RebalancingView from './components/RebalancingView';
import RebalanceLogo from './components/RebalanceLogo';
import AdminLogin from './components/AdminLogin';
import AdminDashboard from './components/AdminDashboard';
import MonitoringDashboard from './components/MonitoringDashboard';
import './styles/App.css';

const ProtectedRoute = ({ children }) => {
    const isAuthenticated = sessionStorage.getItem('isAdmin') === 'true';
    return isAuthenticated ? children : <Navigate to="/admin" replace />;
};

function App() {
    return (
        <StationProvider>
            <Router>
                <div className="app">

                    <header className="app-header">
                        <div className="header-content">

                            {/* LEFT SIDE — TITLE */}
                            <div className="left-title">
                                <h1 className="app-title">
                                    <span className="bike-icon">🚴</span>
                                    Bluebikes Station Map
                                </h1>
                            </div>

                            {/* CENTER — LOGO */}
                            <div className="center-logo">
                                <RebalanceLogo size={95} />
                            </div>

                            {/* RIGHT SIDE — NAVIGATION */}
                            <nav className="nav-links right-nav">
                                <Link to="/" className="nav-link">Map View</Link>
                                <Link to="/stations" className="nav-link">List View</Link>
                                <Link to="/rebalancing" className="nav-link">Rebalancing</Link>
                                <Link to="/monitoring" className="nav-link">Monitoring</Link>
                            </nav>

                        </div>
                    </header>

                    <main className="app-main">
                        <Routes>
                            <Route path="/" element={<MapView />} />
                            <Route path="/stations" element={<StationList />} />
                            <Route path="/stations/:stationId" element={<StationDetail />} />
                            <Route path="/rebalancing" element={<RebalancingView />} />
                            <Route path="/monitoring" element={<MonitoringDashboard />} />
                            <Route path="/admin" element={<AdminLogin />} />
                            <Route path="/admin/dashboard" element={
                                <ProtectedRoute>
                                    <AdminDashboard />
                                </ProtectedRoute>
                            } />
                        </Routes>
                    </main>

                    <footer className="app-footer">
                        <p>
                            Real-time data from Bluebikes GBFS API |
                            ML Predictions powered by XGBoost |
                            Built for MLOps Final Project
                        </p>
                        <p style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
                            <Link to="/admin" style={{ color: 'rgba(255,255,255,0.3)', textDecoration: 'none' }}>Admin Access</Link>
                        </p>
                    </footer>

                </div>
            </Router>
        </StationProvider>
    );
}

export default App;