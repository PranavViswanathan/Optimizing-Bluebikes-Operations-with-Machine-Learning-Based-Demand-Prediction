import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { StationProvider } from './context/StationContext';
import MapView from './components/MapView';
import StationList from './components/StationList';
import StationDetail from './components/StationDetail';

function App() {
    return (
        <StationProvider>
            <Router>
                <div className="app">
                    <header className="app-header">
                        <div className="header-content">
                            <h1 className="app-title">
                                <span className="bike-icon">🚴</span>
                                Bluebikes Station Map
                            </h1>
                            <nav className="nav-links">
                                <Link to="/" className="nav-link">Map View</Link>
                                <Link to="/stations" className="nav-link">List View</Link>
                            </nav>
                        </div>
                    </header>

                    <main className="app-main">
                        <Routes>
                            <Route path="/" element={<MapView />} />
                            <Route path="/stations" element={<StationList />} />
                            <Route path="/stations/:id" element={<StationDetail />} />
                        </Routes>
                    </main>

                    <footer className="app-footer">
                        <p>
                            Real-time data from Bluebikes GBFS API |
                            ML Predictions powered by XGBoost |
                            Built for MLOps Final Project
                        </p>
                    </footer>
                </div>
            </Router >
        </StationProvider >
    );
}

export default App;
