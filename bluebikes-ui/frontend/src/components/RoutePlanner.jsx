import React, { useState, useEffect } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import { useStations } from '../context/StationContext';

const RoutePlanner = ({ userLocation, stations }) => {
    const map = useMap();
    const [routingControl, setRoutingControl] = useState(null);
    const [showPlanner, setShowPlanner] = useState(false);
    const [nearestStations, setNearestStations] = useState([]);
    const [selectedStation, setSelectedStation] = useState(null);
    const [isRoutingLibraryLoaded, setIsRoutingLibraryLoaded] = useState(false);
    const { stationStatus } = useStations();

    // Dynamically load Leaflet Routing Machine
    useEffect(() => {
        // Load CSS
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.css';
        document.head.appendChild(link);

        // Load JS
        const script = document.createElement('script');
        script.src = 'https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.js';
        script.async = true;
        script.onload = () => {
            console.log('Leaflet Routing Machine loaded');
            setIsRoutingLibraryLoaded(true);
        };
        document.body.appendChild(script);

        return () => {
            document.head.removeChild(link);
            document.body.removeChild(script);
        };
    }, []);

    // Clean up routing control on unmount or when hidden
    useEffect(() => {
        return () => {
            if (routingControl) {
                map.removeControl(routingControl);
            }
        };
    }, [routingControl, map]);

    const calculateDistance = (lat1, lon1, lat2, lon2) => {
        const R = 3958.8; // Radius in miles
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a =
            Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    };

    const findNearestDropoff = () => {
        if (!userLocation || !stations.length) return;

        // Filter stations with available docks (at least 1)
        const availableStations = stations.filter(station => {
            const status = stationStatus[station.station_id];
            return status && status.num_docks_available > 0;
        });

        // Calculate distances
        const withDistances = availableStations.map(station => ({
            ...station,
            distance: calculateDistance(
                userLocation.lat,
                userLocation.lng,
                station.lat,
                station.lon
            )
        }));

        // Sort by distance and take top 5
        const sorted = withDistances
            .sort((a, b) => a.distance - b.distance)
            .slice(0, 5);

        setNearestStations(sorted);
        setShowPlanner(true);
    };

    const handleRouteToStation = (station) => {
        if (routingControl) {
            map.removeControl(routingControl);
            setRoutingControl(null);
        }

        setSelectedStation(station);

        // Function to create routing control
        const createRouting = () => {
            // Check for Routing in window.L or global L
            // The Leaflet Routing Machine attaches to the L instance it was loaded with.
            // Since we load it via CDN, it attaches to window.L
            const Routing = (window.L && window.L.Routing) || (L && L.Routing);

            if (Routing) {
                console.log('Found Leaflet Routing Machine');
                try {
                    const control = Routing.control({
                        waypoints: [
                            L.latLng(userLocation.lat, userLocation.lng),
                            L.latLng(station.lat, station.lon)
                        ],
                        routeWhileDragging: false,
                        lineOptions: {
                            styles: [{ color: '#3b82f6', weight: 6, opacity: 0.8 }]
                        },
                        show: true,
                        addWaypoints: false,
                        draggableWaypoints: false,
                        fitSelectedRoutes: true,
                        showAlternatives: false,
                        containerClassName: 'routing-container'
                    }).addTo(map);

                    setRoutingControl(control);
                } catch (err) {
                    console.error('Error creating routing control:', err);
                    alert('Error creating route. See console for details.');
                }
            } else {
                console.error('Leaflet Routing Machine not loaded. window.L:', window.L, 'L.Routing:', L.Routing);
                alert('Routing component not ready yet. Please try again in current route planner session.');
            }
        };

        if (isRoutingLibraryLoaded) {
            createRouting();
        } else {
            alert('Routing library loading... please wait a moment and try clicking Route again.');
        }
    };

    const closePlanner = () => {
        setShowPlanner(false);
        if (routingControl) {
            map.removeControl(routingControl);
            setRoutingControl(null);
        }
        setSelectedStation(null);

        // Reset map view
        if (userLocation) {
            map.setView([userLocation.lat, userLocation.lng], 14);
        }
    };

    if (!userLocation) {
        return null;
    }

    return (
        <>
            {!showPlanner && (
                <button
                    className="route-planner-btn"
                    onClick={findNearestDropoff}
                    title="Find nearest drop-off station"
                >
                    <span className="btn-icon">🅿️</span>
                    <span className="btn-text">Find Drop-off</span>
                </button>
            )}

            {showPlanner && (
                <div className="route-planner-panel">
                    <div className="panel-header">
                        <h3>Nearest Drop-off (Open Docks)</h3>
                        <button className="close-btn" onClick={closePlanner}>×</button>
                    </div>

                    <div className="station-list">
                        {nearestStations.map(station => {
                            const status = stationStatus[station.station_id];
                            const docks = status ? status.num_docks_available : 0;
                            const isSelected = selectedStation?.station_id === station.station_id;

                            return (
                                <div
                                    key={station.station_id}
                                    className={`planner-station-item ${isSelected ? 'selected' : ''}`}
                                    onClick={() => handleRouteToStation(station)}
                                >
                                    <div className="station-info">
                                        <h4>{station.name}</h4>
                                        <div className="station-meta">
                                            <span className="distance">{station.distance.toFixed(2)} mi</span>
                                            <span className="docks">{docks} docks</span>
                                        </div>
                                    </div>
                                    <button className="navigate-btn">
                                        {isSelected ? 'Routing...' : 'Route'}
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </>
    );
};

export default RoutePlanner;
