import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/App.css';
import './styles/FilterButtons.css';
import './styles/NearestStations.css';
import './styles/Rebalancing.css';
import App from './App';

import ErrorBoundary from './components/ErrorBoundary';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
