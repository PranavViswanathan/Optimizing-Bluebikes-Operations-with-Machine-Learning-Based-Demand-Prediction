import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const AdminLogin = () => {
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleLogin = (e) => {
        e.preventDefault();
        if (password === 'admin') {
            sessionStorage.setItem('isAdmin', 'true');
            navigate('/admin/dashboard');
        } else {
            setError('Invalid password');
        }
    };

    return (
        <div className="admin-login-container">
            <div className="admin-login-card glass-card">
                <h2> Admin Access</h2>
                <form onSubmit={handleLogin}>
                    <div className="form-group">
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="Enter Password"
                            className="glass-input"
                            autoFocus
                        />
                    </div>
                    {error && <p className="error-message">{error}</p>}
                    <button type="submit" className="glass-btn primary">
                        Login
                    </button>
                </form>
            </div>
        </div>
    );
};

export default AdminLogin;
