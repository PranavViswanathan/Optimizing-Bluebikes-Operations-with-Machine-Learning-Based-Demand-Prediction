"""
Random Forest Training Module for BlueBikes Demand Prediction
Adapted to work with MLflow tracking pipeline.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import logging
from typing import Optional, Dict, List
from datetime import datetime
import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')
# plt.style.use('seaborn-v0_8-darkgrid')
# sns.set_palette("husl")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, mlflow_client=None, config: Optional[Dict] = None):
    """
    Train Random Forest model with MLflow tracking
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        mlflow_client: MLflow instance for logging
    
    Returns:
        model: Trained Random Forest model
        metrics: Dictionary of performance metrics
    """
    
    # Start nested MLflow run for Random Forest
    with mlflow.start_run(nested=True, run_name=f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        # Optimized parameters for large datasets
        rf_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42,
            'verbose': 1,
            'oob_score': True,
            'warm_start': False
        }
        
        if config is not None:
            params = {**rf_params, **config}
        else:
            params = rf_params
        # Log parameters
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("optimizer", "manual_tuning")
        
        # Train model
        logger.info("Training Random Forest model...")
        rf_model = RandomForestRegressor(**params)
        
        rf_model.fit(X_train, y_train)
        
        logger.info(f"Training completed. OOB Score: {rf_model.oob_score_:.4f}")
        mlflow.log_metric("oob_score", rf_model.oob_score_)
        
        # Make predictions
        y_pred_train = rf_model.predict(X_train)
        y_pred_val = rf_model.predict(X_val)
        y_pred_test = rf_model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mape = np.mean(np.abs((y_train - y_pred_train) / (y_train + 1e-10))) * 100
        
        val_r2 = r2_score(y_val, y_pred_val)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        val_mape = np.mean(np.abs((y_val - y_pred_val) / (y_val + 1e-10))) * 100

        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-10))) * 100
        
        metrics = {
            'train_r2': train_r2,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_mape': train_mape,
            'val_r2': val_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_mape': val_mape,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        logger.info(f"Model Performance:")
        logger.info(f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
        logger.info(f"  Val   R²: {val_r2:.4f}, MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
        logger.info(f"  Test R²: {test_r2:.4f}, MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")
        
        # Feature importance
        feature_columns = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance_df.head(20)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 20 Feature Importances - Random Forest')
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance_random_forest.png")
        plt.close()
        
        # Plot predictions scatter
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
        
        # Training set
        ax1.scatter(y_train, y_pred_train, alpha=0.5, s=1)
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Rides')
        ax1.set_ylabel('Predicted Rides')
        ax1.set_title(f'Training Set (R² = {train_r2:.4f})')
        ax1.grid(True, alpha=0.3)
        
        # Validation set
        ax2.scatter(y_val, y_pred_val, alpha=0.5, s=1)
        ax2.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Rides')
        ax2.set_ylabel('Predicted Rides')
        ax2.set_title(f'Validation Set (R² = {val_r2:.4f})')
        ax2.grid(True, alpha=0.3)

        # Test set
        ax3.scatter(y_test, y_pred_test, alpha=0.5, s=1)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax3.set_xlabel('Actual Rides')
        ax3.set_ylabel('Predicted Rides')
        ax3.set_title(f'Test Set (R² = {test_r2:.4f})')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        mlflow.log_figure(fig, "predictions_scatter_random_forest.png")
        plt.close()
        
        # Plot residuals
        residuals_test = y_test - y_pred_test
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.scatter(y_pred_test, residuals_test, alpha=0.5, s=1)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Rides')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Test Residuals')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Residual Distribution (Mean: {residuals_test.mean():.2f}, Std: {residuals_test.std():.2f})')
        
        plt.tight_layout()
        mlflow.log_figure(fig, "residuals_random_forest.png")
        plt.close()
        
        # Save model
        mlflow.sklearn.log_model(
            rf_model,
            "model",
            registered_model_name=None  # Will be registered by main script if selected as best
        )
        
        joblib.dump(rf_model, 'random_forest_bikeshare_model.pkl')
        mlflow.log_artifact('random_forest_bikeshare_model.pkl')
        
        metadata = {
            'model': 'RandomForest',
            'features': list(feature_columns),
            'performance': metrics,
            'oob_score': rf_model.oob_score_,
            'hyperparameters': params,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(metadata, 'random_forest_model_metadata.pkl')
        mlflow.log_artifact('random_forest_model_metadata.pkl')
        
        return rf_model, metrics
    
def tune_random_forest(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    mlflow=None,
    param_grid=None,
    max_combinations: int = 6,
):
    """
    Lightweight hyperparameter tuning for Random Forest using validation MAE.
    Calls train_random_forest for each combination (nested MLflow runs).
    Selects best config based on val_mae.
    """

    import random
    from itertools import product

    # Small, safe grid by default
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [15, 25],
            "min_samples_leaf": [5, 10],
            # min_samples_split is often tied to leaf size; you can add if needed:
            # "min_samples_split": [10, 20],
            "max_features": ["sqrt"],  # keep simple for now
        }

    # All parameter combinations
    all_combos = [
        dict(zip(param_grid.keys(), values))
        for values in product(*param_grid.values())
    ]

    # Randomly down-sample if too many combinations
    if len(all_combos) > max_combinations:
        all_combos = random.sample(all_combos, max_combinations)

    print(f"Testing {len(all_combos)} Random Forest parameter combinations...")

    best_val_mae = float("inf")
    best_model = None
    best_params = None
    best_metrics = None

    for cfg in all_combos:
        print(f"\nRandom Forest tuning config: {cfg}")

        model, metrics = train_random_forest(
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            mlflow_client=mlflow,
            config=cfg,
        )

        val_mae = metrics["val_mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
            best_params = cfg
            best_metrics = metrics

    print("\nBest Random Forest parameters found:")
    print(f"  Params: {best_params}")
    print(f"  Val MAE: {best_val_mae:.2f}")

    return best_model, best_metrics
