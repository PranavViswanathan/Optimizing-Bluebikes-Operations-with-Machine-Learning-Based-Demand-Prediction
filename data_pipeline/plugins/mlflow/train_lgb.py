"""
LightGBM Training Module for BlueBikes Demand Prediction
Adapted from original W&B version to work with MLflow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import logging
from datetime import datetime
import mlflow
import mlflow.lightgbm

warnings.filterwarnings('ignore')
#plt.style.use('seaborn-v0_8-darkgrid')
#sns.set_palette("husl")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_lightgbm(X_train, y_train, X_test, y_test, mlflow_client=None, use_cv=False):
    """
    Train LightGBM model with MLflow tracking
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        mlflow_client: MLflow instance for logging
        use_cv: Whether to use cross-validation
    
    Returns:
        model: Trained LightGBM model
        metrics: Dictionary of performance metrics
    """
    
    # Start nested MLflow run for LightGBM
    with mlflow.start_run(nested=True, run_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        # Best parameters from your tuning
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 255,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'max_depth': -1,
            'min_gain_to_split': 0.0,
            'n_estimators': 1000,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }
        
        # Log parameters
        mlflow.log_params(lgb_params)
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("optimizer", "manual_tuning")
        
        # Optional: Cross-validation
        if use_cv:
            logger.info("Performing 5-fold cross-validation...")
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                X_fold_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                
                train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
                valid_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
                
                lgb_model = lgb.train(
                    lgb_params,
                    train_data,
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(0)
                    ]
                )
                
                y_pred = lgb_model.predict(X_fold_val, num_iteration=lgb_model.best_iteration)
                fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                cv_scores.append(fold_rmse)
                
                mlflow.log_metric(f'fold_{fold}_rmse', fold_rmse)
            
            mean_cv_rmse = np.mean(cv_scores)
            std_cv_rmse = np.std(cv_scores)
            mlflow.log_metrics({
                'mean_cv_rmse': mean_cv_rmse,
                'std_cv_rmse': std_cv_rmse
            })
            logger.info(f"CV RMSE: {mean_cv_rmse:.2f} (+/- {std_cv_rmse:.2f})")
        
        # Train final model
        logger.info("Training final LightGBM model...")
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        evals_result = {}
        lgb_model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[valid_data],
            valid_names=['test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        # Log training progress
        for i, loss in enumerate(evals_result['test']['rmse']):
            if i % 50 == 0:  # Log every 50 iterations to avoid too many metrics
                mlflow.log_metric("iteration_rmse", loss, step=i)
        
        # Make predictions
        y_pred_train = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)
        y_pred_test = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mape = np.mean(np.abs((y_train - y_pred_train) / (y_train + 1e-10))) * 100
        
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-10))) * 100
        
        metrics = {
            'train_r2': train_r2,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_mape': train_mape,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'best_iteration': lgb_model.best_iteration
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        logger.info(f"Model Performance:")
        logger.info(f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
        logger.info(f"  Test R²: {test_r2:.4f}, MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")
        logger.info(f"  Best Iteration: {lgb_model.best_iteration}")
        
        # Feature importance
        feature_importance = lgb_model.feature_importance(importance_type='gain')
        feature_columns = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance_df.head(20)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 20 Feature Importances - LightGBM')
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance_lightgbm.png")
        plt.close()
        
        # Plot predictions scatter
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training set
        ax1.scatter(y_train, y_pred_train, alpha=0.5, s=1)
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Rides')
        ax1.set_ylabel('Predicted Rides')
        ax1.set_title(f'Training Set (R² = {train_r2:.4f})')
        ax1.grid(True, alpha=0.3)
        
        # Test set
        ax2.scatter(y_test, y_pred_test, alpha=0.5, s=1)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Rides')
        ax2.set_ylabel('Predicted Rides')
        ax2.set_title(f'Test Set (R² = {test_r2:.4f})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        mlflow.log_figure(fig, "predictions_scatter_lightgbm.png")
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
        mlflow.log_figure(fig, "residuals_lightgbm.png")
        plt.close()
        
        # Save model
        mlflow.lightgbm.log_model(
            lgb_model,
            "model",
            registered_model_name=None  # Will be registered by main script if selected as best
        )
        
        # Save model locally as backup
        joblib.dump(lgb_model, 'lightgbm_bikeshare_model.pkl')
        mlflow.log_artifact('lightgbm_bikeshare_model.pkl')
        
        # Save metadata
        metadata = {
            'model': 'LightGBM',
            'features': list(feature_columns),
            'performance': metrics,
            'best_iteration': lgb_model.best_iteration,
            'hyperparameters': lgb_params,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(metadata, 'lightgbm_model_metadata.pkl')
        mlflow.log_artifact('lightgbm_model_metadata.pkl')
        
        return lgb_model, metrics