"""
XGBoost Model Training Module for BlueBikes Demand Prediction
This module contains the XGBoost training function with MLflow tracking
"""
from __future__ import annotations
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
from artifact_manager import ArtifactManager

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, mlflow, config=None):
    """
    Train XGBoost model with MLflow tracking
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        mlflow: MLflow module (passed in to avoid import issues)
        config: Optional dict with custom parameters
    
    Returns:
        tuple: (trained_model, metrics_dict)
    """
    
    with mlflow.start_run(nested=True, run_name="xgboost"):
        
        # Model parameters (can be overridden by config)
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'min_child_weight': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'enable_categorical': False,
            'early_stopping_rounds': 50,
            'verbosity': 0
        }
        
        # Override with custom config if provided
        if config:
            params = {**default_params, **config}
        else:
            params = default_params
        
        # Log model type and parameters
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.log_params(params)
        
        # Log dataset information
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        
        print(f"\nTraining XGBoost with {len(X_train):,} samples...")
        
        # Create and train model
        model = xgb.XGBRegressor(**params)
        
        # Setup evaluation set for early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Log training metrics
        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.log_metric("best_score", model.best_score)
        
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best score: {model.best_score:.4f}")
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, y_val, y_pred_val)
        
        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Print performance summary
        print_performance_summary(metrics)
        
        # Log feature importance
        log_feature_importance(model, X_train.columns if hasattr(X_train, 'columns') else None, mlflow)
        
        # Create and log visualization plots
        create_and_log_plots(y_train, y_pred_train, y_test, y_pred_test, y_val, y_pred_val, mlflow)
        
        # Log the model
        mlflow.xgboost.log_model(
            model, 
            "model"
        )
        
        # Save model summary
        save_model_summary(model, metrics, mlflow)
        
        return model, metrics


def calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, y_val, y_pred_val):
    """Calculate comprehensive evaluation metrics"""
    
    metrics = {
        # Training metrics
        'train_r2': r2_score(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_mape': np.mean(np.abs((y_train - y_pred_train) / (y_train + 1))) * 100,
        
        #Validation metrics
        'val_r2': r2_score(y_val, y_pred_val),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)), 
        'val_mae': mean_absolute_error(y_val, y_pred_val),
        'val_mape': np.mean(np.abs((y_val - y_pred_val) / (y_val + 1))) * 100,
        # Test metrics
        'test_r2': r2_score(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_mape': np.mean(np.abs((y_test - y_pred_test) / (y_test + 1))) * 100,
        
        # Additional test metrics
        'test_max_error': np.max(np.abs(y_test - y_pred_test)),
        'test_median_absolute_error': np.median(np.abs(y_test - y_pred_test)),
        'test_90th_percentile_error': np.percentile(np.abs(y_test - y_pred_test), 90),
        'test_95th_percentile_error': np.percentile(np.abs(y_test - y_pred_test), 95)
    }
    
    return metrics


def print_performance_summary(metrics):
    """Print formatted performance summary"""
    
    print("\n" + "="*50)
    print("XGBOOST MODEL PERFORMANCE")
    print("="*50)
    print(f"Training Set:")
    print(f"  R² Score: {metrics['train_r2']:.4f}")
    print(f"  RMSE: {metrics['train_rmse']:.2f} rides")
    print(f"  MAE: {metrics['train_mae']:.2f} rides")
    print(f"  MAPE: {metrics['train_mape']:.2f}%")
    print(f"\nValidation Set:")
    print(f"  R² Score: {metrics['val_r2']:.4f}")
    print(f"  RMSE: {metrics['val_rmse']:.2f} rides")
    print(f"  MAE: {metrics['val_mae']:.2f} rides")
    print(f"  MAPE: {metrics['val_mape']:.2f}%")
    print(f"\nTest Set:")
    print(f"  R² Score: {metrics['test_r2']:.4f}")
    print(f"  RMSE: {metrics['test_rmse']:.2f} rides")
    print(f"  MAE: {metrics['test_mae']:.2f} rides")
    print(f"  MAPE: {metrics['test_mape']:.2f}%")
    print(f"  90th Percentile Error: {metrics['test_90th_percentile_error']:.2f} rides")
    print("="*50)


def log_feature_importance(model, feature_names, mlflow):
    """Calculate and log feature importance"""
    
    try:
        # Get feature importance
        importance_dict = model.get_booster().get_score(importance_type='gain')
        
        if feature_names:
            # Map to feature names
            importance_data = []
            for i, fname in enumerate(feature_names):
                importance_data.append({
                    'feature': fname,
                    'importance': importance_dict.get(f'f{i}', 0)
                })
            
            # Sort by importance
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Log top 10 features
            for i, row in importance_df.head(10).iterrows():
                mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
            
            # Save full importance as artifact
            # importance_df.to_csv("xgboost_feature_importance.csv", index=False)
            save_path = ArtifactManager.get_feature_importance_csv_path("xgboost")
            importance_df.to_csv(save_path, index=False)
            # mlflow.log_artifact("xgboost_feature_importance.csv")
            mlflow.log_artifact(str(save_path))
            
    except Exception as e:
        print(f"Warning: Could not log feature importance: {e}")


def create_and_log_plots(y_train, y_pred_train, y_test, y_pred_test, y_val, y_pred_val, mlflow):
    """Create visualization plots and log them as artifacts"""
    
    try:
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Training: Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(y_train, y_pred_train, alpha=0.5, s=10)
        ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Rides')
        ax.set_ylabel('Predicted Rides')
        ax.set_title('Training Set: Actual vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # 2. Test: Actual vs Predicted
        ax = axes[0, 1]
        ax.scatter(y_test, y_pred_test, alpha=0.5, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Rides')
        ax.set_ylabel('Predicted Rides')
        ax.set_title('Test Set: Actual vs Predicted')
        ax.grid(True, alpha=0.3)

        # 3. Validation: Actual vs Predicted
        ax = axes[0, 1]
        ax.scatter(y_val, y_pred_val, alpha=0.5, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Rides')
        ax.set_ylabel('Predicted Rides')
        ax.set_title('Test Set: Actual vs Predicted')
        ax.grid(True, alpha=0.3)


        # 4. Training Residuals
        ax = axes[1, 0]
        train_residuals = y_train - y_pred_train
        ax.scatter(y_pred_train, train_residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Rides')
        ax.set_ylabel('Residuals')
        ax.set_title('Training Set: Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # 6. validation Residuals
        ax = axes[1, 1]
        val_residuals = y_val - y_pred_val
        ax.scatter(y_pred_val, val_residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Rides')
        ax.set_ylabel('Residuals')
        ax.set_title('Validation Set: Residual Plot')
        ax.grid(True, alpha=0.3)

        # 5. Test Residuals
        ax = axes[1, 1]
        test_residuals = y_test - y_pred_test
        ax.scatter(y_pred_test, test_residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Rides')
        ax.set_ylabel('Residuals')
        ax.set_title('Test Set: Residual Plot')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.savefig("xgboost_predictions_analysis.png", dpi=100, bbox_inches='tight')
        save_path = ArtifactManager.get_predictions_plot_path("xgboost")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(str(save_path))   
        plt.close()
        
        # Log the plot
        # mlflow.log_artifact("xgboost_predictions_analysis.png")
        
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")


def save_model_summary(model, metrics, mlflow):
    """Save a JSON summary of the model and its performance"""
    
    summary = {
        "model_type": "XGBoost",
        "timestamp": datetime.now().isoformat(),
        "best_iteration": int(model.best_iteration) if hasattr(model, 'best_iteration') else None,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "performance_summary": {
            "test_mae": float(metrics['test_mae']),
            "test_rmse": float(metrics['test_rmse']),
            "test_r2": float(metrics['test_r2']),
            "improvement_over_baseline": None  # Can be calculated if baseline exists
        }
    }
    
    # with open("xgboost_model_summary.json", "w") as f:
    save_path = ArtifactManager.get_training_summary_path("xgboost")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # mlflow.log_artifact("xgboost_model_summary.json")
    mlflow.log_artifact(str(save_path))


# Hyperparameter tuning function
def tune_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, mlflow, param_grid=None, max_combinations=8):
    """
    Perform lightweight hyperparameter tuning for XGBoost using the validation set.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (used for model selection)
        X_test, y_test: Test data (held out, only for final evaluation)
        mlflow: MLflow module
        param_grid: Dict of parameters to tune
        max_combinations: Maximum number of parameter combinations to try

    Returns:
        best_model, best_metrics
    """
    import random
    from itertools import product

    # Much smaller grid
    if param_grid is None:
        param_grid = {
            "max_depth": [6, 8],
            "learning_rate": [0.05],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        }

    # All combinations
    all_combos = [
        dict(zip(param_grid.keys(), v))
        for v in product(*param_grid.values())
    ]

    # Randomly sample at most max_combinations combinations
    if len(all_combos) > max_combinations:
        all_combos = random.sample(all_combos, max_combinations)

    print(f"Testing {len(all_combos)} XGBoost parameter combinations...")

    best_val_mae = float("inf")
    best_model = None
    best_params = None
    best_metrics = None

    for params in all_combos:
        # IMPORTANT: use smaller n_estimators during tuning
        print(f"\n Combination: {params}")
        tuning_config = {
            "n_estimators": 300,          # smaller than 1000
            "early_stopping_rounds": 30,  # faster stopping
        }
        tuning_config.update(params)

        model, metrics = train_xgboost(
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            mlflow,
            config=tuning_config,
        )

        val_mae = metrics["val_mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
            best_params = tuning_config
            best_metrics = metrics

    print("\nBest XGBoost parameters found:")
    print(f"  Params: {best_params}")
    print(f"  Val MAE: {best_val_mae:.2f}")

    return best_model, best_metrics
