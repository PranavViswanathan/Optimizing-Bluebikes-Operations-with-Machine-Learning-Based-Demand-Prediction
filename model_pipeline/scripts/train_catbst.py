"""
CatBoost Training Module for BlueBikes Demand Prediction
Structured similarly to train_lgb.py for consistency.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import logging
from typing import Optional, Dict, List
from datetime import datetime
import mlflow
import mlflow.catboost

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _auto_detect_categoricals(X):
    """
    Detect categorical feature columns if X is a DataFrame.
    Returns a list of column names, or an empty list if not applicable.
    """
    if hasattr(X, "select_dtypes"):
        return X.select_dtypes(include=["object", "category"]).columns.tolist()
    return []


def train_catboost(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    categorical_features=None,
    mlflow_client=None,   # kept for API symmetry with train_lightgbm
    use_cv: bool = False,
    config: Optional[Dict] = None,
):
    """
    Train CatBoost model with MLflow tracking.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        categorical_features: List of categorical feature names or indices.
                              If None and X_train is a DataFrame, they are auto-detected.
        mlflow_client: Unused; kept for consistency with train_lightgbm signature.
        use_cv: Whether to run K-fold cross-validation before final training.

    Returns:
        model: Trained CatBoostRegressor model
        metrics: Dictionary of performance metrics on the test set
    """

    # Auto-detect categoricals if not provided
    if categorical_features is None:
        categorical_features = _auto_detect_categoricals(X_train)

    with mlflow.start_run(
        nested=True,
        run_name=f"catboost_{datetime.now().strftime('%Y%m%d_%H%M')}"
    ):
        # Hyperparameters (adapted from your previous CatBoost config)
        cb_params = {
            "iterations": 500,
            "depth": 8,
            "learning_rate": 0.1,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": 100,
        }
        if config is not None:
            params = {**cb_params, **config}
        else:
            params = cb_params
        # Log parameters + tags
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "CatBoost")
        mlflow.set_tag("optimizer", "manual_tuning")

        # Cross-validation (optional, 5-fold)
        if use_cv:
            logger.info("Performing 5-fold cross-validation with CatBoost...")
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            # Need indices if data is numpy; iloc if DataFrame
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                if hasattr(X_train, "iloc"):
                    X_fold_train = X_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    y_fold_val = y_train.iloc[val_idx]
                else:
                    X_fold_train = X_train[train_idx]
                    X_fold_val = X_train[val_idx]
                    y_fold_train = y_train[train_idx]
                    y_fold_val = y_train[val_idx]

                fold_model = CatBoostRegressor(
                    **params,
                    cat_features=categorical_features
                    if isinstance(categorical_features, list) else None
                )

                fold_model.fit(
                    X_fold_train,
                    y_fold_train,
                    eval_set=(X_fold_val, y_fold_val),
                    use_best_model=True,
                    verbose=False
                )

                y_pred_val = fold_model.predict(X_fold_val)
                fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred_val))
                cv_scores.append(fold_rmse)

                mlflow.log_metric(f"fold_{fold}_rmse", fold_rmse)

            mean_cv_rmse = float(np.mean(cv_scores))
            std_cv_rmse = float(np.std(cv_scores))
            mlflow.log_metrics({
                "mean_cv_rmse": mean_cv_rmse,
                "std_cv_rmse": std_cv_rmse
            })
            logger.info(
                f"CV RMSE: {mean_cv_rmse:.2f} (+/- {std_cv_rmse:.2f})"
            )

        # Final model training on full training set
        logger.info("Training final CatBoost model...")
        model = CatBoostRegressor(
            **params,
            cat_features=categorical_features
            if isinstance(categorical_features, list) else None
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=cb_params["verbose"]
        )

        # Log training curve
        evals_result = model.get_evals_result()
        # For CatBoost, typically: evals_result['learn']['RMSE'], evals_result['validation']['RMSE']
        if "validation" in evals_result and "RMSE" in evals_result["validation"]:
            val_rmse_history = evals_result["validation"]["RMSE"]
            for i, loss in enumerate(val_rmse_history):
                if i % 50 == 0:
                    mlflow.log_metric("iteration_rmse", float(loss), step=i)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        # Metrics
        def _mape(y_true, y_pred):
            return float(
                np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            )

        train_r2 = float(r2_score(y_train, y_pred_train))
        train_mae = float(mean_absolute_error(y_train, y_pred_train))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        train_mape = _mape(np.array(y_train), np.array(y_pred_train))

        val_r2 = float(r2_score(y_val, y_pred_val))
        val_mae = float(mean_absolute_error(y_val, y_pred_val))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        val_mape = _mape(np.array(y_val), np.array(y_pred_val))

        test_r2 = float(r2_score(y_test, y_pred_test))
        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        test_mape = _mape(np.array(y_test), np.array(y_pred_test))

        metrics = {
            "train_r2": train_r2,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_mape": train_mape,
            "val_r2": val_r2,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_mape": val_mape,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "best_iteration": int(model.get_best_iteration()),
        }

        mlflow.log_metrics(metrics)

        logger.info("CatBoost Model Performance:")
        logger.info(
            f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}"
        )
        logger.info(
            f"  Val   R²: {val_r2:.4f}, MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}"
        )
        logger.info(
            f"  Test  R²: {test_r2:.4f}, MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}"
        )
        logger.info(f"  Best Iteration: {metrics['best_iteration']}")

        # Feature importance
        feature_importance = model.get_feature_importance()
        if hasattr(X_train, "columns"):
            feature_columns = list(X_train.columns)
        else:
            feature_columns = [f"feature_{i}" for i in range(len(feature_importance))]

        feature_importance_df = (
            pd.DataFrame(
                {"feature": feature_columns, "importance": feature_importance}
            )
            .sort_values("importance", ascending=False)
        )

        # Plot: Top 20 feature importances
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance_df.head(20)
        ax.barh(range(len(top_features)), top_features["importance"])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.set_xlabel("Feature Importance")
        ax.set_title("Top 20 Feature Importances - CatBoost")
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance_catboost.png")
        plt.close(fig)

        # Scatter plots: actual vs predicted (train & test)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

        # Training set
        ax1.scatter(y_train, y_pred_train, alpha=0.5, s=1)
        ax1.plot(
            [np.min(y_train), np.max(y_train)],
            [np.min(y_train), np.max(y_train)],
            "r--",
            lw=2,
        )
        ax1.set_xlabel("Actual Rides")
        ax1.set_ylabel("Predicted Rides")
        ax1.set_title(f"Training Set (R² = {train_r2:.4f})")
        ax1.grid(True, alpha=0.3)
        #Validation set
        ax2.scatter(y_val, y_pred_val, alpha=0.5, s=1)
        ax2.plot(
            [np.min(y_val), np.max(y_val)],
            [np.min(y_val), np.max(y_val)],
            "r--",
            lw=2,
        )
        ax2.set_xlabel("Actual Rides")
        ax2.set_ylabel("Predicted Rides")
        ax2.set_title(f"Validation Set (R² = {val_r2:.4f})")
        ax2.grid(True, alpha=0.3)

        # Test set
        ax3.scatter(y_test, y_pred_test, alpha=0.5, s=1)
        ax3.plot(
            [np.min(y_test), np.max(y_test)],
            [np.min(y_test), np.max(y_test)],
            "r--",
            lw=2,
        )
        ax3.set_xlabel("Actual Rides")
        ax3.set_ylabel("Predicted Rides")
        ax3.set_title(f"Test Set (R² = {test_r2:.4f})")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        mlflow.log_figure(fig, "predictions_scatter_catboost.png")
        plt.close(fig)

        # Residual plots (test set)
        residuals_test = np.array(y_test) - np.array(y_pred_test)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.scatter(y_pred_test, residuals_test, alpha=0.5, s=1)
        ax1.axhline(y=0, color="r", linestyle="--")
        ax1.set_xlabel("Predicted Rides")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Test Residuals - CatBoost")
        ax1.grid(True, alpha=0.3)

        ax2.hist(residuals_test, bins=50, edgecolor="black", alpha=0.7)
        ax2.set_xlabel("Residuals")
        ax2.set_ylabel("Frequency")
        ax2.set_title(
            f"Residual Distribution "
            f"(Mean: {residuals_test.mean():.2f}, Std: {residuals_test.std():.2f})"
        )

        plt.tight_layout()
        mlflow.log_figure(fig, "residuals_catboost.png")
        plt.close(fig)

        # Log model with MLflow
        mlflow.catboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=None  # optional: set externally if needed
        )

        # Also save model locally as backup
        joblib.dump(model, "catboost_bikeshare_model.pkl")
        mlflow.log_artifact("catboost_bikeshare_model.pkl")

        # Save metadata (features, performance, hyperparameters)
        metadata = {
            "model": "CatBoost",
            "features": feature_columns,
            "performance": metrics,
            "hyperparameters": params,
            "categorical_features": categorical_features,
            "timestamp": datetime.now().isoformat(),
        }
        joblib.dump(metadata, "catboost_model_metadata.pkl")
        mlflow.log_artifact("catboost_model_metadata.pkl")

        return model, metrics

def tune_catboost(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    mlflow_client=None,
    param_grid=None,
    max_combinations=6
):
    """
    Lightweight hyperparameter tuning for CatBoost using validation MAE.
    Calls train_catboost for each combination (nested MLflow runs).
    """

    import random
    from itertools import product

    if param_grid is None:
        # Small grid to keep runtime reasonable
        param_grid = {
            "depth": [6, 8],
            "learning_rate": [0.05, 0.1],
            "iterations": [300],  # fewer trees during tuning
        }

    all_combos = [
        dict(zip(param_grid.keys(), v))
        for v in product(*param_grid.values())
    ]

    if len(all_combos) > max_combinations:
        all_combos = random.sample(all_combos, max_combinations)

    print(f"Testing {len(all_combos)} CatBoost parameter combinations...")

    best_val_mae = float("inf")
    best_model = None
    best_params = None
    best_metrics = None

    for cfg in all_combos:
        print(f"\nCatBoost tuning config: {cfg}")

        model, metrics = train_catboost(
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            categorical_features=None,
            mlflow_client=mlflow_client,
            use_cv=False,
            config=cfg,
        )

        val_mae = metrics["val_mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
            best_params = cfg
            best_metrics = metrics

    print("\nBest CatBoost parameters found:")
    print(f"  Params: {best_params}")
    print(f"  Val MAE: {best_val_mae:.2f}")

    return best_model, best_metrics
