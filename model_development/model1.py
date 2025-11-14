import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import logging
from datetime import datetime
import wandb
import argparse
import sys

# Custom module imports
from feature_generation import load_and_prepare_data
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_with_config(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        X, y, feature_columns = load_and_prepare_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        wandb.log({
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(feature_columns)
        })
        
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': config.num_leaves,
            'learning_rate': config.learning_rate,
            'feature_fraction': config.feature_fraction,
            'bagging_fraction': config.bagging_fraction,
            'bagging_freq': config.bagging_freq,
            'min_child_samples': config.min_child_samples,
            'reg_alpha': config.reg_alpha,
            'reg_lambda': config.reg_lambda,
            'max_depth': config.max_depth,
            'min_gain_to_split': config.min_gain_to_split,
            'n_estimators': 1000,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }
        
        if config.use_cv:
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_val = y_train.iloc[val_idx]
                
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
                
                wandb.log({f'fold_{fold}_rmse': fold_rmse})
            
            mean_cv_rmse = np.mean(cv_scores)
            std_cv_rmse = np.std(cv_scores)
            wandb.log({
                'mean_cv_rmse': mean_cv_rmse,
                'std_cv_rmse': std_cv_rmse
            })
        
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
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        y_pred_train = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)
        y_pred_test = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
        
        train_r2 = r2_score(y_train, y_pred_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        wandb.log({
            "train_r2": train_r2,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "best_iteration": lgb_model.best_iteration
        })
        
        if config.log_plots:
            feature_importance = lgb_model.feature_importance(importance_type='gain')
            feature_importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance_df.head(20)
            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 20 Feature Importances')
            plt.tight_layout()
            wandb.log({"feature_importance": wandb.Image(fig)})
            plt.close()

def run_hyperparameter_sweep():
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'test_rmse',
            'goal': 'minimize'
        },
        'parameters': {
            'num_leaves': {
                'distribution': 'int_uniform',
                'min': 31,
                'max': 512
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 0.01,
                'max': 0.3
            },
            'feature_fraction': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1.0
            },
            'bagging_fraction': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1.0
            },
            'bagging_freq': {
                'values': [1, 3, 5, 7, 10]
            },
            'min_child_samples': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 100
            },
            'reg_alpha': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,
                'max': 10.0
            },
            'reg_lambda': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,
                'max': 10.0
            },
            'max_depth': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 20
            },
            'min_gain_to_split': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,
                'max': 1.0
            },
            'use_cv': {
                'value': False
            },
            'log_plots': {
                'value': False
            }
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="bluebikes-hyperparameter-tuning")
    wandb.agent(sweep_id, train_with_config, count=50)
    
    return sweep_id

def train_final_model(best_params=None):
    logger.info("="*60)
    logger.info("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    logger.info("="*60)
    
    if best_params is None:
        best_params = {
            'num_leaves': 255,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'max_depth': -1,
            'min_gain_to_split': 0.0
        }
    
    wandb.init(
        project="bluebikes-demand-prediction",
        name=f"final-model-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=best_params,
        tags=["final", "production"]
    )
    
    X, y, feature_columns = load_and_prepare_data()
    
    wandb.log({"final_dataset_size": len(X)})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    logger.info(f"Train-test split: {len(X_train):,} train, {len(X_test):,} test samples")
    wandb.log({
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": len(feature_columns)
    })
    
    logger.info("Training final LightGBM model...")
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        **best_params,
        'n_estimators': 1000,
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }
    
    wandb.config.update({"lgb_params": lgb_params})
    
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
            lgb.log_evaluation(period=0),
            lgb.record_evaluation(evals_result)
        ]
    )
    
    for i, loss in enumerate(evals_result['test']['rmse']):
        wandb.log({"iteration": i, "test_rmse": loss})
    
    y_pred_train = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)
    y_pred_test = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    
    train_r2 = r2_score(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    logger.info(f"Final Model Results:")
    logger.info(f"  Train R²: {train_r2:.4f}")
    logger.info(f"  Train MAE: {train_mae:.0f} rides")
    logger.info(f"  Train RMSE: {train_rmse:.0f} rides")
    logger.info(f"  Test R²: {test_r2:.4f}")
    logger.info(f"  Test MAE: {test_mae:.0f} rides")
    logger.info(f"  Test RMSE: {test_rmse:.0f} rides")
    
    wandb.log({
        "train_r2": train_r2,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "best_iteration": lgb_model.best_iteration
    })
    
    feature_importance = lgb_model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance_df.head(20)
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Feature Importances - Final Model')
    plt.tight_layout()
    wandb.log({"feature_importance": wandb.Image(fig)})
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(y_train, y_pred_train, alpha=0.5, s=1)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Rides')
    ax1.set_ylabel('Predicted Rides')
    ax1.set_title(f'Training Set (R² = {train_r2:.4f})')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(y_test, y_pred_test, alpha=0.5, s=1)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Rides')
    ax2.set_ylabel('Predicted Rides')
    ax2.set_title(f'Test Set (R² = {test_r2:.4f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({"predictions_scatter": wandb.Image(fig)})
    plt.close()
    
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(y_pred_train, residuals_train, alpha=0.5, s=1)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Rides')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Training Residuals')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(y_pred_test, residuals_test, alpha=0.5, s=1)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Rides')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Test Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({"residuals_plot": wandb.Image(fig)})
    plt.close()
    
    joblib.dump(lgb_model, 'lightgbm_bikeshare_model.pkl')
    
    artifact = wandb.Artifact(
        'bikeshare-model',
        type='model',
        description='Best LightGBM model for bike demand prediction',
        metadata=best_params
    )
    artifact.add_file('lightgbm_bikeshare_model.pkl')
    wandb.log_artifact(artifact)
    
    metadata = {
        'model': 'LightGBM',
        'features': feature_columns,
        'performance': {
            'train_r2': train_r2,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        },
        'best_iteration': lgb_model.best_iteration,
        'hyperparameters': best_params,
        'wandb_run_id': wandb.run.id,
        'wandb_run_name': wandb.run.name
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    
    wandb.summary["final_test_r2"] = test_r2
    wandb.summary["final_test_rmse"] = test_rmse
    wandb.summary["final_test_mae"] = test_mae
    wandb.summary["model_type"] = "LightGBM"
    
    logger.info(f"\nFinal model saved successfully!")
    logger.info("Pipeline complete!")
    logger.info("="*60)
    
    wandb.finish()
    
    return lgb_model, metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bike Demand Prediction Pipeline')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'sweep', 'both'],
                        help='Mode: train (single run), sweep (hyperparameter tuning), or both')
    parser.add_argument('--sweep-count', type=int, default=50,
                        help='Number of sweep runs for hyperparameter tuning')
    
    args = parser.parse_args()
    
    if args.mode == 'sweep':
        logger.info("Starting hyperparameter sweep...")
        sweep_id = run_hyperparameter_sweep()
        logger.info(f"Sweep completed. Sweep ID: {sweep_id}")
        logger.info("Check W&B dashboard for best parameters")
        
    elif args.mode == 'train':
        logger.info("Training final model with default/best parameters...")
        model, metadata = train_final_model()
        
    elif args.mode == 'both':
        logger.info("Running hyperparameter sweep followed by final training...")
        sweep_id = run_hyperparameter_sweep()
        logger.info(f"Sweep completed. Sweep ID: {sweep_id}")
        
        api = wandb.Api()
        sweep = api.sweep(f"bluebikes-hyperparameter-tuning/{sweep_id}")
        best_run = sweep.best_run()
        best_params = best_run.config
        
        logger.info(f"Best parameters found: {best_params}")
        logger.info("Training final model with best parameters...")
        model, metadata = train_final_model(best_params)
    
    logger.info("All operations completed successfully!")