"""
Model Training Module
Handles training of multiple models and selection of the best model
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from pathlib import Path

from train_xgb import train_xgboost, tune_xgboost
from train_lgb import train_lightgbm, tune_lightgbm
from train_random_forest import train_random_forest, tune_random_forest
from data_module import DataLoader
from pipeline_config import TrainingConfig

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and manage multiple models"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config if config else TrainingConfig()
        self.setup_mlflow()
        self.client = MlflowClient()
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_model_path = None
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        self.experiment = mlflow.set_experiment(self.config.experiment_name)
        print(f"MLflow Experiment: {self.config.experiment_name}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        
    def train_all_models(self, data: dict):
        """Train all configured models"""
        print("\n" + "="*80)
        print(" MODEL TRAINING ".center(80))
        print("="*80)
        
        # Extract data
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        
        # Drop 'date' column for training
        X_train_clean = X_train.drop('date', axis=1, errors='ignore')
        X_val_clean = X_val.drop('date', axis=1, errors='ignore')
        X_test_clean = X_test.drop('date', axis=1, errors='ignore')
        
        print(f"Models to train: {self.config.models_to_train}")
        print(f"Hyperparameter tuning: {self.config.tune_hyperparameters}")
        
        self.results = {}
        
        with mlflow.start_run(run_name=f"baseline_training_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tag("pipeline_stage", "baseline_training")
            mlflow.log_param("models_trained", self.config.models_to_train)
            mlflow.log_param("train_samples", len(X_train_clean))
            mlflow.log_param("test_samples", len(X_test_clean))
            
            for model_name in self.config.models_to_train:
                print(f"\n{'='*40}")
                print(f"Training {model_name.upper()}")
                print(f"{'='*40}")
                
                try:
                    model, metrics, run_id = self._train_single_model(
                        model_name, X_train_clean, y_train, 
                        X_val_clean, y_val, X_test_clean, y_test
                    )
                    
                    self.results[model_name] = (model, metrics, run_id)
                    
                    print(f"\n{model_name.upper()} Results:")
                    print(f"  Test MAE: {metrics['test_mae']:.2f}")
                    print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
                    print(f"  Test R²: {metrics['test_r2']:.4f}")
                    
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        return self.results
    
    def _train_single_model(self, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train a single model and return model, metrics, and run_id"""
        
        if model_name == 'xgboost':
            if self.config.tune_hyperparameters:
                model, metrics = tune_xgboost(
                    X_train, y_train, X_val, y_val, X_test, y_test, mlflow
                )
            else:
                model, metrics = train_xgboost(
                    X_train, y_train, X_val, y_val, X_test, y_test, mlflow
                )
            runs = self.client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string="tags.model_type = 'XGBoost'",
                order_by=["attribute.start_time DESC"],
                max_results=1
            )
            run_id = runs[0].info.run_id if runs else None
        
        elif model_name == 'lightgbm':
            if self.config.tune_hyperparameters:
                model, metrics = tune_lightgbm(
                    X_train, y_train, X_val, y_val, X_test, y_test, mlflow
                )
            else:
                model, metrics = train_lightgbm(
                    X_train, y_train, X_val, y_val, X_test, y_test, mlflow, use_cv=False
                )
            runs = self.client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string="tags.model_type = 'LightGBM'",
                order_by=["attribute.start_time DESC"],
                max_results=1
            )
            run_id = runs[0].info.run_id if runs else None
        
        elif model_name == 'randomforest':
            if self.config.tune_hyperparameters:
                model, metrics = tune_random_forest(
                    X_train, y_train, X_val, y_val, X_test, y_test, mlflow
                )
            else:
                model, metrics = train_random_forest(
                    X_train, y_train, X_val, y_val, X_test, y_test, mlflow_client=mlflow
                )
            runs = self.client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string="tags.model_type = 'RandomForest'",
                order_by=["attribute.start_time DESC"],
                max_results=1
            )
            run_id = runs[0].info.run_id if runs else None
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model, metrics, run_id
    
    def select_best_model(self, metric='test_r2', results=None):
        """Select and save the best model"""
        print("\n" + "="*80)
        print(" MODEL SELECTION ".center(80))
        print("="*80)
        
        if results is None:
            results = self.results
        
        if not results:
            raise ValueError("No models trained. Run train_all_models() first.")
        
        if metric == 'test_r2':
            best_model_name = max(results.keys(), 
                                key=lambda x: results[x][1][metric])
        else:
            best_model_name = min(results.keys(), 
                                key=lambda x: results[x][1][metric])
        
        best_model, best_metrics, best_run_id = results[best_model_name]
        
        print(f"Best Model: {best_model_name.upper()}")
        print(f"Selection Metric: {metric}")
        print(f"\nPerformance:")
        print(f"  Test MAE: {best_metrics['test_mae']:.2f}")
        print(f"  Test RMSE: {best_metrics['test_rmse']:.2f}")
        print(f"  Test R²: {best_metrics['test_r2']:.4f}")
        print(f"  Test MAPE: {best_metrics.get('test_mape', 'N/A')}")
        
        # Save the model
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.best_model_path = f"best_model_{best_model_name}.pkl"
        
        joblib.dump(best_model, self.best_model_path)
        print(f"\nBest model saved to: {self.best_model_path}")
        
        # Save metadata
        metadata = {
            'model_name': best_model_name,
            'model_path': self.best_model_path,
            'metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                       for k, v in best_metrics.items()},
            'run_id': best_run_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        import json
        metadata_path = "best_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to: {metadata_path}")
        
        return best_model_name, best_model, best_metrics


def main():
    """Standalone execution for model training"""
    print("="*80)
    print(" BLUEBIKES MODEL TRAINING MODULE ".center(80))
    print("="*80)
    
    # Load data
    data = DataLoader.load_data()
    
    # Initialize trainer
    config = TrainingConfig(
        models_to_train=['xgboost', 'lightgbm', 'randomforest'],
        tune_hyperparameters=False,
        experiment_name="bluebikes_training"
    )
    trainer = ModelTrainer(config)
    
    # Train models
    results = trainer.train_all_models(data)
    
    # Select best model
    best_name, best_model, best_metrics = trainer.select_best_model()
    
    print("\n" + "="*80)
    print(" MODEL TRAINING COMPLETE ".center(80))
    print("="*80)
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()