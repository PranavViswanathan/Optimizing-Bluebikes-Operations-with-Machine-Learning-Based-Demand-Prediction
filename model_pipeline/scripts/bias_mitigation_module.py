"""
Bias Mitigation Module
Applies bias mitigation strategies and retrains models
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import joblib
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_module import DataLoader
from pipeline_config import TrainingConfig


class BiasMitigator:
    """Apply bias mitigation and retrain models"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config if config else TrainingConfig()
        self.setup_mlflow()
        self.mitigated_data = {}
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        self.experiment = mlflow.set_experiment(self.config.experiment_name)
        
    def apply_feature_engineering(self, data: dict):
        """Apply bias mitigation through feature engineering"""
        print("\n" + "="*80)
        print(" APPLYING BIAS MITIGATION ".center(80))
        print("="*80)
        
        print("Strategy: Feature Engineering Only")
        
        X_train = data['X_train'].copy()
        X_val = data['X_val'].copy()
        X_test = data['X_test'].copy()
        y_train = data['y_train'].copy()
        y_val = data['y_val'].copy()
        y_test = data['y_test'].copy()
        
        # Apply feature engineering
        X_train_mit = self._add_optimized_features(X_train)
        X_val_mit = self._add_optimized_features(X_val)
        X_test_mit = self._add_optimized_features(X_test)
        
        print(f"Added 10 bias-aware features to training data")
        print(f"Training set: {len(X_train_mit):,} samples, {X_train_mit.shape[1]} features")
        
        # Store mitigated data
        self.mitigated_data = {
            'X_train': X_train_mit,
            'X_val': X_val_mit,
            'X_test': X_test_mit,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'sample_weights': None
        }
        
        return self.mitigated_data
    
    def _add_optimized_features(self, X):
        """Add bias-aware features"""
        X = X.copy()
        
        # Temporal bias features
        X['is_hour_8'] = (X['hour'] == 8).astype(int)
        X['is_hour_17_18'] = X['hour'].isin([17, 18]).astype(int)
        
        X['rush_intensity'] = 0.0
        X.loc[X['hour'] == 8, 'rush_intensity'] = 1.0
        X.loc[X['hour'].isin([17, 18]), 'rush_intensity'] = 1.0
        X.loc[X['hour'].isin([7, 9]), 'rush_intensity'] = 0.5
        X.loc[X['hour'].isin([16, 19]), 'rush_intensity'] = 0.5
        
        # Interaction bias features
        X['weekday_morning_rush'] = (1 - X['is_weekend']) * X['is_morning_rush']
        X['weekday_evening_rush'] = (1 - X['is_weekend']) * X['is_evening_rush']
        
        # Demand level features
        if 'rides_last_hour' in X.columns:
            X['high_demand_flag'] = (X['rides_last_hour'] > X['rides_last_hour'].quantile(0.75)).astype(int)
            X['low_demand_flag'] = (X['rides_last_hour'] < X['rides_last_hour'].quantile(0.25)).astype(int)
            
            if 'rides_rolling_3h' in X.columns:
                X['demand_volatility'] = np.abs(X['rides_last_hour'] - X['rides_rolling_3h'])
            else:
                X['demand_volatility'] = 0
        else:
            X['high_demand_flag'] = 0
            X['low_demand_flag'] = 0
            X['demand_volatility'] = 0
        
        # Composite features
        X['problem_period'] = (
            X['is_hour_8'] + X['is_hour_17_18'] +
            X['weekday_morning_rush'] + X['weekday_evening_rush']
        ).clip(0, 1)
        
        hour_groups = pd.cut(X['hour'], bins=[0, 6, 10, 14, 18, 24], 
                            labels=[0, 1, 2, 3, 4], include_lowest=True)
        X['hour_group'] = pd.to_numeric(hour_groups, errors='coerce').fillna(0).astype(int)
        
        return X
    
    def save_mitigated_data(self, output_dir: str = "data_splits_mitigated"):
        """Save mitigated data to disk"""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.mitigated_data:
            raise ValueError("No mitigated data to save. Run apply_feature_engineering() first.")
        
        print(f"\nSaving mitigated data to: {output_dir}/")
        
        for key, value in self.mitigated_data.items():
            if value is not None:
                filepath = output_path / f"{key}.pkl"
                joblib.dump(value, filepath)
                print(f"  Saved: {key}.pkl")
        
        print(f"\nMitigated data saved successfully!")
    
    def retrain_model(self, model_metadata_path: str, data: dict = None):
        """Retrain the best model with bias mitigation"""
        print("\n" + "="*80)
        print(" RETRAINING WITH BIAS MITIGATION ".center(80))
        print("="*80)
        
        # Load model metadata
        with open(model_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_name = metadata['model_name']
        
        # Use mitigated data if not provided
        if data is None:
            data = self.mitigated_data
        
        if not data:
            raise ValueError("No data provided for retraining")
        
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        sample_weights = data.get('sample_weights')
        
        # Drop 'date' column
        X_train_clean = X_train.drop('date', axis=1, errors='ignore')
        X_val_clean = X_val.drop('date', axis=1, errors='ignore')
        X_test_clean = X_test.drop('date', axis=1, errors='ignore')
        
        with mlflow.start_run(run_name=f"mitigated_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tag("pipeline_stage", "bias_mitigated")
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("mitigation_strategy", "optimized_multi_strategy")
            mlflow.log_param("has_sample_weights", sample_weights is not None)
            mlflow.log_param("n_features", X_train_clean.shape[1])
            mlflow.log_param("n_samples", len(X_train_clean))
            mlflow.log_param("mitigation_components", "weighting+features+augmentation")
            
            print(f"Retraining {model_name.upper()} with optimized bias mitigation...")
            
            # Train model based on type
            if model_name == 'xgboost':
                model, metrics = self._train_xgboost(
                    X_train_clean, y_train, X_val_clean, y_val, 
                    X_test_clean, y_test, sample_weights
                )
            elif model_name == 'lightgbm':
                model, metrics = self._train_lightgbm(
                    X_train_clean, y_train, X_val_clean, y_val, 
                    X_test_clean, y_test, sample_weights
                )
            elif model_name == 'randomforest':
                model, metrics = self._train_randomforest(
                    X_train_clean, y_train, X_val_clean, y_val, 
                    X_test_clean, y_test, sample_weights
                )
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            print(f"\nMitigated Model Performance:")
            print(f"  Test MAE: {metrics['test_mae']:.2f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
            print(f"  Test R²: {metrics['test_r2']:.4f}")
            print(f"  Test MAPE: {metrics['test_mape']:.2f}%")
            
            # Save mitigated model
            mitigated_model_path = f"mitigated_model_{model_name}.pkl"
            joblib.dump(model, mitigated_model_path)
            print(f"\nMitigated model saved to: {mitigated_model_path}")
            
            # Save metadata
            mitigated_metadata = {
                'model_name': model_name,
                'model_path': mitigated_model_path,
                'metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                           for k, v in metrics.items()},
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'mitigation_applied': True
            }
            
            metadata_path = "mitigated_model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(mitigated_metadata, f, indent=2)
            print(f"Model metadata saved to: {metadata_path}")
            
            return model, metrics
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, X_test, y_test, sample_weights):
        """Train XGBoost model"""
        import xgboost as xgb
        
        params = {
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
            'early_stopping_rounds': 50
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        mlflow.xgboost.log_model(model, "model")
        
        metrics = self._calculate_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test)
        return model, metrics
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, X_test, y_test, sample_weights):
        """Train LightGBM model"""
        import lightgbm as lgb
        
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        mlflow.lightgbm.log_model(model, "model")
        
        metrics = self._calculate_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test)
        return model, metrics
    
    def _train_randomforest(self, X_train, y_train, X_val, y_val, X_test, y_test, sample_weights):
        """Train Random Forest model"""
        from sklearn.ensemble import RandomForestRegressor
        
        params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        mlflow.sklearn.log_model(model, "model")
        
        metrics = self._calculate_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test)
        return model, metrics
    
    def _calculate_metrics(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """Calculate evaluation metrics"""
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            'val_mae': mean_absolute_error(y_val, y_pred_val),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'val_r2': r2_score(y_val, y_pred_val),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mape': np.mean(np.abs((y_test - y_pred_test) / (y_test + 1))) * 100
        }
        
        return metrics


def main():
    """Standalone execution for bias mitigation"""
    print("="*80)
    print(" BLUEBIKES BIAS MITIGATION MODULE ".center(80))
    print("="*80)
    
    # Load original data
    data = DataLoader.load_data()
    
    # Initialize mitigator
    mitigator = BiasMitigator()
    
    # Apply mitigation
    mitigated_data = mitigator.apply_feature_engineering(data)
    
    # Save mitigated data
    mitigator.save_mitigated_data()
    
    # Retrain model
    model, metrics = mitigator.retrain_model("best_model_metadata.json", mitigated_data)
    
    print("\n" + "="*80)
    print(" BIAS MITIGATION COMPLETE ".center(80))
    print("="*80)
    
    return mitigator, model, metrics


if __name__ == "__main__":
    mitigator, model, metrics = main()