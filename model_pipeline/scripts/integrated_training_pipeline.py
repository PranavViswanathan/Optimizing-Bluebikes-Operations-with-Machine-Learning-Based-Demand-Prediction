"""
Integrated BlueBikes Training Pipeline with Optimized Bias Mitigation
Complete workflow: Train -> Select Best -> Bias Analysis -> Mitigation -> Retrain -> Re-analyze

Uses the BEST bias mitigation approach combining:
1. Targeted sample weighting (focuses on your specific bias patterns)
2. Strategic feature engineering (addresses temporal, rush hour, and interaction biases)
3. Selective augmentation (for severely underrepresented critical slices)
"""

from __future__ import annotations
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from bias_detection import BikeShareBiasDetector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
import joblib
import json
import os

warnings.filterwarnings('ignore')

from train_xgb import train_xgboost, tune_xgboost
from train_lgb import train_lightgbm, tune_lightgbm
from train_random_forest import train_random_forest, tune_random_forest
from feature_generation import load_and_prepare_data
from sensitivity_analysis import run_sensitivity_analysis
from artifact_manager import ArtifactManager


class IntegratedBlueBikesTrainer:
    """
    Complete training pipeline with integrated bias detection and mitigation.
    Uses optimized multi-strategy approach for maximum bias reduction.
    """
    
    def __init__(self, experiment_name="bluebikes_integrated_pipeline"):
        self.experiment_name = experiment_name
        self.setup_mlflow()
        self.client = MlflowClient()
        
        # Pipeline state
        self.best_model = None
        self.best_model_name = None
        self.best_model_path = None
        self.baseline_bias_report = None
        self.mitigated_model = None
        self.mitigated_model_path = None
        self.final_bias_report = None
        
    def setup_mlflow(self):
        mlflow.set_tracking_uri("./mlruns")
        self.experiment = mlflow.set_experiment(self.experiment_name)
        print(f"MLflow Experiment: {self.experiment_name}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    
    def _calculate_sample_weights(self, X_train, y_train):
        """
        Calculate sample weights - CONSERVATIVE VERSION.
        
        Key insight: Aggressive weighting made things WORSE.
        This version uses mild weights that won't destabilize the model.
        
        Args:
            X_train: Training features DataFrame
            y_train: Training target Series
            
        Returns:
            numpy array of weights, one per training sample
        """
        print("\n" + "-"*60)
        print(" CALCULATING SAMPLE WEIGHTS (CONSERVATIVE) ".center(60))
        print("-"*60)
        
        n_samples = len(X_train)
        sample_weights = np.ones(n_samples)
        
        # =============================================================
        # CONSERVATIVE APPROACH: Small adjustments only
        # Max weight should be ~1.5x, not 4x
        # =============================================================
        
        # Strategy 1: Mild upweight for worst hours (not aggressive)
        if 'hour' in X_train.columns:
            # Only the WORST hour gets a boost
            hour_17_mask = X_train['hour'] == 17
            hour_16_mask = X_train['hour'] == 16
            
            sample_weights[hour_17_mask] *= 1.3  # Was 2.0
            sample_weights[hour_16_mask] *= 1.2  # Was 1.8
            
            print(f"  ✓ Hour 17 (+30%): {hour_17_mask.sum():,} samples")
            print(f"  ✓ Hour 16 (+20%): {hour_16_mask.sum():,} samples")
        
        # Strategy 2: Mild upweight for very high demand only (top 10%)
        y_q90 = y_train.quantile(0.90)
        very_high_demand_mask = y_train > y_q90
        sample_weights[very_high_demand_mask] *= 1.2  # Was 1.95 (1.3 * 1.5)
        print(f"  ✓ Very high demand (>q90, +20%): {very_high_demand_mask.sum():,} samples")
        
        # Strategy 3: Mild upweight for weekday evening rush
        if 'is_weekend' in X_train.columns and 'hour' in X_train.columns:
            weekday_evening = (X_train['is_weekend'] == 0) & (X_train['hour'].isin([16, 17, 18]))
            sample_weights[weekday_evening] *= 1.15  # Was 1.3
            print(f"  ✓ Weekday evening rush (+15%): {weekday_evening.sum():,} samples")
        
        # =============================================================
        # NO DOWNWEIGHTING - This was destabilizing the model
        # =============================================================
        print(f"  ✓ No downweighting applied (keeps model stable)")
        
        # =============================================================
        # Normalize weights
        # =============================================================
        sample_weights = sample_weights * n_samples / sample_weights.sum()
        
        # Print statistics
        print(f"\n  Weight Statistics:")
        print(f"    Min:    {sample_weights.min():.3f}")
        print(f"    Max:    {sample_weights.max():.3f}")
        print(f"    Mean:   {sample_weights.mean():.3f}")
        print(f"    Std:    {sample_weights.std():.3f}")
        
        # Verify weights are reasonable
        if sample_weights.max() > 2.0:
            print(f" WARNING: Max weight > 2.0, may cause instability")
        else:
            print(f"  ✓ Weights are in safe range")
        
        return sample_weights

    def load_and_prepare_data(self):
        """Load and split data with proper validation set."""
        print("\n" + "="*80)
        print(" STEP 1: LOADING AND PREPARING DATA ".center(80))
        print("="*80)
                
        X, y, feature_columns = load_and_prepare_data()
        X["date"] = pd.to_datetime(X["date"])

        # Date boundaries
        train_start = pd.Timestamp("2024-01-01")
        val_start   = pd.Timestamp("2025-08-01")   # Validation: Aug-Sep 2025
        train_end   = pd.Timestamp("2025-09-30")   # Training ends here (includes val)
        test_start  = pd.Timestamp("2025-10-01")
        test_end    = pd.Timestamp("2025-11-30")

        # Create masks - validation carved from end of training period
        train_mask = (X["date"] >= train_start) & (X["date"] < val_start)
        val_mask   = (X["date"] >= val_start) & (X["date"] <= train_end)
        test_mask  = (X["date"] >= test_start) & (X["date"] <= test_end)

        X_train = X.loc[train_mask].copy()
        y_train = y.loc[train_mask].copy()
        X_val   = X.loc[val_mask].copy()
        y_val   = y.loc[val_mask].copy()
        X_test  = X.loc[test_mask].copy()
        y_test  = y.loc[test_mask].copy()
        
        print(f"Dataset shape: {X.shape}")
        print(f"\nData Split:")
        print(f"  Training:   {train_start.date()} to {(val_start - pd.Timedelta(days=1)).date()} ({len(X_train):,} samples)")
        print(f"  Validation: {val_start.date()} to {train_end.date()} ({len(X_val):,} samples)")
        print(f"  Test:       {test_start.date()} to {test_end.date()} ({len(X_test):,} samples)")
        print(f"  Features: {X_train.shape[1]}")
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def train_all_models(self, X_train, X_test, X_val, y_train, y_test, y_val, 
                        models_to_train=None, tune=False):
        """Reuse existing model training logic."""
        print("\n" + "="*80)
        print(" STEP 2: TRAINING MODELS ".center(80))
        print("="*80)
        
        if models_to_train is None:
            models_to_train = ['xgboost', 'lightgbm', 'randomforest']
        
        print(f"Models to train: {models_to_train}")
        
        # Drop 'date' column for training
        X_train_clean = X_train.drop('date', axis=1, errors='ignore')
        X_val_clean = X_val.drop('date', axis=1, errors='ignore')
        X_test_clean = X_test.drop('date', axis=1, errors='ignore')
        
        results = {}       
        with mlflow.start_run(run_name=f"baseline_training_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tag("pipeline_stage", "baseline_training")
            mlflow.log_param("models_trained", models_to_train)
            mlflow.log_param("train_samples", len(X_train_clean))
            mlflow.log_param("test_samples", len(X_test_clean))
     
            for model_name in models_to_train:
                print(f"\n{'='*40}")
                print(f"Training {model_name.upper()}")
                print(f"{'='*40}")
                
                try:
                    if model_name == 'xgboost':
                        if tune:
                            model, metrics = tune_xgboost(
                                X_train_clean, y_train, X_val_clean, y_val, X_test_clean, y_test, mlflow
                            )
                        else:
                            model, metrics = train_xgboost(
                                X_train_clean, y_train, X_val_clean, y_val, X_test_clean, y_test, mlflow
                            )
                        runs = self.client.search_runs(
                            experiment_ids=[self.experiment.experiment_id],
                            filter_string="tags.model_type = 'XGBoost'",
                            order_by=["attribute.start_time DESC"],
                            max_results=1
                        )
                        run_id = runs[0].info.run_id if runs else None
                    
                    elif model_name == 'lightgbm':
                        if tune:
                            model, metrics = tune_lightgbm(
                                X_train_clean, y_train, X_val_clean, y_val, X_test_clean, y_test, mlflow
                            )
                        else:
                            model, metrics = train_lightgbm(
                                X_train_clean, y_train, X_val_clean, y_val, X_test_clean, y_test, mlflow, use_cv=False
                            )
                        runs = self.client.search_runs(
                            experiment_ids=[self.experiment.experiment_id],
                            filter_string="tags.model_type = 'LightGBM'",
                            order_by=["attribute.start_time DESC"],
                            max_results=1
                        )
                        run_id = runs[0].info.run_id if runs else None

                    elif model_name == 'randomforest':
                        if tune:
                            model, metrics = tune_random_forest(
                                X_train_clean, y_train, X_val_clean, y_val, X_test_clean, y_test, mlflow
                            )
                        else:
                            model, metrics = train_random_forest(
                                X_train_clean, y_train, X_val_clean, y_val, X_test_clean, y_test, mlflow_client=mlflow
                            )
                        runs = self.client.search_runs(
                            experiment_ids=[self.experiment.experiment_id],
                            filter_string="tags.model_type = 'RandomForest'",
                            order_by=["attribute.start_time DESC"],
                            max_results=1
                        )
                        run_id = runs[0].info.run_id if runs else None
                    
                    else:
                        print(f"Warning: Model {model_name} not implemented")
                        continue
                    
                    results[model_name] = (model, metrics, run_id)
                    
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        return results
    
    def select_best_model(self, results, metric='test_mae'):
        """Select and save the best model."""
        print("\n" + "="*80)
        print(" STEP 3: SELECTING BEST MODEL ".center(80))
        print("="*80)
        
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
        # self.best_model_path = f"best_model_{best_model_name}.pkl"
        
        # joblib.dump(best_model, self.best_model_path)
        self.best_model_path = ArtifactManager.get_best_model_path(best_model_name)
        joblib.dump(best_model, self.best_model_path)
        print(f"\nBest model saved to: {self.best_model_path}")
        
        return best_model_name, best_model, best_metrics
    
    def run_bias_analysis(self, model_path, X_test, y_test, stage="baseline"):
        """Run bias detection on a model."""
        print("\n" + "="*80)
        print(f" STEP {'4' if stage == 'baseline' else '7'}: BIAS ANALYSIS ({stage.upper()}) ".center(80))
        print("="*80)
        
        detector = BikeShareBiasDetector(
            model_path=model_path,
            X_test=X_test,
            y_test=y_test
        )
        
        bias_report = detector.run_full_analysis()
        
        # Save report with stage prefix
        report_filename = f'bias_detection_report_{stage}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_filename, 'w') as f:
            json.dump(bias_report, f, indent=2, default=str)
        
        print(f"\nBias report saved to: {report_filename}")
        
        # Rename the bias_analysis_plots.png to include stage name
        import os
        if os.path.exists('bias_analysis_plots.png'):
            plot_filename = f'bias_analysis_plots_{stage}.png'
            # Remove existing file if it exists (Windows compatibility)
            if os.path.exists(plot_filename):
                os.remove(plot_filename)
            os.rename('bias_analysis_plots.png', plot_filename)
            print(f"Bias plots saved to: {plot_filename}")
        
        if stage == "baseline":
            self.baseline_bias_report = bias_report
        else:
            self.final_bias_report = bias_report
        
        return bias_report
    
    def apply_optimized_bias_mitigation(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Apply bias mitigation with PROPER quantile handling.
        
        UPDATED: Now includes SAMPLE WEIGHTING for bias reduction!
        
        CRITICAL: Quantiles calculated from TRAINING data only to prevent data leakage.
        """
        print("\n" + "="*80)
        print(" STEP 5: APPLYING BIAS MITIGATION ".center(80))
        print("="*80)
        
        # =========================================================
        # CALCULATE QUANTILES FROM TRAINING DATA ONLY
        # =========================================================
        self.train_quantiles = {}
        
        if 'rides_last_hour' in X_train.columns:
            self.train_quantiles['rides_last_hour_q75'] = float(X_train['rides_last_hour'].quantile(0.75))
            self.train_quantiles['rides_last_hour_q25'] = float(X_train['rides_last_hour'].quantile(0.25))
            self.train_quantiles['rides_last_hour_median'] = float(X_train['rides_last_hour'].median())
        
        print("Training quantiles (used for all splits):")
        for k, v in self.train_quantiles.items():
            print(f"  {k}: {v:.2f}")
        
        # =========================================================
        # APPLY FEATURES USING TRAINING QUANTILES
        # =========================================================
        X_train = self._add_bias_features(X_train, self.train_quantiles, is_training=True)
        X_val = self._add_bias_features(X_val, self.train_quantiles, is_training=False)
        X_test = self._add_bias_features(X_test, self.train_quantiles, is_training=False)
        
        print(f"\nAdded bias-aware features:")
        print(f"  Training:   {len(X_train):,} samples, {X_train.shape[1]} features")
        print(f"  Validation: {len(X_val):,} samples, {X_val.shape[1]} features")
        print(f"  Test:       {len(X_test):,} samples, {X_test.shape[1]} features")
        
        # Save quantiles for monitoring baseline
        self._save_quantiles_for_monitoring()
        
        # =========================================================
        # CALCULATE SAMPLE WEIGHTS
        # =========================================================
        sample_weights = self._calculate_sample_weights(X_train, y_train)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, sample_weights

    def _add_bias_features(self, X, quantiles, is_training=False):
        """
        Add bias-aware features using pre-calculated quantiles.
        
        Args:
            X: DataFrame to add features to
            quantiles: Dict of quantile values from training data
            is_training: Whether this is training data (for logging)
        """
        X = X.copy()
        
        # ---------------------------------------------------------
        # TEMPORAL FEATURES (no leakage - based on hour values)
        # ---------------------------------------------------------
        X['is_hour_8'] = (X['hour'] == 8).astype(int)
        X['is_hour_17_18'] = X['hour'].isin([17, 18]).astype(int)
        
        X['rush_intensity'] = 0.0
        X.loc[X['hour'] == 8, 'rush_intensity'] = 1.0
        X.loc[X['hour'].isin([17, 18]), 'rush_intensity'] = 1.0
        X.loc[X['hour'].isin([7, 9]), 'rush_intensity'] = 0.5
        X.loc[X['hour'].isin([16, 19]), 'rush_intensity'] = 0.5
        
        # ---------------------------------------------------------
        # INTERACTION FEATURES (no leakage)
        # ---------------------------------------------------------
        # Only add if not already present (from feature_generation.py)
        if 'weekday_morning_rush' not in X.columns:
            X['weekday_morning_rush'] = (1 - X['is_weekend']) * X['is_morning_rush']
        if 'weekday_evening_rush' not in X.columns:
            X['weekday_evening_rush'] = (1 - X['is_weekend']) * X['is_evening_rush']
        
        # ---------------------------------------------------------
        # DEMAND FEATURES (using TRAINING quantiles - no leakage)
        # ---------------------------------------------------------
        if 'rides_last_hour' in X.columns and quantiles:
            q75 = quantiles.get('rides_last_hour_q75', 50)
            q25 = quantiles.get('rides_last_hour_q25', 10)
            
            X['high_demand_flag'] = (X['rides_last_hour'] > q75).astype(int)
            X['low_demand_flag'] = (X['rides_last_hour'] < q25).astype(int)
            
            if 'rides_rolling_3h' in X.columns:
                X['demand_volatility'] = np.abs(X['rides_last_hour'] - X['rides_rolling_3h'])
            else:
                X['demand_volatility'] = 0.0
        else:
            X['high_demand_flag'] = 0
            X['low_demand_flag'] = 0
            X['demand_volatility'] = 0.0
        
        # ---------------------------------------------------------
        # COMPOSITE FEATURES
        # ---------------------------------------------------------
        X['problem_period'] = (
            X['is_hour_8'] + X['is_hour_17_18'] +
            X['weekday_morning_rush'] + X['weekday_evening_rush']
        ).clip(0, 1)
        
        # Hour grouping (deterministic, no leakage)
        hour_groups = pd.cut(
            X['hour'], 
            bins=[-0.1, 6, 10, 14, 18, 24],  # -0.1 to include 0
            labels=[0, 1, 2, 3, 4], 
            include_lowest=True
        )
        X['hour_group'] = pd.to_numeric(hour_groups, errors='coerce').fillna(0).astype(int)
        
        return X
    

    def _save_quantiles_for_monitoring(self):
        """Save training quantiles for use in drift monitoring baseline."""
        import json
        from pathlib import Path
        
        quantiles_path = Path("/opt/airflow/models/production/training_quantiles.json")
        quantiles_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(quantiles_path, 'w') as f:
                json.dump(self.train_quantiles, f, indent=2)
            print(f"Training quantiles saved to: {quantiles_path}")
        except Exception as e:
            print(f"Warning: Could not save quantiles: {e}")

    
    def retrain_best_model(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                          sample_weights=None):
        """Retrain the best model with bias mitigation."""
        print("\n" + "="*80)
        print(" STEP 6: RETRAINING WITH BIAS MITIGATION ".center(80))
        print("="*80)
        
        X_train_clean = X_train.drop('date', axis=1, errors='ignore')
        X_val_clean = X_val.drop('date', axis=1, errors='ignore')
        X_test_clean = X_test.drop('date', axis=1, errors='ignore')
        
        with mlflow.start_run(run_name=f"mitigated_{self.best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tag("pipeline_stage", "bias_mitigated")
            mlflow.set_tag("model_type", self.best_model_name)
            mlflow.set_tag("mitigation_strategy", "optimized_multi_strategy")
            mlflow.log_param("has_sample_weights", sample_weights is not None)
            mlflow.log_param("n_features", X_train_clean.shape[1])
            mlflow.log_param("n_samples", len(X_train_clean))
            mlflow.log_param("mitigation_components", "weighting+features+augmentation")
            
            print(f"Retraining {self.best_model_name.upper()} with optimized bias mitigation...")
            
            if self.best_model_name == 'xgboost':
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
                    X_train_clean, y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_train_clean, y_train), (X_val_clean, y_val)],
                    verbose=False
                )
                mlflow.xgboost.log_model(model, "model")
                
            elif self.best_model_name == 'lightgbm':
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
                    X_train_clean, y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val_clean, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                mlflow.lightgbm.log_model(model, "model")
                
            elif self.best_model_name == 'randomforest':
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
                model.fit(X_train_clean, y_train, sample_weight=sample_weights)
                mlflow.sklearn.log_model(model, "model")
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            y_pred_train = model.predict(X_train_clean)
            y_pred_val = model.predict(X_val_clean)
            y_pred_test = model.predict(X_test_clean)
            
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
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            print(f"\nMitigated Model Performance:")
            print(f"  Test MAE: {metrics['test_mae']:.2f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
            print(f"  Test R²: {metrics['test_r2']:.4f}")
            print(f"  Test MAPE: {metrics['test_mape']:.2f}%")
            
            # Save mitigated model
            self.mitigated_model = model
            # self.mitigated_model_path = f"mitigated_model_{self.best_model_name}.pkl"
            self.mitigated_model_path = ArtifactManager.get_mitigated_model_path(self.best_model_name)
            joblib.dump(model, self.mitigated_model_path)

            print(f"\nMitigated model saved to: {self.mitigated_model_path}")
            
            return model, metrics
    
    def compare_results(self):
        """Compare baseline and mitigated model performance."""
        print("\n" + "="*80)
        print(" STEP 8: COMPARING RESULTS ".center(80))
        print("="*80)
        
        if not self.baseline_bias_report or not self.final_bias_report:
            print("Warning: Missing bias reports for comparison")
            return None
        
        baseline_overall = self.baseline_bias_report.get('overall_performance', {})
        final_overall = self.final_bias_report.get('overall_performance', {})
        
        # Helper to safely convert to float
        def to_float(val, default=0.0):
            if val is None or val == 'N/A':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Helper to safely convert to int
        def to_int(val, default=0):
            if val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default
        
        print("\nOverall Performance Comparison:")
        print("="*60)
        
        metrics = ['mae', 'rmse', 'r2', 'mape']
        
        for metric in metrics:
            baseline_val = to_float(baseline_overall.get(metric))
            final_val = to_float(final_overall.get(metric))
            
            if metric == 'r2':
                diff = final_val - baseline_val
                better = diff > 0
                pct_change = (diff / baseline_val * 100) if baseline_val != 0 else 0
            else:
                diff = baseline_val - final_val
                better = diff > 0
                pct_change = (diff / baseline_val * 100) if baseline_val != 0 else 0
            
            symbol = "+" if better else "-"
            print(f"{metric.upper():6s}: {baseline_val:8.4f} -> {final_val:8.4f} "
                f"({symbol}{abs(pct_change):.2f}%)")
        
        print("\nBias Issues Comparison:")
        print("="*60)
        
        baseline_issues = len(self.baseline_bias_report.get('bias_detected', []))
        final_issues = len(self.final_bias_report.get('bias_detected', []))
        
        print(f"Baseline: {baseline_issues} issues detected")
        print(f"Mitigated: {final_issues} issues detected")
        
        reduction = baseline_issues - final_issues
        if reduction > 0:
            print(f"Reduced bias issues by {reduction} ({reduction/baseline_issues*100:.1f}%)")
        elif reduction < 0:
            print(f"Bias issues increased by {abs(reduction)}")
        else:
            print(f"No change in number of bias issues")
        
        # Create comparison report with EXPLICIT type conversions
        # This prevents numpy types from causing JSON serialization issues
        comparison_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'baseline_model': str(self.best_model_path) if self.best_model_path else '',
            'mitigated_model': str(self.mitigated_model_path) if self.mitigated_model_path else '',
            'baseline_metrics': {
                k: to_float(v) for k, v in baseline_overall.items()
            },
            'mitigated_metrics': {
                k: to_float(v) for k, v in final_overall.items()
            },
            'baseline_bias_issues': to_int(baseline_issues),
            'mitigated_bias_issues': to_int(final_issues),
            'improvement': {
                'mae_improvement': to_float(baseline_overall.get('mae')) - to_float(final_overall.get('mae')),
                'rmse_improvement': to_float(baseline_overall.get('rmse')) - to_float(final_overall.get('rmse')),
                'r2_improvement': to_float(final_overall.get('r2')) - to_float(baseline_overall.get('r2')),
                'bias_issues_reduction': to_int(reduction)
            }
        }
        
        # Save comparison report
        comparison_filename = ArtifactManager.get_comparison_report_path(timestamp=True)
        try:
            with open(comparison_filename, 'w') as f:
                json.dump(comparison_report, f, indent=2)
            print(f"\nComparison report saved to: {comparison_filename}")
        except Exception as e:
            print(f"Warning: Could not save comparison report: {e}")
        
        return comparison_report
    
    def run_complete_pipeline(self, models_to_train=None, tune=False):
        """
        Run the complete integrated pipeline.
        """
        ArtifactManager.setup()
        # Step 1: Load data
        X_train, X_test, X_val, y_train, y_test, y_val = self.load_and_prepare_data()
        
        # Step 2: Train models
        results = self.train_all_models(
            X_train, X_test, X_val, y_train, y_test, y_val,
            models_to_train=models_to_train,
            tune=tune
        )
        
        if not results:
            print("No models trained successfully. Exiting.")
            return None
        
        # Step 3: Select best model
        best_model_name, best_model, best_metrics = self.select_best_model(
            results, metric='test_r2'
        )
        
        # Step 4: Bias analysis on baseline model
        baseline_bias_report = self.run_bias_analysis(
            self.best_model_path, X_test, y_test, stage="baseline"
        )
        
        # Step 5: Apply optimized bias mitigation
        X_train_mit, X_val_mit, X_test_mit, y_train_mit, y_val_mit, y_test_mit, sample_weights = \
            self.apply_optimized_bias_mitigation(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
        
        # Step 6: Retrain with mitigation
        mitigated_model, mitigated_metrics = self.retrain_best_model(
            X_train_mit, y_train_mit, X_val_mit, y_val_mit, X_test_mit, y_test_mit,
            sample_weights=sample_weights
        )
        
        # Step 7: Bias analysis on mitigated model
        final_bias_report = self.run_bias_analysis(
            self.mitigated_model_path, X_test_mit, y_test_mit, stage="mitigated"
        )

        sensitivity_report = run_sensitivity_analysis(
            self.best_model_path, X_test, y_test, stage="baseline"
        )
        
        # Step 8: Compare results
        comparison_report = self.compare_results()
        
        print("\n" + "="*80)
        print(" PIPELINE COMPLETE ".center(80))
        print("="*80)
        print(f"\nBaseline model: {self.best_model_path}")
        print(f"Mitigated model: {self.mitigated_model_path}")
        print(f"\nView detailed results in MLflow UI:")
        print("  $ mlflow ui --port 5000")
        print("  Then open: http://localhost:5000")
        
        return {
            'baseline_model': self.best_model,
            'baseline_metrics': best_metrics,
            'baseline_bias_report': baseline_bias_report,
            'mitigated_model': mitigated_model,
            'mitigated_metrics': mitigated_metrics,
            'final_bias_report': final_bias_report,
            'comparison_report': comparison_report
        }


def main():
    """Main entry point."""
    
    print("="*80)
    print(" BLUEBIKES DEMAND PREDICTION - INTEGRATED PIPELINE ".center(80))
    print("="*80)
    
    trainer = IntegratedBlueBikesTrainer(
        experiment_name="bluebikes_integrated_pipeline_v1"
    )
    
    # Configuration
    models_to_train = ['xgboost', 'lightgbm', 'randomforest']
    tune_hyperparameters = True
    
    print("\nConfiguration:")
    print(f"  Models: {', '.join(models_to_train)}")
    print(f"  Hyperparameter Tuning: {'Enabled' if tune_hyperparameters else 'Disabled'}")
    print(f"  Bias Mitigation: Feature Engineering")
    
    # Run complete pipeline
    results = trainer.run_complete_pipeline(
        models_to_train=models_to_train,
        tune=tune_hyperparameters
    )
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
