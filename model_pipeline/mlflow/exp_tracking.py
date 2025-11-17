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
warnings.filterwarnings('ignore')

from train_xgb import train_xgboost, tune_xgboost
from train_lgb import train_lightgbm, tune_lightgbm
# from train_catbst import train_catboost, tune_catboost
from feature_generation import load_and_prepare_data
from train_random_forest import train_random_forest, tune_random_forest


class BlueBikesModelTrainer:
    
    def __init__(self, experiment_name="bluebikes_model_comparison"):
        self.experiment_name = experiment_name
        self.setup_mlflow()
        self.client = MlflowClient()
        self.models_to_train = [] 
        
    def setup_mlflow(self):
        mlflow.set_tracking_uri("./mlruns")
        self.experiment = mlflow.set_experiment(self.experiment_name)
        print(f"MLflow Experiment: {self.experiment_name}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        
    def load_and_prepare_data(self, data_source=None):
        print("LOADING AND PREPARING DATA")
                
        X, y, feature_columns = load_and_prepare_data()
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=42, shuffle=True
        # )
        X["date"] = pd.to_datetime(X["date"])

        train_start = pd.Timestamp("2024-06-01")
        train_end   = pd.Timestamp("2025-06-30")
        test_start  = pd.Timestamp("2025-07-01")
        test_end = pd.Timestamp("2025-07-31")

        train_mask_full = (X["date"] >= train_start) & (X["date"] <= train_end)
        test_mask  = (X["date"] >= test_start) & (X["date"] <= test_end)

        X_train_full = X.loc[train_mask_full].copy()
        y_train_full = y.loc[train_mask_full].copy()
        X_test = X.loc[test_mask].copy()
        y_test = y.loc[test_mask].copy()

        val_start = pd.Timestamp("2025-05-01")

        val_mask = X_train_full['date'] >= val_start
        tr_mask = X_train_full['date'] < val_start

        X_train = X_train_full.loc[tr_mask].drop(columns=["date"])
        y_train = y_train_full.loc[tr_mask]

        X_val   = X_train_full.loc[val_mask].drop(columns=["date"])
        y_val   = y_train_full.loc[val_mask]

        X_test = X_test.drop(columns=["date"])
        
        print(f"Dataset shape: {X.shape}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def train_all_models(self, X_train, X_test, X_val, y_train, y_test, y_val, models_to_train=None, tune=False):
        
        if models_to_train is None:
            models_to_train = ['xgboost', 'lightgbm', 'randomforest']  
        
        print("TRAINING MODELS")
        print(f"Models to train: {models_to_train}")
        
        results = {}       
        with mlflow.start_run(run_name=f"training_session_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_param("models_trained", models_to_train)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
     
            for model_name in models_to_train:
                print(f"\n{'='*40}")
                print(f"Training {model_name.upper()}")
                print(f"{'='*40}")
                
                try:
                    if model_name == 'xgboost':
                        if tune:
                            print("With Hyperparameter Tuning...")
                            model, metrics = tune_xgboost(
                                X_train, y_train, X_val, y_val, X_test, y_test, mlflow
                            )
                            runs = self.client.search_runs(
                                experiment_ids=[self.experiment.experiment_id],
                                filter_string="tags.model_type = 'XGBoost'",
                                order_by=["attribute.start_time DESC"],
                                max_results=1
                            )
                            run_id = runs[0].info.run_id if runs else None
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
                        if tune:
                            print("With Hyperparameter Tuning...")
                            model, metrics = tune_lightgbm(
                                X_train, y_train, X_val, y_val, X_test, y_test, mlflow
                            )
                            runs = self.client.search_runs(
                                experiment_ids=[self.experiment.experiment_id],
                                filter_string="tags.model_type = 'LightGBM'",
                                order_by=["attribute.start_time DESC"],
                                max_results=1
                            )
                            run_id = runs[0].info.run_id if runs else None
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

                    # elif model_name == 'catboost':
                    #     if tune:
                    #         print("With Hyperparameter Tuning...")
                    #         model, metrics = tune_catboost(
                    #             X_train, y_train, X_val, y_val, X_test, y_test, mlflow
                    #         )
                    #         runs = self.client.search_runs(
                    #             experiment_ids=[self.experiment.experiment_id],
                    #             filter_string="tags.model_type = 'CatBoost'",
                    #             order_by=["start_time DESC"],
                    #             max_results=1
                    #         )
                    #         run_id = runs[0].info.run_id if runs else None
                    #     else:
                    #         model, metrics = train_catboost(
                    #             X_train, y_train, X_val, y_val, X_test, y_test, mlflow_client=mlflow
                    #         )
                    #         runs = self.client.search_runs(
                    #             experiment_ids=[self.experiment.experiment_id],
                    #             filter_string="tags.model_type = 'CatBoost'",
                    #             order_by=["start_time DESC"],
                    #             max_results=1
                    #         )
                    #         run_id = runs[0].info.run_id if runs else None
                    
                    elif model_name == 'randomforest':
                        if tune:
                            print("With Hyperparameter Tuning...")
                            model, metrics = tune_random_forest(
                                X_train, y_train, X_val, y_val, X_test, y_test, mlflow
                            )
                            runs = self.client.search_runs(
                                experiment_ids=[self.experiment.experiment_id],
                                filter_string="tags.model_type = 'RandomForest'",
                                order_by=["attribute.start_time DESC"],
                                max_results=1
                            )
                            run_id = runs[0].info.run_id if runs else None
                        else:
                            model, metrics = train_random_forest(
                                X_train, y_train,X_val, y_val, X_test, y_test, mlflow_client=mlflow
                            )
                            runs = self.client.search_runs(
                                experiment_ids=[self.experiment.experiment_id],
                                filter_string="tags.model_type = 'RandomForest'",
                                order_by=["attribute.start_time DESC"],
                                max_results=1
                            )
                            run_id = runs[0].info.run_id if runs else None
                    
                    else:
                        print(f"Warning: Model {model_name} not implemented yet")
                        continue
                    
                    results[model_name] = (model, metrics, run_id)
                    
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
                    
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            if results:
                best_model_name = min(results.keys(), 
                                    key=lambda x: results[x][1]['test_mae'])
                mlflow.log_metric("best_test_mae", results[best_model_name][1]['test_mae'])
                mlflow.log_metric("best_test_r2", results[best_model_name][1]['test_r2'])
                mlflow.set_tag("best_model", best_model_name)
                
                self.create_comparison_plot(results)
        
        return results
    
    def create_comparison_plot(self, results):
        """Create a comparison plot of model performances"""
        import matplotlib.pyplot as plt
        
        if not results:
            return
        
        models = list(results.keys())
        metrics_names = ['test_mae', 'test_rmse', 'test_r2', 'test_mape']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics_names):
            values = []
            labels = []
            for model_name in models:
                if metric in results[model_name][1]:
                    values.append(results[model_name][1][metric])
                    labels.append(model_name.upper())
            
            if values:
                colors = ['green' if v == min(values) and metric != 'test_r2' else 
                         'green' if v == max(values) and metric == 'test_r2' else 'steelblue' 
                         for v in values]
                
                axes[idx].bar(labels, values, color=colors)
                axes[idx].set_title(f'{metric.upper()}')
                axes[idx].set_ylabel('Score')
                axes[idx].grid(True, alpha=0.3)
                
                for i, v in enumerate(values):
                    axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Model Comparison', fontsize=16)
        plt.tight_layout()
        mlflow.log_figure(fig, "model_comparison.png")
        plt.close()
    
    def compare_models(self, results):
        print("MODEL COMPARISON")

        comparison_data = []
        for model_name, (model, metrics, run_id) in results.items():
            comparison_data.append({
                'model': model_name,
                'test_mae': metrics['test_mae'],
                'test_rmse': metrics['test_rmse'],
                'test_r2': metrics['test_r2'],
                'test_mape': metrics.get('test_mape', np.nan),
                'train_mae': metrics['train_mae'],
                'train_r2': metrics['train_r2'],
                'run_id': run_id
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('test_mae')
        
        print("\nModel Performance Ranking (by Test MAE):")
        print("="*80)
        for idx, row in comparison_df.iterrows():
            print(f"{idx+1}. {row['model'].upper()}")
            print(f"   Test MAE: {row['test_mae']:.2f} | "
                  f"RMSE: {row['test_rmse']:.2f} | "
                  f"R²: {row['test_r2']:.4f} | "
                  f"MAPE: {row['test_mape']:.2f}%")
            print(f"   Train R²: {row['train_r2']:.4f} | "
                  f"Train MAE: {row['train_mae']:.2f}")
            print(f"   Run ID: {row['run_id']}")
            print()
        
        best_r2_model = comparison_df.loc[comparison_df['test_r2'].idxmax()]
        print(f"\nBest Model by R²: {best_r2_model['model'].upper()} (R² = {best_r2_model['test_r2']:.4f})")

        best_mae_model = comparison_df.iloc[0]
        print(f"Best Model by MAE: {best_mae_model['model'].upper()} (MAE = {best_mae_model['test_mae']:.2f})")
        
        return comparison_df
    
    def select_best_model(self, results, metric='test_mae'):
        
        if metric == 'test_r2':
            best_model_name = max(results.keys(), 
                                key=lambda x: results[x][1][metric])
        else:
            best_model_name = min(results.keys(), 
                                key=lambda x: results[x][1][metric])
        
        best_model, best_metrics, best_run_id = results[best_model_name]
        
        print("BEST MODEL SELECTED")
        print(f"Model: {best_model_name.upper()}")
        print(f"Selection Metric: {metric}")
        print(f"\nPerformance:")
        print(f"  Test MAE: {best_metrics['test_mae']:.2f}")
        print(f"  Test RMSE: {best_metrics['test_rmse']:.2f}")
        print(f"  Test R²: {best_metrics['test_r2']:.4f}")
        print(f"  Test MAPE: {best_metrics.get('test_mape', 'N/A')}")
        print(f"\nRun ID: {best_run_id}")
        
        return best_model_name, best_model, best_metrics, best_run_id
    
    def register_model(self, model_name, run_id, model_type):
        print("REGISTERING MODEL")
        
        try:
            model_uri = f"runs:/{run_id}/model"
            registered_model_name = f"bluebikes_{model_type}_model"
            try:
                self.client.create_registered_model(registered_model_name)
                print(f"Created new registered model: {registered_model_name}")
            except:
                print(f"Using existing registered model: {registered_model_name}")
            model_version = self.client.create_model_version(
                name=registered_model_name,
                source=model_uri,
                run_id=run_id,
                description=f"{model_type.upper()} model trained on {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            print(f"Model registered as version {model_version.version}")
            self.client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version.version,
                stage="Staging"
            )
            print(f"Model version {model_version.version} moved to Staging")
            
            return model_version
            
        except Exception as e:
            print(f"Error registering model: {e}")
            return None


def main():
    
    print(" BLUEBIKES DEMAND PREDICTION - MODEL TRAINING PIPELINE ".center(80))
    
    trainer = BlueBikesModelTrainer(
        experiment_name="bluebikes_model_comparison_v3"
    )
    
    X_train, X_test, X_val, y_train, y_test, y_val = trainer.load_and_prepare_data()
    models_to_train = ['xgboost', 'lightgbm', 'randomforest']  # removed 'catboost'
    results = trainer.train_all_models(
        X_train, X_test, X_val, y_train, y_test, y_val,
        models_to_train=models_to_train, tune=True
    )
    
    if results:
        comparison_df = trainer.compare_models(results)
        best_model_name, best_model, best_metrics, best_run_id = trainer.select_best_model(
            results, 
            metric='test_r2'
        )
        
        
        comparison_df.to_csv("model_comparison.csv", index=False)
        print(f"\n Comparison saved to: model_comparison.csv")
        
        import joblib
        joblib.dump(best_model, f"best_model_{best_model_name}.pkl")
        print(f"Best model saved to: best_model_{best_model_name}.pkl")
        
        register = input("\nRegister the best model for deployment? (y/n): ").lower()
        trainer.register_model(best_model_name, best_run_id, best_model_name)
    
    print("PIPELINE COMPLETE")
    print("\nView detailed results in MLflow UI:")
    print("   $ mlflow ui --port 5000")
    print("   Then open: http://localhost:5000")
    print(f"\nExperiment: {trainer.experiment_name}")
    
    detector = BikeShareBiasDetector(
        model_path=f"best_model_{best_model_name}.pkl",
        X_test=X_test,
        y_test=y_test
    )
    
    # Run full bias analysis
    bias_report = detector.run_full_analysis()
    
    print("Bias detection complete. Check generated reports and visualizations.")


    return results


if __name__ == "__main__":
    results = main()