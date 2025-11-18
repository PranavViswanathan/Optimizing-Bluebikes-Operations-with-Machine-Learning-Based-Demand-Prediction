from __future__ import annotations
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add these imports for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    
    def create_residual_plots(self, y_true, y_pred, model_name):
        """Create residual analysis plots"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title(f'{model_name} - Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of Residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'{model_name} - Residual Distribution')
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{model_name} - Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        axes[1, 1].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                        'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title(f'{model_name} - Actual vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Model Diagnostics', fontsize=14, y=1.02)
        plt.tight_layout()
        
        return fig
    
    def create_feature_importance_plot(self, model, feature_names, model_name, top_n=20):
        """Create feature importance plot for tree-based models"""
        try:
            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'feature_importance'):
                # For LightGBM
                importance = model.feature_importance(importance_type='gain')
            else:
                return None
                
            # Create DataFrame and sort
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
            ax.set_title(f'{model_name} - Top {top_n} Feature Importances')
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Feature')
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Could not create feature importance plot: {e}")
            return None
    
    def create_error_distribution_plot(self, results):
        """Create error distribution comparison across models"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data for box plots
        errors_data = []
        model_names = []
        
        for model_name in results.keys():
            # You'll need to store predictions in your training functions
            # For now, using stored metrics
            metrics = results[model_name][1]
            errors_data.append([
                metrics['test_mae'],
                metrics['test_rmse'],
                metrics.get('test_mape', 0)
            ])
            model_names.append(model_name.upper())
        
        # Error metrics comparison
        metrics_df = pd.DataFrame(errors_data, 
                                 columns=['MAE', 'RMSE', 'MAPE'],
                                 index=model_names)
        
        metrics_df.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Error Metrics Comparison')
        axes[0].set_ylabel('Error Value')
        axes[0].set_xlabel('Model')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # R² Score comparison
        r2_scores = [results[model][1]['test_r2'] for model in results.keys()]
        axes[1].bar(model_names, r2_scores, color='green', alpha=0.7)
        axes[1].set_title('R² Score Comparison')
        axes[1].set_ylabel('R² Score')
        axes[1].set_xlabel('Model')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(r2_scores):
            axes[1].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.suptitle('Model Performance Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        
        return fig
    
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
            
            # Log data statistics
            mlflow.log_metric("target_mean", float(y_train.mean()))
            mlflow.log_metric("target_std", float(y_train.std()))
            mlflow.log_metric("target_min", float(y_train.min()))
            mlflow.log_metric("target_max", float(y_train.max()))
     
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
                    
                    # Generate predictions for visualization
                    y_pred_test = model.predict(X_test)
                    y_pred_train = model.predict(X_train)
                    
                    # Create and log residual plots
                    residual_fig = self.create_residual_plots(y_test, y_pred_test, model_name.upper())
                    if residual_fig:
                        mlflow.log_figure(residual_fig, f"{model_name}_residuals.png")
                        plt.close(residual_fig)
                    
                    # Create and log feature importance plot
                    feature_imp_fig = self.create_feature_importance_plot(
                        model, X_train.columns, model_name.upper()
                    )
                    if feature_imp_fig:
                        mlflow.log_figure(feature_imp_fig, f"{model_name}_feature_importance.png")
                        plt.close(feature_imp_fig)
                    
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
                
                # Create and log comparison plots
                self.create_comparison_plot(results)
                
                # Create and log error distribution plot
                error_dist_fig = self.create_error_distribution_plot(results)
                if error_dist_fig:
                    mlflow.log_figure(error_dist_fig, "error_distribution_comparison.png")
                    plt.close(error_dist_fig)
                
        
        return results
    
    def create_comparison_plot(self, results):
        """Create an enhanced comparison plot of model performances"""
        import matplotlib.pyplot as plt
        
        if not results:
            return
        
        models = list(results.keys())
        metrics_names = ['test_mae', 'test_rmse', 'test_r2', 'test_mape']
        
        # Create a more comprehensive comparison plot
        fig = plt.figure(figsize=(16, 10))
        
        # Create a 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Individual metric plots
        for idx, metric in enumerate(metrics_names):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
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
                
                bars = ax.bar(labels, values, color=colors)
                ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Combined normalized comparison (spider/radar chart style)
        ax_spider = fig.add_subplot(gs[:, 2], projection='polar')
        
        # Normalize metrics for spider chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model_name in models:
            values = []
            for metric in metrics_names:
                if metric in results[model_name][1]:
                    val = results[model_name][1][metric]
                    # Normalize (0-1 scale)
                    if metric == 'test_r2':
                        values.append(val)  # R2 is already 0-1
                    else:
                        # Invert error metrics (lower is better)
                        max_val = max([results[m][1].get(metric, 0) for m in models])
                        values.append(1 - (val / max_val) if max_val > 0 else 0)
            
            values += values[:1]  # Complete the circle
            ax_spider.plot(angles, values, 'o-', linewidth=2, label=model_name.upper())
            ax_spider.fill(angles, values, alpha=0.25)
        
        ax_spider.set_xticks(angles[:-1])
        ax_spider.set_xticklabels([m.upper() for m in metrics_names])
        ax_spider.set_ylim(0, 1)
        ax_spider.set_title('Normalized Performance Comparison\n(Higher is Better)', 
                           fontsize=12, fontweight='bold', pad=20)
        ax_spider.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax_spider.grid(True)
        
        plt.suptitle('Model Performance Comparison Dashboard', fontsize=16, fontweight='bold')
        
        mlflow.log_figure(fig, "model_comparison_dashboard.png")
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
        if register == 'y':
            trainer.register_model(best_model_name, best_run_id, best_model_name)
    
    print("PIPELINE COMPLETE")
    print("\nView detailed results in MLflow UI:")
    print("   $ mlflow ui --port 5000")
    print("   Then open: http://localhost:5000")
    print(f"\nExperiment: {trainer.experiment_name}")
    
    return results


if __name__ == "__main__":
    results = main()