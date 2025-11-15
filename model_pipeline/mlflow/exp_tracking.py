"""
Main Training Script for BlueBikes Demand Prediction
This script orchestrates training multiple models and selecting the best one
"""

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

# Import model modules
from train_xgb import train_xgboost
from feature_generation import load_and_prepare_data
# from models.lightgbm_model import train_lightgbm  # Add when created
# from models.random_forest_model import train_random_forest  # Add when created
# from models.neural_network_model import train_neural_network  # Add when created

# Import your feature extraction
# from feature_extraction import prepare_bluebikes_features


class BlueBikesModelTrainer:
    """Main class for training and comparing models"""
    
    def __init__(self, experiment_name="bluebikes_model_comparison"):
        """Initialize the trainer with MLflow setup"""
        
        self.experiment_name = experiment_name
        self.setup_mlflow()
        self.client = MlflowClient()
        self.models_to_train = []  # Will be populated based on config
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("./mlruns")
        self.experiment = mlflow.set_experiment(self.experiment_name)
        print(f"MLflow Experiment: {self.experiment_name}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        
    def load_and_prepare_data(self, data_source=None):
        """
        Load and prepare the data for training
        
        Args:
            data_source: Path to data or None to use feature extraction function
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        
        print("\n" + "="*60)
        print("LOADING AND PREPARING DATA")
        print("="*60)
                
        X, y, feature_columns = load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Dataset shape: {X.shape}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
        
        return X_train, X_test, y_train, y_test
    
    def train_all_models(self, X_train, X_test, y_train, y_test, models_to_train=None):
        """
        Train all specified models
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
            models_to_train: List of model names to train
        
        Returns:
            Dict of {model_name: (model, metrics, run_id)}
        """
        
        if models_to_train is None:
            models_to_train = ['xgboost']  # Default to just XGBoost for now
            # Add more as you create them: ['xgboost', 'lightgbm', 'random_forest']
        
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        print(f"Models to train: {models_to_train}")
        
        results = {}
        
        # Start parent run for this training session
        with mlflow.start_run(run_name=f"training_session_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            
            # Log session information
            mlflow.log_param("models_trained", models_to_train)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Train each model
            for model_name in models_to_train:
                print(f"\n{'='*40}")
                print(f"Training {model_name.upper()}")
                print(f"{'='*40}")
                
                try:
                    if model_name == 'xgboost':
                        model, metrics = train_xgboost(
                            X_train, y_train, X_test, y_test, mlflow
                        )
                    
                    # Add more models as you create them:
                    # elif model_name == 'lightgbm':
                    #     model, metrics = train_lightgbm(
                    #         X_train, y_train, X_test, y_test, mlflow
                    #     )
                    # elif model_name == 'random_forest':
                    #     model, metrics = train_random_forest(
                    #         X_train, y_train, X_test, y_test, mlflow
                    #     )
                    
                    else:
                        print(f"Warning: Model {model_name} not implemented yet")
                        continue
                    
                    # Store results
                    run_id = mlflow.active_run().info.run_id
                    results[model_name] = (model, metrics, run_id)
                    
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    continue
            
            # Log comparison metrics
            if results:
                best_model_name = min(results.keys(), 
                                    key=lambda x: results[x][1]['test_mae'])
                mlflow.log_metric("best_test_mae", results[best_model_name][1]['test_mae'])
                mlflow.set_tag("best_model", best_model_name)
        
        return results
    
    def compare_models(self, results):
        """
        Compare trained models and display results
        
        Args:
            results: Dict of model results
        
        Returns:
            DataFrame with comparison
        """
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for model_name, (model, metrics, run_id) in results.items():
            comparison_data.append({
                'model': model_name,
                'test_mae': metrics['test_mae'],
                'test_rmse': metrics['test_rmse'],
                'test_r2': metrics['test_r2'],
                'test_mape': metrics.get('test_mape', np.nan),
                'train_mae': metrics['train_mae'],
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
                  f"R²: {row['test_r2']:.4f}")
            print(f"   Run ID: {row['run_id']}")
            print()
        
        return comparison_df
    
    def select_best_model(self, results, metric='test_mae'):
        """
        Select the best model based on specified metric
        
        Args:
            results: Dict of model results
            metric: Metric to use for selection
        
        Returns:
            best_model_name, best_model, best_metrics, best_run_id
        """
        
        best_model_name = min(results.keys(), 
                            key=lambda x: results[x][1][metric])
        best_model, best_metrics, best_run_id = results[best_model_name]
        
        print("\n" + "="*60)
        print("BEST MODEL SELECTED")
        print("="*60)
        print(f"Model: {best_model_name.upper()}")
        print(f"Test MAE: {best_metrics['test_mae']:.2f}")
        print(f"Test RMSE: {best_metrics['test_rmse']:.2f}")
        print(f"Test R²: {best_metrics['test_r2']:.4f}")
        print(f"Run ID: {best_run_id}")
        
        return best_model_name, best_model, best_metrics, best_run_id
    
    def register_model(self, model_name, run_id, model_type):
        """
        Register the best model in MLflow Model Registry
        
        Args:
            model_name: Name for the registered model
            run_id: MLflow run ID
            model_type: Type of model (xgboost, lightgbm, etc.)
        """
        
        print("\n" + "="*60)
        print("REGISTERING MODEL")
        print("="*60)
        
        try:
            # Create model URI
            model_uri = f"runs:/{run_id}/model"
            
            # Register model
            registered_model_name = f"bluebikes_{model_type}_model"
            
            # Create registered model if it doesn't exist
            try:
                self.client.create_registered_model(registered_model_name)
                print(f"Created new registered model: {registered_model_name}")
            except:
                print(f"Using existing registered model: {registered_model_name}")
            
            # Create new version
            model_version = self.client.create_model_version(
                name=registered_model_name,
                source=model_uri,
                run_id=run_id,
                description=f"{model_type.upper()} model trained on {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            print(f"✅ Model registered as version {model_version.version}")
            
            # Transition to staging
            self.client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version.version,
                stage="Staging"
            )
            print(f"📤 Model version {model_version.version} moved to Staging")
            
            return model_version
            
        except Exception as e:
            print(f"❌ Error registering model: {e}")
            return None


def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print(" BLUEBIKES DEMAND PREDICTION - MODEL TRAINING PIPELINE ".center(80))
    print("="*80)
    
    # Initialize trainer
    trainer = BlueBikesModelTrainer(
        experiment_name="bluebikes_model_comparison_v2"
    )
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = trainer.load_and_prepare_data()
    
    # Define which models to train
    models_to_train = ['xgboost']  # Add more as you implement them
    # models_to_train = ['xgboost', 'lightgbm', 'random_forest', 'neural_network']
    
    # Train all models
    results = trainer.train_all_models(
        X_train, X_test, y_train, y_test,
        models_to_train=models_to_train
    )
    
    if results:
        # Compare models
        comparison_df = trainer.compare_models(results)
        
        # Select best model
        best_model_name, best_model, best_metrics, best_run_id = trainer.select_best_model(results)
        
        # Save comparison results
        comparison_df.to_csv("model_comparison.csv", index=False)
        print(f"\n📊 Comparison saved to: model_comparison.csv")
        
        # Optionally register the best model
        register = input("\n📦 Register the best model for deployment? (y/n): ").lower()
        if register == 'y':
            trainer.register_model(best_model_name, best_run_id, best_model_name)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\n📊 View detailed results in MLflow UI:")
    print("   $ mlflow ui --port 5000")
    print("   Then open: http://localhost:5000")
    print(f"\n🔍 Experiment: {trainer.experiment_name}")
    
    return results


if __name__ == "__main__":
    results = main()