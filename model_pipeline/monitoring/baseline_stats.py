"""
Baseline Statistics Generator for BlueBikes Model Monitoring
Creates and manages reference data baselines for Evidently AI drift detection.

Captures:
- Reference data (training/test data) for comparison
- Feature statistics for quick checks
- Model performance baseline
- Prediction distributions
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle
import joblib
import logging

from monitoring_config import (
    MonitoringConfig,
    get_config,
    get_baseline_path,
    BASELINES_DIR,
    PRODUCTION_MODEL_PATH,
    PRODUCTION_METADATA_PATH,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineGenerator:
    """
    Generates and manages baseline data for Evidently AI monitoring.
    
    Stores:
    - Reference DataFrame (sampled from training data)
    - Feature statistics summary
    - Model performance metrics
    - Prediction distribution baseline
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or get_config()
        self.baseline: Dict[str, Any] = {}
        
    def generate_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model: Any,
        model_metrics: Dict[str, float],
        model_version: int = 1,
        model_name: str = "unknown",
        sample_size: int = 5000,
        feature_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate complete baseline from training data and model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features  
            y_test: Test target
            model: Trained model object
            model_metrics: Performance metrics from training
            model_version: Version number for this baseline
            model_name: Name of the model (xgboost, lightgbm, etc.)
            sample_size: Number of samples to store as reference (for memory efficiency)
            feature_columns: List of feature columns the model expects (optional)
            
        Returns:
            Dictionary containing baseline data and statistics
        """
        logger.info("="*60)
        logger.info("GENERATING BASELINE FOR EVIDENTLY AI")
        logger.info("="*60)
        
        # Drop date column for model operations
        X_train_clean = X_train.drop('date', axis=1, errors='ignore')
        X_test_clean = X_test.drop('date', axis=1, errors='ignore')
        
        # If feature_columns provided, ensure correct column order
        if feature_columns:
            # Filter to only columns that exist
            available_cols = [c for c in feature_columns if c in X_train_clean.columns]
            X_train_clean = X_train_clean[available_cols]
            X_test_clean = X_test_clean[available_cols]
            logger.info(f"Using {len(available_cols)} features from feature_columns")
        
        # Sample reference data (use test set as reference for drift detection)
        # This is what we compare production data against
        if len(X_test) > sample_size:
            sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
            reference_data = X_test_clean.iloc[sample_idx].copy()
            reference_target = y_test.iloc[sample_idx].copy()
        else:
            reference_data = X_test_clean.copy()
            reference_target = y_test.copy()
        
        # Add predictions to reference data
        reference_preds = model.predict(reference_data)
        reference_data = reference_data.copy()
        reference_data['prediction'] = reference_preds
        reference_data[self.config.features.target_column] = reference_target.values
        
        # Generate predictions on full test set for statistics
        y_pred_test = model.predict(X_test_clean)
        
        baseline = {
            "metadata": {
                "version": model_version,
                "model_name": model_name,
                "created_at": datetime.now().isoformat(),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "reference_samples": len(reference_data),
                "n_features": X_train_clean.shape[1],
                "feature_names": list(X_train_clean.columns),
            },
            "reference_data": reference_data,  # DataFrame for Evidently
            "feature_stats": self._compute_feature_stats(X_train_clean),
            "target_stats": self._compute_target_stats(y_train),
            "prediction_stats": self._compute_prediction_stats(y_pred_test),
            "performance_baseline": self._normalize_metrics(model_metrics),
        }
        
        self.baseline = baseline
        
        logger.info(f"Baseline generated for model v{model_version}")
        logger.info(f"  Reference samples: {len(reference_data)}")
        logger.info(f"  Features tracked: {len(baseline['feature_stats'])}")
        logger.info(f"  Test R²: {model_metrics.get('test_r2', 'N/A')}")
        
        return baseline
    
    def _compute_feature_stats(self, X: pd.DataFrame) -> Dict[str, Dict]:
        """Compute summary statistics for each feature."""
        stats = {}
        
        for col in X.columns:
            if col in self.config.features.skip_features:
                continue
                
            col_data = X[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            if col in self.config.features.categorical_features:
                # Categorical statistics
                value_counts = col_data.value_counts(normalize=True)
                stats[col] = {
                    "type": "categorical",
                    "proportions": {str(k): float(v) for k, v in value_counts.items()},
                    "n_unique": int(col_data.nunique()),
                    "n_samples": len(col_data),
                }
            else:
                # Numerical statistics
                stats[col] = {
                    "type": "numerical",
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "q25": float(col_data.quantile(0.25)),
                    "q75": float(col_data.quantile(0.75)),
                    "n_samples": len(col_data),
                }
        
        return stats
    
    def _compute_target_stats(self, y: pd.Series) -> Dict:
        """Compute statistics for the target variable."""
        y_clean = y.dropna()
        
        return {
            "mean": float(y_clean.mean()),
            "std": float(y_clean.std()),
            "min": float(y_clean.min()),
            "max": float(y_clean.max()),
            "median": float(y_clean.median()),
            "n_samples": len(y_clean),
        }
    
    def _compute_prediction_stats(self, predictions: np.ndarray) -> Dict:
        """Compute statistics for model predictions."""
        preds = predictions.flatten()
        
        return {
            "mean": float(np.mean(preds)),
            "std": float(np.std(preds)),
            "min": float(np.min(preds)),
            "max": float(np.max(preds)),
            "median": float(np.median(preds)),
            "q10": float(np.percentile(preds, 10)),
            "q90": float(np.percentile(preds, 90)),
            "n_predictions": len(preds),
        }
    
    def _normalize_metrics(self, metrics: Dict) -> Dict[str, float]:
        """Normalize metrics to ensure serializability."""
        normalized = {}
        
        for key, value in metrics.items():
            try:
                if isinstance(value, (np.floating, np.integer)):
                    normalized[key] = float(value)
                elif isinstance(value, (int, float)):
                    normalized[key] = float(value)
                else:
                    normalized[key] = float(value)
            except (TypeError, ValueError):
                logger.warning(f"Could not normalize metric {key}")
        
        return normalized
    
    def save_baseline(
        self, 
        path: Optional[Path] = None, 
        version: Optional[int] = None
    ) -> Path:
        """Save baseline to pickle file (includes DataFrame)."""
        if not self.baseline:
            raise ValueError("No baseline generated. Run generate_baseline() first.")
        
        if path is None:
            path = get_baseline_path(version)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle (to preserve DataFrame)
        with open(path, 'wb') as f:
            pickle.dump(self.baseline, f)
        
        # Also save as current baseline
        current_path = get_baseline_path(None)
        with open(current_path, 'wb') as f:
            pickle.dump(self.baseline, f)
        
        # Save metadata as JSON for easy inspection
        metadata_path = path.parent / f"baseline_v{version}_metadata.json"
        metadata = {k: v for k, v in self.baseline.items() if k != 'reference_data'}
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Baseline saved to: {path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return path
    
    @staticmethod
    def load_baseline(path: Optional[Path] = None) -> Dict:
        """Load baseline from pickle file."""
        if path is None:
            path = get_baseline_path(None)
        
        if not path.exists():
            raise FileNotFoundError(f"Baseline not found at {path}")
        
        with open(path, 'rb') as f:
            baseline = pickle.load(f)
        
        logger.info(f"Loaded baseline v{baseline['metadata']['version']} from {path}")
        logger.info(f"  Reference samples: {len(baseline.get('reference_data', []))}")
        
        return baseline


class BaselineManager:
    """Manages multiple baseline versions."""
    
    def __init__(self):
        self.baselines_dir = BASELINES_DIR
    
    def list_baselines(self) -> List[Dict]:
        """List all available baselines."""
        baselines = []
        
        for path in self.baselines_dir.glob("baseline_v*.pkl"):
            try:
                # Load just metadata to avoid loading full DataFrame
                metadata_path = path.parent / f"{path.stem}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                    baselines.append({
                        "version": data["metadata"]["version"],
                        "created_at": data["metadata"]["created_at"],
                        "model_name": data["metadata"].get("model_name", "unknown"),
                        "reference_samples": data["metadata"].get("reference_samples", 0),
                        "path": str(path),
                    })
                else:
                    # Fallback: load pickle
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    baselines.append({
                        "version": data["metadata"]["version"],
                        "created_at": data["metadata"]["created_at"],
                        "model_name": data["metadata"].get("model_name", "unknown"),
                        "path": str(path),
                    })
            except Exception as e:
                logger.warning(f"Could not read baseline {path}: {e}")
        
        return sorted(baselines, key=lambda x: x["version"], reverse=True)
    
    def get_latest_version(self) -> int:
        """Get the latest baseline version number."""
        baselines = self.list_baselines()
        if not baselines:
            return 0
        return baselines[0]["version"]
    
    def cleanup_old_baselines(self, keep_count: int = 5):
        """Remove old baselines, keeping the most recent N."""
        baselines = self.list_baselines()
        
        if len(baselines) <= keep_count:
            return
        
        to_remove = baselines[keep_count:]
        
        for baseline in to_remove:
            try:
                path = Path(baseline["path"])
                path.unlink()
                
                # Also remove metadata JSON
                metadata_path = path.parent / f"{path.stem}_metadata.json"
                if metadata_path.exists():
                    metadata_path.unlink()
                
                logger.info(f"Removed old baseline: {path}")
            except Exception as e:
                logger.warning(f"Could not remove {baseline['path']}: {e}")


def generate_baseline_from_training(
    model_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> Path:
    """
    Generate baseline from existing production model and training data.
    
    Called after model training/deployment to create monitoring baseline.
    Uses IntegratedBlueBikesTrainer to ensure feature consistency with trained model.
    """
    import sys
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline')
    
    logger.info("="*60)
    logger.info("GENERATING BASELINE FROM PRODUCTION MODEL")
    logger.info("="*60)
    
    # Use defaults if not provided
    if model_path is None:
        model_path = str(PRODUCTION_MODEL_PATH)
    if metadata_path is None:
        metadata_path = str(PRODUCTION_METADATA_PATH)
    
    # Load model metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Model: {metadata.get('model_type', 'unknown')}")
    logger.info(f"Version: {metadata.get('version', 1)}")
    
    # Load model
    model = joblib.load(model_path)
    
    # Get expected features from model
    try:
        # XGBoost
        if hasattr(model, 'get_booster'):
            expected_features = model.get_booster().feature_names
        # LightGBM
        elif hasattr(model, 'feature_name_'):
            expected_features = model.feature_name_
        # Sklearn models
        elif hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        else:
            expected_features = None
            logger.warning("Could not extract feature names from model")
    except Exception as e:
        expected_features = None
        logger.warning(f"Error extracting feature names: {e}")
    
    if expected_features:
        logger.info(f"Model expects {len(expected_features)} features")
        logger.info(f"Features: {expected_features[:5]}... (showing first 5)")
    
    # Load data using IntegratedBlueBikesTrainer to ensure feature consistency
    # This includes all bias mitigation features
    from integrated_training_pipeline import IntegratedBlueBikesTrainer
    
    logger.info("Loading data via IntegratedBlueBikesTrainer for feature consistency...")
    trainer = IntegratedBlueBikesTrainer(experiment_name='baseline_generation_temp')
    
    # load_and_prepare_data returns 6 values: X_train, X_test, X_val, y_train, y_test, y_val
    X_train, X_test, X_val, y_train, y_test, y_val = trainer.load_and_prepare_data()
    
    logger.info(f"Loaded data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Features: {X_train.shape[1]} columns")
    
    # If we have expected features, filter and reorder columns to match model
    if expected_features:
        # Check for missing features
        available_features = [f for f in expected_features if f in X_train.columns]
        missing_features = [f for f in expected_features if f not in X_train.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features}")
            # Add missing features with default values to all splits
            for feat in missing_features:
                X_train[feat] = 0
                X_test[feat] = 0
                X_val[feat] = 0
                logger.info(f"  Added missing feature '{feat}' with default value 0")
        
        # Reorder columns to match model's expected order (keep date if present)
        cols_to_keep = [c for c in expected_features if c in X_train.columns]
        if 'date' in X_train.columns and 'date' not in cols_to_keep:
            cols_to_keep = ['date'] + cols_to_keep
        
        X_train = X_train[cols_to_keep]
        X_test = X_test[cols_to_keep]
        X_val = X_val[cols_to_keep]
        logger.info(f"Filtered to {len(cols_to_keep)} columns to match model")
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Get version and metrics
    version = metadata.get("version", 1)
    model_name = metadata.get("model_type", "unknown")
    metrics = metadata.get("metrics", {})
    
    # Generate baseline
    generator = BaselineGenerator()
    baseline = generator.generate_baseline(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        model_metrics=metrics,
        model_version=version,
        model_name=model_name,
        sample_size=5000,
        feature_columns=expected_features
    )
    
    # Save baseline
    path = generator.save_baseline(version=version)
    
    logger.info("="*60)
    logger.info("BASELINE GENERATION COMPLETE")
    logger.info("="*60)
    
    return path


def generate_baseline_after_training(
    trainer,  # IntegratedBlueBikesTrainer instance
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: Dict[str, float],
    version: int,
    feature_columns: Optional[List[str]] = None
) -> Path:
    """
    Generate baseline immediately after training completes.
    
    This is the PREFERRED method as it uses the exact same data and features
    that were used during training, avoiding any feature mismatch issues.
    
    Called from integrated_training_pipeline.py after model is trained.
    
    Args:
        trainer: IntegratedBlueBikesTrainer instance with trained model
        X_train: Training features (exactly as used in training)
        y_train: Training target
        X_test: Test features (exactly as used in training)
        y_test: Test target
        metrics: Performance metrics from training
        version: Model version number
        feature_columns: List of feature columns used by model
        
    Returns:
        Path to saved baseline file
    """
    logger.info("="*60)
    logger.info("GENERATING BASELINE FROM TRAINING RUN")
    logger.info("="*60)
    logger.info("Using exact training data - guaranteed feature consistency")
    
    model = trainer.mitigated_model if trainer.mitigated_model else trainer.best_model
    model_name = trainer.best_model_name
    
    # Get feature columns from model if not provided
    if feature_columns is None:
        try:
            if hasattr(model, 'get_booster'):
                feature_columns = model.get_booster().feature_names
            elif hasattr(model, 'feature_name_'):
                feature_columns = model.feature_name_
            elif hasattr(model, 'feature_names_in_'):
                feature_columns = list(model.feature_names_in_)
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
    
    if feature_columns:
        logger.info(f"Model uses {len(feature_columns)} features")
    
    generator = BaselineGenerator()
    baseline = generator.generate_baseline(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        model_metrics=metrics,
        model_version=version,
        model_name=model_name,
        feature_columns=feature_columns
    )
    
    path = generator.save_baseline(version=version)
    
    logger.info("="*60)
    logger.info("BASELINE GENERATION COMPLETE")
    logger.info("="*60)
    
    return path


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Statistics Generator")
    parser.add_argument("--generate", action="store_true", help="Generate baseline from production model")
    parser.add_argument("--list", action="store_true", help="List available baselines")
    parser.add_argument("--show", type=int, help="Show details for baseline version N")
    parser.add_argument("--cleanup", type=int, default=None, help="Keep only N most recent baselines")
    
    args = parser.parse_args()
    
    if args.list:
        manager = BaselineManager()
        baselines = manager.list_baselines()
        
        print("\n" + "="*60)
        print("AVAILABLE BASELINES")
        print("="*60)
        
        if baselines:
            for b in baselines:
                print(f"\n  v{b['version']}")
                print(f"    Model: {b['model_name']}")
                print(f"    Created: {b['created_at']}")
                print(f"    Samples: {b.get('reference_samples', 'N/A')}")
                print(f"    Path: {b['path']}")
        else:
            print("  No baselines found")
            print(f"  Looking in: {BASELINES_DIR}")
    
    elif args.show:
        try:
            path = get_baseline_path(args.show)
            baseline = BaselineGenerator.load_baseline(path)
            
            print(f"\n" + "="*60)
            print(f"BASELINE v{args.show} DETAILS")
            print("="*60)
            
            meta = baseline['metadata']
            print(f"\nMetadata:")
            print(f"  Model: {meta['model_name']}")
            print(f"  Created: {meta['created_at']}")
            print(f"  Train samples: {meta['train_samples']}")
            print(f"  Reference samples: {meta['reference_samples']}")
            print(f"  Features: {meta['n_features']}")
            
            print(f"\nFeature Names:")
            for i, feat in enumerate(meta['feature_names']):
                print(f"  {i+1}. {feat}")
            
            print(f"\nPerformance Baseline:")
            for k, v in baseline['performance_baseline'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            
            print(f"\nPrediction Stats:")
            pred = baseline['prediction_stats']
            print(f"  Mean: {pred['mean']:.2f}")
            print(f"  Std: {pred['std']:.2f}")
            print(f"  Range: [{pred['min']:.2f}, {pred['max']:.2f}]")
            
            print(f"\nTarget Stats:")
            target = baseline['target_stats']
            print(f"  Mean: {target['mean']:.2f}")
            print(f"  Std: {target['std']:.2f}")
            print(f"  Range: [{target['min']:.2f}, {target['max']:.2f}]")
            
            print(f"\nReference Data Shape: {baseline['reference_data'].shape}")
            
        except FileNotFoundError:
            print(f"Baseline v{args.show} not found")
    
    elif args.cleanup:
        manager = BaselineManager()
        print(f"Cleaning up baselines, keeping {args.cleanup} most recent...")
        manager.cleanup_old_baselines(keep_count=args.cleanup)
        print("Cleanup complete.")
    
    elif args.generate:
        try:
            path = generate_baseline_from_training()
            print(f"\n✓ Baseline generated: {path}")
        except FileNotFoundError as e:
            print(f"\n✗ Error: {e}")
            print("  Make sure production model exists at:")
            print(f"    {PRODUCTION_MODEL_PATH}")
        except Exception as e:
            print(f"\n✗ Error generating baseline: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        parser.print_help()