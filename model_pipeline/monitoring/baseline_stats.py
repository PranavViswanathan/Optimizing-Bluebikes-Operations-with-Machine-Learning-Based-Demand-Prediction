"""
Baseline Statistics Generator for BlueBikes Model Monitoring
Creates and manages reference data baselines for Evidently AI drift detection.

KEY CHANGE: Uses TRAINING data as baseline (not test data)
- Baseline = what the model learned from
- Current = new production data to compare against baseline
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
    DATA_SPLITS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineGenerator:
    """
    Generates and manages baseline data for Evidently AI monitoring.
    
    IMPORTANT: Baseline is generated from TRAINING data, not test data.
    This represents the distribution the model was trained on.
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
        Generate baseline from TRAINING data.
        
        Args:
            X_train: Training features - THIS IS THE REFERENCE
            y_train: Training target
            X_test: Test features (for metrics only)
            y_test: Test target (for metrics only)
            model: Trained model object
            model_metrics: Performance metrics from training
            model_version: Version number for this baseline
            model_name: Name of the model
            sample_size: Number of samples to store as reference
            feature_columns: List of feature columns the model expects
        """
        logger.info("="*60)
        logger.info("GENERATING BASELINE FROM TRAINING DATA")
        logger.info("="*60)
        logger.info("Baseline = Training distribution (what model learned from)")
        
        # Drop date column for model operations
        X_train_clean = X_train.drop('date', axis=1, errors='ignore')
        X_test_clean = X_test.drop('date', axis=1, errors='ignore')
        
        # If feature_columns provided, ensure correct column order
        if feature_columns:
            available_cols = [c for c in feature_columns if c in X_train_clean.columns]
            X_train_clean = X_train_clean[available_cols]
            X_test_clean = X_test_clean[available_cols]
            logger.info(f"Using {len(available_cols)} features from feature_columns")
        
        # =====================================================
        # KEY: Sample from TRAINING data as reference
        # This is what "normal" looks like for the model
        # =====================================================
        if len(X_train_clean) > sample_size:
            sample_idx = np.random.choice(len(X_train_clean), sample_size, replace=False)
            reference_data = X_train_clean.iloc[sample_idx].copy()
            reference_target = y_train.iloc[sample_idx].copy()
        else:
            reference_data = X_train_clean.copy()
            reference_target = y_train.copy()
        
        logger.info(f"Reference data sampled from TRAINING set: {len(reference_data)} samples")
        
        # Add predictions to reference data
        reference_preds = model.predict(reference_data)
        reference_data = reference_data.copy()
        reference_data['prediction'] = reference_preds
        reference_data[self.config.features.target_column] = reference_target.values
        
        # Generate predictions on test set for baseline metrics
        y_pred_test = model.predict(X_test_clean)
        
        baseline = {
            "metadata": {
                "version": model_version,
                "model_name": model_name,
                "created_at": datetime.now().isoformat(),
                "baseline_source": "training_data",  # Explicitly note source
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "reference_samples": len(reference_data),
                "n_features": X_train_clean.shape[1],
                "feature_names": list(X_train_clean.columns),
                "data_splits": {
                    "train_start": self.config.data_splits.train_start,
                    "train_end": self.config.data_splits.train_end,
                    "test_start": self.config.data_splits.test_start,
                    "test_end": self.config.data_splits.test_end,
                }
            },
            "reference_data": reference_data,  # DataFrame for Evidently
            "feature_stats": self._compute_feature_stats(X_train_clean),  # From training!
            "target_stats": self._compute_target_stats(y_train),  # From training!
            "prediction_stats": self._compute_prediction_stats(y_pred_test),
            "performance_baseline": self._normalize_metrics(model_metrics),
        }
        
        self.baseline = baseline
        
        logger.info(f"Baseline generated for model v{model_version}")
        logger.info(f"  Source: Training data ({self.config.data_splits.train_start} to {self.config.data_splits.train_end})")
        logger.info(f"  Reference samples: {len(reference_data)}")
        logger.info(f"  Features tracked: {len(baseline['feature_stats'])}")
        
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
                value_counts = col_data.value_counts(normalize=True)
                stats[col] = {
                    "type": "categorical",
                    "proportions": {str(k): float(v) for k, v in value_counts.items()},
                    "n_unique": int(col_data.nunique()),
                    "n_samples": len(col_data),
                }
            else:
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
        """Save baseline to pickle file."""
        if not self.baseline:
            raise ValueError("No baseline generated. Run generate_baseline() first.")
        
        if path is None:
            path = get_baseline_path(version)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.baseline, f)
        
        # Also save as current baseline
        current_path = get_baseline_path(None)
        with open(current_path, 'wb') as f:
            pickle.dump(self.baseline, f)
        
        # Save metadata as JSON for inspection
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
        logger.info(f"  Source: {baseline['metadata'].get('baseline_source', 'unknown')}")
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
                metadata_path = path.parent / f"{path.stem}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                    baselines.append({
                        "version": data["metadata"]["version"],
                        "created_at": data["metadata"]["created_at"],
                        "model_name": data["metadata"].get("model_name", "unknown"),
                        "baseline_source": data["metadata"].get("baseline_source", "unknown"),
                        "reference_samples": data["metadata"].get("reference_samples", 0),
                        "path": str(path),
                    })
                else:
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
    Generate baseline from production model using TRAINING data as reference.
    """
    import sys
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline')
    
    logger.info("="*60)
    logger.info("GENERATING BASELINE FROM TRAINING DATA")
    logger.info("="*60)
    
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
        if hasattr(model, 'get_booster'):
            expected_features = model.get_booster().feature_names
        elif hasattr(model, 'feature_name_'):
            expected_features = model.feature_name_
        elif hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        else:
            expected_features = None
    except Exception as e:
        expected_features = None
        logger.warning(f"Error extracting feature names: {e}")
    
    # Load data and split according to config
    from feature_generation import load_and_prepare_data
    
    X, y, feature_columns = load_and_prepare_data()
    X["date"] = pd.to_datetime(X["date"])
    
    config = get_config()
    
    # Split data according to config dates
    train_mask = (X["date"] >= config.data_splits.get_train_start()) & \
                 (X["date"] <= config.data_splits.get_train_end())
    test_mask = (X["date"] >= config.data_splits.get_test_start()) & \
                (X["date"] <= config.data_splits.get_test_end())
    
    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()
    
    logger.info(f"Training data: {len(X_train)} samples ({config.data_splits.train_start} to {config.data_splits.train_end})")
    logger.info(f"Test data: {len(X_test)} samples ({config.data_splits.test_start} to {config.data_splits.test_end})")
    
    # Handle feature alignment
    if expected_features:
        for df in [X_train, X_test]:
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
        
        cols_to_keep = ['date'] + [c for c in expected_features if c in X_train.columns]
        X_train = X_train[cols_to_keep]
        X_test = X_test[cols_to_keep]
    
    # Get version and metrics
    version = metadata.get("version", 1)
    model_name = metadata.get("model_type", "unknown")
    metrics = metadata.get("metrics", {})
    
    # Generate baseline from TRAINING data
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
    
    path = generator.save_baseline(version=version)
    
    logger.info("="*60)
    logger.info("BASELINE GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Baseline source: Training data")
    logger.info(f"Saved to: {path}")
    
    return path
def generate_monthly_baselines():
    """
    Generate separate baseline for each month.
    This allows month-to-month comparisons instead of comparing against all months.
    """
    import sys
    sys.path.insert(0, '/opt/airflow/scripts/model_pipeline')
    
    from feature_generation import load_and_prepare_data
    from monitoring_config import get_config, BASELINES_DIR, DATA_SPLITS
    import pandas as pd
    import json
    
    logger.info("="*60)
    logger.info("GENERATING MONTHLY BASELINES")
    logger.info("="*60)
    
    config = get_config()
    X, y, _ = load_and_prepare_data()
    X["date"] = pd.to_datetime(X["date"])
    
    # Use only training period data
    train_end = DATA_SPLITS.get_train_end()
    training_data = X[X["date"] <= train_end].copy()
    
    logger.info(f"Total training data: {len(training_data)} samples")
    logger.info(f"Date range: {training_data['date'].min()} to {training_data['date'].max()}")
    
    metadata = {}
    month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december']
    
    # Create baseline for each month
    for month_num in range(1, 13):
        month_name = month_names[month_num - 1]
        
        # Get all data from this month across all years in training period
        month_data = training_data[training_data["date"].dt.month == month_num].copy()
        
        if len(month_data) < config.min_samples_per_baseline:
            logger.warning(f"  {month_name}: Only {len(month_data)} samples (min: {config.min_samples_per_baseline}), skipping")
            continue
        
        # Drop date column for Evidently (it doesn't need it)
        month_data_for_baseline = month_data.drop(columns=['date'], errors='ignore')
        
        # Save baseline
        baseline_path = BASELINES_DIR / f"baseline_{month_name}.pkl"
        month_data_for_baseline.to_pickle(baseline_path)
        
        # Store metadata
        years = sorted(month_data['date'].dt.year.unique().tolist())
        metadata[month_name] = {
            'month_number': month_num,
            'samples': len(month_data),
            'date_range': f"{month_data['date'].min().date()} to {month_data['date'].max().date()}",
            'years': years,
            'path': str(baseline_path),
            'created': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"✓ {month_name:12s}: {len(month_data):5d} samples from {years}")
    
    # Save metadata
    metadata_path = BASELINES_DIR / "monthly_baselines_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n✓ Generated {len(metadata)} monthly baselines")
    logger.info(f"✓ Metadata saved to: {metadata_path}")
    
    return metadata


def get_month_name(month_number):
    """Helper function to convert month number to name"""
    month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december']
    return month_names[month_number - 1]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Statistics Generator")
    parser.add_argument("--generate", action="store_true", help="Generate baseline")
    parser.add_argument("--list", action="store_true", help="List available baselines")
    parser.add_argument("--show", type=int, help="Show details for baseline version N")
    
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
                print(f"    Source: {b.get('baseline_source', 'unknown')}")
                print(f"    Created: {b['created_at']}")
                print(f"    Samples: {b.get('reference_samples', 'N/A')}")
        else:
            print("  No baselines found")
    
    elif args.generate:
        try:
            path = generate_baseline_from_training()
            print(f"\n  Baseline generated: {path}")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        parser.print_help()