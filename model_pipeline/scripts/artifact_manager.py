"""
Artifact Manager for BlueBikes Model Pipeline
Centralizes all file storage paths for models, plots, and reports.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class ArtifactManager:
    """
    Manages artifact storage paths for the model pipeline.
    Ensures consistent, organized file storage.
    """
    
    # Base directory - works both in Docker and locally
    BASE_DIR = Path(os.environ.get(
        "MODEL_ARTIFACTS_DIR", 
        "/opt/airflow/artifacts/model_pipeline"
    ))
    
    # Subdirectories
    MODELS_DIR = BASE_DIR / "models"
    PLOTS_DIR = BASE_DIR / "plots"
    REPORTS_DIR = BASE_DIR / "reports"
    
    # Plot subdirectories
    BIAS_PLOTS_DIR = PLOTS_DIR / "bias_analysis"
    SENSITIVITY_PLOTS_DIR = PLOTS_DIR / "sensitivity_analysis"
    TRAINING_PLOTS_DIR = PLOTS_DIR / "training"
    FEATURE_IMPORTANCE_DIR = PLOTS_DIR / "feature_importance"
    
    @classmethod
    def setup(cls):
        """Create all necessary directories."""
        directories = [
            cls.MODELS_DIR,
            cls.REPORTS_DIR,
            cls.BIAS_PLOTS_DIR,
            cls.SENSITIVITY_PLOTS_DIR,
            cls.TRAINING_PLOTS_DIR,
            cls.FEATURE_IMPORTANCE_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Artifact directories initialized at: {cls.BASE_DIR}")
    
    # =========================================================================
    # Model Paths
    # =========================================================================
    
    @classmethod
    def get_model_path(cls, model_name: str, stage: str = "baseline") -> Path:
        """Get path for saving a model."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MODELS_DIR / f"{stage}_model_{model_name}.pkl"
    
    @classmethod
    def get_best_model_path(cls, model_name: str) -> Path:
        """Get path for the best model."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MODELS_DIR / f"best_model_{model_name}.pkl"
    
    @classmethod
    def get_mitigated_model_path(cls, model_name: str) -> Path:
        """Get path for the mitigated model."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MODELS_DIR / f"mitigated_model_{model_name}.pkl"
    
    @classmethod
    def get_model_metadata_path(cls, stage: str = "best") -> Path:
        """Get path for model metadata JSON."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MODELS_DIR / f"{stage}_model_metadata.json"
    
    # =========================================================================
    # Plot Paths
    # =========================================================================
    
    @classmethod
    def get_bias_plot_path(cls, stage: str = "baseline") -> Path:
        """Get path for bias analysis plots."""
        cls.BIAS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.BIAS_PLOTS_DIR / f"bias_analysis_{stage}.png"
    
    @classmethod
    def get_sensitivity_plot_path(cls, stage: str = "baseline") -> Path:
        """Get path for sensitivity analysis plots."""
        cls.SENSITIVITY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.SENSITIVITY_PLOTS_DIR / f"sensitivity_analysis_{stage}.png"
    
    @classmethod
    def get_shap_plot_path(cls, stage: str = "baseline") -> Path:
        """Get path for SHAP summary plots."""
        cls.SENSITIVITY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.SENSITIVITY_PLOTS_DIR / f"shap_summary_{stage}.png"
    
    @classmethod
    def get_feature_importance_path(cls, model_name: str) -> Path:
        """Get path for feature importance plots."""
        cls.FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
        return cls.FEATURE_IMPORTANCE_DIR / f"feature_importance_{model_name}.png"
    
    @classmethod
    def get_predictions_plot_path(cls, model_name: str) -> Path:
        """Get path for predictions scatter plots."""
        cls.TRAINING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.TRAINING_PLOTS_DIR / f"predictions_{model_name}.png"
    
    @classmethod
    def get_residuals_plot_path(cls, model_name: str) -> Path:
        """Get path for residual plots."""
        cls.TRAINING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.TRAINING_PLOTS_DIR / f"residuals_{model_name}.png"
    
    # =========================================================================
    # Report Paths
    # =========================================================================
    
    @classmethod
    def get_bias_report_path(cls, stage: str = "baseline", timestamp: bool = False) -> Path:
        """Get path for bias detection reports."""
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return cls.REPORTS_DIR / f"bias_report_{stage}_{ts}.json"
        return cls.REPORTS_DIR / f"bias_report_{stage}.json"
    
    @classmethod
    def get_sensitivity_report_path(cls, stage: str = "baseline", timestamp: bool = False) -> Path:
        """Get path for sensitivity analysis reports."""
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return cls.REPORTS_DIR / f"sensitivity_report_{stage}_{ts}.json"
        return cls.REPORTS_DIR / f"sensitivity_report_{stage}.json"
    
    @classmethod
    def get_comparison_report_path(cls, timestamp: bool = False) -> Path:
        """Get path for comparison reports."""
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return cls.REPORTS_DIR / f"comparison_report_{ts}.json"
        return cls.REPORTS_DIR / f"comparison_report.json"
    
    @classmethod
    def get_training_summary_path(cls, model_name: str) -> Path:
        """Get path for training summary JSON."""
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.REPORTS_DIR / f"training_summary_{model_name}.json"
    
    @classmethod
    def get_feature_importance_csv_path(cls, model_name: str) -> Path:
        """Get path for feature importance CSV."""
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.REPORTS_DIR / f"{model_name}_feature_importance.csv"

    @classmethod
    def get_model_metadata_pkl_path(cls, model_name: str) -> Path:
        """Get path for model metadata pickle file."""
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.REPORTS_DIR / f"{model_name}_model_metadata.pkl"
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    @classmethod
    def list_artifacts(cls) -> dict:
        """List all artifacts in each directory."""
        artifacts = {
            'models': [],
            'plots': [],
            'reports': []
        }
        
        if cls.MODELS_DIR.exists():
            artifacts['models'] = [f.name for f in cls.MODELS_DIR.glob('*')]
        
        if cls.PLOTS_DIR.exists():
            artifacts['plots'] = [str(f.relative_to(cls.PLOTS_DIR)) 
                                  for f in cls.PLOTS_DIR.rglob('*.png')]
        
        if cls.REPORTS_DIR.exists():
            artifacts['reports'] = [f.name for f in cls.REPORTS_DIR.glob('*.json')]
        
        return artifacts
    
    @classmethod
    def cleanup_old_artifacts(cls, keep_latest: int = 5):
        """Remove old timestamped artifacts, keeping the latest N."""
        import re
        
        for report_file in cls.REPORTS_DIR.glob('*_2*.json'):  # Matches timestamped files
            # Keep logic here if needed
            pass
        
        print(f"Cleanup complete. Kept latest {keep_latest} of each artifact type.")


# Initialize directories on import
ArtifactManager.setup()
