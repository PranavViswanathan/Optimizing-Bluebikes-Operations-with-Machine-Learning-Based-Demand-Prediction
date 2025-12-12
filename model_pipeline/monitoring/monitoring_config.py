"""
Monitoring Configuration for BlueBikes Model Pipeline
Centralized settings for drift detection, alerting, and retraining triggers.

Updated to match integrated_training_pipeline.py data splits:
- Training: Jan 2024 - Sep 2025
- Test: Oct 2025 - Nov 2025
- Production (for drift monitoring): Dec 2025+
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import os
import pandas as pd


# =============================================================================
# DATA SPLIT CONFIGURATION (Must match integrated_training_pipeline.py!)
# =============================================================================

@dataclass
class DataSplitConfig:
    """
    Data split dates - MUST match integrated_training_pipeline.py
    Single source of truth for date boundaries.
    """
    train_start: str = "2024-01-01"
    train_end: str = "2025-09-30"
    test_start: str = "2025-10-01"
    test_end: str = "2025-11-30"
    
    # For drift monitoring: anything after test_end is "production"
    production_start: str = "2025-12-01"
    
    def get_train_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_start)
    
    def get_train_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_end)
    
    def get_test_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_start)
    
    def get_test_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_end)
    
    def get_production_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.production_start)


# Global instance for easy access
DATA_SPLITS = DataSplitConfig()


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

MONITORING_BASE_DIR = Path(os.environ.get(
    "MONITORING_DIR",
    "/opt/airflow/scripts/model_pipeline/monitoring"
))

BASELINES_DIR = MONITORING_BASE_DIR / "baselines"
REPORTS_DIR = MONITORING_BASE_DIR / "reports"
HTML_REPORTS_DIR = REPORTS_DIR / "html"
JSON_REPORTS_DIR = REPORTS_DIR / "json"
LOGS_DIR = MONITORING_BASE_DIR / "logs"
PREDICTIONS_DIR = MONITORING_BASE_DIR / "predictions"

MODELS_DIR = Path("/opt/airflow/models")
PRODUCTION_MODEL_PATH = MODELS_DIR / "production" / "current_model.pkl"
PRODUCTION_METADATA_PATH = MODELS_DIR / "production" / "current_metadata.json"
MODEL_VERSIONS_PATH = MODELS_DIR / "model_versions.json"

DATA_DIR = Path("/opt/airflow/data")
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "bluebikes" / "after_duplicates.pkl"

# Ensure directories exist
for dir_path in [BASELINES_DIR, HTML_REPORTS_DIR, JSON_REPORTS_DIR, LOGS_DIR, PREDICTIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

@dataclass
class FeatureConfig:
    """
    Configuration for features to monitor.
    Must match features from feature_generation.py
    """
    
    # Base features from feature_generation.py
    numerical_features: List[str] = field(default_factory=lambda: [
        # Temporal
        "hour", "day_of_week", "month", "year", "day",
        # Cyclical encodings
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Weather
        "TMAX", "TMIN", "PRCP", "temp_avg", "temp_range",
        # Lag features
        "rides_last_hour", "rides_same_hour_yesterday", "rides_same_hour_last_week",
        "rides_rolling_3h", "rides_rolling_24h",
        # Aggregated statistics
        "duration_mean", "duration_std", "duration_median",
        "distance_mean", "distance_std", "distance_median", "member_ratio",
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: [
        "is_weekend", "is_morning_rush", "is_evening_rush", 
        "is_night", "is_midday",
        "is_rainy", "is_heavy_rain", "is_cold", "is_hot",
        "weekend_night", "weekday_morning_rush", "weekday_evening_rush",
    ])
    
    # Additional features added during bias mitigation
    # These are added by apply_optimized_bias_mitigation()
    bias_mitigation_features: List[str] = field(default_factory=lambda: [
        "is_hour_8", "is_hour_17_18", "rush_intensity",
        "high_demand_flag", "low_demand_flag", "demand_volatility",
        "problem_period", "hour_group",
    ])
    
    # High-importance features for stricter monitoring
    critical_features: List[str] = field(default_factory=lambda: [
        "rides_last_hour", "rides_rolling_3h", "temp_avg",
        "hour", "is_weekend", "is_morning_rush", "is_evening_rush",
    ])
    
    skip_features: List[str] = field(default_factory=lambda: ["date"])
    
    target_column: str = "ride_count"
    
    def get_all_features(self, include_bias_features: bool = False) -> List[str]:
        """Get all feature names."""
        features = self.numerical_features + self.categorical_features
        if include_bias_features:
            features += self.bias_mitigation_features
        return features


# =============================================================================
# DRIFT THRESHOLDS
# =============================================================================

@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""
    psi_no_drift: float = 0.1
    psi_minor_drift: float = 0.2
    psi_major_drift: float = 0.25
    ks_pvalue_threshold: float = 0.05
    mean_shift_warning: float = 15.0
    mean_shift_critical: float = 25.0
    std_change_warning: float = 20.0
    std_change_critical: float = 35.0
    proportion_change_warning: float = 0.10
    proportion_change_critical: float = 0.20
    min_features_for_warning: int = 3
    min_features_for_critical: int = 5


@dataclass
class PredictionDriftThresholds:
    """Thresholds for prediction distribution monitoring."""
    mean_shift_warning: float = 15.0
    mean_shift_critical: float = 25.0
    std_change_warning: float = 20.0
    std_change_critical: float = 35.0
    min_prediction: float = 0.0
    max_prediction: float = 500.0
    out_of_range_warning: float = 5.0
    out_of_range_critical: float = 10.0


@dataclass
class PerformanceThresholds:
    """Thresholds for model performance monitoring."""
    r2_minimum: float = 0.65
    r2_drop_warning: float = 0.03
    r2_drop_critical: float = 0.05
    mae_increase_warning: float = 15.0
    mae_increase_critical: float = 25.0
    rmse_increase_warning: float = 15.0
    rmse_increase_critical: float = 25.0
    mape_warning: float = 20.0
    mape_critical: float = 30.0
    rolling_window_days: int = 7
    min_samples_for_evaluation: int = 100


# =============================================================================
# ALERTING & RETRAINING CONFIG
# =============================================================================

@dataclass
class AlertConfig:
    """Configuration for alerting and notifications."""
    discord_webhook_url: str = field(
        default_factory=lambda: os.environ.get("DISCORD_WEBHOOK_URL", "")
    )
    enable_info_alerts: bool = False
    enable_warning_alerts: bool = True
    enable_critical_alerts: bool = True
    alert_cooldown_hours: int = 6
    include_drift_details: bool = True
    include_feature_breakdown: bool = True
    max_features_in_alert: int = 5


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining triggers."""
    auto_retrain_enabled: bool = True
    retraining_dag_id: str = "bluebikes_integrated_bias_training"
    trigger_on_major_data_drift: bool = True
    trigger_on_major_prediction_drift: bool = True
    trigger_on_performance_decay: bool = True
    retraining_cooldown_hours: int = 24
    max_retrains_per_week: int = 3
    retraining_log_path: Path = field(
        default_factory=lambda: LOGS_DIR / "retraining_history.json"
    )


@dataclass
class ScheduleConfig:
    """Configuration for monitoring schedule."""
    monitoring_schedule: str = "0 6 * * *"
    analysis_window_hours: int = 24
    ground_truth_delay_days: int = 1
    report_retention_days: int = 90
    baseline_retention_count: int = 10


@dataclass
class EvidentlyConfig:
    """Configuration for Evidently AI drift detection."""
    drift_method: str = "stattest"
    numerical_stattest: str = "ks"
    categorical_stattest: str = "chisquare"
    stattest_threshold: float = 0.05
    dataset_drift_share: float = 0.3
    generate_html_reports: bool = True
    include_detailed_stats: bool = True


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================

@dataclass 
class MonitoringConfig:
    """Master configuration combining all monitoring settings."""
    data_splits: DataSplitConfig = field(default_factory=DataSplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evidently: EvidentlyConfig = field(default_factory=EvidentlyConfig)
    drift: DriftThresholds = field(default_factory=DriftThresholds)
    prediction: PredictionDriftThresholds = field(default_factory=PredictionDriftThresholds)
    performance: PerformanceThresholds = field(default_factory=PerformanceThresholds)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        import dataclasses
        
        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        return convert(self)
    
    @classmethod
    def load_from_env(cls) -> "MonitoringConfig":
        """Load configuration with environment variable overrides."""
        config = cls()
        
        if os.environ.get("DRIFT_PSI_CRITICAL"):
            config.drift.psi_major_drift = float(os.environ["DRIFT_PSI_CRITICAL"])
        if os.environ.get("PERFORMANCE_R2_MIN"):
            config.performance.r2_minimum = float(os.environ["PERFORMANCE_R2_MIN"])
        if os.environ.get("AUTO_RETRAIN_ENABLED"):
            config.retraining.auto_retrain_enabled = (
                os.environ["AUTO_RETRAIN_ENABLED"].lower() == "true"
            )
        
        return config


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config() -> MonitoringConfig:
    """Get the monitoring configuration."""
    return MonitoringConfig.load_from_env()


def get_baseline_path(version: Optional[int] = None) -> Path:
    """Get path to baseline file."""
    if version is None:
        return BASELINES_DIR / "current_baseline.pkl"
    return BASELINES_DIR / f"baseline_v{version}.pkl"


def get_report_path(date_str: str, report_type: str = "json") -> Path:
    """Get path to monitoring report."""
    if report_type == "html":
        return HTML_REPORTS_DIR / f"drift_report_{date_str}.html"
    return JSON_REPORTS_DIR / f"monitoring_report_{date_str}.json"


def get_predictions_log_path(date_str: str) -> Path:
    """Get path to predictions log for a given date."""
    return PREDICTIONS_DIR / f"predictions_{date_str}.pkl"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config(config: MonitoringConfig) -> List[str]:
    """Validate configuration and return list of warnings."""
    warnings = []
    
    if config.drift.psi_no_drift >= config.drift.psi_minor_drift:
        warnings.append("PSI no_drift threshold >= minor_drift threshold")
    
    if config.drift.psi_minor_drift >= config.drift.psi_major_drift:
        warnings.append("PSI minor_drift threshold >= major_drift threshold")
    
    if config.performance.r2_minimum > 1.0 or config.performance.r2_minimum < 0:
        warnings.append(f"Invalid R² minimum: {config.performance.r2_minimum}")
    
    if config.alerts.enable_warning_alerts and not config.alerts.discord_webhook_url:
        warnings.append("Warning alerts enabled but DISCORD_WEBHOOK_URL not set")
    
    if not config.features.numerical_features and not config.features.categorical_features:
        warnings.append("No features configured for monitoring")
    
    # Validate date splits
    splits = config.data_splits
    if pd.Timestamp(splits.train_end) >= pd.Timestamp(splits.test_start):
        warnings.append("train_end should be before test_start")
    
    return warnings


if __name__ == "__main__":
    config = get_config()
    
    print("=" * 60)
    print("MONITORING CONFIGURATION")
    print("=" * 60)
    
    print(f"\nData Splits:")
    print(f"  Training: {config.data_splits.train_start} to {config.data_splits.train_end}")
    print(f"  Test: {config.data_splits.test_start} to {config.data_splits.test_end}")
    print(f"  Production starts: {config.data_splits.production_start}")
    
    print(f"\nPaths:")
    print(f"  Baselines: {BASELINES_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    
    print(f"\nFeatures to monitor:")
    print(f"  Numerical: {len(config.features.numerical_features)}")
    print(f"  Categorical: {len(config.features.categorical_features)}")
    print(f"  Critical: {config.features.critical_features}")
    
    warnings = validate_config(config)
    if warnings:
        print(f"\n   Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print(f"\n  Configuration valid")