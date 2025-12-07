"""
Monitoring Configuration for BlueBikes Model Pipeline
Centralized settings for drift detection, alerting, and retraining triggers.

Uses Evidently AI for drift detection as recommended in deployment guidelines.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import os


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directories - aligned with your existing structure
MONITORING_BASE_DIR = Path(os.environ.get(
    "MONITORING_DIR",
    "/opt/airflow/scripts/model_pipeline/monitoring"
))

# Subdirectories for monitoring artifacts
BASELINES_DIR = MONITORING_BASE_DIR / "baselines"
REPORTS_DIR = MONITORING_BASE_DIR / "reports"
HTML_REPORTS_DIR = REPORTS_DIR / "html"  # Evidently HTML reports
JSON_REPORTS_DIR = REPORTS_DIR / "json"  # JSON summaries
LOGS_DIR = MONITORING_BASE_DIR / "logs"
PREDICTIONS_DIR = MONITORING_BASE_DIR / "predictions"  # Store predictions for later comparison

# Model paths - aligned with your Model_pipeline_withBias_check.py
MODELS_DIR = Path("/opt/airflow/models")
PRODUCTION_MODEL_PATH = MODELS_DIR / "production" / "current_model.pkl"
PRODUCTION_METADATA_PATH = MODELS_DIR / "production" / "current_metadata.json"
MODEL_VERSIONS_PATH = MODELS_DIR / "model_versions.json"

# Data paths - aligned with your data_pipeline
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
    Aligned with features from your feature_generation.py
    """
    
    # Numerical features - from your feature_generation.py
    numerical_features: List[str] = field(default_factory=lambda: [
        # Temporal
        "hour",
        "day_of_week", 
        "month",
        "year",
        "day",
        
        # Cyclical encodings
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        
        # Weather - from NOAA data
        "TMAX",
        "TMIN",
        "PRCP",
        "temp_avg",
        "temp_range",
        
        # Lag features (critical for demand prediction)
        "rides_last_hour",
        "rides_same_hour_yesterday",
        "rides_same_hour_last_week",
        "rides_rolling_3h",
        "rides_rolling_24h",
        
        # Aggregated ride statistics
        "duration_mean",
        "duration_std",
        "duration_median",
        "distance_mean",
        "distance_std",
        "distance_median",
        "member_ratio",
    ])
    
    # Categorical/binary features - from your feature_generation.py
    categorical_features: List[str] = field(default_factory=lambda: [
        "is_weekend",
        "is_morning_rush",
        "is_evening_rush",
        "is_night",
        "is_midday",
        "is_rainy",
        "is_heavy_rain",
        "is_cold",
        "is_hot",
        "weekend_night",
        "weekday_morning_rush",
        "weekday_evening_rush",
    ])
    
    # High-importance features for stricter monitoring
    # Based on your bias_detection.py analysis
    critical_features: List[str] = field(default_factory=lambda: [
        "rides_last_hour",
        "rides_rolling_3h", 
        "temp_avg",
        "hour",
        "is_weekend",
        "is_morning_rush",
        "is_evening_rush",
    ])
    
    # Features to exclude from monitoring
    skip_features: List[str] = field(default_factory=lambda: [
        "date",  # Not a model feature, used for splitting only
    ])
    
    # Target column name
    target_column: str = "ride_count"


# =============================================================================
# EVIDENTLY AI CONFIGURATION
# =============================================================================

@dataclass
class EvidentlyConfig:
    """Configuration for Evidently AI drift detection."""
    
    # Data drift detection method
    # Options: "stattest" (statistical tests), "domain_classifier", "ratio"
    drift_method: str = "stattest"
    
    # Statistical test for numerical features
    # Options: "ks" (Kolmogorov-Smirnov), "wasserstein", "psi", "kl_div", "jensenshannon"
    numerical_stattest: str = "ks"
    
    # Statistical test for categorical features  
    # Options: "chisquare", "z", "jensenshannon"
    categorical_stattest: str = "chisquare"
    
    # Threshold for statistical tests (p-value or distance threshold)
    stattest_threshold: float = 0.05
    
    # Dataset drift threshold - fraction of drifted features to consider dataset drifted
    dataset_drift_share: float = 0.3  # 30% of features drifted = dataset drift
    
    # Generate HTML reports
    generate_html_reports: bool = True
    
    # Include detailed statistics in reports
    include_detailed_stats: bool = True


# =============================================================================
# DRIFT THRESHOLDS (kept for custom logic and alerting)
# =============================================================================

@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""
    
    # Population Stability Index (PSI) thresholds
    # PSI < 0.1: No significant change
    # 0.1 <= PSI < 0.2: Moderate change, monitor closely
    # PSI >= 0.2: Significant change, action required
    psi_no_drift: float = 0.1
    psi_minor_drift: float = 0.2
    psi_major_drift: float = 0.25
    
    # Kolmogorov-Smirnov test p-value threshold
    # p < 0.05 indicates significant distribution difference
    ks_pvalue_threshold: float = 0.05
    
    # Mean shift thresholds (percentage)
    mean_shift_warning: float = 15.0   # 15% shift triggers warning
    mean_shift_critical: float = 25.0  # 25% shift triggers alert
    
    # Standard deviation change thresholds (percentage)
    std_change_warning: float = 20.0
    std_change_critical: float = 35.0
    
    # Proportion change for categorical features (absolute)
    proportion_change_warning: float = 0.10  # 10% absolute change
    proportion_change_critical: float = 0.20  # 20% absolute change
    
    # Number of features drifted to trigger action
    min_features_for_warning: int = 3
    min_features_for_critical: int = 5


# =============================================================================
# PREDICTION DRIFT THRESHOLDS
# =============================================================================

@dataclass
class PredictionDriftThresholds:
    """Thresholds for prediction distribution monitoring."""
    
    # Mean prediction shift (percentage from baseline)
    mean_shift_warning: float = 15.0
    mean_shift_critical: float = 25.0
    
    # Standard deviation change (percentage)
    std_change_warning: float = 20.0
    std_change_critical: float = 35.0
    
    # Prediction range checks
    # Alert if predictions fall outside expected range
    min_prediction: float = 0.0      # Rides can't be negative
    max_prediction: float = 500.0    # Upper bound sanity check
    
    # Percentage of predictions outside normal range to trigger alert
    out_of_range_warning: float = 5.0   # 5% outside range
    out_of_range_critical: float = 10.0  # 10% outside range


# =============================================================================
# PERFORMANCE DECAY THRESHOLDS
# =============================================================================

@dataclass
class PerformanceThresholds:
    """Thresholds for model performance monitoring."""
    
    # R² score thresholds
    r2_minimum: float = 0.65           # Absolute minimum acceptable
    r2_drop_warning: float = 0.03      # 3% drop from baseline
    r2_drop_critical: float = 0.05     # 5% drop triggers retraining
    
    # MAE thresholds (percentage increase from baseline)
    mae_increase_warning: float = 15.0
    mae_increase_critical: float = 25.0
    
    # RMSE thresholds (percentage increase from baseline)
    rmse_increase_warning: float = 15.0
    rmse_increase_critical: float = 25.0
    
    # MAPE thresholds (absolute percentage points)
    mape_warning: float = 20.0
    mape_critical: float = 30.0
    
    # Rolling window for performance calculation (days)
    rolling_window_days: int = 7
    
    # Minimum samples needed to calculate performance
    min_samples_for_evaluation: int = 100


# =============================================================================
# ALERTING CONFIGURATION
# =============================================================================

@dataclass
class AlertConfig:
    """Configuration for alerting and notifications."""
    
    # Discord webhook (from environment)
    discord_webhook_url: str = field(
        default_factory=lambda: os.environ.get("DISCORD_WEBHOOK_URL", "")
    )
    
    # Alert levels
    enable_info_alerts: bool = False    # Log only, no notification
    enable_warning_alerts: bool = True  # Discord notification
    enable_critical_alerts: bool = True # Discord + trigger action
    
    # Cooldown to prevent alert spam (hours)
    alert_cooldown_hours: int = 6
    
    # Include detailed stats in alerts
    include_drift_details: bool = True
    include_feature_breakdown: bool = True
    max_features_in_alert: int = 5  # Top N drifted features to show


# =============================================================================
# RETRAINING CONFIGURATION  
# =============================================================================

@dataclass
class RetrainingConfig:
    """Configuration for automated retraining triggers."""
    
    # Enable automatic retraining
    auto_retrain_enabled: bool = True
    
    # DAG to trigger for retraining
    retraining_dag_id: str = "bluebikes_integrated_bias_training"
    
    # Conditions that trigger retraining (ANY of these)
    trigger_on_major_data_drift: bool = True
    trigger_on_major_prediction_drift: bool = True
    trigger_on_performance_decay: bool = True
    
    # Cooldown between retraining runs (hours)
    retraining_cooldown_hours: int = 24
    
    # Maximum retraining attempts per week
    max_retrains_per_week: int = 3
    
    # File to track retraining history
    retraining_log_path: Path = field(
        default_factory=lambda: LOGS_DIR / "retraining_history.json"
    )


# =============================================================================
# MONITORING SCHEDULE
# =============================================================================

@dataclass
class ScheduleConfig:
    """Configuration for monitoring schedule."""
    
    # How often to run monitoring (cron expression)
    monitoring_schedule: str = "0 6 * * *"  # Daily at 6 AM
    
    # Data window to analyze (hours)
    analysis_window_hours: int = 24
    
    # How far back to look for ground truth (days)
    # BlueBikes data typically available next day
    ground_truth_delay_days: int = 1
    
    # Retention for monitoring reports (days)
    report_retention_days: int = 90
    
    # Retention for baselines (keep last N versions)
    baseline_retention_count: int = 10


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================

@dataclass 
class MonitoringConfig:
    """Master configuration combining all monitoring settings."""
    
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evidently: EvidentlyConfig = field(default_factory=EvidentlyConfig)
    drift: DriftThresholds = field(default_factory=DriftThresholds)
    prediction: PredictionDriftThresholds = field(default_factory=PredictionDriftThresholds)
    performance: PerformanceThresholds = field(default_factory=PerformanceThresholds)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging/serialization."""
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
        
        # Override thresholds from environment if set
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
    """Get the monitoring configuration (singleton pattern)."""
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
    
    # Check thresholds are sensible
    if config.drift.psi_no_drift >= config.drift.psi_minor_drift:
        warnings.append("PSI no_drift threshold >= minor_drift threshold")
    
    if config.drift.psi_minor_drift >= config.drift.psi_major_drift:
        warnings.append("PSI minor_drift threshold >= major_drift threshold")
    
    if config.performance.r2_minimum > 1.0 or config.performance.r2_minimum < 0:
        warnings.append(f"Invalid R² minimum: {config.performance.r2_minimum}")
    
    # Check Discord webhook
    if config.alerts.enable_warning_alerts and not config.alerts.discord_webhook_url:
        warnings.append("Warning alerts enabled but DISCORD_WEBHOOK_URL not set")
    
    # Check feature lists
    if not config.features.numerical_features and not config.features.categorical_features:
        warnings.append("No features configured for monitoring")
    
    return warnings


# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    
    print("=" * 60)
    print("MONITORING CONFIGURATION")
    print("=" * 60)
    
    print(f"\nPaths:")
    print(f"  Baselines: {BASELINES_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  Logs: {LOGS_DIR}")
    
    print(f"\nFeatures to monitor:")
    print(f"  Numerical: {len(config.features.numerical_features)}")
    print(f"  Categorical: {len(config.features.categorical_features)}")
    print(f"  Critical: {config.features.critical_features}")
    
    print(f"\nDrift Thresholds:")
    print(f"  PSI (no/minor/major): {config.drift.psi_no_drift}/{config.drift.psi_minor_drift}/{config.drift.psi_major_drift}")
    print(f"  KS p-value: {config.drift.ks_pvalue_threshold}")
    
    print(f"\nPerformance Thresholds:")
    print(f"  R² minimum: {config.performance.r2_minimum}")
    print(f"  R² drop critical: {config.performance.r2_drop_critical}")
    
    print(f"\nRetraining:")
    print(f"  Auto-retrain enabled: {config.retraining.auto_retrain_enabled}")
    print(f"  Target DAG: {config.retraining.retraining_dag_id}")
    
    # Validate
    warnings = validate_config(config)
    if warnings:
        print(f"\n⚠️  Configuration Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print(f"\n✓ Configuration valid")