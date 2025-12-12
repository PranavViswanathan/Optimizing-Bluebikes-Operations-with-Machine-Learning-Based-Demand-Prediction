"""
Drift Detection Module for BlueBikes Model Monitoring
Uses Evidently AI for drift detection as recommended in deployment guidelines.

Implements:
- Data drift detection (feature distribution changes)
- Prediction drift detection (model output changes)
- Target drift detection (when ground truth available)
- Model performance monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import pickle
import logging
import joblib

# Evidently AI imports for v0.6.x classic API
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric,
    RegressionQualityMetric,
    # RegressionPredictedVsActual,
    RegressionErrorDistribution,
)
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset, RegressionTestPreset
from evidently.tests import (
    TestColumnDrift,
    TestShareOfDriftedColumns,
    TestMeanInNSigmas,
)

from monitoring_config import (
    MonitoringConfig,
    get_config,
    get_report_path,
    get_baseline_path,
    HTML_REPORTS_DIR,
    BASELINES_DIR,
    JSON_REPORTS_DIR,
    PRODUCTION_MODEL_PATH,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvidentlyDriftDetector:
    """
    Drift detection using Evidently AI.
    
    Provides:
    - Data drift reports with statistical tests
    - Prediction drift monitoring
    - Regression quality metrics (when ground truth available)
    - HTML reports for visualization
    - JSON summaries for programmatic access
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or get_config()
        self.reference_data: Optional[pd.DataFrame] = None
        self.column_mapping: Optional[ColumnMapping] = None
        self.model = None
        self.drift_report: Dict = {}
        
    def load_reference_data(self, month=None):
        """
        Load baseline reference data.
        
        Args:
            month: Optional month number (1-12). If provided, loads month-specific baseline.
                Falls back to overall baseline if month-specific doesn't exist.
        """
        config = get_config()
        
        # Try to load month-specific baseline if requested and enabled
        if month is not None and config.use_monthly_baselines:
            from baseline_stats import get_month_name
            month_name = get_month_name(month)
            baseline_path = BASELINES_DIR / f"baseline_{month_name}.pkl"
            
            if baseline_path.exists():
                self.reference_data = pd.read_pickle(baseline_path)
                logger.info(f"✓ Loaded {month_name} baseline: {len(self.reference_data)} samples")
                logger.info(f"  Comparing {month_name} current data vs {month_name} historical baseline")
                return
            else:
                logger.warning(f"  No baseline for {month_name}, falling back to overall baseline")
        
        # Fall back to overall baseline
        baseline_path = get_baseline_path()
        if baseline_path.exists():
            self.reference_data = pd.read_pickle(baseline_path)
            logger.info(f"✓ Loaded overall baseline: {len(self.reference_data)} samples")
            logger.info(f"  Using overall baseline (all months combined)")
        else:
            raise FileNotFoundError(
                f"No baseline found. Run baseline generation first.\n"
                f"Expected at: {baseline_path}"
            )
    
    def load_model(self, model_path: Optional[Path] = None):
        """Load the production model for making predictions."""
        if model_path is None:
            model_path = PRODUCTION_MODEL_PATH
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def _setup_column_mapping(self):
        """Setup Evidently column mapping based on config."""
        numerical = [f for f in self.config.features.numerical_features 
                    if f in self.reference_data.columns]
        categorical = [f for f in self.config.features.categorical_features 
                      if f in self.reference_data.columns]
        
        self.column_mapping = ColumnMapping(
            target=self.config.features.target_column if self.config.features.target_column in self.reference_data.columns else None,
            prediction='prediction' if 'prediction' in self.reference_data.columns else None,
            numerical_features=numerical,
            categorical_features=categorical,
        )
        
        logger.info(f"Column mapping: {len(numerical)} numerical, {len(categorical)} categorical features")
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        generate_html: bool = True,
        save_report: bool = True,
        report_date: Optional[str] = None
    ) -> Dict:
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data: Current production data to check for drift
            generate_html: Whether to generate HTML report
            save_report: Whether to save reports to disk
            report_date: Date string for report naming (default: today)
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded. Call load_reference_data() first.")
        
        logger.info("="*60)
        logger.info("RUNNING DATA DRIFT DETECTION (Evidently AI)")
        logger.info("="*60)
        
        if report_date is None:
            report_date = datetime.now().strftime("%Y%m%d")
        
        # Clean data - remove date column if present
        reference_clean = self.reference_data.drop('date', axis=1, errors='ignore')
        current_clean = current_data.drop('date', axis=1, errors='ignore')
        
        # Ensure same columns
        common_cols = list(set(reference_clean.columns) & set(current_clean.columns))
        reference_clean = reference_clean[common_cols]
        current_clean = current_clean[common_cols]
        
        logger.info(f"Reference samples: {len(reference_clean)}")
        logger.info(f"Current samples: {len(current_clean)}")
        logger.info(f"Features analyzed: {len(common_cols)}")
        
        # Build Evidently Report
        data_drift_report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        
        data_drift_report.run(
            reference_data=reference_clean,
            current_data=current_clean,
            column_mapping=self.column_mapping
        )
        
        # Extract results
        report_dict = data_drift_report.as_dict()
        
        # Parse drift results
        drift_results = self._parse_drift_results(report_dict)
        
        # Add metadata
        drift_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'report_date': report_date,
            'reference_samples': len(reference_clean),
            'current_samples': len(current_clean),
            'n_features': len(common_cols),
        }
        
        # Determine overall status and action
        drift_results['overall_status'], drift_results['recommended_action'] = \
            self._determine_status(drift_results)
        
        # Generate alerts
        drift_results['alerts'] = self._generate_alerts(drift_results)
        
        self.drift_report = drift_results
        
        # Save reports
        if save_report:
            self._save_json_report(drift_results, report_date)
        
        if generate_html:
            html_path = self._save_html_report(data_drift_report, report_date, "data_drift")
            drift_results['html_report_path'] = str(html_path)
        
        # Log summary
        self._log_drift_summary(drift_results)
        
        return drift_results
    
    def detect_prediction_drift(
        self,
        current_data: pd.DataFrame,
        generate_html: bool = True,
        report_date: Optional[str] = None
    ) -> Dict:
        """
        Detect drift in model predictions.
        
        Makes predictions on current data and compares distribution
        to reference predictions.
        """
        if self.model is None:
            self.load_model()
        
        if self.reference_data is None:
            raise ValueError("Reference data not loaded.")
        
        logger.info("Running prediction drift detection...")
        
        if report_date is None:
            report_date = datetime.now().strftime("%Y%m%d")
        
        # Prepare data for prediction
        feature_cols = [c for c in current_data.columns 
                       if c not in ['date', self.config.features.target_column, 'prediction']]
        
        # Make predictions
        current_preds = self.model.predict(current_data[feature_cols])
        
        # Get reference predictions (or make them)
        if 'prediction' in self.reference_data.columns:
            ref_preds = self.reference_data['prediction'].values
        else:
            ref_feature_cols = [c for c in self.reference_data.columns 
                               if c not in ['date', self.config.features.target_column, 'prediction']]
            ref_preds = self.model.predict(self.reference_data[ref_feature_cols])
        
        # Calculate prediction statistics
        pred_results = {
            'reference_stats': {
                'mean': float(np.mean(ref_preds)),
                'std': float(np.std(ref_preds)),
                'min': float(np.min(ref_preds)),
                'max': float(np.max(ref_preds)),
                'median': float(np.median(ref_preds)),
            },
            'current_stats': {
                'mean': float(np.mean(current_preds)),
                'std': float(np.std(current_preds)),
                'min': float(np.min(current_preds)),
                'max': float(np.max(current_preds)),
                'median': float(np.median(current_preds)),
            },
        }
        
        # Calculate drift metrics
        ref_mean = pred_results['reference_stats']['mean']
        curr_mean = pred_results['current_stats']['mean']
        
        if ref_mean != 0:
            mean_shift_pct = abs(curr_mean - ref_mean) / abs(ref_mean) * 100
        else:
            mean_shift_pct = abs(curr_mean) * 100
        
        ref_std = pred_results['reference_stats']['std']
        curr_std = pred_results['current_stats']['std']
        
        if ref_std != 0:
            std_change_pct = abs(curr_std - ref_std) / ref_std * 100
        else:
            std_change_pct = abs(curr_std) * 100
        
        pred_results['drift_metrics'] = {
            'mean_shift_pct': mean_shift_pct,
            'std_change_pct': std_change_pct,
        }
        
        # Determine if drift detected
        pred_results['drift_detected'] = (
            mean_shift_pct >= self.config.prediction.mean_shift_warning or
            std_change_pct >= self.config.prediction.std_change_warning
        )
        
        pred_results['drift_severity'] = 'none'
        if mean_shift_pct >= self.config.prediction.mean_shift_critical:
            pred_results['drift_severity'] = 'major'
        elif pred_results['drift_detected']:
            pred_results['drift_severity'] = 'minor'
        
        logger.info(f"Prediction drift - Mean shift: {mean_shift_pct:.1f}%, Std change: {std_change_pct:.1f}%")
        logger.info(f"Drift detected: {pred_results['drift_detected']} ({pred_results['drift_severity']})")
        
        return pred_results
    
    def detect_performance_drift(
        self,
        current_data: pd.DataFrame,
        current_actuals: pd.Series,
        generate_html: bool = True,
        report_date: Optional[str] = None
    ) -> Dict:
        """
        Detect model performance degradation using ground truth.
        
        Args:
            current_data: Current features
            current_actuals: Actual target values (ground truth)
            generate_html: Whether to generate HTML report
            report_date: Date for report naming
        """
        if self.model is None:
            self.load_model()
        
        logger.info("Running performance drift detection...")
        
        if report_date is None:
            report_date = datetime.now().strftime("%Y%m%d")
        
        # Prepare data
        feature_cols = [c for c in current_data.columns 
                       if c not in ['date', self.config.features.target_column, 'prediction']]
        
        # Make predictions
        predictions = self.model.predict(current_data[feature_cols])
        
        # Create DataFrame for Evidently
        eval_df = current_data.copy()
        eval_df['prediction'] = predictions
        eval_df['target'] = current_actuals.values
        
        # Reference data with predictions and targets
        ref_df = self.reference_data.copy()
        if 'prediction' not in ref_df.columns:
            ref_feature_cols = [c for c in ref_df.columns 
                               if c not in ['date', self.config.features.target_column, 'prediction']]
            ref_df['prediction'] = self.model.predict(ref_df[ref_feature_cols])
        if 'target' not in ref_df.columns and self.config.features.target_column in ref_df.columns:
            ref_df['target'] = ref_df[self.config.features.target_column]
        
        # Build regression quality report
        regression_report = Report(metrics=[
            RegressionQualityMetric(),
            # RegressionPredictedVsActual(),
            RegressionErrorDistribution(),
        ])
        
        # Update column mapping for regression
        reg_column_mapping = ColumnMapping(
            target='target',
            prediction='prediction',
        )
        
        regression_report.run(
            reference_data=ref_df[['target', 'prediction']],
            current_data=eval_df[['target', 'prediction']],
            column_mapping=reg_column_mapping
        )
        
        # Extract metrics
        report_dict = regression_report.as_dict()
        
        # Parse regression metrics
        perf_results = self._parse_regression_results(report_dict)
        
        # Determine if performance degraded
        perf_results['performance_degraded'] = self._check_performance_degradation(perf_results)
        
        if generate_html:
            html_path = self._save_html_report(regression_report, report_date, "performance")
            perf_results['html_report_path'] = str(html_path)
        
        return perf_results
    
    def run_full_monitoring(
        self,
        current_data: pd.DataFrame,
        current_actuals: Optional[pd.Series] = None,
        generate_html: bool = True,
        save_reports: bool = True
    ) -> Dict:
        """
        Run complete monitoring suite.
        
        Args:
            current_data: Current production data
            current_actuals: Optional ground truth (if available)
            generate_html: Whether to generate HTML reports
            save_reports: Whether to save reports
            
        Returns:
            Complete monitoring report
        """
        logger.info("="*60)
        logger.info("RUNNING FULL MONITORING SUITE")
        logger.info("="*60)
        
        report_date = datetime.now().strftime("%Y%m%d")
        
        full_report = {
            'timestamp': datetime.now().isoformat(),
            'report_date': report_date,
        }
        
        # 1. Data Drift Detection
        try:
            full_report['data_drift'] = self.detect_data_drift(
                current_data, generate_html, save_reports, report_date
            )
        except Exception as e:
            logger.error(f"Data drift detection failed: {e}")
            full_report['data_drift'] = {'error': str(e)}
        
        # 2. Prediction Drift Detection
        try:
            full_report['prediction_drift'] = self.detect_prediction_drift(
                current_data, generate_html, report_date
            )
        except Exception as e:
            logger.error(f"Prediction drift detection failed: {e}")
            full_report['prediction_drift'] = {'error': str(e)}
        
        # 3. Performance Monitoring (if ground truth available)
        if current_actuals is not None:
            try:
                full_report['performance'] = self.detect_performance_drift(
                    current_data, current_actuals, generate_html, report_date
                )
            except Exception as e:
                logger.error(f"Performance monitoring failed: {e}")
                full_report['performance'] = {'error': str(e)}
        else:
            full_report['performance'] = {'status': 'skipped', 'reason': 'No ground truth provided'}
        
        # 4. Determine overall status
        full_report['overall_status'], full_report['recommended_action'] = \
            self._determine_overall_status(full_report)
        
        # 5. Generate combined alerts
        full_report['alerts'] = self._generate_combined_alerts(full_report)
        
        # Save combined report
        if save_reports:
            json_path = get_report_path(report_date, "json")
            with open(json_path, 'w') as f:
                json.dump(full_report, f, indent=2, default=str)
            logger.info(f"Full report saved to: {json_path}")
        
        self.drift_report = full_report
        
        return full_report
    
    def _parse_drift_results(self, report_dict: Dict) -> Dict:
        """Parse Evidently report dictionary into structured results."""
        results = {
            'dataset_drift': False,
            'drift_share': 0.0,
            'n_drifted_features': 0,
            'n_features_analyzed': 0,
            'drifted_features': [],
            'feature_details': {},
        }
        
        try:
            metrics = report_dict.get('metrics', [])
            
            for metric in metrics:
                metric_id = metric.get('metric', '')
                result = metric.get('result', {})
                
                if 'DatasetDriftMetric' in metric_id:
                    results['dataset_drift'] = result.get('dataset_drift', False)
                    results['drift_share'] = result.get('drift_share', 0.0)
                    results['n_drifted_features'] = result.get('number_of_drifted_columns', 0)
                    results['n_features_analyzed'] = result.get('number_of_columns', 0)
                
                elif 'DataDriftTable' in metric_id:
                    drift_by_columns = result.get('drift_by_columns', {})
                    for col_name, col_data in drift_by_columns.items():
                        is_drifted = col_data.get('drift_detected', False)
                        results['feature_details'][col_name] = {
                            'drift_detected': is_drifted,
                            'drift_score': col_data.get('drift_score', 0.0),
                            'stattest_name': col_data.get('stattest_name', ''),
                            'stattest_threshold': col_data.get('stattest_threshold', 0.0),
                        }
                        if is_drifted:
                            results['drifted_features'].append(col_name)
        
        except Exception as e:
            logger.error(f"Error parsing drift results: {e}")
        
        return results
    
    def _parse_regression_results(self, report_dict: Dict) -> Dict:
        """Parse Evidently regression report."""
        results = {
            'reference_metrics': {},
            'current_metrics': {},
            'metric_changes': {},
        }
        
        try:
            metrics = report_dict.get('metrics', [])
            
            for metric in metrics:
                if 'RegressionQualityMetric' in metric.get('metric', ''):
                    result = metric.get('result', {})
                    
                    # Current metrics
                    current = result.get('current', {})
                    results['current_metrics'] = {
                        'mae': current.get('mean_abs_error', 0),
                        'rmse': np.sqrt(current.get('mean_error', 0)**2 + current.get('error_std', 0)**2),
                        'r2': current.get('r2_score', 0),
                        'mape': current.get('mean_abs_perc_error', 0),
                    }
                    
                    # Reference metrics  
                    reference = result.get('reference', {})
                    if reference:
                        results['reference_metrics'] = {
                            'mae': reference.get('mean_abs_error', 0),
                            'rmse': np.sqrt(reference.get('mean_error', 0)**2 + reference.get('error_std', 0)**2),
                            'r2': reference.get('r2_score', 0),
                            'mape': reference.get('mean_abs_perc_error', 0),
                        }
        
        except Exception as e:
            logger.error(f"Error parsing regression results: {e}")
        
        return results
    
    def _check_performance_degradation(self, perf_results: Dict) -> bool:
        """Check if performance has degraded beyond thresholds."""
        ref = perf_results.get('reference_metrics', {})
        curr = perf_results.get('current_metrics', {})
        
        if not ref or not curr:
            return False
        
        # Check R² drop
        r2_drop = ref.get('r2', 0) - curr.get('r2', 0)
        if r2_drop >= self.config.performance.r2_drop_critical:
            return True
        
        # Check MAE increase
        ref_mae = ref.get('mae', 0)
        if ref_mae > 0:
            mae_increase_pct = (curr.get('mae', 0) - ref_mae) / ref_mae * 100
            if mae_increase_pct >= self.config.performance.mae_increase_critical:
                return True
        
        return False
    
    def _determine_status(self, drift_results: Dict) -> Tuple[str, str]:
        """Determine overall status and recommended action from drift results."""
        if drift_results.get('dataset_drift', False):
            drift_share = drift_results.get('drift_share', 0)
            if drift_share >= 0.5:  # More than 50% features drifted
                return "CRITICAL", "retrain"
            else:
                return "WARNING", "monitor"
        
        n_drifted = drift_results.get('n_drifted_features', 0)
        if n_drifted >= self.config.drift.min_features_for_critical:
            return "WARNING", "monitor"
        
        return "HEALTHY", "none"
    
    def _determine_overall_status(self, full_report: Dict) -> Tuple[str, str]:
        """Determine overall status from complete monitoring report."""
        statuses = []
        
        # Check data drift
        data_drift = full_report.get('data_drift', {})
        if data_drift.get('overall_status') == 'CRITICAL':
            return "CRITICAL", "retrain"
        if data_drift.get('overall_status') == 'WARNING':
            statuses.append('WARNING')
        
        # Check prediction drift
        pred_drift = full_report.get('prediction_drift', {})
        if pred_drift.get('drift_severity') == 'major':
            return "CRITICAL", "retrain"
        if pred_drift.get('drift_detected'):
            statuses.append('WARNING')
        
        # Check performance
        performance = full_report.get('performance', {})
        if performance.get('performance_degraded'):
            return "CRITICAL", "retrain"
        
        if 'WARNING' in statuses:
            return "WARNING", "monitor"
        
        return "HEALTHY", "none"
    
    def _generate_alerts(self, drift_results: Dict) -> List[str]:
        """Generate alert messages from drift results."""
        alerts = []
        
        if drift_results.get('dataset_drift'):
            n_drifted = drift_results.get('n_drifted_features', 0)
            n_total = drift_results.get('n_features_analyzed', 0)
            alerts.append(f"DATA DRIFT DETECTED: {n_drifted}/{n_total} features drifted")
        
        drifted = drift_results.get('drifted_features', [])[:5]
        if drifted:
            alerts.append(f"Drifted features: {', '.join(drifted)}")
        
        return alerts
    
    def _generate_combined_alerts(self, full_report: Dict) -> List[str]:
        """Generate combined alerts from full monitoring report."""
        alerts = []
        
        # Data drift alerts
        data_drift = full_report.get('data_drift', {})
        if data_drift.get('alerts'):
            alerts.extend(data_drift['alerts'])
        
        # Prediction drift alerts
        pred_drift = full_report.get('prediction_drift', {})
        if pred_drift.get('drift_detected'):
            mean_shift = pred_drift.get('drift_metrics', {}).get('mean_shift_pct', 0)
            alerts.append(f"PREDICTION DRIFT: Mean shifted by {mean_shift:.1f}%")
        
        # Performance alerts
        performance = full_report.get('performance', {})
        if performance.get('performance_degraded'):
            alerts.append("PERFORMANCE DEGRADATION: Model accuracy below threshold")
        
        return alerts
    
    def _save_html_report(self, report: Report, date_str: str, report_type: str) -> Path:
        """Save Evidently HTML report."""
        html_path = HTML_REPORTS_DIR / f"{report_type}_report_{date_str}.html"
        report.save_html(str(html_path))
        logger.info(f"HTML report saved: {html_path}")
        return html_path
    
    def _save_json_report(self, results: Dict, date_str: str):
        """Save JSON summary report."""
        json_path = get_report_path(date_str, "json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"JSON report saved: {json_path}")
    
    def _log_drift_summary(self, results: Dict):
        """Log drift detection summary."""
        logger.info("-"*60)
        logger.info("DRIFT DETECTION SUMMARY")
        logger.info("-"*60)
        logger.info(f"Overall Status: {results.get('overall_status', 'UNKNOWN')}")
        logger.info(f"Dataset Drift: {results.get('dataset_drift', False)}")
        logger.info(f"Drift Share: {results.get('drift_share', 0):.1%}")
        logger.info(f"Features Drifted: {results.get('n_drifted_features', 0)}/{results.get('n_features_analyzed', 0)}")
        
        if results.get('drifted_features'):
            logger.info(f"Top Drifted: {results['drifted_features'][:5]}")
        
        logger.info(f"Recommended Action: {results.get('recommended_action', 'none')}")
        
        if results.get('alerts'):
            logger.info("Alerts:")
            for alert in results['alerts']:
                logger.info(f"  ⚠ {alert}")
        
        logger.info("-"*60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_drift_detection(
    current_data: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    reference_path: Optional[Path] = None,
    include_predictions: bool = True,
    generate_html: bool = True
) -> Dict:
    """
    Convenience function to run drift detection.
    
    Args:
        current_data: Current production data
        reference_data: Optional reference DataFrame
        reference_path: Optional path to reference data
        include_predictions: Whether to check prediction drift
        generate_html: Whether to generate HTML reports
        
    Returns:
        Drift detection results
    """
    detector = EvidentlyDriftDetector()
    
    if reference_data is not None:
        detector.load_reference_data(reference_df=reference_data)
    elif reference_path is not None:
        detector.load_reference_data(path=reference_path)
    else:
        detector.load_reference_data()
    
    results = detector.detect_data_drift(current_data, generate_html=generate_html)
    
    if include_predictions:
        pred_results = detector.detect_prediction_drift(current_data)
        results['prediction_drift'] = pred_results
    
    return results


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evidently AI Drift Detection")
    parser.add_argument("--test", action="store_true", help="Run with synthetic test data")
    parser.add_argument("--check", action="store_true", help="Check if baseline exists")
    
    args = parser.parse_args()
    
    if args.check:
        baseline_path = get_baseline_path()
        if baseline_path.exists():
            print(f"✓ Baseline found: {baseline_path}")
        else:
            print(f"✗ No baseline at: {baseline_path}")
    
    elif args.test:
        logger.info("Running drift detection test with synthetic data...")
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        # Reference data (training distribution)
        reference_df = pd.DataFrame({
            'hour': np.random.randint(0, 24, n_samples),
            'temp_avg': np.random.normal(15, 8, n_samples),
            'rides_last_hour': np.random.exponential(40, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples, p=[0.71, 0.29]),
            'ride_count': np.random.poisson(45, n_samples),
        })
        
        # Current data (with some drift)
        current_df = pd.DataFrame({
            'hour': np.random.randint(0, 24, n_samples),
            'temp_avg': np.random.normal(22, 8, n_samples),  # Mean shifted from 15 to 22
            'rides_last_hour': np.random.exponential(55, n_samples),  # Higher demand
            'is_weekend': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),  # Different ratio
            'ride_count': np.random.poisson(50, n_samples),
        })
        
        # Run detection
        detector = EvidentlyDriftDetector()
        detector.load_reference_data(reference_df=reference_df)
        
        results = detector.detect_data_drift(current_df, generate_html=True, save_report=True)
        
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Dataset Drift: {results['dataset_drift']}")
        print(f"Drift Share: {results['drift_share']:.1%}")
        print(f"Drifted Features: {results['drifted_features']}")
        print(f"Overall Status: {results['overall_status']}")
        print(f"Recommended Action: {results['recommended_action']}")
        
        if results.get('html_report_path'):
            print(f"\nHTML Report: {results['html_report_path']}")
    
    else:
        parser.print_help()