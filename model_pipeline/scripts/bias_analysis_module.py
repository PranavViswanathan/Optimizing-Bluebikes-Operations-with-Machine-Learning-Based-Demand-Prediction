"""
Bias Analysis Module
Handles bias detection and reporting for trained models
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

from bias_detection import BikeShareBiasDetector
from data_module import DataLoader
from pipeline_config import BiasConfig


class BiasAnalyzer:
    """Analyze model bias and generate reports"""
    
    def __init__(self, config: BiasConfig = None):
        self.config = config if config else BiasConfig()
        self.reports = {}
        
    def run_bias_analysis(self, model_path: str, data: dict, stage: str = "baseline"):
        """Run bias detection on a model - matches original implementation exactly"""
        print("\n" + "="*80)
        print(f" BIAS ANALYSIS ({stage.upper()}) ".center(80))
        print("="*80)
        
        X_test = data['X_test']
        y_test = data['y_test']
        
        print(f"Analyzing model: {model_path}")
        print(f"Test samples: {len(X_test):,}")
        
        # Create detector and run analysis
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
        if os.path.exists('bias_analysis_plots.png'):
            plot_filename = f'bias_analysis_plots_{stage}.png'
            # Remove existing file if it exists (Windows compatibility)
            if os.path.exists(plot_filename):
                os.remove(plot_filename)
            os.rename('bias_analysis_plots.png', plot_filename)
            print(f"Bias plots saved to: {plot_filename}")
        
        # Print summary (matching original implementation style)
        self._print_bias_summary(bias_report, stage)
        
        # Store report
        self.reports[stage] = bias_report
        
        return bias_report
    
    def _print_bias_summary(self, report: dict, stage: str):
        """Print a summary of the bias analysis - matching original implementation"""
        print(f"\n{stage.upper()} Model Bias Summary:")
        print("="*60)
        
        overall = report.get('overall_performance', {})
        
        # Helper function to safely format numeric values
        def format_value(val, fmt):
            if val is None or val == 'N/A':
                return 'N/A'
            try:
                return f"{float(val):{fmt}}"
            except (ValueError, TypeError):
                return 'N/A'
        
        print(f"Overall Performance:")
        print(f"  MAE: {format_value(overall.get('mae'), '.2f')}")
        print(f"  RMSE: {format_value(overall.get('rmse'), '.2f')}")
        print(f"  R²: {format_value(overall.get('r2'), '.4f')}")
        print(f"  MAPE: {format_value(overall.get('mape'), '.2f')}%")
        
        bias_issues = report.get('bias_detected', [])
        print(f"\nBias Issues Detected: {len(bias_issues)}")
        
        # Note: Detailed bias issues are saved in the JSON report
        # The original implementation doesn't print individual issues here
        print(f"\n→ See detailed bias analysis in the saved JSON report")
        print(f"→ See visualizations in bias_analysis_plots_{stage}.png")
    
    def analyze_from_metadata(self, metadata_path: str, data: dict, stage: str = "baseline"):
        """Run bias analysis using model metadata file"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_path = metadata['model_path']
        return self.run_bias_analysis(model_path, data, stage)
    
    def load_existing_report(self, report_path: str, stage: str = None):
        """Load an existing bias report"""
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        if stage:
            self.reports[stage] = report
        
        return report
    
    def inspect_report_structure(self, report: dict):
        """Inspect and print the structure of a bias report"""
        print("\n" + "="*60)
        print(" BIAS REPORT STRUCTURE ".center(60))
        print("="*60)
        
        print("\nTop-level keys:")
        for key in report.keys():
            print(f"  • {key}")
        
        print("\nOverall performance metrics:")
        overall = report.get('overall_performance', {})
        for key, value in overall.items():
            print(f"  • {key}: {value}")
        
        print("\nBias issues structure:")
        bias_issues = report.get('bias_detected', [])
        print(f"  Total issues: {len(bias_issues)}")
        
        if bias_issues and len(bias_issues) > 0:
            print(f"\n  First issue structure:")
            first_issue = bias_issues[0]
            if isinstance(first_issue, dict):
                for key, value in first_issue.items():
                    print(f"    • {key}: {type(value).__name__} = {value}")
            else:
                print(f"    Type: {type(first_issue).__name__}")
                print(f"    Value: {first_issue}")
        
        # Check for other common sections
        for section in ['temporal_analysis', 'hourly_analysis', 'categorical_analysis', 
                       'feature_importance', 'slice_analysis']:
            if section in report:
                print(f"\n{section}:")
                data = report[section]
                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")
                elif isinstance(data, list):
                    print(f"  Items: {len(data)}")
                else:
                    print(f"  Type: {type(data).__name__}")
    
    def compare_reports(self, baseline_report: dict = None, mitigated_report: dict = None):
        """Compare two bias reports"""
        print("\n" + "="*80)
        print(" BIAS COMPARISON ".center(80))
        print("="*80)
        
        if baseline_report is None:
            baseline_report = self.reports.get('baseline')
        if mitigated_report is None:
            mitigated_report = self.reports.get('mitigated')
        
        if not baseline_report or not mitigated_report:
            print("Warning: Missing reports for comparison")
            print(f"Baseline: {' ' if baseline_report else '✗'}")
            print(f"Mitigated: {' ' if mitigated_report else '✗'}")
            return None
        
        baseline_overall = baseline_report.get('overall_performance', {})
        mitigated_overall = mitigated_report.get('overall_performance', {})
        
        print("\nOverall Performance Comparison:")
        print("="*60)
        
        metrics = ['mae', 'rmse', 'r2', 'mape']
        
        for metric in metrics:
            baseline_val = baseline_overall.get(metric, 0)
            mitigated_val = mitigated_overall.get(metric, 0)
            
            # Safely convert to float
            try:
                baseline_val = float(baseline_val) if baseline_val is not None else 0
                mitigated_val = float(mitigated_val) if mitigated_val is not None else 0
            except (ValueError, TypeError):
                baseline_val = 0
                mitigated_val = 0
            
            if metric == 'r2':
                diff = mitigated_val - baseline_val
                better = diff > 0
                pct_change = (diff / baseline_val * 100) if baseline_val != 0 else 0
            else:
                diff = baseline_val - mitigated_val
                better = diff > 0
                pct_change = (diff / baseline_val * 100) if baseline_val != 0 else 0
            
            symbol = "+" if better else "-"
            print(f"{metric.upper():6s}: {baseline_val:8.4f} -> {mitigated_val:8.4f} "
                  f"({symbol}{abs(pct_change):.2f}%)")
        
        print("\nBias Issues Comparison:")
        print("="*60)
        
        baseline_issues = len(baseline_report.get('bias_detected', []))
        mitigated_issues = len(mitigated_report.get('bias_detected', []))
        
        print(f"Baseline: {baseline_issues} issues detected")
        print(f"Mitigated: {mitigated_issues} issues detected")
        
        reduction = baseline_issues - mitigated_issues
        if reduction > 0:
            print(f"Reduced bias issues by {reduction} ({reduction/baseline_issues*100:.1f}%)")
        elif reduction < 0:
            print(f"Bias issues increased by {abs(reduction)}")
        else:
            print(f"No change in number of bias issues")
        
        # Create comparison report
        comparison = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'baseline_metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                for k, v in baseline_overall.items()},
            'mitigated_metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                 for k, v in mitigated_overall.items()},
            'baseline_bias_issues': int(baseline_issues),
            'mitigated_bias_issues': int(mitigated_issues),
            'improvement': {
                'mae_improvement': float(baseline_overall.get('mae', 0) - mitigated_overall.get('mae', 0)),
                'rmse_improvement': float(baseline_overall.get('rmse', 0) - mitigated_overall.get('rmse', 0)),
                'r2_improvement': float(mitigated_overall.get('r2', 0) - baseline_overall.get('r2', 0)),
                'bias_issues_reduction': int(reduction)
            }
        }
        
        # Save comparison
        comparison_filename = f'bias_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(comparison_filename, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nComparison report saved to: {comparison_filename}")
        
        return comparison


def main():
    """Standalone execution for bias analysis"""
    print("="*80)
    print(" BLUEBIKES BIAS ANALYSIS MODULE ".center(80))
    print("="*80)
    
    # Load data
    data = DataLoader.load_data()
    
    # Load model metadata
    metadata_path = "best_model_metadata.json"
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found. Run model training first.")
        return None
    
    # Initialize analyzer
    analyzer = BiasAnalyzer()
    
    # Run bias analysis
    report = analyzer.analyze_from_metadata(metadata_path, data, stage="baseline")
    
    print("\n" + "="*80)
    print(" BIAS ANALYSIS COMPLETE ".center(80))
    print("="*80)
    
    return analyzer, report


if __name__ == "__main__":
    analyzer, report = main()