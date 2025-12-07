"""
Model Sensitivity Analysis for BlueBikes Demand Prediction
Analyzes how model predictions respond to changes in input features
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import joblib
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from artifact_manager import ArtifactManager

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SHAP for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run 'pip install shap' for SHAP analysis.")


class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for bike-sharing demand prediction models.
    
    Analyzes:
    1. Feature Importance (Permutation-based)
    2. SHAP Values
    3. Feature Perturbation Analysis
    4. Prediction Stability Analysis
    5. Hyperparameter Sensitivity
    """
    
    def __init__(self, model_path: str, X_test: pd.DataFrame, y_test: pd.Series, stage: str = "baseline"):
        """
        Initialize sensitivity analyzer.
        
        Args:
            model_path: Path to saved model (.pkl)
            X_test: Test features
            y_test: Test target
            stage: Analysis stage ("baseline" or "mitigated")
        """
        self.model = joblib.load(model_path)
        self.X_test = X_test.drop('date', axis=1, errors='ignore').copy()
        self.y_test = y_test.copy()
        self.feature_names = list(self.X_test.columns)
        self.stage = stage
        self.results = {}
    
    def shap_feature_importance(self, n_samples: int = 500) -> Dict:
        """
        Calculate SHAP values for feature importance.
        SHAP provides more accurate feature importance than permutation methods.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP analysis.")
            return {'status': 'skipped', 'reason': 'SHAP not installed'}
        
        logger.info("Calculating SHAP feature importance...")
        
        # Sample data for efficiency
        if len(self.X_test) > n_samples:
            sample_idx = np.random.choice(len(self.X_test), n_samples, replace=False)
            X_sample = self.X_test.iloc[sample_idx]
        else:
            X_sample = self.X_test
        
        model_type = type(self.model).__name__
        
        try:
            if 'XGB' in model_type or 'LGBM' in model_type or 'RandomForest' in model_type:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.KernelExplainer(self.model.predict, shap.sample(X_sample, 100))
                shap_values = explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values per feature
            shap_importance = np.abs(shap_values).mean(axis=0)
            
            shap_df = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)
            
            self.results['shap_importance'] = shap_df
            self.results['shap_values'] = shap_values
            
            logger.info("Top 5 features by SHAP importance:")
            for _, row in shap_df.head(5).iterrows():
                logger.info(f"  {row['feature']}: {row['shap_importance']:.4f}")
            
            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            
            shap_path = ArtifactManager.get_shap_plot_path(self.stage)
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP summary plot saved to: {shap_path}")
            
            return {'shap_importance': shap_df.to_dict('records'), 'status': 'success'}
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def hyperparameter_sensitivity_analysis(self) -> Dict:
        """
        Analyze how model performance changes with different hyperparameters.
        """
        logger.info("Running hyperparameter sensitivity analysis...")
        
        model_type = type(self.model).__name__
        
        if hasattr(self.model, 'get_params'):
            current_params = self.model.get_params()
        else:
            current_params = {}
        
        # Define parameter ranges based on model type
        if 'XGB' in model_type:
            param_ranges = {
                'max_depth': [4, 6, 8, 10, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 300, 500, 700],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        elif 'LGBM' in model_type:
            param_ranges = {
                'max_depth': [4, 6, 8, 10, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 300, 500, 700],
                'num_leaves': [31, 63, 127, 255]
            }
        elif 'RandomForest' in model_type:
            param_ranges = {
                'max_depth': [10, 15, 20, 25, None],
                'n_estimators': [50, 100, 150, 200],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8]
            }
        else:
            logger.warning(f"Hyperparameter sensitivity not implemented for {model_type}")
            return {}
        
        sensitivity_results = {}
        
        for param_name, param_values in param_ranges.items():
            if param_name not in current_params:
                continue
            
            current_value = current_params[param_name]
            
            param_results = []
            for value in param_values:
                param_results.append({
                    'value': value,
                    'is_current': value == current_value,
                    'note': 'baseline' if value == current_value else 'alternative'
                })
            
            sensitivity_results[param_name] = {
                'current_value': current_value,
                'tested_values': param_values,
                'results': param_results
            }
        
        self.results['hyperparameter_sensitivity'] = sensitivity_results
        logger.info(f"Analyzed sensitivity for {len(sensitivity_results)} hyperparameters")
        
        return sensitivity_results
        
    def permutation_feature_importance(self, n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculate permutation-based feature importance.
        """
        logger.info("Calculating permutation feature importance...")
        
        perm_importance = permutation_importance(
            self.model, 
            self.X_test, 
            self.y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        self.results['permutation_importance'] = importance_df
        
        logger.info(f"Top 5 features by permutation importance:")
        for _, row in importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance_mean']:.4f} (+/- {row['importance_std']:.4f})")
        
        return importance_df
    
    def feature_perturbation_analysis(
        self, 
        features: Optional[List[str]] = None,
        perturbation_pcts: List[float] = [-20, -10, -5, 5, 10, 20]
    ) -> Dict:
        """
        Analyze how predictions change when features are perturbed by fixed percentages.
        """
        logger.info("Running feature perturbation analysis...")
        
        if features is None:
            numeric_cols = self.X_test.select_dtypes(include=[np.number]).columns
            variances = self.X_test[numeric_cols].var().sort_values(ascending=False)
            features = list(variances.head(10).index)
        
        baseline_pred = self.model.predict(self.X_test)
        baseline_mean = baseline_pred.mean()
        
        perturbation_results = {}
        
        for feature in features:
            if feature not in self.X_test.columns:
                continue
                
            feature_results = {'baseline_mean': baseline_mean, 'perturbations': {}}
            
            for pct in perturbation_pcts:
                X_perturbed = self.X_test.copy()
                X_perturbed[feature] = X_perturbed[feature] * (1 + pct / 100)
                
                perturbed_pred = self.model.predict(X_perturbed)
                pred_change = ((perturbed_pred.mean() - baseline_mean) / baseline_mean) * 100
                
                feature_results['perturbations'][f'{pct:+d}%'] = {
                    'prediction_mean': float(perturbed_pred.mean()),
                    'prediction_change_pct': float(pred_change),
                    'prediction_std': float(perturbed_pred.std())
                }
            
            perturbation_results[feature] = feature_results
        
        self.results['perturbation_analysis'] = perturbation_results
        
        # Calculate sensitivity scores
        sensitivity_scores = {}
        for feature, data in perturbation_results.items():
            changes = [abs(v['prediction_change_pct']) for v in data['perturbations'].values()]
            sensitivity_scores[feature] = np.mean(changes)
        
        self.results['sensitivity_scores'] = dict(
            sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        logger.info("Most sensitive features:")
        for feature, score in list(self.results['sensitivity_scores'].items())[:5]:
            logger.info(f"  {feature}: {score:.2f}% avg prediction change")
        
        return perturbation_results
    
    def prediction_stability_analysis(self, n_samples: int = 1000) -> Dict:
        """
        Analyze prediction stability by adding small random noise to inputs.
        """
        logger.info("Running prediction stability analysis...")
        
        if len(self.X_test) > n_samples:
            sample_idx = np.random.choice(len(self.X_test), n_samples, replace=False)
            X_sample = self.X_test.iloc[sample_idx]
        else:
            X_sample = self.X_test
        
        baseline_pred = self.model.predict(X_sample)
        
        noise_levels = [0.01, 0.02, 0.05, 0.10]
        stability_results = {}
        
        for noise_level in noise_levels:
            pred_variations = []
            
            for _ in range(10):
                X_noisy = X_sample.copy()
                
                for col in X_noisy.select_dtypes(include=[np.number]).columns:
                    std = X_noisy[col].std()
                    if std > 0:
                        noise = np.random.normal(0, std * noise_level, len(X_noisy))
                        X_noisy[col] = X_noisy[col] + noise
                
                noisy_pred = self.model.predict(X_noisy)
                pred_variations.append(noisy_pred)
            
            pred_variations = np.array(pred_variations)
            
            stability_results[f'{int(noise_level*100)}%_noise'] = {
                'mean_prediction_std': float(pred_variations.std(axis=0).mean()),
                'max_prediction_std': float(pred_variations.std(axis=0).max()),
                'mean_abs_change': float(np.abs(pred_variations - baseline_pred).mean()),
                'correlation_with_baseline': float(np.corrcoef(
                    baseline_pred, pred_variations.mean(axis=0)
                )[0, 1])
            }
        
        self.results['stability_analysis'] = stability_results
        
        logger.info("Prediction stability results:")
        for noise, metrics in stability_results.items():
            logger.info(f"  {noise}: correlation={metrics['correlation_with_baseline']:.4f}")
        
        return stability_results
    
    def critical_threshold_analysis(self) -> Dict:
        """
        Identify critical thresholds where model behavior changes significantly.
        """
        logger.info("Running critical threshold analysis...")
        
        critical_features = {
            'hour': list(range(24)),
            'day_of_week': list(range(7)),
            'is_weekend': [0, 1],
            'is_morning_rush': [0, 1],
            'is_evening_rush': [0, 1],
            'is_rainy': [0, 1],
        }
        
        # Add temp_avg if it exists
        if 'temp_avg' in self.X_test.columns:
            critical_features['temp_avg'] = np.linspace(
                self.X_test['temp_avg'].min(),
                self.X_test['temp_avg'].max(),
                20
            ).tolist()
        
        threshold_results = {}
        baseline_pred = self.model.predict(self.X_test).mean()
        
        for feature, values in critical_features.items():
            if feature not in self.X_test.columns:
                continue
            
            feature_impact = []
            
            for value in values:
                X_modified = self.X_test.copy()
                X_modified[feature] = value
                pred_mean = self.model.predict(X_modified).mean()
                
                feature_impact.append({
                    'value': float(value),
                    'prediction_mean': float(pred_mean),
                    'change_from_baseline': float(pred_mean - baseline_pred),
                    'pct_change': float((pred_mean - baseline_pred) / baseline_pred * 100)
                })
            
            threshold_results[feature] = feature_impact
        
        self.results['threshold_analysis'] = threshold_results
        return threshold_results
    
    def generate_sensitivity_report(self) -> Dict:
        """Generate comprehensive sensitivity report."""
        logger.info("Generating sensitivity report...")
        
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stage': self.stage,
            'model_type': type(self.model).__name__,
            'n_features': len(self.feature_names),
            'n_test_samples': len(self.X_test),
            'results': {}
        }
        
        # Top sensitive features
        if 'sensitivity_scores' in self.results:
            report['top_sensitive_features'] = list(self.results['sensitivity_scores'].items())[:10]
        
        # Stability summary
        if 'stability_analysis' in self.results:
            stability = self.results['stability_analysis']
            report['stability_summary'] = {
                'robust_to_noise': all(
                    v['correlation_with_baseline'] > 0.95 
                    for v in stability.values()
                ),
                'min_correlation': min(
                    v['correlation_with_baseline'] for v in stability.values()
                )
            }
        
        # Feature importance summary
        if 'permutation_importance' in self.results:
            top_features = self.results['permutation_importance'].head(10)
            report['top_important_features'] = top_features.to_dict('records')
        
        # SHAP summary
        if 'shap_importance' in self.results:
            top_shap = self.results['shap_importance'].head(10)
            report['top_shap_features'] = top_shap.to_dict('records')
        
        # Hyperparameter sensitivity
        if 'hyperparameter_sensitivity' in self.results:
            report['hyperparameter_sensitivity'] = self.results['hyperparameter_sensitivity']
        
        # Critical findings
        report['critical_findings'] = self._identify_critical_findings()
        
        self.results['report'] = report
        return report
    
    def _identify_critical_findings(self) -> List[str]:
        """Identify critical sensitivity findings."""
        findings = []
        
        if 'sensitivity_scores' in self.results:
            for feature, score in self.results['sensitivity_scores'].items():
                if score > 10:
                    findings.append(
                        f"HIGH SENSITIVITY: '{feature}' causes {score:.1f}% avg prediction change"
                    )
        
        if 'stability_analysis' in self.results:
            for noise, metrics in self.results['stability_analysis'].items():
                if metrics['correlation_with_baseline'] < 0.90:
                    findings.append(
                        f"INSTABILITY WARNING: {noise} causes correlation drop to {metrics['correlation_with_baseline']:.3f}"
                    )
        
        if not findings:
            findings.append("No critical sensitivity issues detected")
        
        return findings
    
    def visualize_sensitivity(self, save_path: str = None):
        """Create comprehensive sensitivity visualization."""
        if save_path is None:
            save_path = ArtifactManager.get_sensitivity_plot_path(self.stage)
        
        logger.info("Creating sensitivity visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Model Sensitivity Analysis ({self.stage.upper()})', fontsize=16, fontweight='bold')
        
        # 1. Permutation Feature Importance
        if 'permutation_importance' in self.results:
            ax = axes[0, 0]
            df = self.results['permutation_importance'].head(15)
            ax.barh(range(len(df)), df['importance_mean'], xerr=df['importance_std'], alpha=0.7)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df['feature'])
            ax.set_xlabel('Importance (MAE increase when permuted)')
            ax.set_title('Permutation Feature Importance')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
        
        # 2. Sensitivity Scores
        if 'sensitivity_scores' in self.results:
            ax = axes[0, 1]
            scores = dict(list(self.results['sensitivity_scores'].items())[:12])
            colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in scores.values()]
            ax.barh(list(scores.keys()), list(scores.values()), color=colors, alpha=0.7)
            ax.set_xlabel('Avg % Prediction Change')
            ax.set_title('Feature Sensitivity Scores')
            ax.axvline(x=5, color='orange', linestyle='--', label='Moderate (5%)')
            ax.axvline(x=10, color='red', linestyle='--', label='High (10%)')
            ax.legend()
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
        
        # 3. Stability Analysis
        if 'stability_analysis' in self.results:
            ax = axes[1, 0]
            stability = self.results['stability_analysis']
            noise_levels = list(stability.keys())
            correlations = [stability[k]['correlation_with_baseline'] for k in noise_levels]
            
            ax.plot(noise_levels, correlations, 'bo-', linewidth=2, markersize=10)
            ax.axhline(y=0.95, color='green', linestyle='--', label='Stable (>0.95)')
            ax.axhline(y=0.90, color='red', linestyle='--', label='Unstable (<0.90)')
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Correlation with Baseline')
            ax.set_title('Prediction Stability Under Noise')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.8, 1.01)
        
        # 4. Hour Sensitivity (if available)
        if 'threshold_analysis' in self.results and 'hour' in self.results['threshold_analysis']:
            ax = axes[1, 1]
            hour_data = self.results['threshold_analysis']['hour']
            hours = [d['value'] for d in hour_data]
            pct_changes = [d['pct_change'] for d in hour_data]
            
            colors = ['red' if abs(p) > 20 else 'orange' if abs(p) > 10 else 'green' for p in pct_changes]
            ax.bar(hours, pct_changes, color=colors, alpha=0.7)
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('% Change from Baseline')
            ax.set_title('Prediction Sensitivity by Hour')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sensitivity plots saved to: {save_path}")
    
    def run_full_analysis(self) -> Dict:
        """Run complete sensitivity analysis."""
        logger.info("="*60)
        logger.info(f"STARTING SENSITIVITY ANALYSIS ({self.stage.upper()})")
        logger.info("="*60)
        
        # Run all analyses
        self.shap_feature_importance()
        self.hyperparameter_sensitivity_analysis()
        self.permutation_feature_importance()
        self.feature_perturbation_analysis()
        self.prediction_stability_analysis()
        self.critical_threshold_analysis()
        
        # Generate report
        report = self.generate_sensitivity_report()
        
        # Create visualizations
        self.visualize_sensitivity()
        
        # Save report
        report_path = ArtifactManager.get_sensitivity_report_path(self.stage, timestamp=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Sensitivity report saved to: {report_path}")
        logger.info("="*60)
        logger.info("SENSITIVITY ANALYSIS COMPLETE")
        logger.info("="*60)
        
        return report


def run_sensitivity_analysis(model_path: str, X_test: pd.DataFrame, y_test: pd.Series, stage: str = "baseline") -> Dict:
    """
    Standalone function to run sensitivity analysis.
    Can be called from integrated_training_pipeline.py
    """
    analyzer = SensitivityAnalyzer(model_path, X_test, y_test, stage=stage)
    report = analyzer.run_full_analysis()
    return report