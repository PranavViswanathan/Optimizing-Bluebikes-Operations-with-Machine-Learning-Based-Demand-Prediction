"""
Comprehensive Model Bias Detection for Bluebikes Demand Prediction
Implements slicing techniques to detect and document bias across multiple dimensions
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
from datetime import datetime
import json
from scipy import stats

warnings.filterwarnings('ignore')

class BikeShareBiasDetector:
    """
    Comprehensive bias detection for bike-sharing demand prediction models.
    Evaluates fairness across temporal, weather, geographic, and usage patterns.
    """
    
    def __init__(self, model_path, X_test, y_test):
        """
        Initialize bias detector with model and test data.
        
        Args:
            model_path: Path to saved model (.pkl file)
            X_test: Test features DataFrame
            y_test: Test target values
        """
        self.model = joblib.load(model_path)
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.y_pred = self.model.predict(self.X_test.drop('date', axis=1, errors='ignore'))
        
        # Store all slice analyses
        self.slice_results = {}
        self.bias_report = {}
        
    def calculate_metrics(self, y_true, y_pred, slice_name="Overall"):
        """Calculate comprehensive metrics for a slice."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return None
            
        metrics = {
            'slice_name': slice_name,
            'n_samples': len(y_true_clean),
            'mae': mean_absolute_error(y_true_clean, y_pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            'r2': r2_score(y_true_clean, y_pred_clean),
            'mape': np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1))) * 100,
            'mean_actual': np.mean(y_true_clean),
            'mean_predicted': np.mean(y_pred_clean),
            'bias': np.mean(y_pred_clean - y_true_clean),
            'std_residual': np.std(y_pred_clean - y_true_clean)
        }
        
        return metrics
    
    def temporal_slice_analysis(self):
        """Analyze bias across temporal dimensions."""
        print("\n" + "="*80)
        print("TEMPORAL BIAS ANALYSIS")
        print("="*80)
        
        temporal_slices = {
            'Hour of Day': 'hour',
            'Day of Week': 'day_of_week',
            'Month': 'month',
            'Year': 'year'
        }
        
        results = {}
        
        for slice_name, col in temporal_slices.items():
            print(f"\n{slice_name} Analysis:")
            print("-" * 60)
            
            slice_metrics = []
            unique_values = sorted(self.X_test[col].unique())
            
            for value in unique_values:
                mask = self.X_test[col] == value
                if mask.sum() > 0:
                    metrics = self.calculate_metrics(
                        self.y_test[mask].values,
                        self.y_pred[mask],
                        f"{slice_name}={value}"
                    )
                    if metrics:
                        slice_metrics.append(metrics)
            
            df_metrics = pd.DataFrame(slice_metrics)
            results[slice_name] = df_metrics
            
            # Display summary
            print(f"\nPerformance Range:")
            print(f"  MAE: {df_metrics['mae'].min():.2f} - {df_metrics['mae'].max():.2f}")
            print(f"  RMSE: {df_metrics['rmse'].min():.2f} - {df_metrics['rmse'].max():.2f}")
            print(f"  R²: {df_metrics['r2'].min():.3f} - {df_metrics['r2'].max():.3f}")
            
            # Identify worst performing slices
            worst_mae = df_metrics.nlargest(3, 'mae')[['slice_name', 'mae', 'n_samples']]
            print(f"\nWorst performing slices (by MAE):")
            print(worst_mae.to_string(index=False))
        
        self.slice_results['temporal'] = results
        return results
    
    def rush_hour_bias_analysis(self):
        """Analyze bias for rush hour vs non-rush hour periods."""
        print("\n" + "="*80)
        print("RUSH HOUR BIAS ANALYSIS")
        print("="*80)
        
        # Define rush hour periods
        self.X_test['time_category'] = 'Off-Peak'
        self.X_test.loc[self.X_test['is_morning_rush'] == 1, 'time_category'] = 'Morning Rush'
        self.X_test.loc[self.X_test['is_evening_rush'] == 1, 'time_category'] = 'Evening Rush'
        self.X_test.loc[self.X_test['is_night'] == 1, 'time_category'] = 'Night'
        self.X_test.loc[self.X_test['is_midday'] == 1, 'time_category'] = 'Midday'
        
        results = []
        for category in ['Morning Rush', 'Evening Rush', 'Midday', 'Night', 'Off-Peak']:
            mask = self.X_test['time_category'] == category
            if mask.sum() > 0:
                metrics = self.calculate_metrics(
                    self.y_test[mask].values,
                    self.y_pred[mask],
                    category
                )
                if metrics:
                    results.append(metrics)
        
        df_results = pd.DataFrame(results)
        print("\nPerformance by Time Category:")
        print(df_results[['slice_name', 'n_samples', 'mae', 'rmse', 'r2', 'bias']].to_string(index=False))
        
        # Statistical significance testing
        print("\n" + "-"*60)
        print("Statistical Significance Testing (ANOVA):")
        
        groups = []
        for category in df_results['slice_name']:
            mask = self.X_test['time_category'] == category
            residuals = self.y_pred[mask] - self.y_test[mask].values
            groups.append(residuals)
        
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("  ⚠️  SIGNIFICANT BIAS DETECTED: Performance varies significantly across time categories")
        else:
            print("  ✓  No significant bias detected across time categories")
        
        self.slice_results['rush_hour'] = df_results
        return df_results
    
    def weather_condition_bias(self):
        """Analyze bias across weather conditions."""
        print("\n" + "="*80)
        print("WEATHER CONDITION BIAS ANALYSIS")
        print("="*80)
        
        # Create weather categories
        self.X_test['weather_category'] = 'Mild'
        self.X_test.loc[self.X_test['is_cold'] == 1, 'weather_category'] = 'Cold'
        self.X_test.loc[self.X_test['is_hot'] == 1, 'weather_category'] = 'Hot'
        self.X_test.loc[self.X_test['is_rainy'] == 1, 'weather_category'] = 'Rainy'
        self.X_test.loc[self.X_test['is_heavy_rain'] == 1, 'weather_category'] = 'Heavy Rain'
        
        results = []
        for category in ['Cold', 'Mild', 'Hot', 'Rainy', 'Heavy Rain']:
            mask = self.X_test['weather_category'] == category
            if mask.sum() > 0:
                metrics = self.calculate_metrics(
                    self.y_test[mask].values,
                    self.y_pred[mask],
                    category
                )
                if metrics:
                    results.append(metrics)
        
        df_results = pd.DataFrame(results)
        print("\nPerformance by Weather Condition:")
        print(df_results[['slice_name', 'n_samples', 'mae', 'rmse', 'mape', 'bias']].to_string(index=False))
        
        # Identify adverse conditions performance
        adverse_conditions = ['Cold', 'Heavy Rain']
        adverse_mask = self.X_test['weather_category'].isin(adverse_conditions)
        normal_mask = ~adverse_mask
        
        adverse_metrics = self.calculate_metrics(
            self.y_test[adverse_mask].values,
            self.y_pred[adverse_mask],
            "Adverse Weather"
        )
        normal_metrics = self.calculate_metrics(
            self.y_test[normal_mask].values,
            self.y_pred[normal_mask],
            "Normal Weather"
        )
        
        print("\n" + "-"*60)
        print("Adverse vs Normal Weather Comparison:")
        print(f"  Adverse MAE: {adverse_metrics['mae']:.2f}")
        print(f"  Normal MAE: {normal_metrics['mae']:.2f}")
        print(f"  Difference: {abs(adverse_metrics['mae'] - normal_metrics['mae']):.2f}")
        
        mae_ratio = adverse_metrics['mae'] / normal_metrics['mae']
        if mae_ratio > 1.2:
            print(f"  ⚠️  BIAS DETECTED: Model performs {((mae_ratio-1)*100):.1f}% worse in adverse weather")
        else:
            print(f"  ✓  Acceptable performance: {((mae_ratio-1)*100):.1f}% difference")
        
        self.slice_results['weather'] = df_results
        return df_results
    
    def weekday_weekend_bias(self):
        """Analyze bias between weekdays and weekends."""
        print("\n" + "="*80)
        print("WEEKDAY VS WEEKEND BIAS ANALYSIS")
        print("="*80)
        
        results = []
        for is_weekend in [0, 1]:
            mask = self.X_test['is_weekend'] == is_weekend
            label = "Weekend" if is_weekend else "Weekday"
            metrics = self.calculate_metrics(
                self.y_test[mask].values,
                self.y_pred[mask],
                label
            )
            if metrics:
                results.append(metrics)
        
        df_results = pd.DataFrame(results)
        print("\nPerformance Comparison:")
        print(df_results[['slice_name', 'n_samples', 'mae', 'rmse', 'r2', 'bias']].to_string(index=False))
        
        # Calculate disparity
        weekday_mae = df_results[df_results['slice_name'] == 'Weekday']['mae'].values[0]
        weekend_mae = df_results[df_results['slice_name'] == 'Weekend']['mae'].values[0]
        disparity_ratio = max(weekday_mae, weekend_mae) / min(weekday_mae, weekend_mae)
        
        print(f"\nDisparity Ratio: {disparity_ratio:.3f}")
        if disparity_ratio > 1.15:
            print(f"  ⚠️  BIAS DETECTED: {((disparity_ratio-1)*100):.1f}% performance difference")
        else:
            print(f"  ✓  Acceptable disparity")
        
        self.slice_results['weekday_weekend'] = df_results
        return df_results
    
    def demand_level_bias(self):
        """Analyze bias across different demand levels."""
        print("\n" + "="*80)
        print("DEMAND LEVEL BIAS ANALYSIS")
        print("="*80)
        
        # Create demand level categories based on actual values
        demand_quartiles = pd.qcut(self.y_test, q=4, labels=['Very Low', 'Low', 'Medium', 'High'])
        
        results = []
        for level in ['Very Low', 'Low', 'Medium', 'High']:
            mask = demand_quartiles == level
            metrics = self.calculate_metrics(
                self.y_test[mask].values,
                self.y_pred[mask],
                f"{level} Demand"
            )
            if metrics:
                results.append(metrics)
        
        df_results = pd.DataFrame(results)
        print("\nPerformance by Demand Level:")
        print(df_results[['slice_name', 'n_samples', 'mae', 'rmse', 'mape', 'bias']].to_string(index=False))
        
        # Check for systematic under/over-prediction
        print("\n" + "-"*60)
        print("Bias Direction Analysis:")
        for _, row in df_results.iterrows():
            direction = "over-predicts" if row['bias'] > 0 else "under-predicts"
            print(f"  {row['slice_name']}: {direction} by {abs(row['bias']):.2f} rides on average")
        
        self.slice_results['demand_level'] = df_results
        return df_results
    
    def interaction_bias_analysis(self):
        """Analyze bias in interaction scenarios (e.g., weekend + rain)."""
        print("\n" + "="*80)
        print("INTERACTION BIAS ANALYSIS")
        print("="*80)
        
        interactions = [
            ('Weekend + Rainy', (self.X_test['is_weekend'] == 1) & (self.X_test['is_rainy'] == 1)),
            ('Weekday + Morning Rush', (self.X_test['is_weekend'] == 0) & (self.X_test['is_morning_rush'] == 1)),
            ('Weekday + Evening Rush', (self.X_test['is_weekend'] == 0) & (self.X_test['is_evening_rush'] == 1)),
            ('Weekend + Night', (self.X_test['is_weekend'] == 1) & (self.X_test['is_night'] == 1)),
            ('Cold + Rainy', (self.X_test['is_cold'] == 1) & (self.X_test['is_rainy'] == 1))
        ]
        
        results = []
        for name, mask in interactions:
            if mask.sum() > 10:  # Only analyze if sufficient samples
                metrics = self.calculate_metrics(
                    self.y_test[mask].values,
                    self.y_pred[mask],
                    name
                )
                if metrics:
                    results.append(metrics)
        
        if results:
            df_results = pd.DataFrame(results)
            print("\nPerformance for Interaction Scenarios:")
            print(df_results[['slice_name', 'n_samples', 'mae', 'mape', 'bias']].to_string(index=False))
            
            self.slice_results['interactions'] = df_results
            return df_results
        else:
            print("Insufficient data for interaction analysis")
            return None
    
    def generate_bias_report(self):
        """Generate comprehensive bias report with mitigation recommendations."""
         
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_samples': len(self.y_test),
            'overall_performance': self.calculate_metrics(self.y_test.values, self.y_pred, "Overall"),
            'bias_detected': [],
            'recommendations': []
        }
        
        for category, results in self.slice_results.items():
            if results is None or (isinstance(results, dict) and len(results) == 0):
                continue
                
            
            if isinstance(results, dict):
                # Handle temporal results (dict of DataFrames)
                for subcategory, df in results.items():
                    mae_range = df['mae'].max() - df['mae'].min()
                    mae_cv = df['mae'].std() / df['mae'].mean() if df['mae'].mean() > 0 else 0
                    
                    if mae_cv > 0.3:  # High variability
                        bias_info = {
                            'category': f"{category} - {subcategory}",
                            'type': 'High Variability',
                            'severity': 'Medium' if mae_cv < 0.5 else 'High',
                            'metric': f"CV={mae_cv:.3f}",
                            'details': f"MAE range: {mae_range:.2f}"
                        }
                        report['bias_detected'].append(bias_info)
                     
            elif isinstance(results, pd.DataFrame):
                # Handle single DataFrame results
                if 'mae' in results.columns:
                    mae_range = results['mae'].max() - results['mae'].min()
                    mae_cv = results['mae'].std() / results['mae'].mean() if results['mae'].mean() > 0 else 0
                    
                    if mae_cv > 0.2:
                        bias_info = {
                            'category': category,
                            'type': 'Performance Disparity',
                            'severity': 'Medium' if mae_cv < 0.4 else 'High',
                            'metric': f"CV={mae_cv:.3f}",
                            'details': results[['slice_name', 'mae']].to_dict('records')
                        }
                        report['bias_detected'].append(bias_info)
                       
               
        if len(report['bias_detected']) == 0:
            report['recommendations'].append("Continue monitoring model performance across slices.")
        else:    
            # Generate specific recommendations
            recommendations = []
            
            for bias in report['bias_detected']:
                if 'temporal' in bias['category'].lower():
                    recommendations.append({
                        'issue': f"Temporal bias in {bias['category']}",
                        'mitigation': [
                            "Add more temporal features (holidays, events)",
                            "Use separate models for different time periods",
                            "Implement time-based sample weighting"
                        ]
                    })
                
                elif 'weather' in bias['category'].lower():
                    recommendations.append({
                        'issue': f"Weather-related bias in {bias['category']}",
                        'mitigation': [
                            "Oversample adverse weather conditions",
                            "Add weather interaction features",
                            "Use weather-specific decision thresholds"
                        ]
                    })
                
                elif 'demand' in bias['category'].lower():
                    recommendations.append({
                        'issue': f"Demand-level bias in {bias['category']}",
                        'mitigation': [
                            "Apply quantile regression for different demand levels",
                            "Use separate models for high/low demand periods",
                            "Implement demand-based sample weighting"
                        ]
                    })
                
                else:
                    recommendations.append({
                        'issue': f"Bias detected in {bias['category']}",
                        'mitigation': [
                            "Analyze feature importance for this slice",
                            "Consider slice-specific model calibration",
                            "Collect more data for underperforming slices"
                        ]
                    })
            
            # Remove duplicates and display
            seen = set()
            unique_recs = []
            for rec in recommendations:
                key = rec['issue']
                if key not in seen:
                    seen.add(key)
                    unique_recs.append(rec)
            
            report['recommendations'] = unique_recs
            
        # Save report
        report_filename = f'bias_detection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Full report saved to: {report_filename}")
        
        self.bias_report = report
        return report
    
    def visualize_bias(self, save_path='bias_analysis_plots.png'):
        """Create comprehensive visualization of bias across slices."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Model Bias Analysis Across Data Slices', fontsize=16, fontweight='bold')
        
        # 1. Temporal bias - Hour of day
        if 'temporal' in self.slice_results and 'Hour of Day' in self.slice_results['temporal']:
            df = self.slice_results['temporal']['Hour of Day']
            ax = axes[0, 0]
            hours = df['slice_name'].str.extract('=(\d+)')[0].astype(int)
            ax.plot(hours, df['mae'], marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Hour of Day', fontweight='bold')
            ax.set_ylabel('MAE', fontweight='bold')
            ax.set_title('Performance by Hour of Day')
            ax.grid(True, alpha=0.3)
            ax.axhline(df['mae'].mean(), color='r', linestyle='--', label='Mean MAE', alpha=0.7)
            ax.legend()
        
        # 2. Day of week bias
        if 'temporal' in self.slice_results and 'Day of Week' in self.slice_results['temporal']:
            df = self.slice_results['temporal']['Day of Week']
            ax = axes[0, 1]
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax.bar(range(7), df['mae'], color='steelblue', alpha=0.7)
            ax.set_xlabel('Day of Week', fontweight='bold')
            ax.set_ylabel('MAE', fontweight='bold')
            ax.set_title('Performance by Day of Week')
            ax.set_xticks(range(7))
            ax.set_xticklabels(days)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Rush hour analysis
        if 'rush_hour' in self.slice_results:
            df = self.slice_results['rush_hour']
            ax = axes[1, 0]
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            ax.barh(df['slice_name'], df['mae'], color=colors[:len(df)])
            ax.set_xlabel('MAE', fontweight='bold')
            ax.set_title('Performance by Time Category')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Weather condition bias
        if 'weather' in self.slice_results:
            df = self.slice_results['weather']
            ax = axes[1, 1]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            ax.bar(df['slice_name'], df['mae'], color=colors[:len(df)], alpha=0.7)
            ax.set_xlabel('Weather Condition', fontweight='bold')
            ax.set_ylabel('MAE', fontweight='bold')
            ax.set_title('Performance by Weather Condition')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 5. Demand level bias
        if 'demand_level' in self.slice_results:
            df = self.slice_results['demand_level']
            ax = axes[2, 0]
            x_pos = range(len(df))
            bars = ax.bar(x_pos, df['mae'], alpha=0.7, label='MAE')
            ax.set_xlabel('Demand Level', fontweight='bold')
            ax.set_ylabel('MAE', fontweight='bold')
            ax.set_title('Performance by Demand Level')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df['slice_name'].str.replace(' Demand', ''))
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add bias direction arrows
            for i, (_, row) in enumerate(df.iterrows()):
                color = 'red' if row['bias'] > 0 else 'blue'
                label = 'Over' if row['bias'] > 0 else 'Under'
                ax.annotate(label, xy=(i, row['mae']), xytext=(i, row['mae'] + 2),
                           ha='center', fontsize=8, color=color, fontweight='bold')
        
        # 6. Weekday vs Weekend
        if 'weekday_weekend' in self.slice_results:
            df = self.slice_results['weekday_weekend']
            ax = axes[2, 1]
            metrics = ['mae', 'rmse', 'mape']
            x = np.arange(len(metrics))
            width = 0.35
            
            weekday = df[df['slice_name'] == 'Weekday'].iloc[0]
            weekend = df[df['slice_name'] == 'Weekend'].iloc[0]
            
            weekday_vals = [weekday['mae'], weekday['rmse'], weekday['mape']]
            weekend_vals = [weekend['mae'], weekend['rmse'], weekend['mape']]
            
            ax.bar(x - width/2, weekday_vals, width, label='Weekday', alpha=0.7)
            ax.bar(x + width/2, weekend_vals, width, label='Weekend', alpha=0.7)
            
            ax.set_xlabel('Metric', fontweight='bold')
            ax.set_ylabel('Value', fontweight='bold')
            ax.set_title('Weekday vs Weekend Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(['MAE', 'RMSE', 'MAPE'])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Bias visualization saved to: {save_path}")
        plt.close()
    
    def run_full_analysis(self):
        """Run complete bias detection analysis."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE BIAS DETECTION ANALYSIS")
        print("="*80)
        
        # Run all analyses
        self.temporal_slice_analysis()
        self.rush_hour_bias_analysis()
        self.weather_condition_bias()
        self.weekday_weekend_bias()
        self.demand_level_bias()
        self.interaction_bias_analysis()
        
        # Generate report and visualizations
        self.generate_bias_report()
        self.visualize_bias()
        
        print("\n" + "="*80)
        print("BIAS DETECTION ANALYSIS COMPLETE")
        print("="*80)
        
        return self.bias_report