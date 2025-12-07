"""
Pipeline Orchestrator
Main orchestrator that ties all modules together
Can be used to run the complete pipeline or individual steps
"""

import argparse
from datetime import datetime
from pathlib import Path

from pipeline_config import PipelineConfig, DataConfig, TrainingConfig, BiasConfig
from data_module import DataLoader
from model_training_module import ModelTrainer
from bias_analysis_module import BiasAnalyzer
from bias_mitigation_module import BiasMitigator


class PipelineOrchestrator:
    """Orchestrates the complete BlueBikes training pipeline"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config if config else PipelineConfig()
        self.state = {}
        
    def run_complete_pipeline(self):
        """Run all pipeline steps in sequence"""
        print("="*80)
        print(" BLUEBIKES COMPLETE PIPELINE ".center(80))
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load and prepare data
        print("\n[1/8] Loading and preparing data...")
        data = self.run_data_loading()
        
        # Step 2: Train models
        print("\n[2/8] Training models...")
        trainer, results = self.run_model_training(data)
        
        # Step 3: Select best model
        print("\n[3/8] Selecting best model...")
        best_name, best_model, best_metrics = trainer.select_best_model(
            metric=self.config.bias.selection_metric
        )
        
        # Step 4: Baseline bias analysis
        print("\n[4/8] Running baseline bias analysis...")
        baseline_report = self.run_bias_analysis(
            data=data, 
            stage=self.config.bias.baseline_stage
        )
        
        # Step 5: Apply bias mitigation
        print("\n[5/8] Applying bias mitigation...")
        mitigated_data = self.run_bias_mitigation(data)
        
        # Step 6: Retrain with mitigation
        print("\n[6/8] Retraining with bias mitigation...")
        mitigated_model, mitigated_metrics = self.retrain_with_mitigation()
        
        # Step 7: Final bias analysis
        print("\n[7/8] Running mitigated model bias analysis...")
        final_report = self.run_bias_analysis(
            data=mitigated_data,
            stage=self.config.bias.mitigated_stage,
            metadata_path="mitigated_model_metadata.json"
        )
        
        # Step 8: Compare results
        print("\n[8/8] Comparing results...")
        comparison = self.compare_results(baseline_report, final_report)
        
        print("\n" + "="*80)
        print(" PIPELINE COMPLETE ".center(80))
        print("="*80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nView detailed results in MLflow UI:")
        print("  $ mlflow ui --port 5000")
        print("  Then open: http://localhost:5000")
        
        return {
            'data': data,
            'mitigated_data': mitigated_data,
            'baseline_model': best_model,
            'baseline_metrics': best_metrics,
            'baseline_bias_report': baseline_report,
            'mitigated_model': mitigated_model,
            'mitigated_metrics': mitigated_metrics,
            'final_bias_report': final_report,
            'comparison': comparison
        }
    
    def run_data_loading(self):
        """Step 1: Load and prepare data"""
        loader = DataLoader(self.config.data)
        data = loader.load_and_split_data()
        loader.save_data()
        self.state['data_loader'] = loader
        return data
    
    def run_model_training(self, data=None):
        """Step 2: Train models"""
        if data is None:
            data = DataLoader.load_data()
        
        trainer = ModelTrainer(self.config.training)
        results = trainer.train_all_models(data)
        self.state['trainer'] = trainer
        self.state['training_results'] = results
        return trainer, results
    
    def run_bias_analysis(self, data=None, stage="baseline", 
                         metadata_path=None):
        """Step 3/7: Run bias analysis"""
        if data is None:
            data = DataLoader.load_data()
        
        # Determine metadata path based on stage if not provided
        if metadata_path is None:
            if stage == "mitigated":
                metadata_path = "mitigated_model_metadata.json"
            else:
                metadata_path = "best_model_metadata.json"
        
        analyzer = BiasAnalyzer(self.config.bias)
        report = analyzer.analyze_from_metadata(metadata_path, data, stage)
        
        if stage not in self.state:
            self.state[stage] = {}
        self.state[stage]['analyzer'] = analyzer
        self.state[stage]['report'] = report
        
        return report
    
    def run_bias_mitigation(self, data=None):
        """Step 4: Apply bias mitigation"""
        if data is None:
            data = DataLoader.load_data()
        
        mitigator = BiasMitigator(self.config.training)
        mitigated_data = mitigator.apply_feature_engineering(data)
        mitigator.save_mitigated_data()
        self.state['mitigator'] = mitigator
        return mitigated_data
    
    def retrain_with_mitigation(self, mitigated_data=None):
        """Step 5: Retrain model with mitigation"""
        mitigator = self.state.get('mitigator')
        if mitigator is None:
            mitigator = BiasMitigator(self.config.training)
        
        if mitigated_data is None:
            # Load from saved mitigated data
            from data_module import DataLoader
            import joblib
            from pathlib import Path
            
            data_path = Path("data_splits_mitigated")
            mitigated_data = {}
            for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
                mitigated_data[key] = joblib.load(data_path / f"{key}.pkl")
            mitigated_data['sample_weights'] = None
        
        model, metrics = mitigator.retrain_model(
            "best_model_metadata.json", 
            mitigated_data
        )
        return model, metrics
    
    def compare_results(self, baseline_report=None, mitigated_report=None):
        """Step 6: Compare baseline and mitigated results"""
        analyzer = BiasAnalyzer(self.config.bias)
        
        if baseline_report:
            analyzer.reports['baseline'] = baseline_report
        if mitigated_report:
            analyzer.reports['mitigated'] = mitigated_report
        
        comparison = analyzer.compare_reports()
        return comparison


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='BlueBikes Training Pipeline Orchestrator'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['all', 'data', 'train', 'bias-baseline', 'mitigate', 
                'retrain', 'bias-final', 'compare'],
        default='all',
        help='Pipeline step to run (default: all)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['xgboost', 'lightgbm', 'randomforest'],
        default=['xgboost', 'lightgbm', 'randomforest'],
        help='Models to train'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning',
        default=True
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='bluebikes_pipeline',
        help='MLflow experiment name'
    )
    
    return parser


def main():
    """Main entry point with CLI"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        training=TrainingConfig(
            models_to_train=args.models,
            tune_hyperparameters=args.tune,
            experiment_name=args.experiment_name
        )
    )
    
    orchestrator = PipelineOrchestrator(config)
    
    # Run based on step argument
    if args.step == 'all':
        results = orchestrator.run_complete_pipeline()
    
    elif args.step == 'data':
        print("="*80)
        print(" STEP: DATA LOADING ".center(80))
        print("="*80)
        results = orchestrator.run_data_loading()
    
    elif args.step == 'train':
        print("="*80)
        print(" STEP: MODEL TRAINING ".center(80))
        print("="*80)
        trainer, results = orchestrator.run_model_training()
        trainer.select_best_model(metric=config.bias.selection_metric)
    
    elif args.step == 'bias-baseline':
        print("="*80)
        print(" STEP: BASELINE BIAS ANALYSIS ".center(80))
        print("="*80)
        results = orchestrator.run_bias_analysis(stage='baseline')
    
    elif args.step == 'mitigate':
        print("="*80)
        print(" STEP: BIAS MITIGATION ".center(80))
        print("="*80)
        results = orchestrator.run_bias_mitigation()
    
    elif args.step == 'retrain':
        print("="*80)
        print(" STEP: RETRAIN WITH MITIGATION ".center(80))
        print("="*80)
        model, results = orchestrator.retrain_with_mitigation()
    
    elif args.step == 'bias-final':
        print("="*80)
        print(" STEP: FINAL BIAS ANALYSIS ".center(80))
        print("="*80)
        # Load mitigated data
        import joblib
        from pathlib import Path
        
        data_path = Path("data_splits_mitigated")
        if not data_path.exists():
            print("Error: Mitigated data not found. Run 'mitigate' step first.")
            return None
            
        mitigated_data = {}
        for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
            file_path = data_path / f"{key}.pkl"
            if file_path.exists():
                mitigated_data[key] = joblib.load(file_path)
            else:
                print(f"Error: {key}.pkl not found in {data_path}")
                return None
        
        results = orchestrator.run_bias_analysis(
            data=mitigated_data,
            stage='mitigated',
            metadata_path='mitigated_model_metadata.json'
        )
    
    elif args.step == 'compare':
        print("="*80)
        print(" STEP: COMPARE RESULTS ".center(80))
        print("="*80)
        # Load existing reports
        import json
        import glob
        
        baseline_reports = sorted(glob.glob('bias_detection_report_baseline_*.json'))
        mitigated_reports = sorted(glob.glob('bias_detection_report_mitigated_*.json'))
        
        if not baseline_reports or not mitigated_reports:
            print("Error: Missing bias reports. Run bias analysis steps first.")
            return None
        
        with open(baseline_reports[-1], 'r') as f:
            baseline_report = json.load(f)
        with open(mitigated_reports[-1], 'r') as f:
            mitigated_report = json.load(f)
        
        results = orchestrator.compare_results(baseline_report, mitigated_report)
    
    return orchestrator, results


if __name__ == "__main__":
    orchestrator, results = main()