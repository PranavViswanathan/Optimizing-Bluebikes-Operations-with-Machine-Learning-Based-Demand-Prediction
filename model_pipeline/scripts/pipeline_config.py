"""
Pipeline Configuration
Centralized configuration for the BlueBikes training pipeline
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class DataConfig:
    """Data loading and splitting configuration"""
    train_start: str = "2024-06-01"
    train_end: str = "2025-06-30"
    test_start: str = "2025-07-01"
    test_end: str = "2025-07-31"
    val_start: str = "2025-05-01"


@dataclass
class TrainingConfig:
    """Model training configuration"""
    models_to_train: List[str] = None
    tune_hyperparameters: bool = False
    experiment_name: str = "bluebikes_pipeline"
    mlflow_tracking_uri: str = "./mlruns"
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = ['xgboost', 'lightgbm', 'randomforest']


@dataclass
class BiasConfig:
    """Bias detection and mitigation configuration"""
    baseline_stage: str = "baseline"
    mitigated_stage: str = "mitigated"
    selection_metric: str = "test_r2"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    data: DataConfig = None
    training: TrainingConfig = None
    bias: BiasConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.bias is None:
            self.bias = BiasConfig()