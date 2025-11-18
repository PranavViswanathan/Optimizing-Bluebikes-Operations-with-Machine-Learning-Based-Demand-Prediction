# MODEL DEVELOPMENT

## Overview

Pipeline for automated training of machine learning models every week to predict how many bike rides happen at every station every hour per day. It makes sure the predictions are fair (unbiased), picks the best model, and automatically updates the production system - all without human intervention.

## Features 

- Multi-Model Training: Trains and compares XGBoost, LightGBM, and Random Forest models
- Bias Detection & Mitigation: Automatically identifies and corrects prediction bias across different features
- Hyperparameter Tuning: Lightweight tuning for optimal model performance
- MLflow Integration: Complete experiment tracking and model versioning
- Airflow Orchestration: Automated weekly training pipeline with failure handling
- Model Validation: Automated quality gates for production deployment
- Version Control: Comprehensive model versioning with metadata tracking
- Discord Notifications: Real-time alerts for pipeline success/failure



## Architecture 
```
data_pipeline/
├── docker-compose.yaml       # Container orchestration configuration
├── Dockerfile               # Custom Airflow image definition
├── .env                     # Environment variables (API keys, webhooks)
├── .env.example            # Template for environment variables
├── dags/                   # Airflow DAG definitions
│   ├── Model_pipeline_withBias_check.py # model pipeline dag
├── scripts/ 
├── working_data/                   # Data storage (gitignored)
│   ├── raw/
│   └── processed/
└── logs/        
```
```
model_pipeline/
├── mlflow/ 
│   ├── data_splits/
│   ├── data_splits_mitigated/    
│   ├── mlruns/               
│   ├── train_xgb.py
│   ├── train_lgb.py
│   ├── train_randomforest.py
│   ├── model_training_module.py
│   ├── pipeline_orchestration.py
│   ├── bias_mitigration_module.py
│   ├── bias_analysis_module.py
│   └── pipeline_config.py

```
## Pipeline Stages

### 1. Model Training 
- Trains three model types in parallel
- Performs optional hyperparameter tuning
- Tracks experiments with MLflow
- Computes comprehensive metrics (R², MAE, RMSE, MAPE)

### 2. Bias Detection
- Analyzes prediction errors across feature distributions
- Identifies systematic bias patterns
- Computes fairness metrics

### 3. Bias Mitigation
- Applies reweighting techniques to reduce bias
- Re-trains model with bias-adjusted data
- Validates bias reduction while maintaining performance

### 4. Model Validation

### 5. Model Promotion
- Compares new model to current production model
- Promotes if:
    - Bias is reduced, OR
    - R² improves by >0.01 AND MAE improves, OR
    - R² improves by >0.02
- Archives previous production version
### 6. Deployment
- Updates production model symlink
- Saves deployment metadata
- Creates version snapshot


## Model Comparison

| Model | Strengths | Use Case |
|-------|-----------|----------|
| **XGBoost** | Best overall performance, handles missing data well | Default choice for production |
| **LightGBM** | Fastest training, memory efficient | Large datasets, frequent retraining |
| **Random Forest** | Robust to outliers, interpretable | Baseline comparisons, feature analysis |


## Setup

### Prerequisites
Navigate to repo root
```bash
# Python 3.7
pip install -r requirements.txt
```

### Required Libraries
- `apache-airflow`
- `mlflow`
- `xgboost`
- `lightgbm`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

### Configuration


## Model Versioning

Models are versioned with comprehensive metadata:

```json
{
  "version": 5,
  "model_type": "xgboost",
  "created_date": "2024-11-18",
  "promoted_date": "2024-11-18",
  "metrics": {
    "test_r2": 0.7834,
    "test_mae": 87.45,
    "baseline_r2": 0.7612
  },
  "bias_metrics": {
    "baseline_issues": 12,
    "mitigated_issues": 4,
    "issues_reduced": 8
  },
  "status": "production"
}
```


## Monitoring

### Key Metrics Tracked
- Model performance (R², MAE, RMSE, MAPE)
- Bias detection results
- Training duration
- Feature importance
- Residual analysis

### Visualization
- Actual vs Predicted scatter plots
- Residual distribution plots
- Feature importance charts
- Training/validation learning curves

## Production Deployment

Production model structure:
```
/opt/airflow/models/
├── production/
│   ├── current_model.pkl              # Symlink to latest
│   ├── current_metadata.json          # Deployment info
│   └── CURRENT_VERSION.txt            # Human-readable version
└── versions/
    ├── model_v1_20241101_bias_mitigated.pkl
    ├── model_v2_20241108_bias_mitigated.pkl
    └── model_v3_20241115_bias_mitigated.pkl
```


## Bias Mitigation Strategy

The pipeline implements a reweighting approach:

1. **Detect**: Analyze prediction errors across feature bins
2. **Quantify**: Compute bias scores for each feature
3. **Mitigate**: Apply sample weights inversely proportional to bias
4. **Validate**: Ensure bias reduction without performance degradation