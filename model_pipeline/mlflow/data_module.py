"""
Data Loading Module
Handles all data loading, preparation, and splitting operations
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from feature_generation import load_and_prepare_data
from pipeline_config import DataConfig


class DataLoader:
    """Load and prepare data for training pipeline"""
    
    def __init__(self, config: DataConfig = None):
        self.config = config if config else DataConfig()
        self.data = {}
        
    def load_and_split_data(self):
        """Load and split data into train/val/test sets"""
        print("\n" + "="*80)
        print(" DATA LOADING AND PREPARATION ".center(80))
        print("="*80)
        
        # Load data using existing function
        X, y, feature_columns = load_and_prepare_data()
        X["date"] = pd.to_datetime(X["date"])
        
        # Convert config dates to timestamps
        train_start = pd.Timestamp(self.config.train_start)
        train_end = pd.Timestamp(self.config.train_end)
        test_start = pd.Timestamp(self.config.test_start)
        test_end = pd.Timestamp(self.config.test_end)
        val_start = pd.Timestamp(self.config.val_start)
        
        # Create masks
        train_mask_full = (X["date"] >= train_start) & (X["date"] <= train_end)
        test_mask = (X["date"] >= test_start) & (X["date"] <= test_end)
        
        # Split data
        X_train_full = X.loc[train_mask_full].copy()
        y_train_full = y.loc[train_mask_full].copy()
        X_test = X.loc[test_mask].copy()
        y_test = y.loc[test_mask].copy()
        
        # Further split training into train and validation
        val_mask = X_train_full['date'] >= val_start
        tr_mask = X_train_full['date'] < val_start
        
        X_train = X_train_full.loc[tr_mask]
        y_train = y_train_full.loc[tr_mask]
        X_val = X_train_full.loc[val_mask]
        y_val = y_train_full.loc[val_mask]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Features: {X_train.shape[1]}")
        
        # Store data
        self.data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_columns': feature_columns
        }
        
        return self.data
    
    def save_data(self, output_dir: str = "data_splits"):
        """Save split data to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.data:
            raise ValueError("No data to save. Run load_and_split_data() first.")
        
        print(f"\nSaving data splits to: {output_dir}/")
        
        for key, value in self.data.items():
            if key != 'feature_columns':
                filepath = output_path / f"{key}.pkl"
                joblib.dump(value, filepath)
                print(f"  Saved: {key}.pkl")
        
        # Save feature columns separately
        joblib.dump(self.data['feature_columns'], output_path / "feature_columns.pkl")
        print(f"  Saved: feature_columns.pkl")
        
        print(f"\nData splits saved successfully!")
        
    @staticmethod
    def load_data(input_dir: str = "data_splits"):
        """Load previously saved data splits"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Data directory not found: {input_dir}")
        
        print(f"\nLoading data splits from: {input_dir}/")
        
        data = {}
        for filename in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 'feature_columns']:
            filepath = input_path / f"{filename}.pkl"
            if filepath.exists():
                data[filename] = joblib.load(filepath)
                print(f"  Loaded: {filename}.pkl")
            else:
                print(f"  Warning: {filename}.pkl not found")
        
        print(f"\nData splits loaded successfully!")
        return data


def main():
    """Standalone execution for data loading"""
    print("="*80)
    print(" BLUEBIKES DATA LOADING MODULE ".center(80))
    print("="*80)
    
    # Initialize and load data
    loader = DataLoader()
    data = loader.load_and_split_data()
    
    # Save data splits
    loader.save_data()
    
    print("\n" + "="*80)
    print(" DATA LOADING COMPLETE ".center(80))
    print("="*80)
    
    return data


if __name__ == "__main__":
    data = main()