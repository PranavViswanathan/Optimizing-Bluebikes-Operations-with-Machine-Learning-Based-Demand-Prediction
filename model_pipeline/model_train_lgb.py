#model_train_lgb.py
from math import sqrt
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import os

# ---------------------------
# Setup logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("train_bluebikes_model")

# ---------------------------
# Config
# ---------------------------
INPUT_PICKLE = r"D:\\MLOps_Coursework\\ML-OPs\data_pipeline\data\\processed\bluebikes\\features_with_lags.pkl"
MODEL_OUTPUT = r"D:\\MLOps_Coursework\\ML-OPs\model_pipeline\\models\\lightgbm_bluebikes.pkl"

ROLLING_WINDOWS = [3, 6, 12, 24]  # hours


# ---------------------------
# Feature Engineering
# ---------------------------
def add_rolling_features(df: pd.DataFrame, target_cols=["trips_started", "trips_ended", "net_flow"]):
    """Add rolling mean and std features for each station."""
    df = df.sort_values(["station_name", "hour_timestamp"])
    for col in target_cols:
        for window in ROLLING_WINDOWS:
            df[f"{col}_rollmean_{window}"] = (
                df.groupby("station_name")[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f"{col}_rollstd_{window}"] = (
                df.groupby("station_name")[col].transform(lambda x: x.rolling(window, min_periods=1).std())
            )
    return df


# ---------------------------
# Train/Test Split (time-based)
# ---------------------------
def time_based_split(df, date_col="hour_timestamp", cutoff="2025-03-01"):
    """Split chronologically into train and test."""
    train_df = df[df[date_col] < cutoff]
    test_df = df[df[date_col] >= cutoff]
    return train_df, test_df


# ---------------------------
# Main training function
# ---------------------------
def train_model():
    logger.info("🔹 Loading feature dataset...")
    df = pd.read_pickle(INPUT_PICKLE)

    # Encode station_name as category codes
    df["station_name"] = df["station_name"].astype("category").cat.codes


    # Drop NA rows that might exist due to lag features
    df = df.dropna(subset=["net_flow"]).copy()

    logger.info("🔹 Adding rolling features...")
    df = add_rolling_features(df)

    logger.info("🔹 Splitting data by time...")
    train_df, test_df = time_based_split(df)

    target = "net_flow"
    feature_cols = [
        c for c in df.columns
        if c not in ["hour_timestamp", "net_flow"]  # exclude time + target
    ]

    logger.info(f"Training rows: {len(train_df):,}, Test rows: {len(test_df):,}")
    logger.info(f"Features used: {len(feature_cols)}")

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]

    # ---------------------------
    # LightGBM Dataset + Training
    # ---------------------------
    logger.info("🔹 Training LightGBM regression model...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(
    objective="regression",
    boosting_type="gbdt",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
    )

    model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # ---------------------------
    # Evaluation
    # ---------------------------
    logger.info("🔹 Evaluating model...")
    preds = model.predict(X_test, num_iteration=model.best_iteration_)
    mae = mean_absolute_error(y_test, preds)
    rmse = sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    logger.info(f"✅ MAE:  {mae:.3f}")
    logger.info(f"✅ RMSE: {rmse:.3f}")
    logger.info(f"✅ R²:   {r2:.3f}")

    # ---------------------------
    # Save Model
    # ---------------------------
    os.makedirs(Path(MODEL_OUTPUT).parent, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    logger.info(f"✅ Model saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    train_model()
