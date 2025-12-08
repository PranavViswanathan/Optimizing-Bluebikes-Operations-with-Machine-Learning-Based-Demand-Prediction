
#xgb_model.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import xgboost as xgb
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
logger = logging.getLogger("train_bluebikes_xgb_trips")

# ---------------------------
# Config
# ---------------------------
INPUT_PICKLE = r"D:\MLOps_Coursework\ML-OPs\data_pipeline\data\processed\bluebikes\features_with_lags.pkl"
MODEL_OUTPUT = r"D:\MLOps_Coursework\ML-OPs\model_pipeline\models\xgboost_bluebikes_trips.pkl"
ROLLING_WINDOWS_STARTS = [3, 6, 12, 24]  # hours
TARGET_COL = "trips_started"

# ---------------------------
# Feature Engineering
# ---------------------------
def add_rolling_features(df: pd.DataFrame, target_cols=["trips_started"]):
    """Add rolling mean and std features for each station."""
    df = df.sort_values(["station_name", "hour_timestamp"])
    for window in ROLLING_WINDOWS_STARTS:
        df[f"{TARGET_COL}_rollmean_{window}"] = (
            df.groupby("station_name")[TARGET_COL]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"{TARGET_COL}_rollstd_{window}"] = (
            df.groupby("station_name")[TARGET_COL]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    return df

# ---------------------------
# Time-based Split
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

    # Drop NA rows that might exist
    df = df.dropna(subset=["trips_started"]).copy()

    logger.info("🔹 Adding rolling features...")
    df = add_rolling_features(df, target_cols=["trips_started"])

    logger.info("🔹 Splitting data by time...")
    train_df, test_df = time_based_split(df)

    target = "trips_started"
    feature_cols = [
        c for c in df.columns
        if c not in ["hour_timestamp", "trips_started", "trips_ended", "net_flow"]
    ]
    # Average trips per station (historical baseline)
    station_avg = df.groupby("station_name")[TARGET_COL].transform("mean")
    df["station_avg_trips"] = station_avg


    logger.info(f"Training rows: {len(train_df):,}, Test rows: {len(test_df):,}")
    logger.info(f"Features used: {len(feature_cols)}")

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]

    if "station_name" in X_train.columns:
        # Convert from pandas StringDtype to regular Python string first
        X_train["station_name"] = X_train["station_name"].astype(str).astype("category")
        X_test["station_name"] = X_test["station_name"].astype(str).astype("category")

    # ---------------------------
    # XGBoost Model
    # ---------------------------
    logger.info("🔹 Training XGBoost regression model...")

    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse",
        "random_state": 42,
        "tree_method": "hist"
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    evals = [(dtrain, "train"), (dtest, "eval")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # ---------------------------
    # Evaluation
    # ---------------------------
    logger.info("🔹 Evaluating model...")
    preds = model.predict(dtest)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    logger.info(f"MAE:  {mae:.3f}")
    logger.info(f"RMSE: {rmse:.3f}")
    logger.info(f"R²:   {r2:.3f}")

    # Optional "pseudo-accuracy" for easier interpretation
    accuracy = r2 * 100
    logger.info(f"Approx. Accuracy: {accuracy:.2f}%")

    # ---------------------------
    # Save Model
    # ---------------------------
    os.makedirs(Path(MODEL_OUTPUT).parent, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    logger.info(f"Model saved to: {MODEL_OUTPUT}")

if __name__ == "__main__":
    train_model()
