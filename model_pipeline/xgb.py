import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")


# ----------------------------
# 🔹 Load preprocessed features
# ----------------------------
features_path = "D:\\MLOps_Coursework\\ML-OPs\\data_pipeline\\data\\processed\\bluebikes\\features_full.pkl"
df = pd.read_pickle(features_path)

print(f"✅ Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop rows with invalid targets
df = df[df["ride_count"] > 0].copy()

# ----------------------------
# 🔹 Define target and features
# ----------------------------
target_col = "ride_count"

# Exclude identifiers and non-numeric columns
exclude_cols = ["date"]
feature_cols = [col for col in df.columns if col not in [target_col] + exclude_cols]

# Keep only numeric features (XGBoost requires numeric)
X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
y = df[target_col]

print(f"✅ Using {len(feature_cols)} features for training")

# ----------------------------
# 🔹 Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# ----------------------------
# 🔹 Train XGBoost Regressor
# ----------------------------
model = xgb.XGBRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42
)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=100,
    # early_stopping_rounds=50
)

# ----------------------------
# 🔹 Evaluate Model
# ----------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance Metrics")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.3f}")

# ----------------------------
# 🔹 Feature Importance
# ----------------------------
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type="gain", max_num_features=20)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()

# ----------------------------
# 🔹 Actual vs Predicted Plot
# ----------------------------
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Hourly Demand")
plt.ylabel("Predicted Hourly Demand")
plt.title("Actual vs Predicted Bluebikes Demand (XGBoost)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.tight_layout()
plt.show()

# ----------------------------
# 🔹 Save Model
# ----------------------------
model_path_json = "D:\\MLOps_Coursework\\ML-OPs\\model_pipeline\\models\\xgb_bluebikes_demand.json"
model_path_pkl = "D:\\MLOps_Coursework\\ML-OPs\\model_pipeline\\models\\xgb_bluebikes_demand.pkl"

model.save_model(model_path_json)
joblib.dump(model, model_path_pkl)

print(f"\n✅ Model saved to: {model_path_json}")
print(f"✅ Backup (joblib) saved to: {model_path_pkl}")
