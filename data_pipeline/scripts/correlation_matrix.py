import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- GLOBAL PATHS ----------
DATA_PIPELINE_DIR = Path(__file__).resolve().parent.parent

# Store assets in a writable Airflow path (NOT under scripts/)
AIRFLOW_ASSETS_DIR = Path(os.environ.get("AIRFLOW_HOME", "/opt/airflow")) / "assets" / "data_pipeline"

# We do NOT create the directory here (Airflow would import this and fail)
ASSETS_DIR = AIRFLOW_ASSETS_DIR

# Add paths to sys.path if not already present
for path in [DATA_PIPELINE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ---------- LOGGER SETUP ----------
from logger import get_logger
logger = get_logger("correlation_matrix")


def correlation_matrix(pkl_path: str, dataset_name: str, method: str = 'pearson') -> pd.DataFrame:
    """
    Loads a DataFrame from a pickle file, computes its correlation matrix,
    visualizes it as a heatmap, and saves the plot.
    """

    # Create folder *now*, during task execution — NOT at module import time
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Validate pickle file ---
    if not os.path.exists(pkl_path):
        logger.error(f"File not found: {pkl_path}")
        return pd.DataFrame()

    # --- Load DataFrame ---
    try:
        df = pd.read_pickle(pkl_path)
        logger.info(f"DataFrame successfully loaded from: {pkl_path}")
    except Exception as e:
        logger.exception(f"Error loading pickle file: {e}")
        return pd.DataFrame()

    # --- Select numeric columns ---
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        logger.warning("No numerical columns found in the DataFrame.")
        return pd.DataFrame()

    # --- Compute correlation matrix ---
    try:
        corr_matrix = numeric_df.corr(method=method)
        corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        logger.info(f"Correlation matrix computed using {method} with shape {corr_matrix.shape}")
    except Exception as e:
        logger.exception(f"Failed to compute correlation matrix: {e}")
        return pd.DataFrame()

    # --- Plot setup ---
    plot_title = f"Correlation Matrix ({method.capitalize()} Method) - {dataset_name}"
    save_filename = f"{dataset_name}_correlation_{method}.png"
    save_path = ASSETS_DIR / save_filename

    n_cols = len(corr_matrix.columns)
    plt.figure(figsize=(max(8, n_cols * 0.8), max(6, n_cols * 0.6)))

    try:
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            square=True,
            cbar=True,
            annot_kws={"size": 8}
        )
        plt.title(plot_title, fontsize=14, pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Correlation matrix plot saved at: {save_path}")
    except Exception as e:
        logger.exception(f"Error generating/saving correlation plot: {e}")

    return corr_matrix
