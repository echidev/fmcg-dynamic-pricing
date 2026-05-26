"""Konfigurasi sentral untuk pipeline FMCG Demand Forecasting.
Single source of truth untuk semua konstanta, parameter, dan feature list.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs/pipeline.yaml"

# Medallion Architecture
BRONZE_DIR = PROJECT_ROOT / "data/bronze"
SILVER_DIR = PROJECT_ROOT / "data/silver"
GOLD_DIR = PROJECT_ROOT / "data/gold"

RAW_PATH = BRONZE_DIR / "online_retail.csv"
DAILY_PATH = SILVER_DIR / "online_retail_daily_product.parquet"
TABULAR_PATH = GOLD_DIR / "online_retail_daily_product_tabular.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Chunked IO ──
CHUNK_SIZE = 200_000
USECOLS = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Country"]
DTYPES = {
    "Invoice": "string",
    "StockCode": "string",
    "Quantity": "float32",
    "Price": "float32",
    "Country": "string",
}

# ── Non-product codes ──
NON_PRODUCT_CODES = {
    "POST", "DOT", "C2", "M", "D", "ADJUST", "ADJUST2",
    "BANK CHARGES", "AMAZONFEE", "B", "S", "PADS",
    "TEST001", "TEST002",
    "GIFT_0001_10", "GIFT_0001_20", "GIFT_0001_30",
    "GIFT_0001_40", "GIFT_0001_50", "GIFT_0001_70", "GIFT_0001_80",
}

# ── Country to holiday mapping ──
COUNTRY_TO_HOLIDAYS = {
    "Australia": "AU", "Austria": "AT", "Belgium": "BE", "Canada": "CA",
    "Channel Islands": "GB", "Czech Republic": "CZ", "Denmark": "DK",
    "EIRE": "IE", "Finland": "FI", "France": "FR", "Germany": "DE",
    "Greece": "GR", "Hong Kong": "HK", "Iceland": "IS", "Israel": "IL",
    "Italy": "IT", "Japan": "JP", "Korea": "KR", "Netherlands": "NL",
    "Nigeria": "NG", "Norway": "NO", "Poland": "PL", "Portugal": "PT",
    "RSA": "ZA", "Saudi Arabia": "SA", "Singapore": "SG", "Spain": "ES",
    "Sweden": "SE", "Switzerland": "CH", "Thailand": "TH", "USA": "US",
    "United Kingdom": "GB", "United Arab Emirates": "AE",
    "European Community": None, "Unspecified": None, "West Indies": None,
    "Bahrain": "BH", "Bermuda": "BM", "Cyprus": "CY", "Lithuania": "LT",
    "Malta": "MT", "Lebanon": "LB",
}

# ── Column names ──
TARGET_COL = "demand_qty"
PRICE_COL = "avg_price"
DATE_COL = "date"
GROUP_COLS = ["stock_code", "country"]

# ── Seasonality features (14) ──
SEASONALITY_FEATURES = [
    "day_of_week", "week_of_year", "month", "quarter", "day_of_month",
    "is_weekend", "is_month_start", "is_month_end",
    "days_to_month_end", "week_of_month",
    "is_month_start_window", "is_month_end_window",
    "is_hari_besar", "is_pre_hari_besar",
]

# ── Holiday intensity features (4) ──
HOLIDAY_INTENSITY_FEATURES = [
    "holiday_intensity", "days_to_next_holiday",
    "is_holiday_season", "holiday_x_weekend",
]

# ── Temporal peak feature (1) ──
TEMPORAL_PEAK_FEATURES = ["is_peak_day"]

# ── Lag / rolling features (33) ──
LAG_ROLL_FEATURES = [
    "demand_lag_1", "demand_lag_2", "demand_lag_7", "demand_lag_14",
    "demand_lag_21", "demand_lag_28", "demand_lag_35", "demand_lag_56", "demand_lag_84",
    "days_since_last_sale", "roll_zero_count_14",
    "roll_max_7", "roll_max_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_56",
    "roll_median_7", "roll_median_14", "roll_median_28",
    "roll_std_7", "roll_std_14", "roll_std_28", "roll_std_56",
    "roll_max_56", "roll_max_84",
    "demand_acceleration_3d",
    "spike_ratio_28", "spike_ratio_56",
    "pct_change_1", "pct_change_7",
    "discount_depth_pct", "price_momentum",
]

# ── Complete feature list (52) ──
FEATURE_COLS = (
    SEASONALITY_FEATURES
    + HOLIDAY_INTENSITY_FEATURES
    + TEMPORAL_PEAK_FEATURES
    + LAG_ROLL_FEATURES
)

# ── Feature index map for audit ──
FEATURE_INDEX_MAP = {i: name for i, name in enumerate(FEATURE_COLS)}

# ── Configurable constants ──
HOLIDAY_INTENSITY_CAP = 100.0

# ── Decoupled Actuarial Architecture params ──
MODEL_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.0339,
    "subsample": 0.82,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "max_bin": 128,
    "random_state": 42,
    "n_jobs": -1,
}

QUANTILE_Q_TARGET = 0.9799
SHORTAGE_MARGIN_MULTIPLIER = 2.2847

# ── Segmentation ──
TOP_SEGMENT_PCT = 0.05
PEAK_DAYS_PCT = 0.05

# ── Threshold grid ──
THRESHOLD_GRID = [round(x, 2) for x in [i * 0.05 for i in range(2, 19)]]

# ── Config loader ──

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r") as f:
            return yaml.safe_load(f)
    return {}


_CFG = load_config()

# ── Config-driven variables ──
DATASET_VERSION = str(_CFG.get("dataset_version", "online_retail_v1"))
FEATURE_SET_ID = str(_CFG.get("feature_set_id", "fmcg_features_v52"))
CUTOFF_DATE = str(_CFG.get("cutoff_date", "2011-12-09"))


def get_git_commit() -> str:
    """Ambil git commit hash dari HEAD."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def set_global_seed(seed: int = 42) -> None:
    """Set global random seed untuk reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


REQUIRED_COLS_SILVER = {
    "stock_code", "country", "date", "demand_qty",
    "revenue", "avg_price", "num_invoices",
}


def validate_schema(df: "pd.DataFrame", required_cols: set, name: str = "dataframe") -> None:
    """Validasi kolom wajib ada di DataFrame."""
    import pandas as pd
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{name}: Missing required columns: {missing}")
    logger = logging.getLogger("schema")
    logger.info("%s: schema OK — %d columns, %d rows", name, len(df.columns), len(df))


def log_env_snapshot(mlflow) -> None:
    """Log environment metadata dan pip freeze ke MLflow."""
    mlflow.log_param("python_version", sys.version.replace("\n", " "))
    mlflow.log_param("os_platform", sys.platform)
    try:
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--no-color"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        mlflow.log_text(pip_freeze, "pip_freeze.txt")
    except Exception:
        mlflow.log_text("pip freeze failed", "pip_freeze.txt")


# ── Data quality ──
MIN_OBS = 60
RANDOM_STATE = int(_CFG.get("random_seed", 42))
USE_LOG_TARGET = True

# ── MLflow ──
MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
MLFLOW_EXPERIMENT = str(_CFG.get("mlflow_experiment", "xgboost_demand_forecasting"))

# ── Business thresholds for model promotion ──
PROMOTION_THRESHOLDS = {
    "ofr_min": float(_CFG.get("promotion_thresholds", {}).get("ofr_min", 0.80)),
    "cls_max": float(_CFG.get("promotion_thresholds", {}).get("cls_max", 50_000)),
}
