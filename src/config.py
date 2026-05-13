"""Konfigurasi sentral untuk pipeline FMCG Demand Forecasting.
Single source of truth untuk semua konstanta, parameter, dan feature list.
"""

from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data/raw/online_retail.csv"
DAILY_PATH = PROJECT_ROOT / "data/transform/online_retail_daily_product.csv"
TABULAR_PATH = PROJECT_ROOT / "data/transform/online_retail_daily_product_tabular.csv"
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

# ── XGBoost classifier params ──
CLF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "max_bin": 256,
    "random_state": 42,
    "n_jobs": -1,
}

# ── XGBoost regressor params ──
REG_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "max_bin": 256,
    "random_state": 42,
    "n_jobs": -1,
}

# ── Twin-XGB Boosted architecture params ──
ALPHA_UNDER = 50.0
QUANTILE_Q = 0.8
QUANTILE_Q_TOP = 0.9
QUANTILE_Q_PEAK = 0.95

# ── Segmentation ──
TOP_SEGMENT_PCT = 0.05
PEAK_DAYS_PCT = 0.05

# ── Event-specific weighting ──
SAMPLE_WEIGHT_ALPHA = 4.0
SAMPLE_WEIGHT_CAP = 5.0
HOLIDAY_BOOST = 1.5
PRE_HOLIDAY_BOOST = 0.5
PEAK_DAYS_BOOST = 2.0

# ── Threshold grid ──
THRESHOLD_GRID = [round(x, 2) for x in [i * 0.05 for i in range(2, 19)]]

# ── Data quality ──
MIN_OBS = 60  # minimum days per item to include
RANDOM_STATE = 42
USE_LOG_TARGET = True

# ── MLflow ──
MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
MLFLOW_EXPERIMENT = "xgboost_demand_forecasting"

# ── Business thresholds for model promotion ──
PROMOTION_THRESHOLDS = {
    "ofr_min": 0.80,
    "cls_max": 45_000,
    "peak_ofr_min": 0.75,
}
