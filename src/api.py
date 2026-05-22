"""FastAPI serving — DecoupledActuarialXGB single-record inference.

Usage:
    uvicorn src.api:app --reload
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import FEATURE_COLS, USE_LOG_TARGET

logger = logging.getLogger("api")
MODEL_DIR = Path("models/decoupled_actuarial_xgb")
MEAN_MODEL_PATH = MODEL_DIR / "mean_model.json"
QUANT_MODEL_PATH = MODEL_DIR / "quant_model.json"
CONFIG_PATH = MODEL_DIR / "model_config.json"

app = FastAPI(
    title="Decoupled Actuarial XGBoost Serving API",
    version="1.0.0",
    description="Single-record inference for FMCG demand forecasting with actuarial optimization.",
)


class FeaturePayload(BaseModel):
    stock_code: Optional[str] = None
    country: Optional[str] = None
    avg_price: float = Field(default=10.0, ge=0.0)
    is_hari_besar: int = Field(default=0, ge=0, le=1)
    is_pre_hari_besar: int = Field(default=0, ge=0, le=1)
    is_peak_day: int = Field(default=0, ge=0, le=1)
    day_of_week: int = Field(default=0, ge=0, le=6)
    week_of_year: int = Field(default=1, ge=1, le=53)
    month: int = Field(default=1, ge=1, le=12)
    quarter: int = Field(default=1, ge=1, le=4)
    day_of_month: int = Field(default=1, ge=1, le=31)
    is_weekend: int = Field(default=0, ge=0, le=1)
    is_month_start: int = Field(default=0, ge=0, le=1)
    is_month_end: int = Field(default=0, ge=0, le=1)
    days_to_month_end: int = Field(default=15, ge=0, le=31)
    week_of_month: int = Field(default=1, ge=1, le=5)
    is_month_start_window: int = Field(default=0, ge=0, le=1)
    is_month_end_window: int = Field(default=0, ge=0, le=1)
    holiday_intensity: float = Field(default=1.0, ge=0.0)
    days_to_next_holiday: int = Field(default=30, ge=0, le=30)
    is_holiday_season: int = Field(default=0, ge=0, le=1)
    holiday_x_weekend: int = Field(default=0, ge=0, le=1)
    demand_lag_1: float = Field(default=0.0, ge=0.0)
    demand_lag_2: float = Field(default=0.0, ge=0.0)
    demand_lag_7: float = Field(default=0.0, ge=0.0)
    demand_lag_14: float = Field(default=0.0, ge=0.0)
    demand_lag_21: float = Field(default=0.0, ge=0.0)
    demand_lag_28: float = Field(default=0.0, ge=0.0)
    demand_lag_35: float = Field(default=0.0, ge=0.0)
    demand_lag_56: float = Field(default=0.0, ge=0.0)
    demand_lag_84: float = Field(default=0.0, ge=0.0)
    days_since_last_sale: int = Field(default=0, ge=0)
    roll_zero_count_14: float = Field(default=0.0, ge=0.0)
    roll_max_7: float = Field(default=0.0, ge=0.0)
    roll_max_28: float = Field(default=0.0, ge=0.0)
    roll_mean_7: float = Field(default=0.0, ge=0.0)
    roll_mean_14: float = Field(default=0.0, ge=0.0)
    roll_mean_28: float = Field(default=0.0, ge=0.0)
    roll_mean_56: float = Field(default=0.0, ge=0.0)
    roll_median_7: float = Field(default=0.0, ge=0.0)
    roll_median_14: float = Field(default=0.0, ge=0.0)
    roll_median_28: float = Field(default=0.0, ge=0.0)
    roll_std_7: float = Field(default=0.0, ge=0.0)
    roll_std_14: float = Field(default=0.0, ge=0.0)
    roll_std_28: float = Field(default=0.0, ge=0.0)
    roll_std_56: float = Field(default=0.0, ge=0.0)
    roll_max_56: float = Field(default=0.0, ge=0.0)
    roll_max_84: float = Field(default=0.0, ge=0.0)
    demand_acceleration_3d: float = Field(default=0.0)
    spike_ratio_28: float = Field(default=0.0)
    spike_ratio_56: float = Field(default=0.0)
    pct_change_1: float = Field(default=0.0)
    pct_change_7: float = Field(default=0.0)
    discount_depth_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    price_momentum: float = Field(default=1.0, ge=0.0)

    margin_multiplier: float = Field(default=1.0, ge=0.5, le=3.0)
    perishability_score: float = Field(default=0.3, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    expected_demand: float
    recommended_stock: float
    risk_status: str


# ── Model cache (lazy load) ──

_mean_model: Optional[xgb.XGBRegressor] = None
_quant_model: Optional[xgb.XGBRegressor] = None


def _get_models():
    global _mean_model, _quant_model
    if _mean_model is not None and _quant_model is not None:
        return _mean_model, _quant_model

    if not MEAN_MODEL_PATH.exists() or not QUANT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifacts tidak ditemukan di {MODEL_DIR}. "
            "Jalankan 'python -m src.train' terlebih dahulu."
        )

    _mean_model = xgb.XGBRegressor()
    _mean_model.load_model(str(MEAN_MODEL_PATH))
    _quant_model = xgb.XGBRegressor()
    _quant_model.load_model(str(QUANT_MODEL_PATH))
    logger.info("Models loaded from %s", MODEL_DIR)
    return _mean_model, _quant_model


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r") as f:
            return json.load(f)
    return {}


# ── Endpoints ──


@app.get("/health")
def health_check():
    models_ok = MEAN_MODEL_PATH.exists() and QUANT_MODEL_PATH.exists()
    return {"status": "healthy" if models_ok else "degraded", "models_loaded": models_ok}


@app.post("/predict-inventory", response_model=PredictResponse)
def predict_inventory(payload: FeaturePayload):
    try:
        mean_model, quant_model = _get_models()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    row = payload.model_dump()
    margin_multiplier = row.pop("margin_multiplier")
    perishability_score = row.pop("perishability_score")

    # Build feature vector in correct order
    data = {}
    for col in FEATURE_COLS:
        data[col] = row.get(col, 0.0)
    data["stock_code"] = row.get("stock_code", "UNKNOWN")
    data["country"] = row.get("country", "UNKNOWN")
    data["avg_price"] = row.get("avg_price", 10.0)

    X = pd.DataFrame([data])[FEATURE_COLS]

    def _pred(model, x):
        p = model.predict(x)
        return float(np.maximum(np.expm1(p) if USE_LOG_TARGET else p, 0.0))

    expected_demand = _pred(mean_model, X)
    quantile_demand = _pred(quant_model, X)

    # Dynamic Critical Fractile
    shortage_penalty = margin_multiplier * (1.0 + 0.3 * perishability_score)
    max_mr_adj = min(margin_multiplier / 3.0, 1.0)
    overstock_cost = (1.0 - max_mr_adj) * (1.0 + 0.5 * perishability_score)
    total_cost = shortage_penalty + overstock_cost + 1e-8
    critical_fractile = float(np.clip(shortage_penalty / total_cost, 0.2, 0.95))

    recommended_stock = expected_demand + (quantile_demand - expected_demand) * critical_fractile
    recommended_stock = float(max(recommended_stock, 0.0))

    if critical_fractile > 0.8:
        risk_status = "HIGH"
    elif critical_fractile > 0.6:
        risk_status = "MEDIUM"
    else:
        risk_status = "LOW"

    return PredictResponse(
        expected_demand=round(expected_demand, 2),
        recommended_stock=round(recommended_stock, 2),
        risk_status=risk_status,
    )
