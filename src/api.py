"""FastAPI serving — DecoupledActuarialXGB via MLflow Model Registry.

Usage:
    uvicorn src.api:app --reload

Environment (from .env):
    MLFLOW_TRACKING_URI        Wajib diisi
    AWS_ACCESS_KEY_ID          Wajib untuk akses S3 artifact
    AWS_SECRET_ACCESS_KEY      Wajib untuk akses S3 artifact
    AWS_DEFAULT_REGION         Wajib untuk akses S3 artifact
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("api")

# ── Bootstrap env ──
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_REGISTERED_MODEL_NAME = "FMCG_Actuarial_Demand_Forecaster"
MLFLOW_MODEL_VERSION = "2"

for _key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
    _val = os.getenv(_key)
    if _val:
        os.environ[_key] = _val

if not MLFLOW_TRACKING_URI:
    raise RuntimeError(
        "MLFLOW_TRACKING_URI tidak ditemukan. Set di file .env "
        "sebelum menjalankan server."
    )

# ── Schemas ──


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


# ── Model cache (singleton, loaded once via lifespan) ──

_pipeline_model: Optional[mlflow.pyfunc.PyFuncModel] = None


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline_model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}/{MLFLOW_MODEL_VERSION}"
        logger.info("Memuat model dari MLflow Registry: %s", model_uri)
        _pipeline_model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model berhasil dimuat dari MLflow Registry")
    except Exception as exc:
        logger.error("Gagal memuat model dari MLflow Registry: %s", exc)
        _pipeline_model = None
    yield


app = FastAPI(
    title="Decoupled Actuarial XGBoost — MLflow Registry",
    version="2.0.0",
    description=(
        "Single-record inference for FMCG demand forecasting. "
        "Model ditarik langsung dari MLflow Model Registry "
        "(FMCG_Actuarial_Demand_Forecaster / v2)."
    ),
    lifespan=lifespan,
)


# ── Endpoints ──


@app.get("/health")
def health_check():
    ok = _pipeline_model is not None
    return {
        "status": "healthy" if ok else "degraded",
        "registry": MLFLOW_REGISTERED_MODEL_NAME,
        "version": MLFLOW_MODEL_VERSION,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict-inventory", response_model=PredictResponse)
def predict_inventory(payload: FeaturePayload):
    if _pipeline_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Mesin prediksi offline atau AWS EC2 sedang mati. "
                "Pastikan MLflow tracking server dapat dijangkau dan "
                "model FMCG_Actuarial_Demand_Forecaster tersedia."
            ),
        )

    try:
        row = payload.model_dump()
        X = pd.DataFrame([row])

        raw = _pipeline_model.predict(X)

        expected_demand, recommended_stock, risk_status = _parse_prediction(raw)

        return PredictResponse(
            expected_demand=round(float(expected_demand), 2),
            recommended_stock=round(float(recommended_stock), 2),
            risk_status=str(risk_status),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediksi gagal: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Prediksi gagal: {exc}",
        )


# ── Helpers ──


def _parse_prediction(raw):
    """Parse pyfunc predict output menjadi (expected, recommended, risk).

    Mendukung format:
      - dict  (keys: expected_demand, recommended_stock, risk_status)
      - DataFrame  (kolom: expected_demand, recommended_stock, risk_status)
      - numpy array 3+ kolom
      - numpy array 1 kolom → treat sebagai recommended_stock
    """
    if isinstance(raw, dict):
        return (
            raw.get("expected_demand", 0.0),
            raw.get("recommended_stock", 0.0),
            raw.get("risk_status", "LOW"),
        )

    if isinstance(raw, pd.DataFrame):
        row = raw.iloc[0]
        cols = row.index.tolist()
        if "expected_demand" in cols:
            return (
                row.get("expected_demand", 0.0),
                row.get("recommended_stock", row.iloc[0]),
                row.get("risk_status", "LOW"),
            )
        arr = row.to_numpy()
    elif isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=float).ravel()
    elif isinstance(raw, np.ndarray):
        arr = raw.ravel()
    elif hasattr(raw, "__iter__"):
        arr = np.asarray(list(raw), dtype=float).ravel()
    else:
        arr = np.atleast_1d(float(raw))

    if len(arr) >= 3:
        return float(arr[0]), float(arr[1]), str(arr[2])
    if len(arr) == 2:
        return float(arr[0]), float(arr[1]), "LOW"
    if len(arr) == 1:
        v = float(arr[0])
        return v * 0.7, v, "LOW"
    return 0.0, 0.0, "LOW"
