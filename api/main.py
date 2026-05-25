import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from api.schemas import FeaturePayload, PredictResponse

logger = logging.getLogger("api")

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_REGISTERED_MODEL_NAME = "FMCG_Actuarial_Demand_Forecaster"

for _key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
    _val = os.getenv(_key)
    if _val:
        os.environ[_key] = _val

if not MLFLOW_TRACKING_URI:
    raise RuntimeError(
        "MLFLOW_TRACKING_URI tidak ditemukan. Set di file .env "
        "sebelum menjalankan server."
    )

_pipeline_model: Optional[mlflow.pyfunc.PyFuncModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline_model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}/latest"
        logger.info("Memuat model dari MLflow Registry: %s", model_uri)
        _pipeline_model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model berhasil dimuat dari MLflow Registry")
    except Exception as exc:
        logger.error("Gagal memuat model dari MLflow Registry: %s", exc)
        _pipeline_model = None
    yield


app = FastAPI(
    title="FMCG Actuarial Demand Forecaster",
    version="3.0.0",
    description=(
        "Single-record inference for FMCG demand forecasting. "
        "Model ditarik langsung dari MLflow Model Registry "
        "(FMCG_Actuarial_Demand_Forecaster / latest). "
        "Artifact model otomatis diunduh dari S3 oleh MLflow."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    ok = _pipeline_model is not None
    return {
        "status": "healthy" if ok else "degraded",
        "registry": MLFLOW_REGISTERED_MODEL_NAME,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: FeaturePayload):
    if _pipeline_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model offline. Pastikan MLflow tracking server dapat dijangkau.",
        )
    try:
        row = payload.model_dump()
        X = pd.DataFrame([row])
        raw = _pipeline_model.predict(X)
        result = _parse_prediction(raw)
        return PredictResponse(
            expected_demand=round(float(result[0]), 2),
            recommended_stock=round(float(result[1]), 2),
            risk_status=str(result[2]),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediksi gagal: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {exc}")


def _parse_prediction(raw):
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
