"""MLflow PyFunc Wrapper — DecoupledActuarialXGB sebagai satu pipeline.

Memuat kedua sub-model (mean + quantile) dari artifact XGBoost JSON
dan menjalankan actuarial layer lengkap, mengembalikan DataFrame
dengan skema: expected_demand, recommended_stock, risk_status.
"""

from __future__ import annotations

import json
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
import xgboost as xgb


class DecoupledActuarialWrapper(mlflow.pyfunc.PythonModel):
    """Custom PyFunc yang membungkus Decoupled Actuarial XGBoost.

    load_context memuat mean_model + quant_model dari artifact JSON,
    lalu predict menjalankan actuarial layer dan mengembalikan 3 kolom.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.model_mean_ = xgb.XGBRegressor()
        self.model_mean_.load_model(context.artifacts["mean_model"])

        self.model_quant_ = xgb.XGBRegressor()
        self.model_quant_.load_model(context.artifacts["quant_model"])

        with open(context.artifacts["config"]) as f:
            self.config: dict[str, Any] = json.load(f)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        feature_cols = self.config.get("feature_cols", list(model_input.columns))
        use_log = self.config.get("use_log_target", True)
        price_col = "avg_price"

        X = model_input[[c for c in feature_cols if c in model_input.columns]]

        def _pred(model: xgb.XGBRegressor, x: pd.DataFrame) -> np.ndarray:
            p = model.predict(x)
            return np.maximum(np.expm1(p) if use_log else p, 0)

        p_mean = _pred(self.model_mean_, X)
        p_quant = _pred(self.model_quant_, X)

        # ── Actuarial optimisation layer ──
        price = model_input.get(
            price_col, pd.Series(np.ones(len(model_input)) * 10.0)
        ).to_numpy(dtype=np.float32)

        median_price = float(np.median(price[price > 0])) if (price > 0).any() else 1.0
        item_price = np.where(price > 0, price, median_price)

        margin_ratio_high = self.config.get("margin_ratio_high", 1.5)
        margin_ratio_low = self.config.get("margin_ratio_low", 0.7)
        smm = self.config.get("shortage_margin_multiplier", 2.2847)

        margin_ratio = np.where(
            item_price > median_price, margin_ratio_high, margin_ratio_low
        )
        shelf_life_perishable = (item_price < float(np.mean(item_price))).astype(float)

        shortage_penalty = margin_ratio * (1.0 + 0.3 * shelf_life_perishable) * smm
        spoilage_penalty = 0.5 * shelf_life_perishable
        max_mr = max(margin_ratio.max(), 1e-8)
        overstock_cost = (1.0 - margin_ratio / max_mr) * (1.0 + spoilage_penalty)
        total_cost = shortage_penalty + overstock_cost + 1e-8
        critical_fractile = np.clip(shortage_penalty / total_cost, 0.2, 0.95)

        final_pred = p_mean + (p_quant - p_mean) * critical_fractile
        final_pred = np.maximum(final_pred, 0)

        recommended_stock = np.maximum(p_quant, final_pred)

        risk_status = np.where(
            recommended_stock > p_mean * 1.5,
            "HIGH",
            "LOW",
        )

        return pd.DataFrame({
            "expected_demand": p_mean,
            "recommended_stock": recommended_stock,
            "risk_status": risk_status,
        })
