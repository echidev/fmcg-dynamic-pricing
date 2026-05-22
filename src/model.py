"""DecoupledActuarialXGB — 3-pronged decoupled architecture.

Step 1: Honest Baseline (reg:squarederror) — unbiased mean estimate.
Step 2: Risk-Aware Layer (reg:quantileerror) — q_target percentile.
Step 3: Actuarial Optimization — dynamic critical fractile per SKU.

Menggantikan TwinXGBBoosted sepenuhnya.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    MODEL_PARAMS,
    QUANTILE_Q_TARGET,
    RANDOM_STATE,
    SHORTAGE_MARGIN_MULTIPLIER,
    USE_LOG_TARGET,
)
from src.metrics import evaluate_prediction


class DecoupledActuarialXGB:
    """3-pronged Decoupled Actuarial XGBoost.

    Args:
        model_params: Parameter XGBoost untuk kedua regressor.
        quantile_q_target: Quantile alpha untuk pinball loss.
        shortage_margin_multiplier: Faktor pengali Cu untuk mendorong OFR.
        margin_ratio_high / margin_ratio_low: Margin ratio threshold.
        use_log_target: Gunakan log1p transform pada target.
        random_state: Seed.
    """

    def __init__(
        self,
        model_params: Optional[Dict] = None,
        quantile_q_target: float = QUANTILE_Q_TARGET,
        shortage_margin_multiplier: float = SHORTAGE_MARGIN_MULTIPLIER,
        margin_ratio_high: float = 1.5,
        margin_ratio_low: float = 0.7,
        use_log_target: bool = USE_LOG_TARGET,
        random_state: int = RANDOM_STATE,
    ):
        self.model_params = dict(model_params or MODEL_PARAMS)
        self.quantile_q_target = quantile_q_target
        self.shortage_margin_multiplier = shortage_margin_multiplier
        self.margin_ratio_high = margin_ratio_high
        self.margin_ratio_low = margin_ratio_low
        self.use_log_target = use_log_target
        self.random_state = random_state
        self._fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_cols: Optional[List[str]] = None,
    ) -> DecoupledActuarialXGB:
        """Train mean + quantile regressors on non-zero rows.

        Args:
            X_train: Training features (DataFrame untuk menjaga feature_names).
            y_train: Training target.
            feature_cols: Feature column names (default: semua kolom di X_train).
        """
        feature_cols = feature_cols or list(X_train.columns)

        nz = y_train > 0
        X_nz = X_train.loc[nz]
        y_nz = y_train[nz]
        y_reg = np.log1p(y_nz) if self.use_log_target else y_nz

        # Step 1: Honest Baseline (MSE)
        self.model_mean_ = xgb.XGBRegressor(
            **self.model_params,
            objective="reg:squarederror",
            feature_names=feature_cols,
        )
        self.model_mean_.fit(X_nz, y_reg)

        # Step 2: Risk-Aware Quantile (Pinball Loss)
        self.model_quant_ = xgb.XGBRegressor(
            **self.model_params,
            objective="reg:quantileerror",
            quantile_alpha=self.quantile_q_target,
            feature_names=feature_cols,
        )
        self.model_quant_.fit(X_nz, y_reg)

        self.feature_cols_ = feature_cols
        self._fitted = True
        return self

    def predict_raw(
        self,
        X_val: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and quantile predictions.

        Returns:
            (p_mean, p_quant) — arrays of shape (n,).
        """
        if not self._fitted:
            raise RuntimeError("Model belum di-fit. Panggil .fit() terlebih dahulu.")

        def _pred(model, x):
            p = model.predict(x)
            return np.maximum(np.expm1(p) if self.use_log_target else p, 0)

        p_mean = _pred(self.model_mean_, X_val)
        p_quant = _pred(self.model_quant_, X_val)
        return p_mean, p_quant

    def predict_actuarial(
        self,
        X_val: pd.DataFrame,
        val_price: np.ndarray,
        val_keys: List[Tuple],
        return_components: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict]:
        """Compute final actuarial prediction.

        Step 3: Dynamic Critical Fractile per SKU.
            CF = Cu / (Cu + Co)
            Cu = margin_ratio * (1 + 0.3 * shelf_life) * shortage_margin_multiplier
            final = mean + (quant - mean) * CF

        Args:
            X_val: Validation features (DataFrame).
            val_price: avg_price per row.
            val_keys: List of (stock_code, country) tuples.
            return_components: If True, return (final_pred, dict_of_components).

        Returns:
            final_pred array, atau (final_pred, components_dict).
        """
        p_mean, p_quant = self.predict_raw(X_val)

        # Synthetic metadata per item from training data
        median_price = float(np.median(val_price[val_price > 0])) if (val_price > 0).any() else 1.0
        item_price_vl = np.where(val_price > 0, val_price, median_price)

        margin_ratio = np.where(
            item_price_vl > median_price,
            self.margin_ratio_high,
            self.margin_ratio_low,
        )
        shelf_life_perishable = (item_price_vl < float(np.mean(item_price_vl))).astype(float)

        # Cu = shortage cost * multiplier
        shortage_penalty = margin_ratio * (1.0 + 0.3 * shelf_life_perishable)
        shortage_penalty = shortage_penalty * self.shortage_margin_multiplier

        # Co = overstock cost
        spoilage_penalty = 0.5 * shelf_life_perishable
        max_mr = max(margin_ratio.max(), 1e-8)
        overstock_cost = (1.0 - margin_ratio / max_mr) * (1.0 + spoilage_penalty)

        total_cost = shortage_penalty + overstock_cost + 1e-8
        critical_fractile = np.clip(shortage_penalty / total_cost, 0.2, 0.95)

        final_pred = p_mean + (p_quant - p_mean) * critical_fractile
        final_pred = np.maximum(final_pred, 0)

        if return_components:
            components = {
                "p_mean": p_mean,
                "p_quant": p_quant,
                "margin_ratio": margin_ratio,
                "shelf_life_perishable": shelf_life_perishable,
                "shortage_penalty": shortage_penalty,
                "critical_fractile": critical_fractile,
            }
            return final_pred, components

        return final_pred

    def predict(
        self,
        X_val: pd.DataFrame,
        val_price: np.ndarray,
        val_keys: List[Tuple],
    ) -> np.ndarray:
        """Alias untuk predict_actuarial (return final_pred only)."""
        return self.predict_actuarial(X_val, val_price, val_keys, return_components=False)

    def evaluate(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        val_price: np.ndarray,
        val_keys: List[Tuple],
        baseline_mae: Optional[float] = None,
    ) -> Dict[str, float]:
        """Predict and evaluate metrics."""
        pred = self.predict(X_val, val_price, val_keys)
        return evaluate_prediction(y_val, pred, val_price, baseline_mae)

    def get_params(self) -> Dict:
        """Return konfigurasi untuk logging / MLflow."""
        return {
            "quantile_q_target": self.quantile_q_target,
            "shortage_margin_multiplier": self.shortage_margin_multiplier,
            "model_params": self.model_params,
            "use_log_target": self.use_log_target,
            "margin_ratio_high": self.margin_ratio_high,
            "margin_ratio_low": self.margin_ratio_low,
        }
