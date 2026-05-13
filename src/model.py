"""Model TwinXGBBoosted — 4 sub‑models + segmented prediction + event weighting.

Digunakan oleh src/tune.py dan src/train.py.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    CLF_PARAMS, REG_PARAMS, ALPHA_UNDER,
    QUANTILE_Q_TOP, QUANTILE_Q_PEAK,
    TOP_SEGMENT_PCT, PEAK_DAYS_PCT,
    SAMPLE_WEIGHT_ALPHA, SAMPLE_WEIGHT_CAP,
    HOLIDAY_BOOST, PRE_HOLIDAY_BOOST, PEAK_DAYS_BOOST,
    THRESHOLD_GRID, USE_LOG_TARGET, GROUP_COLS,
    TARGET_COL, PRICE_COL, DATE_COL,
)
from src.metrics import asymmetric_obj, evaluate_prediction


class TwinXGBBoosted:
    """Twin‑XGB dengan holiday‑aware peak forecasting.

    4 sub‑models:
    - clf: classifier (zero vs non‑zero)
    - reg_top: quantile 0.9 untuk item high‑volume
    - reg_peak: quantile 0.95 untuk peak days
    - reg_tail: asymmetric loss untuk sisanya

    Args:
        clf_params: Parameter untuk XGBClassifier.
        reg_params: Parameter untuk XGBRegressor.
        alpha_under: Penalty untuk under‑forecast di tail regressor.
        quantile_q_peak: Quantile untuk reg_peak.
        sample_weight_alpha / cap: Parameter formula sample weight.
        holiday_boost / pre_holiday_boost / peak_days_boost: Event multipliers.
    """

    def __init__(
        self,
        clf_params: Optional[Dict] = None,
        reg_params: Optional[Dict] = None,
        alpha_under: float = ALPHA_UNDER,
        quantile_q_top: float = QUANTILE_Q_TOP,
        quantile_q_peak: float = QUANTILE_Q_PEAK,
        sample_weight_alpha: float = SAMPLE_WEIGHT_ALPHA,
        sample_weight_cap: float = SAMPLE_WEIGHT_CAP,
        holiday_boost: float = HOLIDAY_BOOST,
        pre_holiday_boost: float = PRE_HOLIDAY_BOOST,
        peak_days_boost: float = PEAK_DAYS_BOOST,
        use_log_target: bool = USE_LOG_TARGET,
        threshold_grid: Optional[list] = None,
    ):
        self.clf_params = clf_params or CLF_PARAMS
        self.reg_params = reg_params or REG_PARAMS
        self.alpha_under = alpha_under
        self.quantile_q_top = quantile_q_top
        self.quantile_q_peak = quantile_q_peak
        self.sample_weight_alpha = sample_weight_alpha
        self.sample_weight_cap = sample_weight_cap
        self.holiday_boost = holiday_boost
        self.pre_holiday_boost = pre_holiday_boost
        self.peak_days_boost = peak_days_boost
        self.use_log_target = use_log_target
        self.threshold_grid = threshold_grid or THRESHOLD_GRID
        self._fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        holiday_train: np.ndarray,
        pre_holiday_train: np.ndarray,
        peak_train: np.ndarray,
        top_keys_train: set,
        train_keys: list,
    ) -> "TwinXGBBoosted":
        """Train semua sub‑models dengan event‑specific sample weights.

        Args:
            X_train: Feature matrix (float32).
            y_train: Target vector (float32).
            holiday_train: Boolean mask (is_hari_besar).
            pre_holiday_train: Boolean mask (is_pre_hari_besar).
            peak_train: Boolean mask (is_peak_day).
            top_keys_train: Set of (stock_code, country) tuples for top items.
            train_keys: List of (stock_code, country) per row.
        """
        # Zero classifier
        yz = (y_train > 0).astype(int)
        spw = (len(yz) - yz.sum()) / (yz.sum() + 1e-8)
        self.clf_ = xgb.XGBClassifier(
            **self.clf_params, objective="binary:logistic",
            scale_pos_weight=spw,
        )
        self.clf_.fit(X_train, yz)
        self.calibrator_ = CalibratedClassifierCV(self.clf_, method="isotonic", cv=3)
        self.calibrator_.fit(X_train, yz)

        # Regressors — only on non‑zero rows
        nz = y_train > 0
        X_nz = X_train[nz]
        y_nz = y_train[nz]
        y_reg = np.log1p(y_nz) if self.use_log_target else y_nz
        q95 = np.quantile(y_nz, 0.95) if y_nz.size else 1.0
        base_w = 1.0 + self.sample_weight_alpha * np.minimum(y_nz / (q95 + 1e-8), self.sample_weight_cap)
        event_mult = (
            1.0
            + self.holiday_boost * holiday_train[nz]
            + self.pre_holiday_boost * pre_holiday_train[nz]
            + self.peak_days_boost * peak_train[nz]
        )
        sw = base_w * event_mult

        self.reg_top_ = xgb.XGBRegressor(
            **self.reg_params, objective="reg:quantileerror",
            quantile_alpha=self.quantile_q_top,
        )
        self.reg_peak_ = xgb.XGBRegressor(
            **self.reg_params, objective="reg:quantileerror",
            quantile_alpha=self.quantile_q_peak,
        )
        self.reg_tail_ = xgb.XGBRegressor(
            **self.reg_params, objective=asymmetric_obj(self.alpha_under),
        )
        self.reg_top_.fit(X_nz, y_reg, sample_weight=sw)
        self.reg_peak_.fit(X_nz, y_reg, sample_weight=sw)
        self.reg_tail_.fit(X_nz, y_reg)

        self.train_keys_ = train_keys
        self.top_keys_ = top_keys_train
        self._fitted = True
        return self

    def predict(
        self,
        X_val: np.ndarray,
        val_keys: list,
        peak_mask: np.ndarray,
        holiday_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Predict + threshold search.

        Returns:
            (final_pred, best_threshold).
        """
        if not self._fitted:
            raise RuntimeError("Model belum di‑fit.")

        vtop = np.array([k in self.top_keys_ for k in val_keys])

        proba = self.calibrator_.predict_proba(X_val)[:, 1]

        def _pred(model, X):
            p = model.predict(X)
            return np.maximum(np.expm1(p) if self.use_log_target else p, 0)

        p_top = _pred(self.reg_top_, X_val)
        p_peak = _pred(self.reg_peak_, X_val)
        p_tail = _pred(self.reg_tail_, X_val)

        pred_reg = np.where(peak_mask, p_peak, np.where(vtop, p_top, p_tail))

        # Bias correction for boosted days
        if holiday_mask is not None and (peak_mask.any() or holiday_mask.any()):
            boosted = peak_mask | holiday_mask
            resid = pred_reg[boosted] - X_val[:, 0]  # dummy, harus pakai y_val
            # NOTE: bias correction membutuhkan y_val; dilakukan di outer function

        best_th = 0.1
        return pred_reg, proba, best_th

    def predict_with_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        price_val: np.ndarray,
        val_keys: list,
        peak_mask: np.ndarray,
        holiday_mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict, float]:
        """Cari threshold optimal (CLS) lalu predict final."""
        pred_reg, proba, _ = self.predict(X_val, val_keys, peak_mask)

        # Bias correction
        boosted = peak_mask | holiday_mask
        if boosted.any():
            resid = pred_reg[boosted] - y_val[boosted]
            mean_resid = np.mean(resid)
            if mean_resid < 0:
                factor = 1.0 - mean_resid / (np.mean(y_val[boosted]) + 1e-8)
                pred_reg[boosted] = pred_reg[boosted] * max(1.0, factor)

        bm = float(np.mean(np.abs(y_val)))
        best_th, best_cls, best_m = None, None, None
        for th in self.threshold_grid:
            pred = pred_reg * (proba >= th).astype(int)
            m = evaluate_prediction(y_val, pred, price_val, bm)
            if best_cls is None or m["cls"] < best_cls:
                best_cls = m["cls"]; best_th = th; best_m = m
        pred_final = pred_reg * (proba >= best_th).astype(int)
        best_m["threshold"] = best_th
        return pred_final, best_m, best_th

    def get_params(self) -> Dict:
        """Return konfigurasi untuk logging / MLflow."""
        return {
            "alpha_under": self.alpha_under,
            "quantile_q_top": self.quantile_q_top,
            "quantile_q_peak": self.quantile_q_peak,
            "sample_weight_alpha": self.sample_weight_alpha,
            "sample_weight_cap": self.sample_weight_cap,
            "holiday_boost": self.holiday_boost,
            "pre_holiday_boost": self.pre_holiday_boost,
            "peak_days_boost": self.peak_days_boost,
            "use_log_target": self.use_log_target,
        }
