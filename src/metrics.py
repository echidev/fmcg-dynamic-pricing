"""Enterprise-grade evaluation metrics + lowercase aliases untuk notebook compatibility."""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score, precision_score, recall_score

ArrayLike = Union[Sequence[float], np.ndarray]


# ═══════════════════════════════════════════════════════════════
#  Helpers (existing)
# ═══════════════════════════════════════════════════════════════

def _to_1d_array(values: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D array-like, got shape {arr.shape}.")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _broadcast_param(values: Optional[ArrayLike], length: int, name: str, default_value: float) -> np.ndarray:
    if values is None:
        return np.full(length, float(default_value))
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(length, float(arr))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D, got shape {arr.shape}.")
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {arr.shape[0]}.")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator != 0 else float(default)


# ═══════════════════════════════════════════════════════════════
#  Standalone metric functions (lowercase — compatible dgn notebook)
# ═══════════════════════════════════════════════════════════════

def calc_cls(y_true: ArrayLike, y_pred: ArrayLike, avg_price: ArrayLike, margin: float = 0.20) -> float:
    """Cost of Lost Sales."""
    y_true, y_pred, avg_price = (
        np.asarray(y_true, dtype=float),
        np.asarray(y_pred, dtype=float),
        np.asarray(avg_price, dtype=float),
    )
    return float(np.sum(np.maximum(y_true - y_pred, 0) * avg_price * margin))


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return 0.0 if mask.sum() == 0 else float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def evaluate_prediction(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    avg_price: ArrayLike,
    baseline_mae: Optional[float] = None,
) -> Dict[str, float]:
    """Lowercase metric dict — kompatibel dengan notebook."""
    y_true_a = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred, dtype=float)
    price_a = np.asarray(avg_price, dtype=float)
    mae = mean_absolute_error(y_true_a, y_pred_a)
    rmse = float(np.sqrt(np.mean((y_true_a - y_pred_a) ** 2)))
    smape_val = smape(y_true_a, y_pred_a)
    ytb = (y_true_a > 0).astype(int)
    ypb = (y_pred_a > 0).astype(int)
    cls = calc_cls(y_true_a, y_pred_a, price_a)
    ofr = float(np.minimum(y_true_a, y_pred_a).sum() / (y_true_a.sum() + 1e-8))
    oos_rate = float(np.mean(y_pred_a < y_true_a))
    fva = None
    if baseline_mae is not None and baseline_mae > 0:
        fva = float((baseline_mae - mae) / baseline_mae)
    return {
        "mae": mae, "rmse": rmse, "smape": smape_val,
        "f1_zero": f1_score(ytb, ypb),
        "precision_zero": precision_score(ytb, ypb, zero_division=0),
        "recall_zero": recall_score(ytb, ypb, zero_division=0),
        "cls": cls, "ofr": ofr, "oos_rate": oos_rate, "fva": fva,
    }


# ═══════════════════════════════════════════════════════════════
#  Enterprise evaluate_model_performance (existing — uppercase)
# ═══════════════════════════════════════════════════════════════

def evaluate_model_performance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_naive: Optional[ArrayLike] = None,
    unit_prices: Optional[ArrayLike] = None,
    ihc_cost: Optional[ArrayLike] = None,
    cls_cost: Optional[ArrayLike] = None,
    peak_threshold_percentile: float = 90,
    gross_margin: Optional[ArrayLike] = None,
    cogs: Optional[ArrayLike] = None,
    holding_daily: Optional[ArrayLike] = None,
) -> Dict[str, float]:
    """Enterprise-grade evaluation — uppercase keys. Dokumentasi lengkap di docstring."""
    y_true_arr = _to_1d_array(y_true, "y_true")
    y_pred_arr = _to_1d_array(y_pred, "y_pred")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    n_obs = y_true_arr.shape[0]
    if n_obs == 0:
        return {k: 0.0 for k in [
            "MAE", "RMSE", "WAPE", "SMAPE_pct", "Global_Bias_pct",
            "F1_Zero", "MAE_NonZero",
            "CLS", "IHC", "Total_Cost", "OOS_Rate", "OFR",
            "Peak_Capture_pct", "Peak_WAPE", "Peak_Bias_pct", "Peak_CLS",
        ]}

    errors = y_true_arr - y_pred_arr
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    wape = _safe_divide(abs_errors.sum(), y_true_arr.sum())
    smape_denom = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0
    smape_mask = smape_denom != 0
    smape_val = float(np.mean(abs_errors[smape_mask] / smape_denom[smape_mask]) * 100) if smape_mask.any() else 0.0
    global_bias_pct = _safe_divide(y_pred_arr.sum() - y_true_arr.sum(), y_true_arr.sum()) * 100
    nonzero_mask = y_true_arr > 0
    mae_nonzero = float(np.mean(abs_errors[nonzero_mask])) if nonzero_mask.any() else 0.0
    yt, yp = (y_true_arr > 0).astype(int), (y_pred_arr > 0).astype(int)
    tp, fp, fn = np.sum((yp == 1) & (yt == 1)), np.sum((yp == 1) & (yt == 0)), np.sum((yp == 0) & (yt == 1))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1_zero = _safe_divide(2 * precision * recall, precision + recall)

    unit_prices_arr = _broadcast_param(unit_prices, n_obs, "unit_prices", 0.0)
    gross_margin_arr = _broadcast_param(gross_margin, n_obs, "gross_margin", 0.20)
    cogs_arr = _broadcast_param(cogs, n_obs, "cogs", 0.80)
    holding_daily_arr = _broadcast_param(holding_daily, n_obs, "holding_daily", 0.20 / 365)
    if cls_cost is not None:
        gross_margin_arr = _broadcast_param(cls_cost, n_obs, "cls_cost", 1.0)
    if ihc_cost is not None:
        holding_daily_arr = _broadcast_param(ihc_cost, n_obs, "ihc_cost", 1.0)

    cls = float(np.sum(np.maximum(y_true_arr - y_pred_arr, 0) * unit_prices_arr * gross_margin_arr))
    ihc = float(np.sum(np.maximum(y_pred_arr - y_true_arr, 0) * unit_prices_arr * cogs_arr * holding_daily_arr))
    oos_rate = float(np.mean(y_pred_arr < y_true_arr))
    ofr = _safe_divide(np.minimum(y_true_arr, y_pred_arr).sum(), y_true_arr.sum())

    peak_threshold = np.percentile(y_true_arr, peak_threshold_percentile)
    peak_mask = y_true_arr > peak_threshold
    if peak_mask.any():
        yt_peak, yp_peak = y_true_arr[peak_mask], y_pred_arr[peak_mask]
        peak_cls = float(np.sum(np.maximum(yt_peak - yp_peak, 0) * unit_prices_arr[peak_mask] * gross_margin_arr[peak_mask]))
        peak_capture = _safe_divide(yp_peak.sum(), yt_peak.sum()) * 100
        peak_wape = _safe_divide(np.abs(yt_peak - yp_peak).sum(), yt_peak.sum())
        peak_bias = _safe_divide(yp_peak.sum() - yt_peak.sum(), yt_peak.sum()) * 100
    else:
        peak_cls = peak_capture = peak_wape = peak_bias = 0.0

    metrics = {
        "MAE": mae, "RMSE": rmse, "WAPE": wape, "SMAPE_pct": smape_val,
        "Global_Bias_pct": global_bias_pct, "F1_Zero": f1_zero, "MAE_NonZero": mae_nonzero,
        "CLS": cls, "IHC": ihc, "Total_Cost": cls + ihc,
        "OOS_Rate": oos_rate, "OFR": ofr,
        "Peak_Capture_pct": peak_capture, "Peak_WAPE": peak_wape,
        "Peak_Bias_pct": peak_bias, "Peak_CLS": peak_cls,
    }
    if y_naive is not None:
        y_naive_arr = _to_1d_array(y_naive, "y_naive")
        metrics["FVA_pct"] = (1.0 - _safe_divide(mae, float(np.mean(np.abs(y_true_arr - y_naive_arr))))) * 100
    return metrics


# ═══════════════════════════════════════════════════════════════
#  Loss functions (untuk XGBoost custom objective)
# ═══════════════════════════════════════════════════════════════

def asymmetric_obj(alpha: float = 2.0):
    """Asymmetric quadratic loss — penalty lebih besar untuk over vs under."""
    def obj(y_true, y_pred, sample_weight=None):
        residual = y_pred - y_true
        grad = np.where(residual > 0, 2 * residual, 2 * alpha * residual)
        hess = np.where(residual > 0, 2.0, 2.0 * alpha)
        if sample_weight is not None:
            grad *= sample_weight
            hess *= sample_weight
        return grad, hess
    return obj


def quantile_obj(q: float = 0.8):
    """Custom quantile objective."""
    def obj(y_true, y_pred, sample_weight=None):
        residual = y_pred - y_true
        grad = np.where(residual >= 0, q, q - 1.0)
        hess = np.ones_like(grad) * 1e-6
        if sample_weight is not None:
            grad *= sample_weight
            hess *= sample_weight
        return grad, hess
    return obj
