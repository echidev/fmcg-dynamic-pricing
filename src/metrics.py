"""Enterprise-grade evaluation metrics for demand forecasting.

This module centralizes all evaluation logic to enforce DRY principles and
ensure consistent metric computation across training, tuning, and inference.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def _to_1d_array(values: ArrayLike, name: str) -> np.ndarray:
    """Convert input to a 1D NumPy array of floats.

    Args:
        values: Array-like input.
        name: Argument name for error messages.

    Returns:
        1D NumPy array of dtype float.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D array-like, got shape {arr.shape}.")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _broadcast_param(
    values: Optional[ArrayLike],
    length: int,
    name: str,
    default_value: float,
) -> np.ndarray:
    """Broadcast a scalar or vector parameter to a 1D array.

    Args:
        values: Scalar or array-like input. If None, defaults are used.
        length: Target length for broadcasting.
        name: Argument name for error messages.
        default_value: Default value when values is None.

    Returns:
        1D NumPy array of length ``length``.
    """
    if values is None:
        return np.full(length, float(default_value))
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(length, float(arr))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D array-like, got shape {arr.shape}.")
    if arr.shape[0] != length:
        raise ValueError(
            f"{name} must have length {length}, got {arr.shape[0]}."
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two scalars, returning default on zero denominator."""
    return float(numerator / denominator) if denominator != 0 else float(default)


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
    """Evaluate demand forecasting performance across statistical and business metrics.

    The function provides a single source of truth for evaluation metrics, including
    global statistical accuracy, intermittent demand quality, business costs, peak
    performance, and forecast value add (FVA).

    Args:
        y_true: Ground truth demand values.
        y_pred: Predicted demand values.
        y_naive: Optional naive forecast for FVA calculation.
        unit_prices: Optional per-unit selling prices aligned with y_true.
        ihc_cost: Optional per-unit inventory holding cost multipliers.
        cls_cost: Optional per-unit cost of lost sales multipliers.
        peak_threshold_percentile: Percentile threshold for peak evaluation.
        gross_margin: Optional gross margin ratio (scalar or vector).
        cogs: Optional cost of goods sold ratio (scalar or vector).
        holding_daily: Optional daily holding cost rate (scalar or vector).

    Returns:
        Dictionary of computed metrics. Keys include:
        MAE, RMSE, WAPE, SMAPE_pct, Global_Bias_pct,
        F1_Zero, MAE_NonZero,
        CLS, IHC, Total_Cost, OOS_Rate, OFR,
        Peak_Capture_pct, Peak_WAPE, Peak_Bias_pct, Peak_CLS,
        FVA_pct (only if y_naive is provided).
    """
    y_true_arr = _to_1d_array(y_true, "y_true")
    y_pred_arr = _to_1d_array(y_pred, "y_pred")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    n_obs = y_true_arr.shape[0]
    if n_obs == 0:
        return {
            "MAE": 0.0,
            "RMSE": 0.0,
            "WAPE": 0.0,
            "SMAPE_pct": 0.0,
            "Global_Bias_pct": 0.0,
            "F1_Zero": 0.0,
            "MAE_NonZero": 0.0,
            "CLS": 0.0,
            "IHC": 0.0,
            "Total_Cost": 0.0,
            "OOS_Rate": 0.0,
            "OFR": 0.0,
            "Peak_Capture_pct": 0.0,
            "Peak_WAPE": 0.0,
            "Peak_Bias_pct": 0.0,
            "Peak_CLS": 0.0,
        }

    errors = y_true_arr - y_pred_arr
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    wape = _safe_divide(abs_errors.sum(), y_true_arr.sum(), default=0.0)

    smape_denom = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0
    smape_mask = smape_denom != 0
    smape = float(
        np.mean(abs_errors[smape_mask] / smape_denom[smape_mask]) * 100
    ) if smape_mask.any() else 0.0

    global_bias_pct = _safe_divide(
        y_pred_arr.sum() - y_true_arr.sum(),
        y_true_arr.sum(),
        default=0.0,
    ) * 100

    nonzero_mask = y_true_arr > 0
    mae_nonzero = float(np.mean(abs_errors[nonzero_mask])) if nonzero_mask.any() else 0.0

    yt = (y_true_arr > 0).astype(int)
    yp = (y_pred_arr > 0).astype(int)
    tp = np.sum((yp == 1) & (yt == 1))
    fp = np.sum((yp == 1) & (yt == 0))
    fn = np.sum((yp == 0) & (yt == 1))
    precision = _safe_divide(tp, tp + fp, default=0.0)
    recall = _safe_divide(tp, tp + fn, default=0.0)
    f1_zero = _safe_divide(2 * precision * recall, precision + recall, default=0.0)

    unit_price_arr = _broadcast_param(unit_prices, n_obs, "unit_prices", 0.0)
    gross_margin_arr = _broadcast_param(gross_margin, n_obs, "gross_margin", 0.20)
    cogs_arr = _broadcast_param(cogs, n_obs, "cogs", 1.0 - 0.20)
    holding_daily_arr = _broadcast_param(holding_daily, n_obs, "holding_daily", 0.20 / 365)

    if cls_cost is not None:
        cls_cost_arr = _broadcast_param(cls_cost, n_obs, "cls_cost", 1.0)
        gross_margin_arr = cls_cost_arr
    if ihc_cost is not None:
        ihc_cost_arr = _broadcast_param(ihc_cost, n_obs, "ihc_cost", 1.0)
        holding_daily_arr = ihc_cost_arr

    cls = float(np.sum(np.maximum(y_true_arr - y_pred_arr, 0) * unit_price_arr * gross_margin_arr))
    ihc = float(
        np.sum(np.maximum(y_pred_arr - y_true_arr, 0) * unit_price_arr * cogs_arr * holding_daily_arr)
    )
    total_cost = cls + ihc

    oos_rate = float(np.mean(y_pred_arr < y_true_arr))
    ofr = _safe_divide(np.minimum(y_true_arr, y_pred_arr).sum(), y_true_arr.sum(), default=0.0)

    peak_threshold = np.percentile(y_true_arr, peak_threshold_percentile)
    peak_mask = y_true_arr > peak_threshold
    if peak_mask.any():
        y_true_peak = y_true_arr[peak_mask]
        y_pred_peak = y_pred_arr[peak_mask]
        peak_errors = y_true_peak - y_pred_peak
        peak_abs_errors = np.abs(peak_errors)
        peak_capture_pct = _safe_divide(y_pred_peak.sum(), y_true_peak.sum(), default=0.0) * 100
        peak_wape = _safe_divide(peak_abs_errors.sum(), y_true_peak.sum(), default=0.0)
        peak_bias_pct = _safe_divide(
            y_pred_peak.sum() - y_true_peak.sum(),
            y_true_peak.sum(),
            default=0.0,
        ) * 100
        peak_cls = float(
            np.sum(
                np.maximum(y_true_peak - y_pred_peak, 0)
                * unit_price_arr[peak_mask]
                * gross_margin_arr[peak_mask]
            )
        )
    else:
        peak_capture_pct = 0.0
        peak_wape = 0.0
        peak_bias_pct = 0.0
        peak_cls = 0.0

    metrics: Dict[str, float] = {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "SMAPE_pct": smape,
        "Global_Bias_pct": global_bias_pct,
        "F1_Zero": f1_zero,
        "MAE_NonZero": mae_nonzero,
        "CLS": cls,
        "IHC": ihc,
        "Total_Cost": total_cost,
        "OOS_Rate": oos_rate,
        "OFR": ofr,
        "Peak_Capture_pct": peak_capture_pct,
        "Peak_WAPE": peak_wape,
        "Peak_Bias_pct": peak_bias_pct,
        "Peak_CLS": peak_cls,
    }

    if y_naive is not None:
        y_naive_arr = _to_1d_array(y_naive, "y_naive")
        if y_naive_arr.shape[0] != n_obs:
            raise ValueError("y_naive must have the same length as y_true.")
        mae_naive = float(np.mean(np.abs(y_true_arr - y_naive_arr)))
        fva = (1.0 - _safe_divide(mae, mae_naive, default=0.0)) * 100
        metrics["FVA_pct"] = fva

    return metrics
