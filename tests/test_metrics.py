"""Unit test untuk semua fungsi metrics."""

import numpy as np
import pytest

from src.metrics import (
    calc_cls,
    evaluate_model_performance,
    evaluate_prediction,
    quantile_obj,
    smape,
)


class TestCalcCLS:
    """Test Cost of Lost Sales."""

    def test_cls_zero_perfect_prediction(self):
        y_true = np.array([10, 20, 30], dtype=float)
        y_pred = np.array([10, 20, 30], dtype=float)
        price = np.array([5, 5, 5], dtype=float)
        assert calc_cls(y_true, y_pred, price) == 0.0

    def test_cls_positive_underforecast(self):
        y_true = np.array([10, 20, 30], dtype=float)
        y_pred = np.array([5, 10, 15], dtype=float)
        price = np.array([5, 5, 5], dtype=float)
        expected = (5 * 5 * 0.2) + (10 * 5 * 0.2) + (15 * 5 * 0.2)
        assert calc_cls(y_true, y_pred, price) == pytest.approx(expected)

    def test_cls_zero_overforecast(self):
        y_true = np.array([10, 20], dtype=float)
        y_pred = np.array([15, 25], dtype=float)
        price = np.array([5, 5], dtype=float)
        assert calc_cls(y_true, y_pred, price) == 0.0

    def test_cls_custom_margin(self):
        y_true = np.array([10], dtype=float)
        y_pred = np.array([5], dtype=float)
        price = np.array([5], dtype=float)
        assert calc_cls(y_true, y_pred, price, margin=0.5) == pytest.approx(5 * 5 * 0.5)

    def test_cls_empty_arrays(self):
        assert calc_cls(np.array([]), np.array([]), np.array([])) == 0.0


class TestSMAPE:
    """Test Symmetric Mean Absolute Percentage Error."""

    def test_smape_perfect_prediction(self):
        assert smape(np.array([10, 20]), np.array([10, 20])) == 0.0

    def test_smape_half_error(self):
        y_true = np.array([100], dtype=float)
        y_pred = np.array([50], dtype=float)
        denom = (100 + 50) / 2
        expected = abs(100 - 50) / denom * 100
        assert smape(y_true, y_pred) == pytest.approx(expected)

    def test_smape_zero_denom_safe(self):
        assert smape(np.array([0, 0]), np.array([0, 0])) == 0.0

    def test_smape_all_positive(self):
        val = smape(np.array([10, 20, 30]), np.array([12, 18, 33]))
        assert val >= 0


class TestEvaluatePrediction:
    """Test evaluate_prediction yang dipakai di model."""

    def test_returns_required_keys(self):
        y_true = np.array([10, 20, 30], dtype=float)
        y_pred = np.array([12, 18, 28], dtype=float)
        price = np.array([5, 5, 5], dtype=float)
        result = evaluate_prediction(y_true, y_pred, price)
        required = {"mae", "rmse", "smape", "cls", "ofr", "oos_rate", "f1_zero", "precision_zero", "recall_zero"}
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_mae_zero_perfect(self):
        y_true = np.array([10, 20], dtype=float)
        result = evaluate_prediction(y_true, y_true, np.array([5, 5]))
        assert result["mae"] == 0.0
        assert result["cls"] == 0.0
        assert result["ofr"] == pytest.approx(1.0, abs=1e-6)

    def test_ofr_formula(self):
        y_true = np.array([10, 0, 30], dtype=float)
        y_pred = np.array([8, 2, 25], dtype=float)
        price = np.array([5, 5, 5], dtype=float)
        result = evaluate_prediction(y_true, y_pred, price)
        expected_ofr = (min(10, 8) + min(0, 2) + min(30, 25)) / (10 + 0 + 30)
        assert result["ofr"] == pytest.approx(expected_ofr)

    def test_fva_with_baseline(self):
        y_true = np.array([10, 20, 30], dtype=float)
        y_pred = np.array([12, 18, 28], dtype=float)
        price = np.array([5, 5, 5], dtype=float)
        result = evaluate_prediction(y_true, y_pred, price, baseline_mae=10.0)
        assert result["fva"] is not None
        assert isinstance(result["fva"], float)

    def test_fva_none_without_baseline(self):
        result = evaluate_prediction(np.array([1, 2]), np.array([1, 2]), np.array([1, 1]))
        assert result["fva"] is None


class TestQuantileObj:
    """Test quantile objective function factory."""

    def test_returns_callable(self):
        obj = quantile_obj(q=0.8)
        assert callable(obj)

    def test_grad_hess_shape(self):
        obj = quantile_obj(q=0.8)
        y_true = np.array([10, 20], dtype=float)
        y_pred = np.array([12, 18], dtype=float)
        grad, hess = obj(y_true, y_pred)
        assert grad.shape == y_true.shape
        assert hess.shape == y_true.shape

    def test_under_prediction_grad(self):
        obj = quantile_obj(q=0.8)
        y_true = np.array([10], dtype=float)
        y_pred = np.array([5], dtype=float)  # under-prediction
        grad, hess = obj(y_true, y_pred)
        assert grad[0] == pytest.approx(-0.2)  # q - 1.0 = -0.2
        assert hess[0] > 0


class TestEnterpriseEvaluate:
    """Test evaluate_model_performance (uppercase keys)."""

    def test_empty_input(self):
        result = evaluate_model_performance(np.array([]), np.array([]))
        assert isinstance(result, dict)
        assert result["MAE"] == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate_model_performance(np.array([1, 2]), np.array([1]))

    def test_perfect_prediction_enterprise(self):
        y = np.array([10, 20, 30], dtype=float)
        result = evaluate_model_performance(y, y, unit_prices=[5, 5, 5])
        assert result["MAE"] == 0.0
        assert result["CLS"] == 0.0
        assert result["OFR"] == 1.0

    def test_peak_metrics(self):
        y_true = np.array([1, 2, 100, 1, 2, 100], dtype=float)
        y_pred = np.array([1, 2, 80, 1, 2, 90], dtype=float)
        result = evaluate_model_performance(y_true, y_pred, peak_threshold_percentile=90)
        assert "Peak_CLS" in result
        assert "Peak_Capture_pct" in result
