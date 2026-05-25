"""Unit test untuk DecoupledActuarialXGB secara terisolasi."""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS, PRICE_COL, TARGET_COL
from src.model import DecoupledActuarialXGB


@pytest.fixture
def tiny_data():
    """Dataset sintetis kecil 100 rows untuk test model."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame(rng.random((n, len(FEATURE_COLS))), columns=FEATURE_COLS).astype("float32")
    X["is_hari_besar"] = 0
    X["is_pre_hari_besar"] = 0
    X["is_weekend"] = (X["day_of_week"] > 0.7).astype("float32")
    X["is_peak_day"] = 0
    X["holiday_x_weekend"] = 0
    X["discount_depth_pct"] = rng.random(n).astype("float32")
    X["price_momentum"] = rng.uniform(0.5, 1.5, n).astype("float32")
    X["days_to_next_holiday"] = rng.integers(0, 30, n).astype("float32")
    X["demand_lag_1"] = 5.0
    X["demand_lag_2"] = 5.0
    y = rng.poisson(lam=10, size=n).astype(np.float32)
    price = rng.uniform(5, 50, n).astype(np.float32)
    keys = [("A", "UK")] * 50 + [("B", "US")] * 50
    return X, y, price, keys


class TestDecoupledActuarialXGBInit:
    """Test konstruktor dan konfigurasi."""

    def test_default_params(self):
        model = DecoupledActuarialXGB()
        assert model.quantile_q_target == 0.9799
        assert model.shortage_margin_multiplier == 2.2847
        assert model.use_log_target is True
        assert model._fitted is False

    def test_custom_params(self):
        model = DecoupledActuarialXGB(
            quantile_q_target=0.85,
            shortage_margin_multiplier=1.5,
            use_log_target=False,
        )
        assert model.quantile_q_target == 0.85
        assert model.shortage_margin_multiplier == 1.5
        assert model.use_log_target is False

    def test_model_params_stored(self):
        custom = {"n_estimators": 100, "max_depth": 4}
        model = DecoupledActuarialXGB(model_params=custom)
        assert model.model_params["n_estimators"] == 100
        assert model.model_params["max_depth"] == 4


class TestDecoupledActuarialXGBFit:
    """Test fit method."""

    def test_fit_creates_models(self, tiny_data):
        X, y, price, keys = tiny_data
        model = DecoupledActuarialXGB(
            model_params={"n_estimators": 5, "max_depth": 2, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )
        model.fit(X, y, feature_cols=FEATURE_COLS)
        assert model._fitted is True
        assert hasattr(model, "model_mean_")
        assert hasattr(model, "model_quant_")
        assert hasattr(model, "feature_cols_")

    def test_fit_stores_feature_cols(self, tiny_data):
        X, y, price, keys = tiny_data
        model = DecoupledActuarialXGB(
            model_params={"n_estimators": 5, "max_depth": 2, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )
        model.fit(X, y, feature_cols=FEATURE_COLS)
        assert model.feature_cols_ == FEATURE_COLS

    def test_fit_mostly_zero_y(self, tiny_data):
        X, y, price, keys = tiny_data
        y_sparse = y.copy()
        y_sparse[y_sparse < 12] = 0
        model = DecoupledActuarialXGB(
            model_params={"n_estimators": 5, "max_depth": 2, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
        )
        model.fit(X, y_sparse, feature_cols=FEATURE_COLS)
        assert model._fitted is True


class TestDecoupledActuarialXGBPredict:
    """Test predict methods."""

    def test_predict_raw_returns_two_arrays(self, tiny_data):
        X, y, price, keys = tiny_data
        model = DecoupledActuarialXGB(
            model_params={"n_estimators": 5, "max_depth": 2, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )
        model.fit(X, y, feature_cols=FEATURE_COLS)
        p_mean, p_quant = model.predict_raw(X)
        assert isinstance(p_mean, np.ndarray)
        assert isinstance(p_quant, np.ndarray)
        assert p_mean.shape == (len(y),)
        assert p_quant.shape == (len(y),)

    def test_predict_returns_non_negative(self, tiny_data):
        X, y, price, keys = tiny_data
        model = DecoupledActuarialXGB(
            model_params={"n_estimators": 5, "max_depth": 2, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )
        model.fit(X, y, feature_cols=FEATURE_COLS)
        pred = model.predict(X, price, keys)
        assert (pred >= 0).all()

    def test_predict_without_fit_raises(self, tiny_data):
        X, y, price, keys = tiny_data
        model = DecoupledActuarialXGB()
        with pytest.raises(RuntimeError, match="belum di-fit"):
            model.predict(X, price, keys)

    def test_predict_raw_without_fit_raises(self, tiny_data):
        X, y, price, keys = tiny_data
        model = DecoupledActuarialXGB()
        with pytest.raises(RuntimeError, match="belum di-fit"):
            model.predict_raw(X)

    def test_predict_actuarial_returns_components(self, tiny_data):
        X, y, price, keys = tiny_data
        model = DecoupledActuarialXGB(
            model_params={"n_estimators": 5, "max_depth": 2, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )
        model.fit(X, y, feature_cols=FEATURE_COLS)
        result = model.predict_actuarial(X, price, keys, return_components=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        pred, components = result
        for key in ("p_mean", "p_quant", "critical_fractile", "margin_ratio"):
            assert key in components


class TestDecoupledActuarialXGBEvaluate:
    """Test evaluate method."""

    def test_evaluate_returns_metrics_dict(self, tiny_data):
        X, y, price, keys = tiny_data
        model = DecoupledActuarialXGB(
            model_params={"n_estimators": 5, "max_depth": 2, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )
        model.fit(X, y, feature_cols=FEATURE_COLS)
        metrics = model.evaluate(X, y, price, keys)
        for key in ("mae", "rmse", "smape", "cls", "ofr"):
            assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics["mae"], float)

    def test_evaluate_with_baseline(self, tiny_data):
        X, y, price, keys = tiny_data
        model = DecoupledActuarialXGB(
            model_params={"n_estimators": 5, "max_depth": 2, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )
        model.fit(X, y, feature_cols=FEATURE_COLS)
        metrics = model.evaluate(X, y, price, keys, baseline_mae=10.0)
        assert "fva" in metrics


class TestDecoupledActuarialXGBGetParams:
    """Test get_params method."""

    def test_get_params_returns_config(self):
        model = DecoupledActuarialXGB(
            quantile_q_target=0.85,
            shortage_margin_multiplier=1.5,
        )
        params = model.get_params()
        assert params["quantile_q_target"] == 0.85
        assert params["shortage_margin_multiplier"] == 1.5
        assert "model_params" in params
        assert "use_log_target" in params
