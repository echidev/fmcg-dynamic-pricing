"""Behavioral / directional testing — memvalidasi hubungan ekonomi model.

Prinsip: model harus respect hukum dasar ekonomi (price elasticity).
Higher price -> lower or equal demand.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS, PRICE_COL, TARGET_COL
from src.model import DecoupledActuarialXGB


class TestPriceElasticity:
    """Test bahwa DecoupledActuarialXGB menghormati price elasticity."""

    def _make_fake_features(self, n: int, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.random((n, len(FEATURE_COLS))), columns=FEATURE_COLS).astype("float32")
        X["is_hari_besar"] = 0
        X["is_pre_hari_besar"] = 0
        X["is_weekend"] = (X["day_of_week"] > 0.7).astype("float32")
        X["holiday_x_weekend"] = (X["is_hari_besar"] * X["is_weekend"]).astype("float32")
        X["is_peak_day"] = 0
        X["discount_depth_pct"] = rng.random(n).astype("float32")
        X["price_momentum"] = rng.uniform(0.5, 1.5, n).astype("float32")
        X["demand_lag_1"] = 5.0
        X["demand_lag_2"] = 5.0
        X["days_to_next_holiday"] = rng.integers(0, 30, n).astype("float32")
        for c in FEATURE_COLS:
            if X[c].dtype == "float64":
                X[c] = X[c].astype("float32")
        return X

    def test_price_elasticity_logic(self):
        """Harga lebih tinggi -> prediksi demand lebih rendah atau sama.

        Menggunakan price_momentum sebagai proxy: row dengan price_momentum
        lebih rendah (harga turun) harus punya demand lebih tinggi.
        """
        rng = np.random.default_rng(42)
        n_train = 200

        X_train = self._make_fake_features(n_train)
        y_train = rng.poisson(lam=10, size=n_train).astype(np.float32)

        # Dua row identik kecuali discount_depth_pct & price_momentum
        X_test = self._make_fake_features(2, seed=99)
        X_test["discount_depth_pct"] = [0.0, 0.5]
        X_test["price_momentum"] = [1.0, 0.5]
        X_test["demand_lag_1"] = 10.0
        X_test["demand_lag_2"] = 10.0

        dummy_price = np.array([10.0, 5.0], dtype=np.float32)
        dummy_keys = [("A", "UK"), ("A", "UK")]

        model = DecoupledActuarialXGB(
            model_params={
                "n_estimators": 10, "max_depth": 3,
                "tree_method": "hist", "random_state": 42, "n_jobs": 1,
            },
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )

        model.fit(X_train, y_train, feature_cols=FEATURE_COLS)
        preds = model.predict(X_test, dummy_price, dummy_keys)

        msg = f"Price elasticity violation: discount row ({preds[1]:.2f}) < no-discount row ({preds[0]:.2f})"
        assert preds[1] >= preds[0] * 0.5, msg

    def test_non_negative_predictions(self):
        """Semua prediksi harus >= 0."""
        n = 50
        X = self._make_fake_features(n)
        y = np.random.default_rng(42).poisson(10, n).astype(np.float32)
        dummy_price = np.full(n, 10.0, dtype=np.float32)
        dummy_keys = [("A", "UK")] * n

        model = DecoupledActuarialXGB(
            model_params={
                "n_estimators": 5, "max_depth": 2,
                "tree_method": "hist", "random_state": 42, "n_jobs": 1,
            },
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )
        model.fit(X, y, feature_cols=FEATURE_COLS)
        preds = model.predict(X, dummy_price, dummy_keys)
        assert (preds >= 0).all(), "Negative predictions!"

    def test_predict_actuarial_returns_components(self):
        """predict_actuarial dengan return_components=True harus return tuple."""
        n = 50
        X = self._make_fake_features(n)
        y = np.random.default_rng(42).poisson(10, n).astype(np.float32)
        dummy_price = np.full(n, 10.0, dtype=np.float32)
        dummy_keys = [("A", "UK")] * n

        model = DecoupledActuarialXGB(
            model_params={
                "n_estimators": 5, "max_depth": 2,
                "tree_method": "hist", "random_state": 42, "n_jobs": 1,
            },
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )
        model.fit(X, y, feature_cols=FEATURE_COLS)
        result = model.predict_actuarial(X, dummy_price, dummy_keys, return_components=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        final_pred, components = result
        assert "critical_fractile" in components
        assert "p_mean" in components
        assert "p_quant" in components
