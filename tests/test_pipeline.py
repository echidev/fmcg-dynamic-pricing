"""Integration test — end-to-end training pipeline dengan synthetic data.

Memvalidasi bahwa pipeline features -> model -> prediction tidak crash
dan mengembalikan output dengan bentuk yang benar.
"""

import numpy as np
import pandas as pd
import pytest


class TestEndToEndPipeline:
    """Test integrasi full pipeline."""

    @pytest.fixture
    def synthetic_panel(self):
        """Panel sintetis 200 rows -- includes zero-demand days."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2011-06-01", periods=n, freq="D")
        demand = rng.poisson(lam=8, size=n).astype(float)
        demand[::5] = 0
        df = pd.DataFrame({
            "stock_code": ["A"] * n,
            "country": ["UK"] * n,
            "date": dates,
            "demand_qty": demand,
            "revenue": rng.uniform(30, 300, n).astype(float),
            "avg_price": rng.uniform(3, 30, n).astype(float),
            "num_invoices": rng.poisson(lam=2, size=n).astype(float),
        })
        return df

    def test_feature_engineering_runs(self, synthetic_panel):
        """Feature engineering tidak crash pada panel kecil."""
        from src.features import build_tabular_dataframe
        result = build_tabular_dataframe(synthetic_panel)
        assert result is not None
        assert len(result) >= len(synthetic_panel)

    def test_end_to_end_training(self, synthetic_panel):
        """Training DecoupledActuarialXGB dengan data sintetis -> valid prediction."""
        from src.features import add_calendar_holiday_features, add_holiday_intensity_features
        from src.features import add_lag_rolling_features
        from src.model import DecoupledActuarialXGB
        from src.config import FEATURE_COLS, PRICE_COL, TARGET_COL

        # Feature engineering
        panel = add_calendar_holiday_features(synthetic_panel)
        panel = add_holiday_intensity_features(panel)
        panel["is_peak_day"] = 0
        panel["is_peak_day"] = panel["is_peak_day"].astype("uint8")
        panel = add_lag_rolling_features(panel)

        # Filter rows with NaN from lag
        panel = panel.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        assert len(panel) > 10, "Too few rows after feature engineering"

        # Train/val split (80/20)
        split = int(len(panel) * 0.8)
        train, val = panel.iloc[:split], panel.iloc[split:]

        X_tr = train[FEATURE_COLS]
        y_tr = train[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        X_vl = val[FEATURE_COLS]
        y_vl = val[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        p_vl = val[PRICE_COL].to_numpy(dtype=np.float32, copy=False)
        val_keys = list(zip(val["stock_code"].to_numpy(), val["country"].to_numpy()))

        # Model with tiny params for speed
        model = DecoupledActuarialXGB(
            model_params={"n_estimators": 2, "max_depth": 2,
                          "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            quantile_q_target=0.90,
            shortage_margin_multiplier=1.5,
        )

        # Train
        model.fit(X_tr, y_tr, feature_cols=FEATURE_COLS)

        # Predict
        pred = model.predict(X_vl, p_vl, val_keys)

        # Assertions
        assert len(pred) == len(y_vl), "Prediction length mismatch"
        assert (pred >= 0).all(), "Negative predictions!"
        assert pred.dtype in (np.float64, np.float32), "Prediction should be float"

        # Evaluate
        metrics = model.evaluate(X_vl, y_vl, p_vl, val_keys)
        assert "mae" in metrics, "Missing MAE in metrics"
        assert "cls" in metrics, "Missing CLS in metrics"
        assert "ofr" in metrics, "Missing OFR in metrics"
        print(f"End-to-end OK: MAE={metrics['mae']:.2f}, CLS={metrics['cls']:.2f}, OFR={metrics['ofr']:.3f}")
