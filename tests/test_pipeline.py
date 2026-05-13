"""Integration test — end‑to‑end training pipeline dengan synthetic data.

Memvalidasi bahwa pipeline features → model → prediction tidak crash
dan mengembalikan output dengan bentuk yang benar.
"""

import numpy as np
import pandas as pd
import pytest


class TestEndToEndPipeline:
    """Test integrasi full pipeline."""

    @pytest.fixture
    def synthetic_panel(self):
        """Panel sintetis 200 rows — includes zero-demand days."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2011-06-01", periods=n, freq="D")
        demand = rng.poisson(lam=8, size=n).astype(float)
        demand[::5] = 0  # every 5th row = zero demand
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
        """Training TwinXGBBoosted dengan data sintetis → valid prediction."""
        from src.features import add_calendar_holiday_features, add_holiday_intensity_features
        from src.features import add_peak_days, add_lag_rolling_features
        from src.model import TwinXGBBoosted
        from src.config import FEATURE_COLS, GROUP_COLS, PRICE_COL, TARGET_COL

        # Feature engineering
        panel = add_calendar_holiday_features(synthetic_panel)
        panel = add_holiday_intensity_features(panel)
        panel = add_peak_days(panel)
        panel = add_lag_rolling_features(panel)

        # Filter rows with NaN from lag
        panel = panel.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        assert len(panel) > 10, "Too few rows after feature engineering"

        # Train/val split (80/20)
        split = int(len(panel) * 0.8)
        train, val = panel.iloc[:split], panel.iloc[split:]

        X_tr = train[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
        y_tr = train[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        X_vl = val[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
        y_vl = val[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        p_vl = val[PRICE_COL].to_numpy(dtype=np.float32, copy=False)

        ht = train["is_hari_besar"].to_numpy(dtype=float)
        ph = train["is_pre_hari_besar"].to_numpy(dtype=float)
        pk = train["is_peak_day"].to_numpy(dtype=float)

        tkeys = list(zip(train["stock_code"].to_numpy(), train["country"].to_numpy()))
        vkeys = list(zip(val["stock_code"].to_numpy(), val["country"].to_numpy()))
        vp = val["is_peak_day"].to_numpy(dtype=bool)
        vh = (val["is_hari_besar"].to_numpy(dtype=bool) | val["is_pre_hari_besar"].to_numpy(dtype=bool))

        seg = train.groupby(GROUP_COLS)[TARGET_COL].sum().sort_values(ascending=False)
        top_keys = set(seg.head(max(1, int(len(seg) * 0.05))).index)

        # Model with tiny params for speed
        model = TwinXGBBoosted(
            clf_params={"n_estimators": 2, "max_depth": 2, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            reg_params={"n_estimators": 2, "max_depth": 2, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            alpha_under=10,
        )

        # Train
        model.fit(X_tr, y_tr, ht, ph, pk, top_keys, tkeys)

        # Predict
        pred_final, metrics, best_th = model.predict_with_threshold(
            X_vl, y_vl, p_vl, vkeys, vp, vh,
        )

        # Assertions
        assert len(pred_final) == len(y_vl), "Prediction length mismatch"
        assert (pred_final >= 0).all(), "Negative predictions!"
        assert "mae" in metrics, "Missing MAE in metrics"
        assert "cls" in metrics, "Missing CLS in metrics"
        assert "ofr" in metrics, "Missing OFR in metrics"
        assert isinstance(best_th, float), "Threshold should be float"
        print(f"End-to-end OK: MAE={metrics['mae']:.2f}, CLS={metrics['cls']:.2f}, OFR={metrics['ofr']:.3f}")
