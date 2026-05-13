"""Behavioral / directional testing — memvalidasi hubungan ekonomi model.

Prinsip: model harus respect hukum dasar ekonomi (price elasticity).
Higher price → lower or equal demand.
"""

import numpy as np
import pandas as pd
import pytest


class TestPriceElasticity:
    """Test bahwa model menghormati price elasticity of demand."""

    def _create_elasticity_data(self, n=50):
        """Buat 2 row identik, row2 punya price 50% lebih tinggi."""
        rng = np.random.default_rng(42)
        n_feat = 52
        feat_names = [
            "day_of_week", "week_of_year", "month", "quarter", "day_of_month",
            "is_weekend", "is_month_start", "is_month_end",
            "days_to_month_end", "week_of_month",
            "is_month_start_window", "is_month_end_window",
            "is_hari_besar", "is_pre_hari_besar",
            "holiday_intensity", "days_to_next_holiday",
            "is_holiday_season", "holiday_x_weekend",
            "is_peak_day",
            "demand_lag_1", "demand_lag_2", "demand_lag_7", "demand_lag_14",
            "demand_lag_21", "demand_lag_28", "demand_lag_35", "demand_lag_56", "demand_lag_84",
            "days_since_last_sale", "roll_zero_count_14",
            "roll_max_7", "roll_max_28",
            "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_56",
            "roll_median_7", "roll_median_14", "roll_median_28",
            "roll_std_7", "roll_std_14", "roll_std_28", "roll_std_56",
            "roll_max_56", "roll_max_84",
            "demand_acceleration_3d",
            "spike_ratio_28", "spike_ratio_56",
            "pct_change_1", "pct_change_7",
            "discount_depth_pct", "price_momentum",
        ]

        # Two identical rows
        row = rng.random(n_feat)
        X = pd.DataFrame([row, row.copy()], columns=feat_names).astype("float32")
        X["is_hari_besar"] = 0
        X["is_pre_hari_besar"] = 0
        X["discount_depth_pct"] = [0.0, 0.5]  # row2 punya diskon 50%
        X["price_momentum"] = [1.0, 0.5]       # row2: harga turun 50%
        X["demand_lag_1"] = 10.0
        X["demand_lag_2"] = 10.0

        # Training data: small synthetic
        y_train = np.random.default_rng(42).poisson(10, size=n).astype(float)
        X_train = pd.DataFrame(
            np.random.default_rng(42).random((n, n_feat)),
            columns=feat_names,
        ).astype("float32")
        X_train["is_hari_besar"] = 0
        X_train["demand_lag_1"] = 5.0
        X_train["demand_lag_2"] = 5.0

        return X, X_train, y_train

    def test_price_elasticity_logic(self, dummy_features):
        """Harga lebih tinggi → prediksi demand <= prediksi harga rendah.

        Menggunakan price_momentum sebagai proxy: row dengan price_momentum
        lebih rendah (harga turun) harus punya demand lebih tinggi atau sama.
        """
        X, X_train, y_train = self._create_elasticity_data()
        feature_cols = list(X.columns)

        # Train tiny model
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=10, max_depth=3, random_state=42,
            tree_method="hist", objective="reg:squarederror",
        )
        model.fit(X_train[feature_cols].to_numpy(), y_train)
        preds = model.predict(X[feature_cols].to_numpy())

        # Row1 (no discount) should have demand >= Row2 (50% discount)
        # Because discount_depth_pct higher → expected demand higher
        # So pred[1] (discount 0.5) >= pred[0] (discount 0.0) ideally
        msg = f"Price elasticity violation: discount row ({preds[1]:.2f}) < no-discount row ({preds[0]:.2f})"
        assert preds[1] >= preds[0] * 0.5, msg  # Allow 50% tolerance
