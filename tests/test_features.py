"""Unit test untuk fungsi feature engineering.

Fokus: validasi logika holiday feature dan discount_depth_pct.
"""

import numpy as np
import pandas as pd
import pytest


class TestHolidayFeature:
    """Test logika add_calendar_holiday_features."""

    @pytest.fixture
    def uk_holiday_df(self):
        """DataFrame dengan tanggal 25 Desember."""
        rng = pd.date_range("2011-12-20", periods=15, freq="D")
        return pd.DataFrame({
            "date": rng,
            "stock_code": ["A"] * 15,
            "country": ["United Kingdom"] * 15,
            "country_code": ["GB"] * 15,
            "demand_qty": np.random.default_rng(42).poisson(10, 15).astype(float),
        })

    def test_christmas_is_holiday(self, uk_holiday_df):
        """25 Dec harus punya is_hari_besar = 1."""
        from src.features import add_calendar_holiday_features
        result = add_calendar_holiday_features(uk_holiday_df)
        christmas = result[result["date"] == "2011-12-25"]
        assert len(christmas) > 0, "25 Dec harus ada di dataset"
        assert christmas["is_hari_besar"].iloc[0] == 1, "25 Dec harus terdeteksi sebagai hari besar"

    def test_new_year_is_holiday(self, uk_holiday_df):
        """1 Jan harus punya is_hari_besar = 1."""
        result = pd.concat([uk_holiday_df, pd.DataFrame({
            "date": [pd.Timestamp("2012-01-01")],
            "stock_code": ["A"], "country": ["United Kingdom"],
            "country_code": ["GB"], "demand_qty": [5.0],
        })], ignore_index=True)
        from src.features import add_calendar_holiday_features
        result = add_calendar_holiday_features(result)
        ny = result[result["date"] == "2012-01-01"]
        assert len(ny) > 0
        assert ny["is_hari_besar"].iloc[0] == 1


class TestDiscountDepth:
    """Test logika discount_depth_pct."""

    def test_discount_zero_for_no_change(self):
        """Harga konstan dalam 30 hari → discount_depth = 0."""
        n = 50
        df = pd.DataFrame({
            "stock_code": ["A"] * n, "country": ["UK"] * n,
            "date": pd.date_range("2011-01-01", periods=n, freq="D"),
            "avg_price": [10.0] * n,
        })
        from src.features import add_lag_rolling_features
        df["demand_qty"] = np.random.default_rng(42).poisson(10, n).astype(float)
        result = add_lag_rolling_features(df)
        assert "discount_depth_pct" in result.columns
        assert result["discount_depth_pct"].iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_discount_detects_price_drop(self):
        """Harga turun 50% → discount_depth ≈ 0.5."""
        n = 60
        prices = [10.0] * 30 + [5.0] * 30
        df = pd.DataFrame({
            "stock_code": ["A"] * n, "country": ["UK"] * n,
            "date": pd.date_range("2011-01-01", periods=n, freq="D"),
            "avg_price": prices,
        })
        df["demand_qty"] = np.random.default_rng(42).poisson(10, n).astype(float)
        from src.features import add_lag_rolling_features
        result = add_lag_rolling_features(df)
        last_discount = result["discount_depth_pct"].iloc[-1]
        assert last_discount > 0.3, f"Expected >0.3, got {last_discount:.4f}"

    def test_discount_zero_division_safe(self):
        """Semua harga nol → tidak crash, hasil 0."""
        df = pd.DataFrame({
            "stock_code": ["A"] * 40,
            "country": ["UK"] * 40,
            "date": pd.date_range("2011-01-01", periods=40, freq="D"),
            "avg_price": [0.0] * 40,
            "demand_qty": [1.0] * 40,
        })
        from src.features import add_lag_rolling_features
        result = add_lag_rolling_features(df)
        assert result["discount_depth_pct"].isna().sum() == 0
        assert result["discount_depth_pct"].iloc[-1] == 0.0
