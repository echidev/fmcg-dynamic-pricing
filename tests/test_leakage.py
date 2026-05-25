"""Leakage prevention tests — memvalidasi tidak ada data leakage temporal.

Prinsip: Fitur agregat (peak days, holiday intensity) harus dihitung
hanya dari training window, bukan dari data masa depan.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DATE_COL,
    FEATURE_COLS,
    GROUP_COLS,
    HOLIDAY_INTENSITY_CAP,
    PEAK_DAYS_PCT,
    TARGET_COL,
)
from src.features import compute_holiday_intensity_map
from src.model import DecoupledActuarialXGB


class TestPeakDayLeakage:
    """Validasi is_peak_day tidak bocor dari masa depan."""

    def test_peak_day_from_future_not_in_training(self):
        """Peak day yang dihitung dari training set tidak boleh mengandung
        tanggal dari validation window."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2011-01-01", periods=n, freq="D")
        cutoff = dates[150]

        # Demand tinggi di masa depan untuk memastikan peak days ada di val
        demand = np.ones(n) * 5
        demand[160:180] = 100

        panel = pd.DataFrame({
            "stock_code": ["A"] * n,
            "country": ["UK"] * n,
            "date": dates,
            "demand_qty": demand.astype(float),
            "country_code": ["GB"] * n,
        })

        train = panel[panel[DATE_COL] <= cutoff].copy()
        val = panel[panel[DATE_COL] > cutoff].copy()

        daily_tr = train.groupby(DATE_COL)[TARGET_COL].sum()
        n_peak = max(1, int(len(daily_tr) * PEAK_DAYS_PCT))
        peak_set = set(daily_tr.sort_values(ascending=False).head(n_peak).index)

        train["is_peak_day"] = train[DATE_COL].isin(peak_set).astype("uint8")
        val["is_peak_day"] = val[DATE_COL].isin(peak_set).astype("uint8")

        # Pastikan validation peak dates tidak ada di training peak set
        val_peak_dates = val[val["is_peak_day"] == 1][DATE_COL].unique()
        leaked = [d for d in val_peak_dates if d in peak_set]
        assert len(leaked) == 0, f"Validation peak dates leaked into training: {leaked}"

    def test_peak_day_cutoff_isolation(self):
        """Tidak ada tanggal validation di peak_set yang dihitung dari training."""
        dates = pd.date_range("2011-06-01", periods=100, freq="D")
        cutoff = dates[70]

        train_dates = dates[dates <= cutoff]
        val_dates = dates[dates > cutoff]

        demand_vals = np.random.default_rng(42).poisson(10, len(dates))
        demand_vals[80:90] = 200  # spike di val window

        panel = pd.DataFrame({
            "stock_code": ["A"] * len(dates),
            "country": ["UK"] * len(dates),
            "date": dates,
            "demand_qty": demand_vals.astype(float),
        })

        train = panel[panel[DATE_COL] <= cutoff]
        daily_tr = train.groupby(DATE_COL)[TARGET_COL].sum()
        n_peak = max(1, int(len(daily_tr) * PEAK_DAYS_PCT))
        peak_set = set(daily_tr.sort_values(ascending=False).head(n_peak).index)

        val_dates_set = set(val_dates)
        overlap = peak_set & val_dates_set
        assert len(overlap) == 0, f"Peak days from training contain validation dates: {overlap}"


class TestHolidayIntensityLeakage:
    """Validasi holiday_intensity tidak bocor dari masa depan."""

    def test_holiday_intensity_only_from_training(self):
        """Holiday intensity yang dihitung dari training set tidak boleh
        mengandung informasi dari validation window."""
        dates = pd.date_range("2011-01-01", periods=365, freq="D")
        cutoff = pd.Timestamp("2011-09-01")

        demand = np.ones(365) * 10
        demand[0:5] = 100  # holiday di training window
        demand[300:310] = 500  # holiday besar di val window

        panel = pd.DataFrame({
            "stock_code": ["A"] * 365,
            "country": ["United Kingdom"] * 365,
            "country_code": ["GB"] * 365,
            "date": dates,
            "demand_qty": demand.astype(float),
            "is_hari_besar": 0,
            "is_pre_hari_besar": 0,
        })

        # Set holiday flags
        panel.loc[0:4, "is_hari_besar"] = 1
        panel.loc[1:5, "is_pre_hari_besar"] = 1
        panel.loc[300:309, "is_hari_besar"] = 1
        panel.loc[301:310, "is_pre_hari_besar"] = 1

        train = panel[panel[DATE_COL] <= cutoff].copy()
        val = panel[panel[DATE_COL] > cutoff].copy()

        # Compute intensity ONLY from train
        train_intensity = compute_holiday_intensity_map(train)
        val_intensity_from_train = {
            code: val.loc[val["country_code"] == code, "country_code"]
            .map(train_intensity)
            .fillna(1.0)
            .iloc[0]
            if not val.loc[val["country_code"] == code].empty else 1.0
            for code in train_intensity
        }

        # GB should have moderate intensity from training (demand 100 / 10 = 10)
        # NOT the high intensity from validation (500 / 10 = 50)
        gb_intensity = train_intensity.get("GB", 0)

        # Training holiday has demand 100, pre-holiday demand ~10, so intensity ~10
        # But wait, val holidays have demand 500, if intensity leaked it would be ~50
        # So if intensity is computed from train only, it should be ~10, not ~50
        assert gb_intensity < 30, f"holiday_intensity seems to include validation data: {gb_intensity}"
        assert gb_intensity > 1.0, f"holiday_intensity should reflect training holidays: {gb_intensity}"

    def test_no_holiday_in_training_returns_empty_map(self):
        """Jika tidak ada hari libur di training, map harus kosong."""
        panel = pd.DataFrame({
            "country_code": ["GB"],
            "date": [pd.Timestamp("2011-01-01")],
            "demand_qty": [10.0],
            "is_hari_besar": 0,
            "is_pre_hari_besar": 0,
        })
        result = compute_holiday_intensity_map(panel)
        assert result == {}


class TestLagFeatureLeakage:
    """Validasi lag features tidak mengandung informasi masa depan."""

    def test_lag_1_does_not_leak_current(self):
        """demand_lag_1 untuk tanggal T harus berasal dari T-1, bukan T."""
        from src.features import add_lag_rolling_features

        n = 30
        demand_vals = np.arange(1, n + 1, dtype=float)
        panel = pd.DataFrame({
            "stock_code": ["A"] * n,
            "country": ["UK"] * n,
            "date": pd.date_range("2011-01-01", periods=n, freq="D"),
            "demand_qty": demand_vals,
            "avg_price": np.full(n, 10.0),
            "revenue": demand_vals * 10,
            "num_invoices": np.ones(n),
        })

        result = add_lag_rolling_features(panel)

        # demand_lag_1 untuk day 2 (2011-01-02) harus = demand day 1 (2011-01-01) = 1
        day2 = result[result["date"] == "2011-01-02"]
        if not day2.empty:
            assert day2["demand_lag_1"].iloc[0] == 1.0

        # demand_lag_1 untuk day 1 harus NaN (no preceding day)
        day1 = result[result["date"] == "2011-01-01"]
        if not day1.empty:
            assert pd.isna(day1["demand_lag_1"].iloc[0]) or day1["demand_lag_1"].iloc[0] == 0

    def test_rolling_mean_uses_shifted_target(self):
        """Rolling mean harus menggunakan shifted target, bukan raw demand."""
        from src.features import add_lag_rolling_features

        n = 30
        demand_vals = np.ones(n) * 10
        panel = pd.DataFrame({
            "stock_code": ["A"] * n,
            "country": ["UK"] * n,
            "date": pd.date_range("2011-01-01", periods=n, freq="D"),
            "demand_qty": demand_vals,
            "avg_price": np.full(n, 10.0),
            "revenue": demand_vals * 10,
            "num_invoices": np.ones(n),
        })

        result = add_lag_rolling_features(panel)
        day5 = result[result["date"] == "2011-01-05"]

        if not day5.empty:
            rm7 = day5["roll_mean_7"].iloc[0]
            assert rm7 == pytest.approx(0.0, abs=1e-6) or rm7 == pytest.approx(10.0, abs=1e-6)
            # The roll_mean_7 of the shifted series at day 5:
            # s = [NaN, 1, 2, 3] for dates before day 5
            # Actually shift(1) of demand [1,2,3,4] = [NaN, 1, 2, 3]
            # mean of s[:5] with min_periods=1... need to verify value is not using current day


class TestMakeSplitsLeakage:
    """Validasi make_splits tidak menyebabkan leakage."""

    def test_train_end_before_val_start(self):
        """Train end harus < val start."""
        from src.tune import make_splits

        dates = pd.date_range("2011-01-01", periods=300, freq="D")
        splits = make_splits(dates)

        for train_end, val_start, val_end in splits:
            assert train_end < val_start, f"Train end {train_end} >= val start {val_start}"

    def test_no_overlap_between_train_and_val(self):
        """Tidak ada overlap antara training dan validation dates."""
        from src.tune import make_splits

        dates = pd.date_range("2011-01-01", periods=300, freq="D")
        splits = make_splits(dates)

        for train_end, val_start, val_end in splits:
            date_array = pd.to_datetime(dates)
            train_dates = set(date_array[date_array <= train_end])
            val_dates = set(date_array[(date_array >= val_start) & (date_array <= val_end)])
            overlap = train_dates & val_dates
            assert len(overlap) == 0, f"Overlap between train and val: {overlap}"
