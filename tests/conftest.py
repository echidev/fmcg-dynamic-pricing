"""Fixtures bersama untuk semua test."""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def dummy_data():
    """Dataset sintetis 100 rows untuk testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2011-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": np.tile(dates, 2)[:100],
        "stock_code": ["A"] * 50 + ["B"] * 50,
        "country": ["UK"] * 50 + ["US"] * 50,
        "demand_qty": rng.poisson(lam=10, size=100).astype(float),
        "revenue": rng.uniform(50, 500, 100).astype(float),
        "avg_price": rng.uniform(5, 50, 100).astype(float),
        "num_invoices": rng.poisson(lam=3, size=100).astype(float),
    })
    return df


@pytest.fixture
def dummy_features(dummy_data):
    """DataFrame dengan FEATURE_COLS sintetis."""
    rng = np.random.default_rng(42)
    n = len(dummy_data)
    cols = [
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
    X = pd.DataFrame(rng.random((n, len(cols))), columns=cols).astype("float32")
    X["is_hari_besar"] = 0
    X["is_pre_hari_besar"] = 0
    X["is_weekend"] = (X["day_of_week"] > 0.7).astype("float32")
    X["holiday_x_weekend"] = (X["is_hari_besar"] * X["is_weekend"]).astype("float32")
    for c in ["is_weekend", "is_month_start", "is_month_end", "is_holiday_season", "is_peak_day"]:
        if c in X.columns:
            X[c] = (X[c] > 0.5).astype("float32")
    X["days_to_next_holiday"] = rng.integers(0, 30, n).astype("float32")
    X["is_hari_besar"].iloc[25] = 1  # satu holiday
    for c in ["demand_lag_1", "demand_lag_2", "demand_lag_7"]:
        X[c] = 1.0
    X["discount_depth_pct"] = rng.random(n).astype("float32")
    X["country_code"] = dummy_data["country"].map(
        {"UK": "GB", "US": "US"}
    ).fillna("GB")
    return X
