from typing import Optional

from pydantic import BaseModel, Field


class FeaturePayload(BaseModel):
    stock_code: Optional[str] = None
    country: Optional[str] = None
    avg_price: float = Field(default=10.0, ge=0.0)
    is_hari_besar: int = Field(default=0, ge=0, le=1)
    is_pre_hari_besar: int = Field(default=0, ge=0, le=1)
    is_peak_day: int = Field(default=0, ge=0, le=1)
    day_of_week: int = Field(default=0, ge=0, le=6)
    week_of_year: int = Field(default=1, ge=1, le=53)
    month: int = Field(default=1, ge=1, le=12)
    quarter: int = Field(default=1, ge=1, le=4)
    day_of_month: int = Field(default=1, ge=1, le=31)
    is_weekend: int = Field(default=0, ge=0, le=1)
    is_month_start: int = Field(default=0, ge=0, le=1)
    is_month_end: int = Field(default=0, ge=0, le=1)
    days_to_month_end: int = Field(default=15, ge=0, le=31)
    week_of_month: int = Field(default=1, ge=1, le=5)
    is_month_start_window: int = Field(default=0, ge=0, le=1)
    is_month_end_window: int = Field(default=0, ge=0, le=1)
    holiday_intensity: float = Field(default=1.0, ge=0.0)
    days_to_next_holiday: int = Field(default=30, ge=0, le=30)
    is_holiday_season: int = Field(default=0, ge=0, le=1)
    holiday_x_weekend: int = Field(default=0, ge=0, le=1)
    demand_lag_1: float = Field(default=0.0, ge=0.0)
    demand_lag_2: float = Field(default=0.0, ge=0.0)
    demand_lag_7: float = Field(default=0.0, ge=0.0)
    demand_lag_14: float = Field(default=0.0, ge=0.0)
    demand_lag_21: float = Field(default=0.0, ge=0.0)
    demand_lag_28: float = Field(default=0.0, ge=0.0)
    demand_lag_35: float = Field(default=0.0, ge=0.0)
    demand_lag_56: float = Field(default=0.0, ge=0.0)
    demand_lag_84: float = Field(default=0.0, ge=0.0)
    days_since_last_sale: int = Field(default=0, ge=0)
    roll_zero_count_14: float = Field(default=0.0, ge=0.0)
    roll_max_7: float = Field(default=0.0, ge=0.0)
    roll_max_28: float = Field(default=0.0, ge=0.0)
    roll_mean_7: float = Field(default=0.0, ge=0.0)
    roll_mean_14: float = Field(default=0.0, ge=0.0)
    roll_mean_28: float = Field(default=0.0, ge=0.0)
    roll_mean_56: float = Field(default=0.0, ge=0.0)
    roll_median_7: float = Field(default=0.0, ge=0.0)
    roll_median_14: float = Field(default=0.0, ge=0.0)
    roll_median_28: float = Field(default=0.0, ge=0.0)
    roll_std_7: float = Field(default=0.0, ge=0.0)
    roll_std_14: float = Field(default=0.0, ge=0.0)
    roll_std_28: float = Field(default=0.0, ge=0.0)
    roll_std_56: float = Field(default=0.0, ge=0.0)
    roll_max_56: float = Field(default=0.0, ge=0.0)
    roll_max_84: float = Field(default=0.0, ge=0.0)
    demand_acceleration_3d: float = Field(default=0.0)
    spike_ratio_28: float = Field(default=0.0)
    spike_ratio_56: float = Field(default=0.0)
    pct_change_1: float = Field(default=0.0)
    pct_change_7: float = Field(default=0.0)
    discount_depth_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    price_momentum: float = Field(default=1.0, ge=0.0)
    margin_multiplier: float = Field(default=1.0, ge=0.5, le=3.0)
    perishability_score: float = Field(default=0.3, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    expected_demand: float
    recommended_stock: float
    risk_status: str
