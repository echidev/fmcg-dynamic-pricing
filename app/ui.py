from __future__ import annotations

import datetime
import json
import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("FMCG_API_URL", "http://localhost:8000")

COUNTRIES = [
    "United Kingdom", "Germany", "France", "EIRE", "Netherlands",
    "Spain", "Belgium", "Switzerland", "Portugal", "Australia",
    "USA", "Canada", "Italy", "Norway", "Sweden", "Denmark",
    "Finland", "Greece", "Japan", "Singapore", "Hong Kong",
    "Israel", "Saudi Arabia", "United Arab Emirates", "Bahrain",
    "Lebanon", "Malta", "Cyprus", "Lithuania", "Iceland",
    "Poland", "Austria", "Czech Republic", "RSA", "Nigeria",
    "Thailand", "Korea", "Bermuda", "Unspecified",
]

STOCK_CODES = [
    "84077", "85099B", "85123A", "21212", "22326",
    "47566", "47559B", "23203", "23217", "23244",
    "23269", "23272", "23281", "23284", "23285",
]


def _build_preview_payload(
    stock_code: str,
    country: str,
    avg_price: float,
    forecast_date: datetime.date,
    is_public_holiday: bool,
    is_peak_day: bool,
    is_promo: bool,
) -> dict[str, Any]:
    dow = forecast_date.weekday()
    # Estimasi demand baseline dari price untuk mengisi lag features
    # Catatan: Untuk production, lag features harus berasal dari historical data nyata
    base_est = max(avg_price * 0.8, 1.0)
    payload: dict[str, Any] = {
        "stock_code": stock_code,
        "country": country,
        "avg_price": avg_price,
        "is_hari_besar": 1 if is_public_holiday else 0,
        "is_pre_hari_besar": 1 if is_public_holiday else 0,
        "is_peak_day": 1 if is_peak_day else 0,
        "day_of_week": dow,
        "week_of_year": forecast_date.isocalendar()[1],
        "month": forecast_date.month,
        "quarter": (forecast_date.month - 1) // 3 + 1,
        "day_of_month": forecast_date.day,
        "is_weekend": 1 if dow >= 5 else 0,
        "is_month_start": 0,
        "is_month_end": 0,
        "days_to_month_end": 15,
        "week_of_month": 1,
        "is_month_start_window": 0,
        "is_month_end_window": 0,
        "holiday_intensity": 3.0 if is_public_holiday else 1.0,
        "days_to_next_holiday": 0 if is_public_holiday else 30,
        "is_holiday_season": 1 if is_public_holiday else 0,
        "holiday_x_weekend": 1 if (is_public_holiday and dow >= 5) else 0,
        "demand_lag_1": base_est,
        "demand_lag_2": base_est * 0.9,
        "demand_lag_7": base_est * 0.85,
        "demand_lag_14": base_est * 0.8,
        "demand_lag_21": base_est * 0.75,
        "demand_lag_28": base_est * 0.7,
        "demand_lag_35": base_est * 0.65,
        "demand_lag_56": base_est * 0.5,
        "demand_lag_84": base_est * 0.3,
        "days_since_last_sale": 1,
        "roll_zero_count_14": 2.0,
        "roll_max_7": base_est * 1.2,
        "roll_max_28": base_est * 1.5,
        "roll_mean_7": base_est * 0.9,
        "roll_mean_14": base_est * 0.85,
        "roll_mean_28": base_est * 0.8,
        "roll_mean_56": base_est * 0.7,
        "roll_median_7": base_est * 0.85,
        "roll_median_14": base_est * 0.8,
        "roll_median_28": base_est * 0.75,
        "roll_std_7": base_est * 0.3,
        "roll_std_14": base_est * 0.35,
        "roll_std_28": base_est * 0.4,
        "roll_std_56": base_est * 0.45,
        "roll_max_56": base_est * 1.8,
        "roll_max_84": base_est * 2.0,
        "demand_acceleration_3d": 1.0,
        "spike_ratio_28": 1.5,
        "spike_ratio_56": 2.0,
        "pct_change_1": 0.0,
        "pct_change_7": 0.0,
        "discount_depth_pct": 0.15 if is_promo else 0.0,
        "price_momentum": 0.85 if is_promo else 1.0,
    }
    return payload


def _render_risk_status(coverage_pct: float) -> tuple[str, str, str]:
    if coverage_pct < 100.0:
        return "STOCKOUT", "Critical", "inverse"
    if coverage_pct <= 150.0:
        return "OPTIMAL", "Safe", "normal"
    return "OVERSTOCK", "Capital Tied", "off"


st.set_page_config(
    page_title="FMCG Demand Forecaster",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: #1C2132;
        border: 1px solid #2A3045;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #818CF8;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8B92A0;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #E6E8EC;
    }
    .prediction-card {
        background: #1C2132;
        border: 1px solid #2A3045;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-card .label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8B92A0;
        margin-bottom: 0.2rem;
    }
    .prediction-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E2E8F0;
    }
    .prediction-card .value.quantile {
        color: #818CF8;
    }
    .prediction-card .divider {
        border-top: 1px solid #2A3045;
        margin: 0.75rem 0;
    }
    .section-header {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8B92A0;
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
    }
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        opacity: 0.9;
    }
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 600; }
    .stSelectbox label, .stDateInput label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #8B92A0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #2A3045;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 0.75rem;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ──

with st.sidebar:
    st.markdown("##### Model Controls")

    avg_price = st.number_input(
        "Price (GBP)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        format="%.2f",
    )
    st.divider()

    st.markdown("##### Calendar Events")
    is_public_holiday = st.checkbox("Public Holiday", value=False)
    is_peak_day = st.checkbox("Peak Demand Day", value=False)
    is_promo = st.checkbox("Promo Season", value=False)

    st.divider()

    st.caption("API Status")
    try:
        h = requests.get(f"{API_URL}/health", timeout=3)
        st.markdown(
            '<span style="color:#34D399;font-size:0.8rem;">&#9679; Connected</span>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            '<span style="color:#F87171;font-size:0.8rem;">&#9679; Disconnected</span>',
            unsafe_allow_html=True,
        )

# ── Main page ──

st.title("FMCG Actuarial Demand Forecaster")
st.markdown(
    '<p style="color:#8B92A0;font-size:0.85rem;margin-top:-0.6rem;margin-bottom:1.5rem;">'
    "Mean Forecast & Quantile Reserve — MLflow Model Registry / latest"
    "</p>",
    unsafe_allow_html=True,
)

col_input, col_result = st.columns([1, 1])

with col_input:
    st.markdown(
        '<p class="section-header">Forecast Inputs</p>',
        unsafe_allow_html=True,
    )

    stock_code = st.selectbox("Stock Code", options=STOCK_CODES, index=0)
    country = st.selectbox("Country", options=COUNTRIES, index=0)

    forecast_date = st.date_input(
        "Forecast Date",
        value=datetime.date.today(),
        min_value=datetime.date.today() - datetime.timedelta(days=365),
        max_value=datetime.date.today() + datetime.timedelta(days=90),
    )

    with st.expander("Technical View — Full Payload", expanded=False):
        st.markdown(
            '<p style="color:#8B92A0;font-size:0.7rem;">'
            "All 52 model features sent to the inference endpoint. "
            "Temporal fields are automatically derived from the Forecast Date."
            "</p>",
            unsafe_allow_html=True,
        )
        preview_payload = _build_preview_payload(
            stock_code=stock_code,
            country=country,
            avg_price=avg_price,
            forecast_date=forecast_date,
            is_public_holiday=is_public_holiday,
            is_peak_day=is_peak_day,
            is_promo=is_promo,
        )
        st.code(
            json.dumps(preview_payload, indent=2, default=str),
            language="json",
        )

    predict_clicked = st.button(
        "Predict Demand",
        type="primary",
        use_container_width=True,
    )

with col_result:
    st.markdown(
        '<p class="section-header">Prediction Result</p>',
        unsafe_allow_html=True,
    )

    if predict_clicked:
        payload = _build_preview_payload(
            stock_code=stock_code,
            country=country,
            avg_price=avg_price,
            forecast_date=forecast_date,
            is_public_holiday=is_public_holiday,
            is_peak_day=is_peak_day,
            is_promo=is_promo,
        )

        with st.spinner("Running actuarial simulation..."):
            try:
                res = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                res.raise_for_status()
                result = res.json()
                st.session_state.last_result = result
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to inference engine at `{API_URL}`.")
                st.session_state.last_result = None
            except requests.exceptions.Timeout:
                st.error("Request timed out. Model may be under heavy load.")
                st.session_state.last_result = None
            except requests.exceptions.HTTPError as e:
                detail = "Unknown error"
                try:
                    detail = e.response.json().get("detail", str(e))
                except Exception:
                    detail = str(e)
                st.error(f"Inference error: {detail}")
                st.session_state.last_result = None
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.session_state.last_result = None

    result = st.session_state.get("last_result")
    if result is not None:
        ed = float(result["expected_demand"])
        rs = float(result["recommended_stock"])
        buffer = rs - ed
        coverage_pct = (rs / (ed + 1e-8)) * 100

        risk_label, risk_subtext, delta_color = _render_risk_status(coverage_pct)

        st.markdown(
            f'<div class="prediction-card">'
            f'<div class="label">Mean Forecast (Expected Demand)</div>'
            f'<div class="value">{ed:.2f} units</div>'
            f'<div class="divider"></div>'
            f'<div class="label">Quantile Reserve (Recommended Stock)</div>'
            f'<div class="value quantile">{rs:.2f} units</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label="Safety Buffer",
                value=f"{buffer:.2f} units",
                delta=f"{coverage_pct - 100:.1f}% vs mean",
                delta_color="normal" if buffer >= 0 else "inverse",
            )
        with m2:
            st.metric(
                label="Risk Status",
                value=risk_label,
                delta=risk_subtext,
                delta_color=delta_color,
            )
        with m3:
            st.metric(
                label="Coverage Ratio",
                value=f"{coverage_pct:.1f}%",
                delta="buffer / expected",
                delta_color="off",
            )
    else:
        st.markdown(
            '<div class="prediction-card">'
            '<div style="color:#8B92A0;font-size:0.85rem;text-align:center;padding:1.5rem 0;">'
            "Set forecast inputs and click Predict Demand."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
