"""Streamlit dashboard — Decoupled Actuarial Inventory Simulator.

Usage:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.config import FEATURE_COLS

API_URL = "http://localhost:8000/predict-inventory"

st.set_page_config(
    page_title="Decoupled Actuarial Inventory",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Notion-inspired dark mode ──

st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    div[data-testid="stMetric"] {
        background: #161A22;
        border: 1px solid #262B36;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #8B92A0;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #E6E8EC;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
    .prediction-card {
        background: #161A22;
        border: 1px solid #262B36;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .prediction-card .label {
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #8B92A0;
        margin-bottom: 0.15rem;
    }
    .prediction-card .value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #E6E8EC;
    }
    .prediction-card .value.quantile {
        color: #60A5FA;
    }
    .prediction-card .divider {
        border-top: 1px solid #262B36;
        margin: 0.75rem 0;
    }
    .risk-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        letter-spacing: 0.02em;
    }
    .risk-badge.LOW { color: #34D399; background: #064E3B; }
    .risk-badge.MEDIUM { color: #FBBF24; background: #78350F; }
    .risk-badge.HIGH { color: #F87171; background: #7F1D1D; }
    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8B92A0;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 0.8rem;
        font-weight: 500;
    }
    .stButton button {
        font-weight: 500;
        border-radius: 6px;
    }
    h1, h2, h3 { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Helpers ──


GOLD_PATH = Path("data/gold/online_retail_daily_product_tabular.parquet")


def load_demo_data() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        st.warning(f"Gold data not found at {GOLD_PATH}. Using generic demo data.")
        return _fallback_demo_data()

    df = pd.read_parquet(GOLD_PATH)
    top_skus = ["84077", "85099B", "85123A", "21212", "22326"]
    rows = []
    for sk in top_skus:
        subset = df[df["stock_code"] == sk].dropna(subset=["demand_lag_1"])
        if subset.empty:
            continue
        row = subset.tail(1).iloc[0]
        record: Dict[str, Any] = {
            "name": sk,
            "stock_code": sk,
            "country": row.get("country", "United Kingdom"),
            "avg_price": float(row["avg_price"]),
            "base_demand": float(row.get("demand_qty", 0)),
            "perishability_score": 0.15,
            "margin_multiplier": 1.5,
        }
        for col in FEATURE_COLS:
            val = row.get(col)
            record[col] = float(val) if pd.notna(val) else 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def _fallback_demo_data() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    fallback_skus = [
        ("84077", "United Kingdom", 0.3, 96.0),
        ("85099B", "United Kingdom", 2.1, 483.0),
        ("85123A", "United Kingdom", 2.9, 120.0),
        ("21212", "EIRE", 0.7, 24.0),
        ("22326", "Germany", 2.5, 48.0),
    ]
    for sk, country, price, demand in fallback_skus:
        record: Dict[str, Any] = {
            "name": sk,
            "stock_code": sk,
            "country": country,
            "avg_price": price,
            "base_demand": demand,
            "perishability_score": 0.15,
            "margin_multiplier": 1.5,
        }
        for col in FEATURE_COLS:
            record[col] = 0.0
        rows.append(record)
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_demo_cached() -> pd.DataFrame:
    return load_demo_data()


def build_payload(
    demo_row: pd.Series,
    new_price: float,
    base_demand: float,
    margin_multiplier: float,
    perishability_score: float,
    is_hari_besar: bool,
    is_peak_day: bool,
    is_promo: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    for col in FEATURE_COLS:
        val = demo_row.get(col)
        payload[col] = float(val) if pd.notna(val) else 0.0

    payload["stock_code"] = str(demo_row.get("stock_code", ""))
    payload["country"] = str(demo_row.get("country", ""))

    old_price = float(demo_row["avg_price"])
    payload["avg_price"] = new_price
    payload["price_momentum"] = round(new_price / (old_price + 1e-8), 4)
    payload["discount_depth_pct"] = round(max(0.0, (old_price - new_price) / (old_price + 1e-8)), 4)

    payload["is_hari_besar"] = 0
    payload["is_pre_hari_besar"] = 0
    payload["is_peak_day"] = 0
    payload["holiday_intensity"] = 1.0
    payload["days_to_next_holiday"] = 30
    payload["is_holiday_season"] = 0
    payload["holiday_x_weekend"] = 0

    if is_hari_besar:
        payload["is_hari_besar"] = 1
        payload["is_pre_hari_besar"] = 1
        payload["holiday_intensity"] = 3.0
        payload["days_to_next_holiday"] = 0
        payload["is_holiday_season"] = 1

    if is_peak_day:
        payload["is_peak_day"] = 1

    if is_promo:
        payload["discount_depth_pct"] = max(float(payload["discount_depth_pct"]), 0.15)
        payload["price_momentum"] = min(float(payload["price_momentum"]), 0.85)

    payload["margin_multiplier"] = margin_multiplier
    payload["perishability_score"] = perishability_score

    return payload


def call_api(payload: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    res = requests.post(API_URL, json=payload, timeout=timeout)
    res.raise_for_status()
    return res.json()


# ── Session state ──

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None
if "new_price" not in st.session_state:
    st.session_state.new_price = None

# ── Sidebar ──

with st.sidebar:
    st.markdown("#### Actuarial Engine Controls")

    margin_multiplier = st.slider(
        "Margin Multiplier",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.05,
        help="Cost of Shortage multiplier. Higher = more aggressive stocking.",
    )
    perishability_score = st.slider(
        "Perishability Score",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Higher = faster spoilage. Suppresses overstock.",
    )
    st.divider()
    base_demand = st.slider(
        "Base Demand Level",
        min_value=1.0,
        max_value=150.0,
        value=50.0,
        step=1.0,
        help="Demand scale used to generate default lag features.",
    )

    st.divider()
    st.caption("API Status")
    try:
        h = requests.get(API_URL.replace("/predict-inventory", "/health"), timeout=3)
        st.markdown(
            '<span style="color:#34D399;font-size:0.8rem;">&#9679; Connected</span>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            '<span style="color:#F87171;font-size:0.8rem;">&#9679; Disconnected</span>',
            unsafe_allow_html=True,
        )

# ── Header ──

st.title("Decoupled Actuarial Inventory")
st.markdown(
    '<p style="color:#8B92A0;font-size:0.9rem;margin-top:-0.5rem;">'
    "FMCG Demand Forecasting &mdash; Mean Baseline + Quantile Boundary + Dynamic Critical Fractile"
    "</p>",
    unsafe_allow_html=True,
)

# ── Demo data ──

demo_df = load_demo_cached()

if st.session_state.new_price is None:
    st.session_state.new_price = float(demo_df.iloc[st.session_state.selected_idx]["avg_price"])

with st.expander("Demo Product Catalog", expanded=False):
    st.markdown(
        '<p style="color:#8B92A0;font-size:0.75rem;margin-bottom:0.5rem;">'
        "Select a product row to auto-populate the simulation form below."
        "</p>",
        unsafe_allow_html=True,
    )
    display_df = demo_df[["name", "stock_code", "country", "avg_price", "base_demand"]].copy()
    display_df.columns = ["Product", "SKU", "Market", "Current Price", "Base Demand"]
    display_df["Current Price"] = display_df["Current Price"].apply(lambda x: f"GBP {x:.2f}")
    display_df["Base Demand"] = display_df["Base Demand"].apply(lambda x: f"{x:.0f} units")

    col_s, col_b = st.columns([3, 1])
    with col_s:
        selected_name = st.selectbox(
            "Select a product",
            options=demo_df["name"].tolist(),
            index=st.session_state.selected_idx,
            label_visibility="collapsed",
        )
    with col_b:
        if st.button("Apply", type="primary", use_container_width=True):
            idx = int(demo_df[demo_df["name"] == selected_name].index[0])
            st.session_state.selected_idx = idx
            demo_row = demo_df.iloc[idx]
            st.session_state.new_price = float(demo_row["avg_price"])
            st.rerun()

    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Top Row: Metrics ──

demo_row = demo_df.iloc[st.session_state.selected_idx]
base_price = float(demo_row["avg_price"])

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Current Price",
    f"GBP {base_price:.2f}",
    delta=None,
)
m2.metric(
    "Base Demand",
    f"{base_demand:.0f} units",
    delta=None,
)
m3.metric(
    "Perishability",
    f"{perishability_score:.0%}",
    delta=None,
)
m4.metric(
    "Margin Target",
    f"{margin_multiplier:.2f}x",
    delta=None,
)

st.divider()

# ── Middle Section: Simulation ──

st.markdown('<p class="section-header">Simulation</p>', unsafe_allow_html=True)

col_input, col_result = st.columns([1, 1])

with col_input:
    st.markdown(
        '<p style="color:#8B92A0;font-size:0.75rem;font-weight:500;text-transform:uppercase;'
        'letter-spacing:0.04em;margin-bottom:0.5rem;">Business Inputs</p>',
        unsafe_allow_html=True,
    )

    current_name = demo_df.iloc[st.session_state.selected_idx]["name"]
    st.markdown(
        f'<p style="font-size:0.85rem;color:#E6E8EC;">Product: '
        f'<strong>{current_name}</strong></p>',
        unsafe_allow_html=True,
    )

    new_price = st.number_input(
        "New Price (GBP)",
        min_value=0.01,
        max_value=50.0,
        value=st.session_state.new_price,
        step=0.25,
        format="%.2f",
        help="Proposed selling price for simulation.",
    )
    st.session_state.new_price = new_price

    is_promo = st.checkbox(
        "Promo Season",
        value=False,
        help="Activates discount depth and price momentum adjustments.",
    )
    is_hari_besar = st.checkbox(
        "Hari Besar Nasional",
        value=False,
        help="National holiday. Increases holiday intensity score.",
    )
    is_peak_day = st.checkbox(
        "Peak Demand Day",
        value=False,
        help="Historically high-demand date.",
    )

    run_clicked = st.button(
        "Run Simulation",
        type="primary",
        use_container_width=False,
    )

    # Technical View expander
    with st.expander("Technical View - Model Payload", expanded=False):
        st.markdown(
            '<p style="color:#8B92A0;font-size:0.75rem;">'
            "How business inputs map to model features before API call."
            "</p>",
            unsafe_allow_html=True,
        )
        preview_payload = build_payload(
            demo_row=demo_row,
            new_price=new_price,
            base_demand=base_demand,
            margin_multiplier=margin_multiplier,
            perishability_score=perishability_score,
            is_hari_besar=is_hari_besar,
            is_peak_day=is_peak_day,
            is_promo=is_promo,
        )
        preview_json = {
            k: v for k, v in preview_payload.items()
            if k not in ("stock_code", "country")
        }
        st.code(
            json.dumps(preview_json, indent=2, default=str),
            language="json",
            line_numbers=False,
        )

with col_result:
    st.markdown(
        '<p style="color:#8B92A0;font-size:0.75rem;font-weight:500;text-transform:uppercase;'
        'letter-spacing:0.04em;margin-bottom:0.5rem;">Prediction Result</p>',
        unsafe_allow_html=True,
    )

    if run_clicked:
        payload = build_payload(
            demo_row=demo_row,
            new_price=new_price,
            base_demand=base_demand,
            margin_multiplier=margin_multiplier,
            perishability_score=perishability_score,
            is_hari_besar=is_hari_besar,
            is_peak_day=is_peak_day,
            is_promo=is_promo,
        )
        st.session_state.last_payload = payload

        with st.spinner("Running actuarial simulation..."):
            try:
                result = call_api(payload)
                st.session_state.last_result = result
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the inference engine. "
                    "Ensure the API server is running at "
                    f"`{API_URL}`."
                )
                st.session_state.last_result = None
            except requests.exceptions.Timeout:
                st.error(
                    "The inference request timed out. "
                    "The model may be under heavy load. Please try again."
                )
                st.session_state.last_result = None
            except requests.exceptions.HTTPError as e:
                detail = "Unknown error"
                try:
                    detail = e.response.json().get("detail", str(e))
                except Exception:
                    detail = str(e)
                st.error(f"Inference engine returned an error: {detail}")
                st.session_state.last_result = None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.last_result = None

    # Display result
    result = st.session_state.get("last_result")
    if result is not None:
        ed = float(result["expected_demand"])
        rs = float(result["recommended_stock"])
        risk = str(result["risk_status"])
        buffer = rs - ed
        pct_over = (buffer / (ed + 1e-8)) * 100

        st.markdown(
            f'<div class="prediction-card">'
            f'<div class="label">Expected Demand (Mean)</div>'
            f'<div class="value">{ed:.2f} units</div>'
            f'<div style="margin-top:0.75rem;"></div>'
            f'<div class="label">Risk Boundary (Quantile)</div>'
            f'<div class="value quantile">{rs:.2f} units</div>'
            f'<div class="divider"></div>'
            f'<div class="label">Recommended Stock</div>'
            f'<div class="value" style="font-size:2rem;">{rs:.2f} units</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        status_col, buffer_col, risk_col = st.columns(3)
        with status_col:
            st.markdown(
                f'<div class="prediction-card">'
                f'<div class="label">Safety Buffer</div>'
                f'<div style="font-size:1.3rem;font-weight:600;">{buffer:.2f}</div>'
                f'<div style="font-size:0.7rem;color:#8B92A0;">({pct_over:.1f}% above mean)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with buffer_col:
            st.markdown(
                f'<div class="prediction-card">'
                f'<div class="label">Risk Status</div>'
                f'<div style="margin-top:0.3rem;">'
                f'<span class="risk-badge {risk}">{risk}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with risk_col:
            st.markdown(
                f'<div class="prediction-card">'
                f'<div class="label">Coverage Ratio</div>'
                f'<div style="font-size:1.3rem;font-weight:600;">{pct_over:.1f}%</div>'
                f'<div style="font-size:0.7rem;color:#8B92A0;">buffer vs. expected</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="prediction-card">'
            '<div style="color:#8B92A0;font-size:0.9rem;text-align:center;padding:1rem 0;">'
            "Configure business inputs on the left and click Run Simulation."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

# ── Bottom Section: 7-Day Forecast ──

st.divider()
st.markdown('<p class="section-header">7-Day Forecast Simulation</p>', unsafe_allow_html=True)

if st.button("Generate 7-Day Forecast", use_container_width=False):
    base_result = st.session_state.get("last_result")
    if base_result is None:
        st.warning("Run a single-day simulation first to establish a baseline.")
    else:
        with st.spinner("Generating 7-day projection..."):
            base_expected = float(base_result["expected_demand"])
            days = [f"D-{i}" for i in range(7, 0, -1)]
            exp_list: List[float] = []
            rec_list: List[float] = []

            base_payload = st.session_state.get("last_payload", None)
            api_ok = True

            for i in range(7):
                if base_payload is not None:
                    sim = dict(base_payload)
                    decay = 1.0 - 0.04 * i
                    noise = np.random.uniform(0.88, 1.12)
                    sim["avg_price"] = base_payload.get("avg_price", 10_000) * (1.0 + 0.015 * i)
                    for lag_k in [1, 2, 7, 14]:
                        if lag_k in sim:
                            sim[lag_k] = base_expected * decay * noise * float(np.random.uniform(0.85, 1.15))
                    sim["margin_multiplier"] = margin_multiplier
                    sim["perishability_score"] = perishability_score

                    try:
                        out = call_api(sim, timeout=10)
                        exp_list.append(float(out["expected_demand"]))
                        rec_list.append(float(out["recommended_stock"]))
                        continue
                    except Exception:
                        api_ok = False

                exp_list.append(base_expected * np.random.uniform(0.75, 1.05))
                rec_list.append(base_expected * np.random.uniform(1.0, 1.4))

            forecast_df = pd.DataFrame(
                {"Day": days, "Expected Demand": exp_list, "Recommended Stock": rec_list}
            ).set_index("Day")

            st.session_state.forecast_data = forecast_df

            if not api_ok:
                st.caption(
                    "Some forecast days fell back to simulated estimates "
                    "(API unavailable for multi-day inference)."
                )

if st.session_state.forecast_data is not None:
    fdf = st.session_state.forecast_data

    st.line_chart(
        fdf,
        use_container_width=True,
        height=350,
        color=["#8B92A0", "#60A5FA"],
    )

    st.markdown(
        '<p style="color:#8B92A0;font-size:0.75rem;margin-top:0.25rem;">'
        "Solid lines: Expected Demand (mean) vs. Recommended Stock (quantile boundary)."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.expander("Forecast Detail", expanded=False):
        display_fdf = fdf.reset_index()
        display_fdf.columns = ["Day", "Expected Demand", "Recommended Stock"]
        display_fdf["Expected Demand"] = display_fdf["Expected Demand"].apply(
            lambda x: f"{x:.2f}"
        )
        display_fdf["Recommended Stock"] = display_fdf["Recommended Stock"].apply(
            lambda x: f"{x:.2f}"
        )
        st.dataframe(display_fdf, use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<p style="color:#8B92A0;font-size:0.85rem;">'
        "Run a single simulation above, then generate the 7-day projection."
        "</p>",
        unsafe_allow_html=True,
    )
