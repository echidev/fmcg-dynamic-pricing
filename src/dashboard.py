"""Streamlit dashboard — Decoupled Actuarial Inventory Simulator.

Usage:
    streamlit run src/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.config import FEATURE_COLS

API_URL = "http://localhost:8000/predict-inventory"

st.set_page_config(
    page_title="Decoupled Actuarial Inventory Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Decoupled Actuarial Inventory Dashboard")
st.caption("FMCG Demand Forecasting | Single-record inference simulation")

# ── Sidebar: Actuarial Controls ──

with st.sidebar:
    st.header("Actuarial Controls")
    margin_multiplier = st.slider(
        "Margin Multiplier",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.05,
        help="Menggandakan Cost of Shortage. Lebih tinggi = stok lebih agresif.",
    )
    perishability_score = st.slider(
        "Perishability Score",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Semakin tinggi = barang semakin mudah busuk. Menekan overstock.",
    )
    st.divider()
    st.caption("Simulation Parameters")
    base_demand = st.slider(
        "Base Demand Level",
        min_value=1.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        help="Skala permintaan dasar untuk mengisi default lag features.",
    )
    holiday_toggle = st.checkbox("Hari Besar", value=False)
    peak_toggle = st.checkbox("Peak Day", value=False)

# ── Build payload from defaults + sidebar ──

default_payload: Dict = {col: 0.0 for col in FEATURE_COLS}
default_payload.update(
    {
        "avg_price": 10.0,
        "day_of_week": 2,
        "week_of_year": 25,
        "month": 6,
        "quarter": 2,
        "day_of_month": 15,
        "is_weekend": 0,
        "is_month_start": 0,
        "is_month_end": 0,
        "days_to_month_end": 15,
        "week_of_month": 3,
        "holiday_intensity": 1.0,
        "days_to_next_holiday": 30,
        "margin_multiplier": margin_multiplier,
        "perishability_score": perishability_score,
    }
)

if holiday_toggle:
    default_payload["is_hari_besar"] = 1
    default_payload["is_pre_hari_besar"] = 1
    default_payload["holiday_intensity"] = 3.0
    default_payload["days_to_next_holiday"] = 0

if peak_toggle:
    default_payload["is_peak_day"] = 1

# Set lag features based on base demand
for lag in [1, 2, 7, 14, 21, 28]:
    default_payload[f"demand_lag_{lag}"] = base_demand * np.random.uniform(0.7, 1.3)

for w in [7, 14, 28]:
    default_payload[f"roll_mean_{w}"] = base_demand * np.random.uniform(0.85, 1.15)
    default_payload[f"roll_std_{w}"] = base_demand * np.random.uniform(0.1, 0.3)

default_payload["roll_max_28"] = base_demand * 1.8
default_payload["roll_max_7"] = base_demand * 1.5

# ── Main Area ──

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Input Features")
    df_display = pd.DataFrame([default_payload])
    st.dataframe(df_display, use_container_width=True)

with col_right:
    st.subheader("Prediction Result")

    if st.button("Run Prediction", type="primary", use_container_width=True):
        payload = dict(default_payload)
        payload["margin_multiplier"] = margin_multiplier
        payload["perishability_score"] = perishability_score

        try:
            res = requests.post(API_URL, json=payload, timeout=10)
            res.raise_for_status()
            out = res.json()

            st.metric(
                label="Recommended Stock",
                value=f"{out['recommended_stock']:.2f}",
                delta=f"{out['recommended_stock'] - out['expected_demand']:.2f} vs expected",
                delta_color="normal",
            )

            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.metric("Expected Demand", f"{out['expected_demand']:.2f}")
            meta_col2.metric("Risk Status", out["risk_status"])
            meta_col3.metric(
                "Safety Buffer",
                f"{out['recommended_stock'] - out['expected_demand']:.2f}",
            )

        except requests.exceptions.ConnectionError:
            st.error(
                f"Tidak dapat terhubung ke API di {API_URL}. "
                "Pastikan FastAPI sudah berjalan: uvicorn src.api:app --reload"
            )
        except Exception as exc:
            st.error(f"Prediction gagal: {exc}")

    else:
        st.info("Klik 'Run Prediction' untuk menjalankan inferensi.")

# ── 7-Day Forecast Simulation ──

st.divider()
st.subheader("7-Day Forecast Simulation")

if st.button("Generate 7-Day Forecast", use_container_width=True):
    days_labels = [f"D-{i}" for i in range(7, 0, -1)]
    simulated_expected = []
    simulated_recommended = []

    base = base_demand
    for i in range(7):
        sim_payload = dict(default_payload)
        decay = 1.0 - 0.05 * i
        noise = np.random.uniform(0.85, 1.15)
        sim_payload["avg_price"] = 10.0 * (1.0 + 0.02 * i)
        for lag in [1, 2, 7, 14]:
            sim_payload[f"demand_lag_{lag}"] = base * decay * noise * np.random.uniform(0.8, 1.2)
        sim_payload["margin_multiplier"] = margin_multiplier
        sim_payload["perishability_score"] = perishability_score

        try:
            sim_res = requests.post(API_URL, json=sim_payload, timeout=10)
            sim_res.raise_for_status()
            sim_out = sim_res.json()
            simulated_expected.append(sim_out["expected_demand"])
            simulated_recommended.append(sim_out["recommended_stock"])
        except Exception:
            simulated_expected.append(0.0)
            simulated_recommended.append(0.0)

    chart_df = pd.DataFrame(
        {
            "Day": days_labels,
            "Expected Demand": simulated_expected,
            "Recommended Stock": simulated_recommended,
        }
    ).set_index("Day")

    st.line_chart(chart_df, use_container_width=True, height=400)

    st.subheader("Forecast Detail")
    st.dataframe(chart_df.reset_index(), use_container_width=True)

else:
    st.info("Klik 'Generate 7-Day Forecast' untuk melihat proyeksi mingguan.")

st.divider()
st.caption("Decoupled Actuarial XGBoost | Model: Mean Baseline + Quantile q=0.98 + Dynamic Critical Fractile")
