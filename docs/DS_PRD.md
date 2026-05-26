# Data Science Product Requirements Document

**Project:** FMCG Actuarial Demand Forecaster  
**Version:** 3.0.0  
**Status:** Active Development  
**Model Architecture:** DecoupledActuarialXGB (3-Pronged Decoupled)

---

## Executive Summary

The FMCG Actuarial Demand Forecaster is an enterprise-grade demand forecasting system purpose-built for Fast-Moving Consumer Goods (FMCG) supply chains. It combines **XGBoost regression** with **actuarial risk optimization** to produce:

- **Expected Demand** — an unbiased mean forecast via MSE-optimized regression.
- **Recommended Stock** — a risk-adjusted inventory target derived from quantile regression and dynamic critical fractile calculus.
- **Risk Status** — a contextual flag (HIGH / LOW) indicating whether the recommended stock exceeds normal variability thresholds.

The system is architected around a **Decoupled 3-Pronged Design** that separates statistical estimation (Step 1 & 2) from business-risk optimisation (Step 3), enabling independent tuning of forecast accuracy versus inventory policy.

---

## Problem Statement

### The FMCG Overstock / Stockout Dilemma

FMCG demand data exhibits structural characteristics that render conventional forecasting approaches inadequate:

| Characteristic | Implication |
|---|---|
| **Zero inflation** | 60–80% of item-day combinations record zero demand. Standard MSE regressors learn toward zero, producing chronic under-forecasting. |
| **Peak-day volatility** | Holiday and promotional periods see 5–20× baseline demand. Models trained on mean behaviour systematically miss these spikes. |
| **Asymmetric cost structure** | Cost of a stockout (lost revenue + customer churn) far exceeds cost of overstock (holding cost + spoilage). Symmetric loss functions (MSE, MAE) fail to reflect this asymmetry. |
| **Short product lifecycles** | Many SKUs trade for <12 months. Insufficient historical data for naïve time-series methods. |

The result is a persistent **under-forecasting bias** that inflates the Cost of Lost Sales (CLS) and depresses the Order Fulfilment Rate (OFR).

---

## Business Objectives

1. **Achieve Order Fulfilment Rate (OFR) >= 80%** across the portfolio, with a stretch target of 89%.
2. **Minimise Cost of Lost Sales (CLS) to < 50,000** per evaluation window.
3. **Maintain peak-day OFR >= 75%** without compromising regular-day inventory efficiency.
4. **Provide per-SKU risk classification** (HIGH / LOW) to enable differentiated replenishment workflows.

---

## Success Metrics

All metrics are computed on a hold-out test window of the most recent 30 days per product.

| Metric | Target | Definition |
|---|---|---|
| **OFR** (Order Fulfilment Rate) | >= 0.80 | `min(y_true, y_pred).sum() / y_true.sum()` |
| **CLS** (Cost of Lost Sales) | < 50,000 | `sum(max(0, y_true - y_pred) * price * margin)` |
| **Peak OFR** | >= 0.75 | OFR computed exclusively on peak-demand days |
| **MAE** | Minimise | Mean Absolute Error |
| **RMSE** | Minimise | Root Mean Squared Error |
| **WAPE** | Minimise | Weighted Absolute Percentage Error |
| **SMAPE** | Minimise | Symmetric MAPE |
| **F1-Zero** | Maximise | F1 score for zero-demand classification |
| **IHC** (Inventory Holding Cost) | Report | Estimated cost of excess stock |
| **Total Cost** | Minimise | CLS + IHC |
| **OOS Rate** (Out-of-Stock) | Minimise | Proportion of true demand left unfulfilled |
| **FVA** (Forecast Value Added) | Positive | Improvement over naive seasonal benchmark |

---

## Scope

### In-Scope

- Demand forecasting for **5,000+ SKUs** across **38 countries** from the Online Retail dataset (2010–2011).
- Feature engineering: **52 features** spanning calendar, holiday intensity, temporal peaks, and lag/rolling statistics.
- Model training via **3-fold expanding-window time-series cross-validation** with 30-day horizons.
- Single-record inference via **FastAPI** REST endpoint.
- Interactive dashboard via **Streamlit** with configurable price, calendar events, and SKU/country selectors.
- Model lifecycle management via **MLflow Model Registry** with automated promotion gates (OFR >= 0.80, CLS < 50,000).
- Data and model versioning via **DVC with S3 remote**.
- All features computed with **zero data leakage** (every lag/statistic is shifted +1, holiday intensity is computed per fold).

### Out-of-Scope

- Real-time streaming inference (batch and single-record only).
- Multi-echelon supply chain simulation (single-warehouse, single-echelon).
- Dynamic pricing optimisation (price is treated as an input feature, not an output).
- Deep learning / transformer models (XGBoost only per current architecture).
- Mobile client applications.

---

## Constraints

| Constraint | Detail |
|---|---|
| **Memory** | Training environment limited to ~16 GB RAM (EC2 t3.large). All data pipelines use chunked IO (`CHUNK_SIZE = 200_000` rows). |
| **Compute** | GPU not available in production training. XGBoost uses `tree_method=hist` for CPU efficiency. |
| **Data window** | 730 days training window, 30-day validation, 30-day test. Minimum 60 observations per SKU. |
| **Latency** | Single-record inference must complete in < 500 ms (FastAPI on EC2). |
| **Leakage** | Zero tolerance for data leakage — enforced via dedicated leakage test suite (`tests/test_leakage.py`). |
| **Artifact size** | Model artifacts stored directly on S3 to bypass EC2 disk/RAM constraints (Direct-to-S3 architecture). |

---

## Stakeholder KPIs

| Stakeholder | Primary KPI | Acceptable Threshold |
|---|---|---|
| **Supply Chain Manager** | OFR | >= 80% |
| **Finance / CFO** | CLS | < 50,000 per window |
| **Warehouse Ops** | IHC + Total Cost | Reported, no hard threshold |
| **Data Science Team** | FVA | Positive vs. naive seasonal baseline |
| **Inventory Planner** | Risk Status (HIGH/LOW) | < 30% HIGH-risk items |

---

## Assumptions

1. Historical sales data is representative of future demand patterns (stationarity within product lifecycle).
2. Price elasticity effects are captured by the existing feature set (avg_price, discount_depth_pct, price_momentum).
3. Holiday calendar is deterministic and available for all 38 countries via the `holidays` Python library.
4. All products are independent (no substitution or cannibalisation effects are modelled).
5. The cost margin multiplier (`shortage_margin_multiplier = 2.2847`) is a reasonable proxy for the true shortage-to-overstock cost ratio across the portfolio.

---

## Milestones

| Phase | Deliverable | Status |
|---|---|---|
| **P0 — EDA & Baseline** | Data audit, cleaning pipeline, two-stage baseline | ✅ Complete |
| **P1 — Feature Engineering** | 52 features with leakage-proof computation | ✅ Complete |
| **P2 — Hyperparameter Tuning** | Optuna CV + MLflow tracking + auto-registration | ✅ Complete |
| **P3 — Actuarial Layer** | DecoupledActuarialXGB with critical fractile | ✅ Complete |
| **P4 — Inference Serving** | FastAPI + Streamlit UI | ✅ Complete |
| **P5 — Production Hardening** | API hardening, model registry, Direct-to-S3 architecture | ✅ Complete |
