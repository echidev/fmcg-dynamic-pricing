# FMCG Actuarial Demand Forecaster

**Enterprise-grade demand forecasting for Fast-Moving Consumer Goods — combining XGBoost regression with actuarial risk optimisation.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/mlflow-2.10+-blueviolet.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/dvc-3.0+-green.svg)](https://dvc.org/)

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.12+ | Core development |
| **ML Framework** | XGBoost 2.0 | Gradient-boosted regression (mean + quantile) — default & final model |
| **Feature Engineering** | pandas, numpy | 52 features: calendar, holiday, lags, rolling stats |
| **Hyperparameter Tuning** | Optuna | TPE-sampled search with 3-fold time-series CV |
| **Experiment Tracking** | MLflow | Run logging, metric comparison, model registry |
| **Data Versioning** | DVC + S3 | Medallion data lineage (bronze → silver → gold) |
| **Cloud Storage** | AWS S3 (ap-southeast-2) | Artifact store for DVC remote & MLflow models |
| **Serving API** | FastAPI | Single-record inference REST endpoint |
| **Dashboard** | Streamlit | Interactive UI for inventory planners |
| **Orchestration** | Custom CLI | `run_pipeline.py` (prep → features → tune → train) |
| **Testing** | pytest, coverage | Unit, behaviour, leakage, integration tests |
| **Linting** | Ruff | Code quality enforcement |

---

## Directory Structure

```
fmcg-dynamic-pricing/
│
├── api/                          # FastAPI inference server
│   ├── main.py                   #   REST endpoints (health, predict)
│   └── schemas.py                #   Pydantic request/response models
│
├── app/                          # Streamlit dashboard
│   └── ui.py                     #   Interactive inventory planner UI
│
├── configs/
│   └── pipeline.yaml             #   Central pipeline configuration
│
├── docs/                         # Technical documentation
│   ├── DS_PRD.md                 #   Product Requirements Document
│   ├── model_architecture.md     #   Model architecture & actuarial formulas
│   ├── system_architecture.md    #   MLOps infrastructure & data flow
│   └── dataset_card_and_audit.md #   Data provenance, schema, quality audit
│
├── notebooks/                    # Jupyter notebooks (chronological experiments)
│   ├── 01_eda_online_retail.ipynb                 # Exploratory data analysis
│   ├── 02_two_stage_baseline.ipynb                # Two-stage (classification + regression) baseline
│   ├── 03_xgboost_improvement.ipynb               # XGBoost vs baseline comparison
│   ├── 04_baseline_tabular_models.ipynb            # Benchmark: RF, GB, Linear models
│   ├── 05_seasonality_feature_analysis.ipynb       # Calendar/holiday feature engineering
│   ├── 06_twin_xgb_asymmetric_loss_cv.ipynb        # Twin XGBoost with asymmetric loss + CV
│   ├── 07_twin_xgb_boosted_holiday_v1.ipynb        # Holiday-boosted Twin XGBoost v1
│   ├── 08_twin_xgb_boosted_v2_tail_alpha.ipynb     # Twin XGBoost v2 with tail-alpha tuning
│   └── 09_decoupled_quantile_actuarial.ipynb       # Final DecoupledActuarialXGB architecture
│
├── src/                          # Python modules (core pipeline)
│   ├── config.py                 #   Single source of truth (constants, params, features)
│   ├── data_prep.py              #   Chunked cleaning → daily panel (bronze → silver)
│   ├── features.py               #   52 features, leakage-safe (silver → gold)
│   ├── metrics.py                #   Evaluation: CLS, OFR, SMAPE, WAPE, FVA, etc.
│   ├── model.py                  #   DecoupledActuarialXGB (3-pronged decoupled)
│   ├── model_wrapper.py          #   MLflow PyFunc wrapper
│   ├── tune.py                   #   Optuna + MLflow + 3-fold CV + auto-registration
│   ├── train.py                  #   Production training + model registration
│   └── infer.py                  #   Batch inference CLI
│
├── tests/                        # Test suite (18 test classes)
│   ├── conftest.py               #   Shared fixtures
│   ├── test_model.py             #   Model unit tests
│   ├── test_model_behavior.py    #   Behavioural tests (price elasticity)
│   ├── test_features.py          #   Holiday & discount logic
│   ├── test_leakage.py           #   Zero data leakage enforcement
│   ├── test_data_quality.py      #   Data integrity checks
│   ├── test_metrics.py           #   Metric correctness
│   ├── test_config.py            #   Config validation
│   └── test_pipeline.py          #   End-to-end integration
│
├── .dvc/                         # DVC remote configuration
├── .env                          # Environment variables (AWS + MLflow)
├── .gitignore                    # Git exclusion rules
├── .streamlit/config.toml        # Streamlit app configuration
├── run_pipeline.py               # CLI orchestrator
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## Model — DecoupledActuarialXGB

A **3-Pronged Decoupled Architecture** that separates statistical estimation from business-risk optimisation:

| Step | Component | Objective | Output |
|---|---|---|---|
| **1** | Honest Baseline | `reg:squarederror` (MSE) | Unbiased conditional mean `E[X]` |
| **2** | Risk-Aware Quantile | `reg:quantileerror` (pinball, q=0.9799) | Upper-tail quantile `Q_q[X]` |
| **3** | Actuarial Layer | Dynamic critical fractile per SKU | Recommended stock `= max(Q, E + (Q-E)×CF)` |

**Key Parameters:**

| Parameter | Value | Search Range |
|---|---|---|
| `q_target` | 0.9799 | [0.85, 0.98] |
| `shortage_margin_multiplier` | 2.2847 | [1.0, 3.0] |
| `n_estimators` | 400 | [200, 600] |
| `max_depth` | 5 | [3, 7] |
| `learning_rate` | 0.0339 | [0.01, 0.2] |

**Feature Set:** 52 features across 4 groups — Calendar/Holiday (14), Holiday Intensity (4), Temporal Peak (1), Lag/Rolling (33). All features leakage-proofed with `+1 shift`.

**ONNX Export:** The mean regressor can be exported to ONNX format for low-latency inference (`src/train.py --skip-onnx` skips this step).

---

## Setup & Installation

### Prerequisites

- Python >= 3.12
- DVC >= 3.0
- AWS credentials with S3 access to `dvc-demand-forecast-*` bucket

### Local Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd fmcg-dynamic-pricing

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env          # Create if not exists
```

### Pull Data & Models (DVC)

```bash
# Sync data from S3 remote
dvc pull

# Verify data integrity
ls -la data/bronze/
ls -la data/silver/
ls -la data/gold/
```

---

## Running the Pipeline

### Full End-to-End

```bash
python run_pipeline.py
```

Executes in sequence:
1. `src.data_prep` — Bronze CSV → Silver daily panel
2. `src.features` — Silver → Gold with 52 features
3. `src.tune` — Optuna hyperparameter search (100 trials)
4. `src.train` — Final model training + MLflow registration

### Partial Execution

```bash
# Tune only (skip final training)
python run_pipeline.py --skip-train

# Train only (skip tuning, use best existing params)
python run_pipeline.py --skip-tune

# Custom trial count
python run_pipeline.py --trials 200

# Skip ONNX export during training
python run_pipeline.py --skip-onnx

# Clean MLflow runs and cache
python run_pipeline.py --cleanup

# Dry run (preview without executing)
python run_pipeline.py --dry-run
```

### Running Tests

```bash
# Full test suite
python -m pytest tests/ -v

# Specific test modules
python -m pytest tests/test_leakage.py -v    # Data leakage tests
python -m pytest tests/test_model.py -v      # Model unit tests

# With coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Running Inference Services

### 1. FastAPI Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

| Endpoint | Method | Description |
|---|---|---|
| `http://localhost:8000/health` | GET | Service health check |
| `http://localhost:8000/predict` | POST | Single-record inference |
| `http://localhost:8000/docs` | GET | Interactive Swagger UI |

### 2. Streamlit Dashboard

```bash
streamlit run app/ui.py
```

Access at `http://localhost:8501`.

### 3. Batch Inference (CLI)

```bash
python -m src.infer --input data/gold/online_retail_daily_product_tabular.parquet --output predictions.csv
```

---

## Business Metrics

| Metric | Target | Description |
|---|---|---|---|
| **OFR** (Order Fulfilment Rate) | >= 80% | `min(y_true, y_pred).sum() / y_true.sum()` |
| **CLS** (Cost of Lost Sales) | < 50,000 | `sum(max(0, y_true - y_pred) × price × margin)` |
| **Peak OFR** | >= 75% | OFR on peak-demand days only |
| **MAE** | Minimise | Mean Absolute Error |
| **RMSE** | Minimise | Root Mean Squared Error |
| **WAPE** | Minimise | Weighted Absolute Percentage Error |
| **SMAPE** | Minimise | Symmetric MAPE |
| **F1-Zero** | Maximise | F1 score for zero-demand classification |
| **IHC** (Inventory Holding Cost) | Report | Estimated cost of excess stock |
| **Total Cost** | Minimise | CLS + IHC |
| **OOS Rate** | Minimise | Proportion of true demand left unfulfilled |
| **FVA** (Forecast Value Added) | > 0 | Improvement over naive seasonal baseline |

---

## Documentation

| Document | Description |
|---|---|---|
| `docs/DS_PRD.md` | Product Requirements Document — business objectives, success metrics, scope |
| `docs/model_architecture.md` | Model internals — actuarial formulas, parameters, feature inventory |
| `docs/system_architecture.md` | MLOps infrastructure — data flow, Direct-to-S3, serving topology |
| `docs/dataset_card_and_audit.md` | Dataset Card & Data Quality Audit — provenance, schema, quality checks |

---

## License

This project is licensed under the MIT License — see `LICENSE` for details.

---

## Maintainers

**Data Science & MLOps Lead** — [@wildanmaulana1](https://github.com/wildanmaulana1)
