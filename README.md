# FMCG Dynamic Pricing & Demand Forecasting

Enterprise-grade demand forecasting for Fast-Moving Consumer Goods (FMCG) using **Twin-XGB Boosted** — a two-stage XGBoost model with holiday-aware peak segmentation.

## Tech Stack

- **Models:** XGBoost, LightGBM (experimental)
- **Tuning:** Optuna + MLflow tracking
- **Data:** DVC (data/model versioning)
- **Pipeline:** Python `src/` modules
- **Tests:** pytest (17 tests)

## Quick Start

```bash
# Full pipeline (prep → features → tune → train)
python run_pipeline.py

# Skip tuning, just train
python run_pipeline.py --skip-tune

# Run tests
python -m pytest tests/ -v
```

## Governance & Standards

- **Config-first:** Primary settings live in `configs/pipeline.yaml`. CLI args override defaults.
- **Data lineage:** Each data/feature stage writes a manifest under `artifacts/manifests/` with hashes and time range.
- **Logging:** Pipeline uses structured logging (no raw prints) for reproducibility.
- **CI/CD:** Workflow exists under `.github/workflows/pytest_mlops.yml` but is disabled (`on: []`).

## Pipeline Flow

```
Raw CSV
  → src.data_prep.py   (chunked cleaning → daily panel)
  → src.features.py     (52 features: calendar, holiday intensity, peak days, lag/rolling)
  → src.tune.py         (Optuna + MLflow + 3-fold CV + auto model registry)
  → src.train.py        (final model → models/twin_xgb_boosted/)
  → src.infer.py        (load model → predict)
```

## Model: Twin-XGB Boosted

4 sub-models with event-specific segmentation:

| Model | Objective | Target |
|-------|-----------|--------|
| `clf` | binary:logistic | Zero vs non-zero classifier |
| `reg_top` | quantile (q=0.9) | High-volume items |
| `reg_peak` | quantile (q=0.95) | Peak demand days |
| `reg_tail` | asymmetric (α=50) | Low-demand items |

## Key Features (52 total)

- Calendar (14): day_of_week, month, quarter, holiday flags, etc.
- Holiday Intensity (4): `holiday_intensity`, `days_to_next_holiday`, etc.
- Peak Days (1): `is_peak_day` (top 5% global demand)
- Lag/Rolling (33): lags up to 84 days, rolling stats, spike ratios, price momentum

## Business Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| CLS | < 45,000 | Cost of Lost Sales |
| OFR | ≥ 0.80 | Order Fill Rate |
| Peak OFR | ≥ 0.75 | Fill Rate on peak days |

## Project Structure

```
src/                    # Python modules
├── config.py           # Single source of truth (constants, params, feature lists)
├── data_prep.py        # Chunked cleaning + daily aggregation
├── features.py         # 52 feature engineering
├── metrics.py          # Evaluation metrics (enterprise + notebook compatible)
├── model.py            # TwinXGBBoosted class
├── tune.py             # Optuna + MLflow tuning
├── train.py            # Final training
└── infer.py            # Inference

tests/                  # pytest test suite (17 tests)
├── test_features.py    # Unit: holiday, discount logic
├── test_data_quality.py# Data: no negatives, completeness
├── test_model_behavior.py # Behavioral: price elasticity
└── test_pipeline.py    # Integration: end-to-end training

notebooks/              # Jupyter notebooks
├── experiments/        # Active experiments
│   ├── twin_xgb_boosted.ipynb          # Boosted model (latest)
│   ├── twin_xgb_asym_ts_cv.ipynb       # Original twin-xgb
│   ├── seasonality_feature_relevance.ipynb  # Feature analysis
│   └── _archive/                       # Obsolete experiments
└── eda/                # Archived EDA notebooks
```

## MLflow

```bash
# View experiments
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Models that meet business thresholds are automatically registered to the **Model Registry**.
