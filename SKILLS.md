# Agent Technical Skills & Context

This document outlines the technical stack and standard operating procedures for this project. Use these tools when assisting the user.

## 1. DVC (Data Version Control)

- **Concept:** Used for tracking large datasets and model artifacts.
- **Execution:** When generating a new transformed dataset or a final production model, execute `dvc add <path>` and append the instruction for the user to commit the `.dvc` file.

## 2. MLflow (Experiment Tracking)

- **Concept:** Used to log hyperparameter tuning, metrics, and models.
- **Execution:** `src/tune.py` MUST use `mlflow.start_run()`.
- **Metrics to log:** MAE, FVA (Forecast Value Added), OFR (Order Fill Rate), and CLS (Cost of Lost Sales).
- **Backend:** Local SQLite backend (`sqlite:///mlruns/mlflow.db`).

## 3. Optuna (Hyperparameter Tuning)

- **Concept:** Used within `src/tune.py` to find optimal model parameters.
- **Execution:** Create an `objective(trial)` function. Suggest tuning parameters specifically tailored for Zero-Inflated time-series data (e.g., `objective='reg:tweedie'` for XGBoost/LightGBM).

## 4. Modeling Stack

- **Libraries:** `xgboost`, `lightgbm`, `scikit-learn`.
- **Focus:** FMCG Demand Forecasting involves heavily zero-inflated data. Models must be robust against predicting "safe zeros" and should be evaluated on both "Zero Detection" (F1-score) and "Non-Zero Magnitude" (MAE).

## 5. Unified Scripts

- Always parameterize scripts using `argparse`.
- Example: `python -m src.tune --model xgboost --trials 50`. Avoid creating duplicate files for different algorithms unless fundamentally necessary (e.g., Tabular vs. Deep Learning).
