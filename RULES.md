# MLOps Project Rules for AI Agent

As an AI Agent working on this FMCG Dynamic Pricing & Demand Forecasting project, you MUST adhere to the following strict rules:

## 1. Directory Access & Modularity
- **`src/`**: ONLY raw Python scripts (`.py`). All core logic (data prep, feature engineering, training, tuning, inference) MUST reside here. Do NOT put experimental code here. All scripts must be executable via `python -m src.<module>` from the project root.
- **`notebooks/`**: ONLY for Exploratory Data Analysis (EDA), visualization, and result presentation. DO NOT write core pipeline functions or training loops here. Import functions from `src/` instead.
- **`models/`**: THIS IS A SACRED PRODUCTION REGISTRY.
  - Do NOT save hyperparameter tuning artifacts here.
  - Only the final, winning model goes to `models/demand_forecasting/production/` via the `src/train.py` script.
- **`data/`**: Treat `data/raw/` as READ-ONLY. Never overwrite raw data.

## 2. Version Control (Git & DVC)
- **Data & Models**: NEVER commit `.csv`, `.parquet`, `.pkl`, `.h5`, or any binary/large files to Git.
- ALWAYS use `dvc add <file_path>` for data and models. Only commit the resulting `.dvc` files to Git.
- Ensure `mlruns/` and `mlflow.db` are in `.gitignore`.

## 3. Data Leakage Prevention (Crucial)
- When doing multi-step time-series forecasting, NEVER use target variables (e.g., `demand_qty`, `revenue` on day *t*) as features.
- All momentum features (e.g., `diff_1`) MUST be calculated using historical lags (`lag_1 - lag_2`), NEVER `actual - lag_1`.
- Aggregations (e.g., product popularity) must be calculated ONLY using the training set (`df_train`) and mapped to the test set to avoid future leakage.

## 4. Data Cutoff, Splits & Lineage (Mandatory)
- **Cutoff Policy**: Every dataset split MUST be defined by an explicit `cutoff_date` and window sizes (`train_window`, `valid_window`, `test_window`).
- **Train-Only Aggregation**: Any aggregation, encoding, or normalization MUST be fit on the training window only and applied to validation/test windows.
- **Lineage Evidence**: Every run MUST record a dataset manifest containing dataset id, time span, row counts, and a checksum/hash for each artifact.

## 5. Config Governance
- **Config First**: All scripts MUST read a config file (YAML/TOML). `argparse` is only for overrides.
- **Required Params**: `dataset_version`, `feature_set_id`, `cutoff_date`, `model_type`, `random_seed`, `git_commit` MUST be logged for every run.

## 6. Enforcement & Evidence
- **Evidence Required**: Before any training or promotion, provide evidence for dataset manifest, split integrity, leakage checks, and reproducibility snapshot.
- **Pre-Commit Guardrails**: Use pre-commit hooks to block large/binary files and secrets before commit.

## 7. Coding Standards
- **Absolute Imports**: Use absolute imports based on the root directory (e.g., `from src.features import create_lags`).
- **Path Management**: Do NOT hardcode absolute system paths. Always use relative paths (e.g., `data/raw/data.csv`) or `pathlib`.
- **CLI Execution**: All scripts must use `argparse` for CLI parameterization. Do NOT use `sys.path.insert()` workarounds. Run scripts as `python -m src.<module>` from the project root.
- **Professional Tone**: Strictly NO emojis or emoticons in code, comments, print statements, or commit messages. Maintain a professional, enterprise-grade environment.
- **Enterprise Documentation**: Document code clearly and neatly. Use standard docstrings (e.g., Google or NumPy format) for all classes and functions. Keep inline comments concise; focus on explaining the *why* (business logic or architectural decision) rather than stating the obvious *what*.

## 8. Security, Privacy & Confidentiality
- **Restricted Access**: You are STRICTLY FORBIDDEN from reading, modifying, exfiltrating, or printing the contents of `.env` files, `.aws/credentials`, or any local configuration files that contain secrets.
- **Credential Isolation**: NEVER hardcode API keys, database URIs, passwords, or cloud access tokens in any script, notebook, or documentation. All credentials must be loaded dynamically via `os.environ` or a secure secret manager.
- **Git Ignore Enforcement**: Ensure that `.env`, `*.key`, `*.pem`, `*.sqlite`, and any files containing authentication details are explicitly listed in `.gitignore` before performing any version control operations.
- **Data Anonymization & Logging**: Do not log, print, or export sensitive FMCG operational metrics, unmasked financial data, or credentials into MLflow tracking logs, standard output, or error tracebacks.
- **Redaction Rule**: If an exception includes sensitive values, replace them with `REDACTED` before logging.

## 9. Dependency & Environment Strictness
- **Dependency Management**: DO NOT dynamically execute shell commands to install libraries (e.g., `os.system("pip install")`). All dependencies must be explicitly defined in `requirements.txt` or `pyproject.toml`.
- **Environment Parity**: Assume execution within a Docker container or WSL environment. Avoid OS-specific functions that break cross-platform compatibility.

## 10. Experiment Tracking Protocol (MLflow)
- **Run Naming Convention**: Assign structured, descriptive run names (e.g., `xgboost_baseline_v1`, `prophet_holiday_features`). Do not use default or sequential names like `run_1`.
- **Tagging Requirements**: Always attach essential tags to MLflow runs (e.g., model type, forecasting window size, pipeline stage).
- **Artifact Limits**: Do NOT log massive DataFrames as artifacts on every iteration. Restrict artifact logging to final model summaries, evaluation metrics, and specific validation plots.

## 11. Execution Safety & Idempotency
- **Idempotent Operations**: All data preparation and feature engineering scripts MUST be idempotent. Executing them multiple times must yield the exact same output without duplicating records or corrupting the state.
- **State Validations**: Always verify directory existence and file integrity before executing I/O operations to prevent pipeline crashes.

## 12. Resource & Cloud Governance
- **Memory Management**: Implement chunking or batch processing for large datasets. DO NOT load entire datasets into memory if they exceed reasonable RAM limits.
- **Execution Limits**: Strictly prohibit infinite loops (`while True` without rigid `break` conditions) during hyperparameter tuning or API calls to prevent Out-Of-Memory (OOM) errors and uncontrolled cloud compute billing.

## 13. Automated Testing Baseline
- **Pytest Required**: Use `pytest` for automated checks. Tests must include leakage, split integrity, and null checks.
- **Sanity Checks**: Implement basic validation checks (using `assert` or simple validation functions) at the end of data engineering stages. Ensure no unexpected null values exist in critical features and array dimensions match expectations before proceeding to training.

## 14. Absolute Reproducibility
- **Global Seed Setting**: You MUST initialize a global random seed at the beginning of every training or processing script for all relevant libraries (e.g., `random`, `numpy.random`, `torch`, `xgboost`). Reproducibility is non-negotiable.
- **Deterministic Flags**: Set deterministic flags for LightGBM/XGBoost where supported, and set `PYTHONHASHSEED`.
- **Environment Snapshot**: Log `python --version`, OS info, and `pip freeze` to MLflow for every training/tuning run.

## 15. Enterprise Logging
- **Standardized Logging**: DO NOT use `print()` for process tracking. Implement the standard Python `logging` module. Configure logging levels (`INFO`, `WARNING`, `ERROR`) appropriately to track execution flow, data shapes, and pipeline progression.
- **Allowed Fields**: Log only non-sensitive metadata (row counts, shapes, null ratios, feature counts, window ranges).
- **Forbidden Fields**: Never log raw values, COGS, price floors/ceilings, or customer-level records.

## 16. Model Registry Governance
- **Metadata Schema**: Any model placed in `models/demand_forecasting/production/` MUST include metadata: dataset id, checksum, cutoff date, git commit, metrics, and approval date.
- **Promotion Criteria**: A model can be promoted only if leakage tests pass and required metrics meet defined thresholds.

## 17. Mathematical & Business Constraints
- **Pricing Optimization Boundaries**: When generating dynamic pricing recommendations, apply strict mathematical boundary conditions. Prices must never fall below minimum margin thresholds and must adhere to predefined maximum price ceilings.
- **Demand Constraints**: Ensure forecasted demand matrices do not contain negative values. Apply `max(0, forecasted_value)` or appropriate mathematical constraints to all final output vectors.
