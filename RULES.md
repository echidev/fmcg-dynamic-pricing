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

## 4. Coding Standards

- **Absolute Imports**: Use absolute imports based on the root directory (e.g., `from src.features import create_lags`).
- **Path Management**: Do NOT hardcode absolute system paths (like `C:/Users/...`). Always use relative paths (e.g., `data/raw/data.csv`) or `pathlib`.
- **CLI Execution**: All scripts must use `argparse` for CLI parameterization. Do NOT use `sys.path.insert()` workarounds. Run scripts as `python -m src.<module>` from the project root.
- **Professional Tone**: Strictly NO emojis or emoticons in code, comments, print statements, or commit messages. Maintain a professional, enterprise-grade environment.
- **Enterprise Documentation**: Document code clearly and neatly. Use standard docstrings (e.g., Google or NumPy format) for all classes and functions. Keep inline comments concise; focus on explaining the *why* (business logic or architectural decision) rather than stating the obvious *what*.