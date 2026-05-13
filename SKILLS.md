# Agent Technical Skills & Context

This document outlines the technical stack, standard operating procedures, strict hardware constraints (16GB RAM limit), and business acumen requirements for this FMCG Dynamic Pricing & Demand Forecasting project. You MUST use these tools, rules, and commercial constraints when writing code or providing solutions.

## 1. DVC & Data Format Optimization
- **Concept:** Used for tracking large datasets and model artifacts.
- **Execution:** Execute `dvc add <path>` when generating artifacts and prompt the user to commit the `.dvc` file.
- **Data Format MUST BE Parquet:** To save RAM and disk space, NEVER save intermediate or processed data as `.csv`. ALWAYS use `.parquet` to preserve data types and utilize Snappy/GZIP compression.
- **CSV Ingestion Only:** CSV is permitted only for raw ingestion. Convert to Parquet immediately after ingestion.

## 2. Strict Memory Constraints (16GB RAM Limit)
Your execution environment has a strict 16GB RAM limit. You MUST employ the following memory-saving techniques to prevent Out-Of-Memory (OOM) crashes:
- **Use Polars over Pandas:** Whenever possible, use `polars` for large-scale data wrangling instead of `pandas` due to its multithreaded and memory-efficient Rust backend.
- **Aggressive Downcasting:** If `pandas` must be used, immediately downcast data types after loading (e.g., `float64` to `float32`, `int64` to `int32/int16`, and strings to `category`).
- **Chunking:** Use `chunksize` when reading unavoidably large CSV files during raw ingestion. Do not load everything into memory at once.
- **Garbage Collection:** Explicitly use `del variable_name` and `gc.collect()` immediately after large dataframes are no longer needed in the script.
- **Model Efficiency:** Prefer LightGBM (`lightgbm`) over XGBoost for large datasets, as LightGBM uses histogram-based algorithms that consume significantly less memory.

## 3. MLflow & Experiment Tracking
- **Concept:** Used to log hyperparameter tuning, metrics, and models.
- **Execution:** `src/tune.py` MUST use `mlflow.start_run()`. Local SQLite backend (`sqlite:///mlruns/mlflow.db`).
- **Metrics to log:** MAE, WMAPE, FVA (Forecast Value Added), OFR (Order Fill Rate), and CLS (Cost of Lost Sales).
- **Enforce Model Signatures:** You MUST use `mlflow.models.infer_signature(X_train, y_pred)` when logging models to prevent shape mismatch hallucinations during production inference.

## 4. Optuna (Hyperparameter Tuning)
- **Concept:** Used within `src/tune.py` to find optimal model parameters.
- **Execution & Efficiency:** Create an `objective(trial)` function. 
- **Mandatory Pruning:** You MUST use a pruner (e.g., `optuna.pruners.MedianPruner`) and implement Early Stopping in your boosting models to instantly kill trials that perform poorly, saving massive amounts of compute time.

## 5. Modeling Stack & Zero-Inflated Data
- **Libraries:** `xgboost`, `lightgbm`, `scikit-learn`.
- **Zero-Inflated Strategy:** FMCG Demand Forecasting involves heavily zero-inflated data.
  - *Approach A:* Use `objective='reg:tweedie'`.
  - *Approach B (Hurdle Model):* If Tweedie fails, implement a Two-Stage model: Stage 1 (Classifier for Zero vs. Non-Zero probability) -> Stage 2 (Regressor for magnitude if Non-Zero).
- **Time-Series Validation:** NEVER use random `train_test_split` or `KFold`. You MUST use `TimeSeriesSplit` or a custom sliding/expanding window validation to prevent temporal data leakage.
- **Model Selection Rule:** Use LightGBM by default. Switch to XGBoost only when LightGBM fails to meet baseline WMAPE after tuning. Trigger the Hurdle Model when zero-rate exceeds 40% or Tweedie underperforms baseline by more than 5% WMAPE.

## 6. Code Quality, Typing & Linting
- **Type Hinting:** All functions in `src/` MUST use standard Python type hints (e.g., `def create_features(df: pd.DataFrame) -> pd.DataFrame:`).
- **Validation:** Use `pydantic` for validating configurations and API payloads if building an inference endpoint.
- **Linting:** Code must be clean and compliant with `flake8` or `ruff` standards. Professional tone only; strictly no emojis in codebase.

## 7. LLM Context Window Optimization (Self-Regulation)
- **Do NOT Print Huge DataFrames:** When executing code to explore data, NEVER `print(df)` or `print(df.head(100))`. This floods your own context window and causes you to lose focus.
- **Print Summaries Only:** Use `print(df.info())`, `print(df.columns)`, or `print(df.describe())` to understand data structures.
- **Unified Scripts:** Always parameterize scripts using `argparse`. Example: `python -m src.tune --model xgboost --trials 50`.

## 8. High-Cardinality Feature Engineering (Scalability)
FMCG data contains high-cardinality categorical variables (e.g., `sku_id`, `store_id`).
- **NEVER use One-Hot Encoding (`pd.get_dummies`)** for these variables. It will cause immediate OOM crashes.
- **Tree Models:** For LightGBM/XGBoost, cast these columns as `category` data types and let the algorithms handle them natively using their built-in categorical splitters. Log the categorical column list to MLflow.

## 9. Deep Learning & Resource Scaling (PyTorch/TensorFlow)
If the pipeline upgrades to Deep Learning architectures (e.g., LSTMs, Temporal Fusion Transformers):
- **Mixed Precision Training:** ALWAYS use Mixed Precision (e.g., `torch.cuda.amp` or FP16).
- **Gradient Accumulation:** To simulate large batch sizes without exceeding the 16GB limit, use small actual batch sizes and accumulate gradients over multiple steps before calling `optimizer.step()`.
- **Dataloaders:** NEVER load the entire PyTorch dataset into memory. Use efficient custom `Dataset` classes that read from disk sequentially.

## 10. Production-Ready Inference (ONNX & Quantization)
Pickle files are slow and heavy. For dynamic pricing, inference latency must be sub-second.
- **ONNX Export:** When finalizing a model for the `production/` folder, ALWAYS attempt to export it to ONNX format (`.onnx`) for highly optimized C++ backend execution. If export fails, log the failure reason and keep the native model format.
- **Model Quantization:** Apply Post-Training Quantization (e.g., INT8 quantization) to reduce model size and increase inference speed.

## 11. Advanced FMCG Metrics & Custom Loss Functions
Standard metrics do not fully capture FMCG business realities.
- **WMAPE (Weighted Mean Absolute Percentage Error):** You MUST calculate WMAPE alongside MAE to handle varying-volume data accurately. Canonical formula: `sum(|y - y_hat|) / sum(|y|)`.
- **Asymmetric Loss Functions:** Stockouts often cost more than overstocking. Be prepared to implement custom asymmetric objective functions where under-predictions are penalized differently than over-predictions.

## 12. FMCG Business Acumen & Commercial Logic
You are not just a statistical calculator; you are a business optimizer. All technical decisions must align with these commercial realities:
- **Margin-Aware Optimization:** Dynamic pricing is NOT about maximizing volume; it is about maximizing gross profit. Your objective functions must explicitly factor in Cost of Goods Sold (COGS). Never recommend a price that drops below the minimum margin threshold unless explicitly clearing perishable stock.
- **Price Elasticity & Cannibalization:** Consider Cross-Price Elasticity. Understand that drastically dropping the price of SKU A (e.g., a 500ml shampoo) will cannibalize sales of SKU B (e.g., a 250ml shampoo of the same brand).
- **Calendar & Promo Centricity:** FMCG demand is heavily dictated by external temporal events. Always engineer features that account for payday cycles (e.g., end/beginning of the month), major religious holidays, and double-digit promo events (e.g., 11.11, 12.12).
- **Inventory Lifecycle (Perishability):** FMCG items have strict expiration dates. Pricing algorithms must switch from "Margin Optimization" to "Clearance/Markdown Optimization" when an item's shelf-life reaches critical thresholds to minimize outright spoilage loss.

## 13. Polars/Pandas Boundary
- **Polars First:** Use Polars for heavy ETL and feature engineering.
- **Pandas Boundary:** Convert to Pandas only at model-fitting boundaries when a library requires it.
