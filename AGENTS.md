# MASTER AGENT DIRECTORY

Welcome to the FMCG Dynamic Pricing & Demand Forecasting Project. 

This repository follows a strict enterprise-grade MLOps architecture. Before executing any code, writing scripts, or modifying the pipeline, you MUST read the core instruction files.

## Core Instruction Files (READ THESE FIRST)
- `RULES.md`: Contains non-negotiable boundaries regarding data leakage prevention, folder access, and version control limits.
- `SKILLS.md`: Outlines the specific tech stack (DVC, MLflow, Optuna, XGBoost/LightGBM) and execution expectations.

## Current Repo State & Entrypoints
This repository is NOT empty. The data pipeline is modularized within the `src/` directory.

**Execution Flow:**
1. **Data Preparation:** `src/data_prep.py` (Handles cleaning and basic transformations).
2. **Feature Engineering:** `src/features.py` (Handles zero-leakage lag creation, momentum, and static product mapping).
3. **Experimentation:** `src/tune.py` (Runs Optuna trials, logs to MLflow).
4. **Production Training:** `src/train.py` (Final training script for the champion model - *in development*).
5. **Inference:** `src/infer.py` (Serves the model for predictions - *in development*).

## Data and Artifact Management
- **Git:** Standard source control. `.gitignore` explicitly blocks `data/`, `models/`, and `mlruns/`.
- **DVC:** Used for tracking all raw/transformed data and production model binaries. 
  - *Agent Note:* If you generate a new dataset or finalize a model, remind the user to run `dvc add <path>`.
- **MLflow:** Local tracking server (`mlruns/` and `mlflow.db`). Used exclusively during `src/tune.py` execution.

## Active Development Focus
Currently optimizing the demand forecasting models for highly zero-inflated FMCG data. Focus is on eliminating under-forecasting bias during peak days and managing Cost of Lost Sales (CLS).