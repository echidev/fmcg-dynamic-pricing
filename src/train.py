"""Pelatihan produksi — train DecoupledActuarialXGB final model on full data.

Menggunakan best params dari Optuna tuning:
  learning_rate=0.0339, max_depth=5, subsample=0.82
  q_target=0.9799, shortage_margin_multiplier=2.2847

MLflow Tracking URI dibaca dari environment variable MLFLOW_TRACKING_URI
via python-dotenv (.env file atau environment variable).

Usage:
    python -m src.train
    python -m src.train --force-preprocess
    python -m src.train --output-dir models/decoupled_actuarial_xgb
"""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import mlflow
import numpy as np

from src.config import (
    DATE_COL,
    FEATURE_COLS,
    GOLD_DIR,
    GROUP_COLS,
    MIN_OBS,
    MODEL_PARAMS,
    MODELS_DIR,
    PEAK_DAYS_PCT,
    PRICE_COL,
    QUANTILE_Q_TARGET,
    SHORTAGE_MARGIN_MULTIPLIER,
    TABULAR_PATH,
    TARGET_COL,
)
from src.model import DecoupledActuarialXGB


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("train")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def train(panel, output_dir: Path, params_override: dict = None):
    """Train final DecoupledActuarialXGB pada full panel dan simpan."""
    logger = _setup_logger()

    # Use best known params from experiment, allow override via dict
    p = dict(params_override) if params_override else {}
    model_params = dict(
        MODEL_PARAMS,
        n_estimators=p.get("n_estimators", MODEL_PARAMS["n_estimators"]),
        max_depth=p.get("max_depth", MODEL_PARAMS["max_depth"]),
        learning_rate=p.get("learning_rate", MODEL_PARAMS["learning_rate"]),
        subsample=p.get("subsample", MODEL_PARAMS["subsample"]),
    )
    q_target = p.get("q_target", QUANTILE_Q_TARGET)
    smm = p.get("shortage_margin_multiplier", SHORTAGE_MARGIN_MULTIPLIER)

    logger.info("Training DecoupledActuarialXGB...")
    logger.info("  model_params=%s", model_params)
    logger.info("  q_target=%.4f, shortage_margin_multiplier=%.2f", q_target, smm)

    x_all = panel[FEATURE_COLS]
    y_all = panel[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
    price_all = panel[PRICE_COL].to_numpy(dtype=np.float32, copy=False)
    keys_all = list(zip(panel["stock_code"].to_numpy(), panel["country"].to_numpy()))

    model = DecoupledActuarialXGB(
        model_params=model_params,
        quantile_q_target=q_target,
        shortage_margin_multiplier=smm,
    )
    model.fit(x_all, y_all, feature_cols=FEATURE_COLS)

    # Evaluate on full data (in-sample sanity check)
    logger.info("Evaluasi pada full dataset...")
    metrics = model.evaluate(x_all, y_all, price_all, keys_all)
    logger.info("  CLS=%.0f  OFR=%.4f  MAE=%.2f", metrics["cls"], metrics["ofr"], metrics["mae"])

    # Save models
    output_dir.mkdir(parents=True, exist_ok=True)
    model.model_mean_.save_model(str(output_dir / "mean_model.json"))
    model.model_quant_.save_model(str(output_dir / "quant_model.json"))

    # Save config
    config = model.get_params()
    config["metrics"] = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()}
    config["feature_cols"] = FEATURE_COLS
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(config, f, default=str, indent=2)

    logger.info("Model disimpan ke %s", output_dir)
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train DecoupledActuarialXGB final model")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--input", type=str, default=str(GOLD_DIR / "online_retail_daily_product_tabular.parquet"))
    parser.add_argument("--tabular-parquet", type=str, default=str(TABULAR_PATH))
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "decoupled_actuarial_xgb"))
    args = parser.parse_args()

    logger = _setup_logger()

    # ── Load MLflow Tracking URI dari environment variable ──
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        raise ValueError(
            "Environment variable MLFLOW_TRACKING_URI tidak ditemukan. "
            "Setel di file .env atau sebagai environment variable sebelum menjalankan script."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("FMCG-Actuarial-Optimization")

    # Load data
    if args.force_preprocess:
        logger.info("Menjalankan full ETL: bronze → silver → gold...")
        from src.data_prep import process_bronze
        from src.features import process_silver
        process_bronze()
        panel = process_silver()
    else:
        logger.info("Memuat gold tabular: %s", args.tabular_parquet)
        import pandas as pd
        panel = pd.read_parquet(args.tabular_parquet)

    # Peak days
    daily_total = panel.groupby(DATE_COL)[TARGET_COL].sum().sort_values(ascending=False)
    peak_n = max(1, int(len(daily_total) * PEAK_DAYS_PCT))
    peak_set = set(daily_total.head(peak_n).index)
    panel["is_peak_day"] = panel[DATE_COL].isin(peak_set).astype("uint8")

    # Filter low-obs items
    obs = panel.groupby(GROUP_COLS).size()
    panel = panel[panel.set_index(GROUP_COLS).index.isin(obs[obs >= MIN_OBS].index)].copy()
    logger.info("Panel shape setelah filter MIN_OBS: %s", panel.shape)

    # ── MLflow Run ──
    with mlflow.start_run(run_name="Production_Tuned_Model") as run:
        model, metrics = train(panel, Path(args.output_dir))

        # Log parameter pemenang dari Optuna tuning
        mlflow.log_params({
            "learning_rate": MODEL_PARAMS["learning_rate"],
            "max_depth": MODEL_PARAMS["max_depth"],
            "subsample": MODEL_PARAMS["subsample"],
            "q_target": QUANTILE_Q_TARGET,
            "shortage_margin_multiplier": SHORTAGE_MARGIN_MULTIPLIER,
        })

        # Log metrik bisnis final pada validation set
        mlflow.log_metrics({
            "Global_OFR": metrics["ofr"],
            "Max_CLS": metrics["cls"],
        })

        # Log artifact model XGBoost
        mlflow.xgboost.log_model(
            xgb_model=model.model_mean_,
            artifact_path="mean_model",
            registered_model_name="FMCG_Actuarial_Demand_Forecaster",
        )
        mlflow.xgboost.log_model(
            xgb_model=model.model_quant_,
            artifact_path="quant_model",
            registered_model_name="FMCG_Actuarial_Demand_Forecaster",
        )

        mlflow.set_tag("model", "decoupled-actuarial-xgb")
        mlflow.set_tag("stage", "production")

    logger.info("MLflow Run ID: %s", run.info.run_id)
    logger.info("Pelatihan selesai. Model di: %s", args.output_dir)


if __name__ == "__main__":
    main()
