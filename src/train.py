"""Production training — train DecoupledActuarialXGB final model on full data.

Menggunakan best params dari Optuna tuning:
  learning_rate=0.0339, max_depth=5, subsample=0.82
  q_target=0.9799, shortage_margin_multiplier=2.2847

Usage:
    python -m src.train
    python -m src.train --force-preprocess
    python -m src.train --output-dir models/decoupled_actuarial_xgb
"""

import argparse
import gc
import json
import logging
from pathlib import Path

import mlflow
import numpy as np

from src.config import (
    DATE_COL,
    FEATURE_COLS,
    GROUP_COLS,
    MIN_OBS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_PARAMS,
    MODELS_DIR,
    PEAK_DAYS_PCT,
    PRICE_COL,
    QUANTILE_Q_TARGET,
    RAW_PATH,
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
    logger.info("Evaluating on full dataset...")
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

    logger.info("Model saved to %s", output_dir)
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train DecoupledActuarialXGB final model")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--input", type=str, default=str(RAW_PATH))
    parser.add_argument("--tabular-parquet", type=str, default=str(TABULAR_PATH))
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "decoupled_actuarial_xgb"))
    args = parser.parse_args()

    logger = _setup_logger()

    # Load data
    if args.force_preprocess:
        logger.info("Running preprocessing from raw...")
        from src.data_prep import aggregate_daily_from_chunks, build_full_panel
        from src.features import build_tabular_dataframe
        daily = aggregate_daily_from_chunks(Path(args.input))
        panel = build_full_panel(daily)
        del daily
        gc.collect()
        panel = build_tabular_dataframe(panel)
    else:
        logger.info("Loading tabular: %s", args.tabular_parquet)
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
    logger.info("Panel shape after MIN_OBS filter: %s", panel.shape)

    # MLflow logging
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="decoupled_actuarial_production_train") as run:
        _, metrics = train(panel, Path(args.output_dir))
        mlflow.log_params({
            "model_params": str(MODEL_PARAMS),
            "q_target": QUANTILE_Q_TARGET,
            "shortage_margin_multiplier": SHORTAGE_MARGIN_MULTIPLIER,
        })
        mlflow.log_metrics({
            "train_cls": metrics["cls"],
            "train_ofr": metrics["ofr"],
            "train_mae": metrics["mae"],
        })
        mlflow.set_tag("model", "decoupled-actuarial-xgb")
        mlflow.set_tag("stage", "production")

    logger.info("MLflow Run ID: %s", run.info.run_id)
    logger.info("Training selesai. Model di: %s", args.output_dir)


if __name__ == "__main__":
    main()
