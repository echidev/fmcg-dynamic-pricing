"""Inference — load trained DecoupledActuarialXGB model and predict.

Usage:
    python -m src.infer --input data/gold/online_retail_daily_product_tabular.parquet --output predictions.csv
"""

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import FEATURE_COLS, MODELS_DIR, PRICE_COL


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("infer")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_model(model_dir: Path) -> dict:
    """Load sub-models dan config dari direktori."""
    mean_model = xgb.XGBRegressor()
    mean_model.load_model(str(model_dir / "mean_model.json"))

    quant_model = xgb.XGBRegressor()
    quant_model.load_model(str(model_dir / "quant_model.json"))

    with open(model_dir / "model_config.json") as f:
        config = json.load(f)

    return {
        "mean_model": mean_model,
        "quant_model": quant_model,
        "config": config,
    }


def predict(model_dict: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict demand menggunakan DecoupledActuarialXGB.

    Args:
        model_dict: Output dari load_model().
        df: DataFrame dengan FEATURE_COLS, GROUP_COLS, PRICE_COL.

    Returns:
        Array prediksi demand_qty.
    """
    from src.model import DecoupledActuarialXGB

    config = model_dict["config"]
    model = DecoupledActuarialXGB(
        model_params=config.get("model_params", {}),
        quantile_q_target=config.get("quantile_q_target", 0.9799),
        shortage_margin_multiplier=config.get("shortage_margin_multiplier", 2.2847),
        use_log_target=config.get("use_log_target", True),
    )
    model.model_mean_ = model_dict["mean_model"]
    model.model_quant_ = model_dict["quant_model"]
    model.feature_cols_ = config.get("feature_cols", FEATURE_COLS)
    model._fitted = True

    X = df[FEATURE_COLS]
    price = df[PRICE_COL].to_numpy(dtype=np.float32, copy=False)
    keys = list(zip(
        df.get("stock_code", pd.Series([""] * len(df))).to_numpy(),
        df.get("country", pd.Series([""] * len(df))).to_numpy(),
    ))

    return model.predict(X, price, keys)


def main():
    parser = argparse.ArgumentParser(description="DecoupledActuarialXGB inference")
    parser.add_argument("--input", type=str, required=True, help="Input Parquet/CSV dengan FEATURE_COLS")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--model-dir", type=str, default=str(MODELS_DIR / "decoupled_actuarial_xgb"))
    parser.add_argument("--format", type=str, choices=["csv", "parquet"], default="csv")
    parser.add_argument("--from-registry", action="store_true", help="Load model from MLflow Model Registry")
    parser.add_argument("--model-uri", type=str, default="models:/FMCG_Actuarial_Demand_Forecaster/latest", help="MLflow model URI (default: latest from registry)")
    args = parser.parse_args()

    logger = _setup_logger()

    if args.from_registry:
        import mlflow
        logger.info("Loading model from MLflow Registry: %s", args.model_uri)
        pyfunc_model = mlflow.pyfunc.load_model(args.model_uri)
        logger.info("Model loaded from MLflow Registry")
        # For registry mode, we need to handle differently
        logger.info("Registry mode: output will use PyFunc schema (expected_demand, recommended_stock, risk_status)")
    else:
        logger.info("Loading model from %s", args.model_dir)
        model_dict = load_model(Path(args.model_dir))

    logger.info("Loading data from %s", args.input)
    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    logger.info("Data shape: %s", df.shape)
    logger.info("Predicting...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.from_registry:
        result_df = pyfunc_model.predict(df)
        result_df.to_csv(output_path, index=False)
        logger.info("Predictions (registry schema) saved to %s", output_path)
        logger.info("Columns: %s", list(result_df.columns))
    else:
        pred = predict(model_dict, df)
        df["forecast"] = pred

        if args.format == "parquet":
            df.to_parquet(output_path.with_suffix(".parquet"), index=False)
            logger.info("Predictions saved to %s", output_path.with_suffix(".parquet"))
        else:
            df.to_csv(output_path, index=False)
            logger.info("Predictions saved to %s", output_path)

        logger.info("Zero rate: %.1f%%", (pred == 0).mean() * 100)
        logger.info("Mean forecast: %.2f", pred.mean())

    del df, pred
    gc.collect()


if __name__ == "__main__":
    main()
