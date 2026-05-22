"""Inference — load trained DecoupledActuarialXGB model and predict.

Usage:
    python -m src.infer --input data/transform/new_data.parquet --output predictions.csv
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
    """Predict demand menggunakan Decoupled Actuarial.

    Args:
        model_dict: Output dari load_model().
        df: DataFrame dengan FEATURE_COLS, GROUP_COLS, PRICE_COL.

    Returns:
        Array prediksi demand_qty.
    """
    X = df[FEATURE_COLS]
    price = df[PRICE_COL].to_numpy(dtype=np.float32, copy=False)

    config = model_dict["config"]
    use_log = config.get("use_log_target", True)

    def _pred(model, x):
        p = model.predict(x)
        return np.maximum(np.expm1(p) if use_log else p, 0)

    p_mean = _pred(model_dict["mean_model"], X)
    p_quant = _pred(model_dict["quant_model"], X)

    # Actuarial layer
    median_price = float(np.median(price[price > 0])) if (price > 0).any() else 1.0
    item_price = np.where(price > 0, price, median_price)

    margin_ratio_high = config.get("margin_ratio_high", 1.5)
    margin_ratio_low = config.get("margin_ratio_low", 0.7)
    margin_ratio = np.where(item_price > median_price, margin_ratio_high, margin_ratio_low)
    shelf_life_perishable = (item_price < float(np.mean(item_price))).astype(float)

    smm = config.get("shortage_margin_multiplier", 2.2847)
    shortage_penalty = margin_ratio * (1.0 + 0.3 * shelf_life_perishable) * smm

    spoilage_penalty = 0.5 * shelf_life_perishable
    max_mr = max(margin_ratio.max(), 1e-8)
    overstock_cost = (1.0 - margin_ratio / max_mr) * (1.0 + spoilage_penalty)

    total_cost = shortage_penalty + overstock_cost + 1e-8
    critical_fractile = np.clip(shortage_penalty / total_cost, 0.2, 0.95)

    final_pred = p_mean + (p_quant - p_mean) * critical_fractile
    final_pred = np.maximum(final_pred, 0)

    return final_pred


def main():
    parser = argparse.ArgumentParser(description="DecoupledActuarialXGB inference")
    parser.add_argument("--input", type=str, required=True, help="Input Parquet/CSV dengan FEATURE_COLS")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--model-dir", type=str, default=str(MODELS_DIR / "decoupled_actuarial_xgb"))
    parser.add_argument("--format", type=str, choices=["csv", "parquet"], default="csv")
    args = parser.parse_args()

    logger = _setup_logger()
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

    pred = predict(model_dict, df)

    df["forecast"] = pred
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
