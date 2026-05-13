"""Inference — load trained TwinXGBBoosted model and predict.

Usage:
    python -m src.infer --input data/raw/new_data.csv --output predictions.csv
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from src.config import FEATURE_COLS, GROUP_COLS, TARGET_COL, PRICE_COL, MODELS_DIR


def load_model(model_dir: Path):
    """Load sub‑models dan config dari direktori."""
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_dir / "clf.json"))
    reg_top = xgb.XGBRegressor()
    reg_top.load_model(str(model_dir / "reg_top.json"))
    reg_peak = xgb.XGBRegressor()
    reg_peak.load_model(str(model_dir / "reg_peak.json"))
    reg_tail = xgb.XGBRegressor()
    reg_tail.load_model(str(model_dir / "reg_tail.json"))

    with open(model_dir / "model_config.json") as f:
        config = json.load(f)

    top_keys = {tuple(k) if isinstance(k, list) else k for k in config.get("top_keys", [])}
    use_log = config.get("use_log_target", True)

    return {"clf": clf, "reg_top": reg_top, "reg_peak": reg_peak,
            "reg_tail": reg_tail, "top_keys": top_keys, "use_log": use_log}


def predict(model_dict, df: pd.DataFrame, peak_mask: np.ndarray) -> np.ndarray:
    """Predict demand untuk DataFrame baru.

    Args:
        model_dict: Output dari load_model().
        df: DataFrame dengan FEATURE_COLS, GROUP_COLS, dan is_peak_day.
        peak_mask: Boolean array (is_peak_day untuk setiap row).

    Returns:
        Array prediksi demand_qty.
    """
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    keys = list(zip(df[GROUP_COLS[0]].to_numpy(), df[GROUP_COLS[1]].to_numpy()))
    vtop = np.array([k in model_dict["top_keys"] for k in keys])

    def _pred(model, x):
        p = model.predict(x)
        use_log = model_dict["use_log"]
        return np.maximum(np.expm1(p) if use_log else p, 0)

    p_top = _pred(model_dict["reg_top"], X)
    p_peak = _pred(model_dict["reg_peak"], X)
    p_tail = _pred(model_dict["reg_tail"], X)

    return np.where(peak_mask, p_peak, np.where(vtop, p_top, p_tail))


def main():
    parser = argparse.ArgumentParser(description="TwinXGBBoosted inference")
    parser.add_argument("--input", type=str, required=True, help="Input CSV dengan FEATURE_COLS")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--model-dir", type=str, default=str(MODELS_DIR / "twin_xgb_boosted"))
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}")
    model_dict = load_model(Path(args.model_dir))

    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    peak_mask = df["is_peak_day"].to_numpy(dtype=bool) if "is_peak_day" in df.columns else np.zeros(len(df), dtype=bool)

    print("Predicting...")
    pred_reg = predict(model_dict, df, peak_mask)

    # Apply classification gating
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    proba = model_dict["clf"].predict_proba(X)[:, 1]
    pred = pred_reg * (proba >= args.threshold).astype(int)

    df["forecast"] = pred
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Zero rate: {(pred == 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
