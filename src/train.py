"""Production training — train TwinXGBBoosted final model on full data.

Usage:
    python -m src.train --run-id <mlflow-run-id>
    python -m src.train --force-preprocess
"""

import argparse, gc, json
from pathlib import Path
import numpy as np
import mlflow

from src.config import (
    RAW_PATH, TABULAR_PATH, MODELS_DIR, FEATURE_COLS, GROUP_COLS,
    TARGET_COL, PRICE_COL, DATE_COL, RANDOM_STATE,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
    REG_PARAMS, CLF_PARAMS,
)
from src.model import TwinXGBBoosted


def load_best_params(run_id: str) -> dict:
    """Load params dari MLflow run."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.params


def train(panel, params, output_dir: Path):
    """Train final model pada full panel dan simpan."""
    x = panel[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    y = panel[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
    ht = panel["is_hari_besar"].to_numpy(dtype=float)
    ph = panel["is_pre_hari_besar"].to_numpy(dtype=float)
    pk = panel["is_peak_day"].to_numpy(dtype=float)
    tkeys = list(zip(panel["stock_code"].to_numpy(), panel["country"].to_numpy()))

    seg = panel.groupby(GROUP_COLS)[TARGET_COL].sum().sort_values(ascending=False)
    top_n = max(1, int(len(seg) * 0.05))
    top_keys = set(seg.head(top_n).index)

    model = TwinXGBBoosted(
        alpha_under=float(params.get("alpha_under", 50)),
        quantile_q_peak=float(params.get("quantile_q_peak", 0.95)),
        sample_weight_alpha=float(params.get("sample_weight_alpha", 4.0)),
        sample_weight_cap=float(params.get("sample_weight_cap", 5.0)),
        holiday_boost=float(params.get("holiday_boost", 1.5)),
        peak_days_boost=float(params.get("peak_days_boost", 2.0)),
        use_log_target=True,
    )
    model.fit(x, y, ht, ph, pk, top_keys, tkeys)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    # XGBoost native save untuk setiap sub‑model
    model.clf_.save_model(str(output_dir / "clf.json"))
    model.reg_top_.save_model(str(output_dir / "reg_top.json"))
    model.reg_peak_.save_model(str(output_dir / "reg_peak.json"))
    model.reg_tail_.save_model(str(output_dir / "reg_tail.json"))

    # Save config
    config = model.get_params()
    config["top_keys"] = list(top_keys)
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(config, f, default=str)

    print(f"Model saved to {output_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train TwinXGBBoosted final model")
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run ID untuk best params")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--input", type=str, default=str(RAW_PATH))
    parser.add_argument("--tabular-csv", type=str, default=str(TABULAR_PATH))
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "twin_xgb_boosted"))
    args = parser.parse_args()

    # Load params
    if args.run_id:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        params = load_best_params(args.run_id)
        print(f"Loaded params from run {args.run_id}")
    else:
        params = {"alpha_under": 50, "quantile_q_peak": 0.95,
                  "sample_weight_alpha": 4.0, "sample_weight_cap": 5.0,
                  "holiday_boost": 1.5, "peak_days_boost": 2.0}
        print("Using default params")

    # Load data
    if args.force_preprocess:
        from src.data_prep import aggregate_daily_from_chunks, build_full_panel
        from src.features import build_tabular_dataframe
        daily = aggregate_daily_from_chunks(Path(args.input))
        panel = build_full_panel(daily)
        del daily; gc.collect()
        panel = build_tabular_dataframe(panel)
    else:
        import pandas as pd
        panel = pd.read_csv(args.tabular_csv, parse_dates=[DATE_COL], low_memory=False)

    # Peak days
    daily_total = panel.groupby(DATE_COL)[TARGET_COL].sum().sort_values(ascending=False)
    peak_set = set(daily_total.head(max(1, int(len(daily_total) * 0.05))).index)
    panel["is_peak_day"] = panel[DATE_COL].isin(peak_set).astype("uint8")

    train(panel, params, Path(args.output_dir))


if __name__ == "__main__":
    main()
