"""Hyperparameter tuning — Optuna + MLflow untuk TwinXGBBoosted.

Usage:
    python -m src.tune --trials 200
    python -m src.tune --trials 50 --force-preprocess
"""

import argparse, os, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import optuna
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    RAW_PATH, TABULAR_PATH, FEATURE_COLS, GROUP_COLS,
    TARGET_COL, PRICE_COL, DATE_COL, RANDOM_STATE,
    PROMOTION_THRESHOLDS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
    REG_PARAMS, CLF_PARAMS,
)
from src.model import TwinXGBBoosted
from src.metrics import evaluate_prediction


# ── Helper: time series CV splits ──

def make_splits(dates, horizon=30, n_splits=3, min_train=180):
    dates = np.array(sorted(pd.to_datetime(dates).unique()))
    total = len(dates)
    splits = []
    for i in range(n_splits):
        ve = total - (n_splits - i - 1) * horizon
        vs = ve - horizon
        te = vs - 1
        if te < min_train:
            continue
        splits.append((dates[te], dates[vs], dates[ve - 1]))
    return splits


# ── Data loader ──

def load_data(tabular_path, horizon=30):
    """Load tabular CSV, return panel_df + FEATURE_COLS."""
    print(f"Loading tabular: {tabular_path}")
    panel = pd.read_csv(tabular_path, parse_dates=[DATE_COL], low_memory=False)
    # Ensure types
    for c in FEATURE_COLS:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    nz = set(GROUP_COLS + [TARGET_COL, PRICE_COL, DATE_COL,
                           "is_hari_besar", "is_pre_hari_besar", "is_peak_day"])
    for c in nz:
        if c in panel.columns:
            panel[c] = panel[c]
    return panel


# ── Optuna objective ──

def objective(trial, panel, splits):
    """Optuna trial untuk TwinXGBBoosted."""
    # Architecture params
    q_peak = trial.suggest_float("quantile_q_peak", 0.90, 0.99)
    alpha_under = trial.suggest_int("alpha_under", 10, 200, log=True)
    peak_pct = trial.suggest_float("peak_days_pct", 0.02, 0.10)

    # Event weighting
    hb = trial.suggest_float("holiday_boost", 0.5, 5.0)
    pb = trial.suggest_float("peak_days_boost", 0.5, 5.0)
    sw_alpha = trial.suggest_float("sample_weight_alpha", 1.0, 8.0)
    sw_cap = trial.suggest_float("sample_weight_cap", 2.0, 10.0)

    # Tree params
    ne = trial.suggest_int("n_estimators", 200, 800, step=100)
    md = trial.suggest_int("max_depth", 4, 10)
    lr = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)

    rp = dict(REG_PARAMS, n_estimators=ne, max_depth=md, learning_rate=lr)
    cp = dict(CLF_PARAMS)

    cls_metrics = []
    for fold_idx, (te, vs, ve) in enumerate(splits, 1):
        train_mask = panel[DATE_COL] <= te
        val_mask = (panel[DATE_COL] >= vs) & (panel[DATE_COL] <= ve)

        # Item-based top keys
        seg = panel.loc[train_mask].groupby(GROUP_COLS)[TARGET_COL].sum().sort_values(ascending=False)
        tn = max(1, int(len(seg) * 0.05))
        top_keys = set(seg.head(tn).index)

        df_tr = panel.loc[train_mask]
        df_vl = panel.loc[val_mask]

        X_tr = df_tr[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
        y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        X_vl = df_vl[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
        y_vl = df_vl[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        pv = df_vl[PRICE_COL].to_numpy(dtype=np.float32, copy=False)

        tkeys = list(zip(df_tr["stock_code"].to_numpy(), df_tr["country"].to_numpy()))
        vkeys = list(zip(df_vl["stock_code"].to_numpy(), df_vl["country"].to_numpy()))

        ht = df_tr["is_hari_besar"].to_numpy(dtype=float)
        ph = df_tr["is_pre_hari_besar"].to_numpy(dtype=float)
        pk = df_tr["is_peak_day"].to_numpy(dtype=float)
        vp = df_vl["is_peak_day"].to_numpy(dtype=bool)
        vh = df_vl["is_hari_besar"].to_numpy(dtype=bool) | df_vl["is_pre_hari_besar"].to_numpy(dtype=bool)

        model = TwinXGBBoosted(
            clf_params=cp,
            reg_params=rp,
            alpha_under=alpha_under,
            quantile_q_peak=q_peak,
            sample_weight_alpha=sw_alpha,
            sample_weight_cap=sw_cap,
            holiday_boost=hb,
            peak_days_boost=pb,
        )
        try:
            model.fit(X_tr, y_tr, ht, ph, pk, top_keys, tkeys)
            _, metrics, _ = model.predict_with_threshold(X_vl, y_vl, pv, vkeys, vp, vh)
            cls_metrics.append(metrics["cls"])
        except Exception as e:
            return 1e9

        del df_tr, df_vl, X_tr, y_tr, X_vl, y_vl, pv; gc.collect()

    if not cls_metrics:
        return 1e9
    return float(np.mean(cls_metrics))


def run_optuna(panel, splits, n_trials=100, seed=42):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda t: objective(t, panel, splits),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study


def promote_if_eligible(metrics, model_params, run_id):
    """Daftarkan ke model registry jika threshold bisnis terpenuhi."""
    th = PROMOTION_THRESHOLDS
    ofr_ok = metrics.get("ofr", 0) >= th["ofr_min"]
    cls_ok = metrics.get("cls", 1e9) <= th["cls_max"]
    peak_ok = metrics.get("peak_ofr", 0) >= th["peak_ofr_min"]
    if ofr_ok and cls_ok and peak_ok:
        name = "twin_xgb_boosted"
        desc = (
            f"OFR={metrics['ofr']:.3f} CLS={metrics['cls']:.0f} "
            f"PeakOFR={metrics.get('peak_ofr',0):.3f}"
        )
        try:
            client = mlflow.tracking.MlflowClient()
            result = mlflow.register_model(
                f"runs:/{run_id}/model",
                name,
            )
            client.set_registered_model_alias(result.name, "Staging")
            client.update_registered_model(
                name=result.name,
                description=desc,
            )
            print(f"  -> Registered to Model Registry: {name} (Staging)")
        except Exception as e:
            print(f"  -> Registry skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="TwinXGBBoosted hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tabular-csv", type=str, default=str(TABULAR_PATH))
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--input", type=str, default=str(RAW_PATH))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    print("=" * 60)
    print("TwinXGBBoosted — Optuna Tuning")
    print("=" * 60)

    if args.force_preprocess:
        print("Running preprocessing from raw...")
        from src.data_prep import aggregate_daily_from_chunks, build_full_panel
        from src.features import build_tabular_dataframe
        daily = aggregate_daily_from_chunks(Path(args.input))
        panel = build_full_panel(daily)
        del daily; gc.collect()
        panel = build_tabular_dataframe(panel)
    else:
        panel = load_data(args.tabular_csv)

    # Compute peak days on full data
    daily_total = panel.groupby(DATE_COL)[TARGET_COL].sum().sort_values(ascending=False)
    peak_n = max(1, int(len(daily_total) * 0.05))
    peak_set = set(daily_total.head(peak_n).index)
    panel["is_peak_day"] = panel[DATE_COL].isin(peak_set).astype("uint8")

    splits = make_splits(panel[DATE_COL])
    print(f"Splits: {len(splits)}, panel: {panel.shape}")

    # Filter low-obs items
    obs = panel.groupby(GROUP_COLS).size()
    panel = panel[panel.set_index(GROUP_COLS).index.isin(obs[obs >= 60].index)].copy()
    print(f"After filter: {panel.shape}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"twin_xgb_boosted_optuna_{args.trials}t_{args.seed}s"
    with mlflow.start_run(run_name=run_name) as run:
        study = run_optuna(panel, splits, args.trials, args.seed)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cls", study.best_value)
        mlflow.set_tag("model", "twin-xgb-boosted")
        mlflow.set_tag("method", "optuna")
        mlflow.set_tag("n_trials", args.trials)

        print(f"\nBest params: {study.best_params}")
        print(f"Best CV CLS: {study.best_value:.2f}")

        # Retrain best model on full training set for logging
        best = study.best_params
        rp_best = dict(REG_PARAMS,
                       n_estimators=best["n_estimators"],
                       max_depth=best["max_depth"],
                       learning_rate=best["learning_rate"])
        model = TwinXGBBoosted(
            reg_params=rp_best,
            alpha_under=best["alpha_under"],
            quantile_q_peak=best["quantile_q_peak"],
            sample_weight_alpha=best["sample_weight_alpha"],
            sample_weight_cap=best["sample_weight_cap"],
            holiday_boost=best["holiday_boost"],
            peak_days_boost=best["peak_days_boost"],
        )
        # Fit on full panel for registry check
        x_all = panel[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
        y_all = panel[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        tkeys_all = list(zip(panel["stock_code"].to_numpy(), panel["country"].to_numpy()))
        seg_all = panel.groupby(GROUP_COLS)[TARGET_COL].sum().sort_values(ascending=False)
        tk_all = set(seg_all.head(max(1, int(len(seg_all) * 0.05))).index)
        model.fit(
            x_all, y_all,
            panel["is_hari_besar"].to_numpy(dtype=float),
            panel["is_pre_hari_besar"].to_numpy(dtype=float),
            panel["is_peak_day"].to_numpy(dtype=float),
            tk_all, tkeys_all,
        )

        # Quick eval on last fold
        last_split = splits[-1]
        vm = (panel[DATE_COL] >= last_split[1]) & (panel[DATE_COL] <= last_split[2])
        df_v = panel.loc[vm]
        x_v = df_v[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_v = df_v[TARGET_COL].to_numpy()
        p_v = df_v[PRICE_COL].to_numpy()
        vk_v = list(zip(df_v["stock_code"].to_numpy(), df_v["country"].to_numpy()))
        vp_v = df_v["is_peak_day"].to_numpy(dtype=bool)
        vh_v = (df_v["is_hari_besar"].to_numpy(dtype=bool) | df_v["is_pre_hari_besar"].to_numpy(dtype=bool))
        _, final_m, _ = model.predict_with_threshold(x_v, y_v, p_v, vk_v, vp_v, vh_v)

        mlflow.log_metrics({
            "val_mae": final_m["mae"],
            "val_rmse": final_m["rmse"],
            "val_cls": final_m["cls"],
            "val_ofr": final_m["ofr"],
        })

        promote_if_eligible(final_m, best, run.info.run_id)

    print(f"\nMLflow Run ID: {run.info.run_id}")
    print(f"Experiment: {MLFLOW_EXPERIMENT}")


if __name__ == "__main__":
    main()
