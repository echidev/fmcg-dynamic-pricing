"""Hyperparameter tuning — Optuna + MLflow untuk DecoupledActuarialXGB.

Usage:
    python -m src.tune --trials 30
    python -m src.tune --trials 30 --force-preprocess
"""

import argparse
import gc
import logging

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner

from src.config import (
    CUTOFF_DATE,
    DATASET_VERSION,
    DATE_COL,
    FEATURE_COLS,
    FEATURE_SET_ID,
    GROUP_COLS,
    HOLIDAY_INTENSITY_CAP,
    MIN_OBS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_PARAMS,
    PEAK_DAYS_PCT,
    PRICE_COL,
    PROMOTION_THRESHOLDS,
    RANDOM_STATE,
    TABULAR_PATH,
    TARGET_COL,
    get_git_commit,
    log_env_snapshot,
    set_global_seed,
)
from src.features import compute_holiday_intensity_map
from src.metrics import evaluate_prediction
from src.model import DecoupledActuarialXGB


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("tune")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def make_splits(dates, horizon=30, n_splits=3, min_train=180, cutoff_date=None):
    """Time-series expanding window splits."""
    dates = np.array(sorted(pd.to_datetime(dates).unique()))
    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        dates = dates[dates <= cutoff]
        if len(dates) == 0:
            raise ValueError(f"cutoff_date {cutoff_date} menghasilkan 0 unique dates")
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


def load_data(tabular_path):
    """Load tabular Parquet."""
    logger = _setup_logger()
    logger.info("Loading tabular: %s", tabular_path)
    panel = pd.read_parquet(tabular_path)
    for c in FEATURE_COLS:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    return panel


def objective(trial, panel, splits, feature_cols):
    """Optuna trial untuk DecoupledActuarialXGB.

    Dual-dimension search:
      - ML: learning_rate, max_depth, subsample, n_estimators
      - Actuarial: q_target, shortage_margin_multiplier

    Constrained optimization: OFR < 0.80 dikenakan penalty 1e6.
    """
    # ML params
    n_estimators = trial.suggest_int("n_estimators", 200, 600, step=100)
    max_depth = trial.suggest_int("max_depth", 3, 7)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    subsample = trial.suggest_float("subsample", 0.7, 1.0)

    # Actuarial params
    q_target = trial.suggest_float("q_target", 0.85, 0.98)
    smm = trial.suggest_float("shortage_margin_multiplier", 1.0, 3.0)

    model_params = dict(
        MODEL_PARAMS,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
    )

    cls_scores = []
    ofr_scores = []

    for fold_idx, (train_end, val_start, val_end) in enumerate(splits, start=1):
        train_mask = panel[DATE_COL] <= train_end
        val_mask = (panel[DATE_COL] >= val_start) & (panel[DATE_COL] <= val_end)

        df_tr = panel.loc[train_mask].reset_index(drop=True)
        df_vl = panel.loc[val_mask].reset_index(drop=True)

        # Compute is_peak_day from training data only (leakage-safe)
        daily_tr = df_tr.groupby(DATE_COL)[TARGET_COL].sum()
        n_peak = max(1, int(len(daily_tr) * PEAK_DAYS_PCT))
        peak_set = set(daily_tr.sort_values(ascending=False).head(n_peak).index)
        df_tr["is_peak_day"] = df_tr[DATE_COL].isin(peak_set).astype("uint8")
        df_vl["is_peak_day"] = df_vl[DATE_COL].isin(peak_set).astype("uint8")

        # Compute holiday_intensity from training data only (leakage-safe)
        intensity_map = compute_holiday_intensity_map(df_tr)
        df_tr["holiday_intensity"] = df_tr["country_code"].map(intensity_map).fillna(1.0).clip(upper=HOLIDAY_INTENSITY_CAP).astype("float32")
        df_vl["holiday_intensity"] = df_vl["country_code"].map(intensity_map).fillna(1.0).clip(upper=HOLIDAY_INTENSITY_CAP).astype("float32")

        X_tr = df_tr[feature_cols]
        y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        X_vl = df_vl[feature_cols]
        y_vl = df_vl[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        price_vl = df_vl[PRICE_COL].to_numpy(dtype=np.float32, copy=False)
        val_keys = list(zip(df_vl["stock_code"].to_numpy(), df_vl["country"].to_numpy()))

        try:
            model = DecoupledActuarialXGB(
                model_params=model_params,
                quantile_q_target=q_target,
                shortage_margin_multiplier=smm,
            )
            model.fit(X_tr, y_tr, feature_cols=feature_cols)

            pred = model.predict(X_vl, price_vl, val_keys)
            metrics = evaluate_prediction(y_vl, pred, price_vl)

            cls_scores.append(metrics["cls"])
            ofr_scores.append(metrics["ofr"])

            # Report intermediate value for pruning
            trial.report(float(metrics["cls"]), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        except Exception:
            return 1e9

        finally:
            del df_tr, df_vl, X_tr, y_tr, X_vl, y_vl, price_vl
            gc.collect()

    if not cls_scores:
        return 1e9

    avg_cls = float(np.mean(cls_scores))
    avg_ofr = float(np.mean(ofr_scores))

    # Constrained optimization: OFR < 0.80 = catastrophic
    if avg_ofr < 0.80:
        return avg_cls + 1_000_000.0

    return avg_cls


def run_optuna(panel, splits, feature_cols, n_trials=100, seed=42):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda t: objective(t, panel, splits, FEATURE_COLS),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study


def promote_if_eligible(metrics, model_params, run_id):
    """Daftarkan ke model registry jika threshold bisnis terpenuhi."""
    logger = _setup_logger()
    th = PROMOTION_THRESHOLDS
    ofr_ok = metrics.get("ofr", 0) >= th["ofr_min"]
    cls_ok = metrics.get("cls", 1e9) <= th["cls_max"]
    if ofr_ok and cls_ok:
        name = "decoupled_actuarial_xgb"
        desc = (
            f"OFR={metrics['ofr']:.3f} CLS={metrics['cls']:.0f} "
            f"params={model_params}"
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
            logger.info("Registered to Model Registry: %s (Staging)", name)
        except Exception as e:
            logger.warning("Registry skipped: %s", e)


def main():
    parser = argparse.ArgumentParser(description="DecoupledActuarialXGB hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tabular-parquet", type=str, default=str(TABULAR_PATH))
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--input", type=str, default="data/bronze/online_retail.csv")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    set_global_seed(args.seed)

    logger = _setup_logger()
    logger.info("DecoupledActuarialXGB — Optuna Tuning")

    if args.force_preprocess:
        logger.info("Menjalankan full ETL: bronze -> silver -> gold...")
        from src.data_prep import process_bronze
        from src.features import process_silver
        process_bronze()
        panel = process_silver()
    else:
        panel = load_data(args.tabular_parquet)

    splits = make_splits(panel[DATE_COL], cutoff_date=CUTOFF_DATE)
    logger.info("Splits=%s panel=%s cutoff=%s", len(splits), panel.shape, CUTOFF_DATE)

    # Filter low-obs items
    obs = panel.groupby(GROUP_COLS).size()
    panel = panel[panel.set_index(GROUP_COLS).index.isin(obs[obs >= MIN_OBS].index)].copy()
    logger.info("After MIN_OBS filter: %s", panel.shape)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"decoupled_actuarial_optuna_{args.trials}t_{args.seed}s"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("git_commit", get_git_commit())
        mlflow.log_param("dataset_version", DATASET_VERSION)
        mlflow.log_param("feature_set_id", FEATURE_SET_ID)
        mlflow.log_param("cutoff_date", CUTOFF_DATE)
        log_env_snapshot(mlflow)

        study = run_optuna(panel, splits, FEATURE_COLS, args.trials, args.seed)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_cls", study.best_value)
        mlflow.set_tag("model", "decoupled-actuarial-xgb")
        mlflow.set_tag("method", "optuna")
        mlflow.set_tag("n_trials", args.trials)

        logger.info("Best params: %s", study.best_params)
        logger.info("Best CV CLS: %.2f", study.best_value)

        # Retrain best model on full set
        bp = study.best_params
        model_params_best = dict(
            MODEL_PARAMS,
            n_estimators=bp["n_estimators"],
            max_depth=bp["max_depth"],
            learning_rate=bp["learning_rate"],
            subsample=bp["subsample"],
        )

        model = DecoupledActuarialXGB(
            model_params=model_params_best,
            quantile_q_target=bp["q_target"],
            shortage_margin_multiplier=bp["shortage_margin_multiplier"],
        )

        x_all = panel[FEATURE_COLS]
        y_all = panel[TARGET_COL].to_numpy(dtype=np.float32, copy=False)
        model.fit(x_all, y_all, feature_cols=FEATURE_COLS)

        # Evaluate on last fold
        last_split = splits[-1]
        vm = (panel[DATE_COL] >= last_split[1]) & (panel[DATE_COL] <= last_split[2])
        df_v = panel.loc[vm]
        x_v = df_v[FEATURE_COLS]
        y_v = df_v[TARGET_COL].to_numpy(dtype=np.float32)
        p_v = df_v[PRICE_COL].to_numpy()
        k_v = list(zip(df_v["stock_code"].to_numpy(), df_v["country"].to_numpy()))
        bm = float(np.mean(np.abs(y_v)))
        final_m = model.evaluate(x_v, y_v, p_v, k_v, baseline_mae=bm)

        # MLflow signature & logging
        try:
            from mlflow.models import infer_signature
            signature = infer_signature(x_all.to_numpy()[:100], y_all[:100])
            mlflow.xgboost.log_model(
                model.model_mean_,
                artifact_path="mean_model",
                signature=signature,
            )
        except Exception:
            pass

        mlflow.log_metrics({
            "val_mae": final_m["mae"],
            "val_rmse": final_m["rmse"],
            "val_smape": final_m["smape"],
            "val_cls": final_m["cls"],
            "val_ofr": final_m["ofr"],
            "val_oos_rate": final_m["oos_rate"],
        })
        if final_m.get("fva") is not None:
            mlflow.log_metric("val_fva", final_m["fva"])

        promote_if_eligible(final_m, dict(bp), run.info.run_id)

    logger.info("MLflow Run ID: %s", run.info.run_id)
    logger.info("Experiment: %s", MLFLOW_EXPERIMENT)


if __name__ == "__main__":
    main()
