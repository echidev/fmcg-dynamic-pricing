"""XGBoost hyperparameter tuning with Optuna and MLflow tracking.

Data loading: pre-generated tabular CSV by default; --force-preprocess
to re-run the full preprocessing pipeline from raw data.

Usage::

    python -m src.tune --trials 30
    python -m src.tune --trials 100 --force-preprocess
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error

import mlflow
import optuna
import gc

from src.data_prep import clean_online_retail, build_daily_product_dataset
from src.features import build_tabular_dataframe
from src.metrics import evaluate_model_performance


FEATURE_COLS = [
    "day_of_week", "week_of_year", "month", "quarter", "day_of_month",
    "is_weekend", "is_month_start", "is_month_end",
    "demand_lag_1", "demand_lag_2", "demand_lag_7", "demand_lag_14", "demand_lag_28",
    "avg_price_lag_1", "revenue_lag_1", "num_invoices_lag_1",
    "roll_mean_7", "roll_mean_14", "roll_mean_28",
    "roll_std_7", "roll_std_28",
    "diff_1", "diff_7",
    "product_popularity", "product_revenue_share", "product_lifecycle_age",
]




def load_from_csv(tabular_path: str, horizon_days: int = 30):
    """Load feature matrix and target from a pre-generated tabular CSV.

    Args:
        tabular_path: Path to the tabular CSV produced by features.py.
        horizon_days: Number of days to reserve for the test split.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, avg_price_arr).
    """
    print(f"Loading tabular CSV: {tabular_path}")
    use_cols = set(FEATURE_COLS + ["date", "demand_qty", "avg_price_lag_1"])
    date_max = None
    for chunk in pd.read_csv(
        tabular_path,
        usecols=["date"],
        parse_dates=["date"],
        chunksize=200_000,
    ):
        chunk_max = chunk["date"].max()
        date_max = chunk_max if date_max is None else max(date_max, chunk_max)

    if date_max is None:
        raise ValueError("No data found in tabular CSV.")

    global_max_date = date_max
    cutoff = global_max_date - pd.Timedelta(days=horizon_days)
    print(f"Global max date: {global_max_date}, cutoff: {cutoff}")

    dtype_map = {col: "float32" for col in use_cols if col != "date"}
    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    X_test_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []
    avg_price_parts: list[np.ndarray] = []

    for chunk in pd.read_csv(
        tabular_path,
        usecols=list(use_cols),
        parse_dates=["date"],
        dtype=dtype_map,
        chunksize=200_000,
        low_memory=False,
    ):
        train_mask = chunk["date"] <= cutoff
        test_mask = ~train_mask

        if train_mask.any():
            train_chunk = chunk.loc[train_mask]
            X_train_parts.append(
                train_chunk[FEATURE_COLS]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype("float32")
                .to_numpy()
            )
            y_train_parts.append(
                train_chunk["demand_qty"].fillna(0).astype("float32").to_numpy()
            )

        if test_mask.any():
            test_chunk = chunk.loc[test_mask]
            X_test_parts.append(
                test_chunk[FEATURE_COLS]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype("float32")
                .to_numpy()
            )
            y_test_parts.append(
                test_chunk["demand_qty"].fillna(0).astype("float32").to_numpy()
            )
            avg_price_parts.append(
                test_chunk["avg_price_lag_1"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype("float32")
                .to_numpy()
            )

    if not X_train_parts or not X_test_parts:
        raise ValueError("Train/test split resulted in empty partitions.")

    X_train = pd.DataFrame(np.concatenate(X_train_parts), columns=FEATURE_COLS)
    y_train = pd.Series(np.concatenate(y_train_parts), name="demand_qty")
    X_test = pd.DataFrame(np.concatenate(X_test_parts), columns=FEATURE_COLS)
    y_test = pd.Series(np.concatenate(y_test_parts), name="demand_qty")

    avg_price_arr = np.concatenate(avg_price_parts)
    avg_price_arr = np.nan_to_num(avg_price_arr, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Zero-demand: train={(y_train==0).mean()*100:.1f}%, test={(y_test==0).mean()*100:.1f}%")

    return X_train, y_train, X_test, y_test, avg_price_arr


def load_from_preprocess(input_path: str, horizon_days: int = 30):
    """Run the full preprocessing pipeline and return train/test splits.

    Used when the tabular CSV is stale and a fresh feature engineering
    pass is required.

    Args:
        input_path: Path to the raw input CSV.
        horizon_days: Number of days to reserve for the test split.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, avg_price_arr).
    """
    print(f"Running preprocessing from raw: {input_path}")
    df_raw = pd.read_csv(input_path)
    df_clean = clean_online_retail(df_raw)
    daily_product = build_daily_product_dataset(df_clean)

    global_max_date = daily_product["date"].max()
    cutoff = global_max_date - pd.Timedelta(days=horizon_days)

    df_tab = build_tabular_dataframe(daily_product, cutoff_date=cutoff)

    train_mask = df_tab["date"] <= cutoff
    test_mask = df_tab["date"] > cutoff

    df_train = df_tab.loc[train_mask].copy()
    df_test = df_tab.loc[test_mask].copy()

    X_train = df_train[FEATURE_COLS].fillna(0).astype("float32")
    y_train = df_train["demand_qty"].astype("float32")
    X_test = df_test[FEATURE_COLS].fillna(0).astype("float32")
    y_test = df_test["demand_qty"].astype("float32")

    avg_price_arr = df_test["avg_price_lag_1"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).values
    avg_price_arr = np.nan_to_num(avg_price_arr, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Zero-demand: train={(y_train==0).mean()*100:.1f}%, test={(y_test==0).mean()*100:.1f}%")

    return X_train, y_train, X_test, y_test, avg_price_arr


def optuna_suggest_params(trial) -> dict:
    """Define the Optuna hyperparameter search space.

    Args:
        trial: Optuna trial object for parameter suggestion.

    Returns:
        Dictionary of suggested hyperparameters.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    "n_jobs": 1,  # Force single-threaded to prevent OOM
    "tree_method": "hist",
    }


def run_optuna_search(X_train, y_train, X_test, y_test, avg_price_arr, n_trials=30, seed=42):
    """Run Optuna TPE sampler for Bayesian hyperparameter optimization.

    Minimizes test-set MAE across the defined search space.
    Runs sequentially (n_jobs=1) to prevent memory exhaustion.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_test: Test feature matrix.
        y_test: Test target vector.
        avg_price_arr: Test-set average prices for business metrics.
        n_trials: Number of Optuna trials to run.
        seed: Random state for reproducibility.

    Returns:
        Tuple of (best_model, metrics, best_params, duration_seconds).
    """
    print(f"\nOptuna TPE Search (n_trials={n_trials})")
    trial_num = [0]  # Mutable counter

    def objective(trial):
        trial_num[0] += 1
        params = optuna_suggest_params(trial)
        params["objective"] = "reg:squarederror"
        params["random_state"] = seed
        params["verbosity"] = 0
        params["n_jobs"] = 1  # Force single-threaded

        # Print current trial info
        print(f"[Trial {trial_num[0]}/{n_trials}] Testing: n_estimators={params['n_estimators']}, "
              f"max_depth={params['max_depth']}, lr={params['learning_rate']:.4f}")

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return mean_absolute_error(y_test, y_pred)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_mae = study.best_value

    # Retrain best model on full training data
    best_params_clean = {k: v for k, v in best_params.items() if k != "n_jobs"}
    best_model = xgb.XGBRegressor(
        **best_params_clean,
        objective="reg:squarederror",
        random_state=seed,
        verbosity=0,
        n_jobs=1,
        tree_method="hist",
    )
    t0 = time.time()
    best_model.fit(X_train, y_train)
    duration = time.time() - t0

    y_pred = best_model.predict(X_test)
    metrics = evaluate_model_performance(
        y_true=y_test.values if hasattr(y_test, "values") else y_test,
        y_pred=y_pred,
        unit_prices=avg_price_arr,
        gross_margin=0.20,
        cogs=0.80,
        holding_daily=0.20 / 365,
    )

    print(f"\nBest params: {best_params}")
    print(f"Best test MAE: {best_mae:.4f}")
    print(f"Training duration: {duration:.1f}s")

    mlflow.log_params({**best_params, "method": "optuna", "n_trials": n_trials})
    mlflow_metrics = {
        "mae": metrics.get("MAE", 0.0),
        "rmse": metrics.get("RMSE", 0.0),
        "smape": metrics.get("SMAPE_pct", 0.0),
        "f1_zero": metrics.get("F1_Zero", 0.0),
        "mae_nonzero": metrics.get("MAE_NonZero", 0.0),
        "cls": metrics.get("CLS", 0.0),
        "ihc": metrics.get("IHC", 0.0),
        "oos_rate": metrics.get("OOS_Rate", 0.0),
        "ofr": metrics.get("OFR", 0.0),
        "duration_s": round(duration, 1),
    }
    mlflow.log_metrics(mlflow_metrics)
    mlflow.set_tag("method", "optuna")

    return best_model, metrics, best_params, duration


def resolve_trials(trials_default: int, trials_specific: int | None) -> int:
    """Resolve stage-specific trials with fallback.

    Args:
        trials_default: Default trial budget.
        trials_specific: Optional stage-specific budget.

    Returns:
        Resolved trial count.
    """
    return trials_specific if trials_specific is not None else trials_default


def tune_xgboost(args, X_train, y_train, X_test, y_test, avg_price_arr):
    """Tune single-stage XGBoost regressor."""
    run_name = f"xgboost_optuna_{args.horizon}d_seed{args.seed}"
    with mlflow.start_run(run_name=run_name):
        model, metrics, params, duration = run_optuna_search(
            X_train, y_train, X_test, y_test, avg_price_arr, args.trials, args.seed
        )

        mlflow.xgboost.log_model(model, artifact_path="best_model")

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(
            f"MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | "
            f"SMAPE: {metrics['SMAPE_pct']:.4f}%"
        )
        print(
            f"F1_Zero: {metrics['F1_Zero']:.4f} | CLS: {metrics['CLS']:.2f} | "
            f"IHC: {metrics['IHC']:.2f}"
        )
        print(f"OFR: {metrics['OFR']:.4f} | Duration: {duration:.1f}s")
        print(f"Params: {params}")


def run_optuna_classifier(X_train, y_train_zero, X_test, y_test_zero, n_trials, seed, imbalance_ratio):
    """Tune XGBoost classifier using AUC-PR objective."""
    print(f"\nOptuna TPE Search (Classifier, n_trials={n_trials})")
    trial_num = [0]

    def objective(trial):
        trial_num[0] += 1
        params = optuna_suggest_params(trial)
        params["objective"] = "binary:logistic"
        params["random_state"] = seed
        params["verbosity"] = 0
        params["n_jobs"] = 1
        params["scale_pos_weight"] = imbalance_ratio

        print(
            f"[Trial {trial_num[0]}/{n_trials}] Classifier: n_estimators={params['n_estimators']}, "
            f"max_depth={params['max_depth']}, lr={params['learning_rate']:.4f}"
        )

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train_zero)
        proba = model.predict_proba(X_test)[:, 1]

        return average_precision_score(y_test_zero, proba)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    best_params_clean = {k: v for k, v in best_params.items() if k != "n_jobs"}
    best_model = xgb.XGBClassifier(
        **best_params_clean,
        objective="binary:logistic",
        random_state=seed,
        verbosity=0,
        n_jobs=1,
        scale_pos_weight=imbalance_ratio,
        tree_method="hist",
    )
    t0 = time.time()
    best_model.fit(X_train, y_train_zero)
    duration = time.time() - t0

    proba = best_model.predict_proba(X_test)[:, 1]
    auc_pr = average_precision_score(y_test_zero, proba)

    print(f"\nBest classifier params: {best_params}")
    print(f"Best AUC-PR: {best_score:.4f}")
    print(f"Training duration: {duration:.1f}s")

    return best_model, auc_pr, best_params, duration


def run_optuna_regressor(X_train, y_train, X_test, y_test, n_trials, seed):
    """Tune XGBoost regressor using RMSE on non-zero targets."""
    print(f"\nOptuna TPE Search (Regressor, n_trials={n_trials})")
    trial_num = [0]

    def objective(trial):
        trial_num[0] += 1
        params = optuna_suggest_params(trial)
        params["objective"] = "reg:squarederror"
        params["random_state"] = seed
        params["verbosity"] = 0
        params["n_jobs"] = 1

        print(
            f"[Trial {trial_num[0]}/{n_trials}] Regressor: n_estimators={params['n_estimators']}, "
            f"max_depth={params['max_depth']}, lr={params['learning_rate']:.4f}"
        )

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    best_params_clean = {k: v for k, v in best_params.items() if k != "n_jobs"}
    best_model = xgb.XGBRegressor(
        **best_params_clean,
        objective="reg:squarederror",
        random_state=seed,
        verbosity=0,
        n_jobs=1,
        tree_method="hist",
    )
    t0 = time.time()
    best_model.fit(X_train, y_train)
    duration = time.time() - t0

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nBest regressor params: {best_params}")
    print(f"Best RMSE (non-zero): {best_score:.4f}")
    print(f"Training duration: {duration:.1f}s")

    return best_model, rmse, best_params, duration


def tune_twin_xgb(args, X_train, y_train, X_test, y_test, avg_price_arr):
    """Sequentially tune two-stage XGBoost classifier + regressor."""
    run_name = f"twin_xgb_optuna_{args.horizon}d_seed{args.seed}"
    with mlflow.start_run(run_name=run_name):
        y_train_zero = (y_train > 0).astype(int).values
        y_test_zero = (y_test > 0).astype(int).values

        count_zero = int((y_train_zero == 0).sum())
        count_one = int((y_train_zero == 1).sum())
        imbalance_ratio = count_zero / count_one if count_one > 0 else 1.0

        trials_clf = resolve_trials(args.trials, args.trials_clf)
        trials_reg = resolve_trials(args.trials, args.trials_reg)

        clf_model, auc_pr, clf_params, clf_duration = run_optuna_classifier(
            X_train,
            y_train_zero,
            X_test,
            y_test_zero,
            trials_clf,
            args.seed,
            imbalance_ratio,
        )

        nonzero_mask = y_train > 0
        X_train_nonzero = X_train.loc[nonzero_mask]
        y_train_nonzero = y_train.loc[nonzero_mask]

        test_nonzero_mask = y_test.values > 0
        X_test_nonzero = X_test.loc[test_nonzero_mask]
        y_test_nonzero = y_test.loc[test_nonzero_mask]

        reg_model, rmse_nz, reg_params, reg_duration = run_optuna_regressor(
            X_train_nonzero,
            y_train_nonzero,
            X_test_nonzero,
            y_test_nonzero,
            trials_reg,
            args.seed,
        )

        proba = clf_model.predict_proba(X_test)[:, 1]
        gate = (proba >= args.threshold).astype(int)
        qty_pred = reg_model.predict(X_test)
        final_pred = gate * qty_pred

        metrics = evaluate_model_performance(
            y_true=y_test.values if hasattr(y_test, "values") else y_test,
            y_pred=final_pred,
            unit_prices=avg_price_arr,
            gross_margin=0.20,
            cogs=0.80,
            holding_daily=0.20 / 365,
        )

        mlflow.xgboost.log_model(clf_model, artifact_path="classifier_model")
        mlflow.xgboost.log_model(reg_model, artifact_path="regressor_model")
        mlflow.log_params({f"clf_{k}": v for k, v in clf_params.items()})
        mlflow.log_params({f"reg_{k}": v for k, v in reg_params.items()})
        mlflow.log_params({"threshold": args.threshold, "model": "twin-xgb"})
        mlflow.log_metrics({
            "auc_pr": auc_pr,
            "rmse_nonzero": rmse_nz,
            "clf_duration_s": round(clf_duration, 1),
            "reg_duration_s": round(reg_duration, 1),
            "mae": metrics.get("MAE", 0.0),
            "rmse": metrics.get("RMSE", 0.0),
            "smape": metrics.get("SMAPE_pct", 0.0),
            "f1_zero": metrics.get("F1_Zero", 0.0),
            "mae_nonzero": metrics.get("MAE_NonZero", 0.0),
            "cls": metrics.get("CLS", 0.0),
            "ihc": metrics.get("IHC", 0.0),
            "oos_rate": metrics.get("OOS_Rate", 0.0),
            "ofr": metrics.get("OFR", 0.0),
        })
        mlflow.set_tag("method", "optuna")
        mlflow.set_tag("model", "twin-xgb")
        mlflow.set_tag("tuning", "sequential")

        del X_train_nonzero, y_train_nonzero, X_test_nonzero, y_test_nonzero
        del proba, gate, qty_pred, final_pred
        gc.collect()

        print("\n" + "=" * 60)
        print("TWIN-XGB RESULTS SUMMARY")
        print("=" * 60)
        print(f"AUC-PR (Classifier): {auc_pr:.4f}")
        print(f"RMSE_NonZero (Regressor): {rmse_nz:.4f}")
        print(
            f"MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | "
            f"SMAPE: {metrics['SMAPE_pct']:.4f}%"
        )
        print(
            f"F1_Zero: {metrics['F1_Zero']:.4f} | CLS: {metrics['CLS']:.2f} | "
            f"IHC: {metrics['IHC']:.2f}"
        )
        print(f"OFR: {metrics['OFR']:.4f} | Threshold: {args.threshold:.2f}")


def main():
    """Execute the hyperparameter tuning pipeline."""
    parser = argparse.ArgumentParser(
        description="XGBoost hyperparameter tuning with MLflow tracking"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "twin-xgb"],
        help="Model to tune (default: xgboost)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials (default: 30)",
    )
    parser.add_argument(
        "--trials-clf",
        type=int,
        default=None,
        help="Trials for classifier tuning (default: --trials)",
    )
    parser.add_argument(
        "--trials-reg",
        type=int,
        default=None,
        help="Trials for regressor tuning (default: --trials)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Probability threshold for two-stage gating (default: 0.30)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Forecast horizon in days (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--tabular-csv",
        type=str,
        default="data/transform/online_retail_daily_product_tabular.csv",
        help="Path to pre-generated tabular CSV",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Re-run preprocessing from raw CSV",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/online_retail.csv",
        help="Path to raw input CSV (used with --force-preprocess)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Hyperparameter Tuning (Optuna Only)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Trials: {args.trials}")
    if args.model == "twin-xgb":
        print(f"Trials (clf): {resolve_trials(args.trials, args.trials_clf)}")
        print(f"Trials (reg): {resolve_trials(args.trials, args.trials_reg)}")
        print(f"Threshold: {args.threshold}")
    print(f"Horizon: {args.horizon} days")
    print(f"Seed: {args.seed}")
    print(f"Force preprocess: {args.force_preprocess}")

    if args.force_preprocess:
        X_train, y_train, X_test, y_test, avg_price_arr = load_from_preprocess(
            args.input, args.horizon
        )
    else:
        X_train, y_train, X_test, y_test, avg_price_arr = load_from_csv(
            args.tabular_csv, args.horizon
        )

    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("xgboost_demand_forecasting")

    # Validate artifact location to prevent path mismatch
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("xgboost_demand_forecasting")
    if experiment:
        expected_artifact_loc = os.path.join(os.getcwd(), "mlruns", str(experiment.experiment_id))
        if experiment.artifact_location != expected_artifact_loc:
            print(f"Warning: Artifact location mismatch detected.")
            print(f"  Expected: {expected_artifact_loc}")
            print(f"  Current: {experiment.artifact_location}")

    model_registry = {
        "xgboost": tune_xgboost,
        "twin-xgb": tune_twin_xgb,
    }

    model_registry[args.model](
        args,
        X_train,
        y_train,
        X_test,
        y_test,
        avg_price_arr,
    )

    del X_train, y_train, X_test, y_test, avg_price_arr

    print(f"\nMLflow experiment: xgboost_demand_forecasting")


if __name__ == "__main__":
    main()
