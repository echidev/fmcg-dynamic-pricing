#!/usr/bin/env python3
"""Pipeline runner — end‑to‑end: data prep → features → tune → train.

Usage:
    python run_pipeline.py                         # full pipeline (100 trials)
    python run_pipeline.py --trials 200            # custom trials
    python run_pipeline.py --skip-tune             # tanpa tuning
    python run_pipeline.py --skip-train            # tanpa train final
    python run_pipeline.py --dry-run               # tanpa eksekusi
    python run_pipeline.py --cleanup               # hapus artifacts sementara

Flow:
    1. src.data_prep    — raw CSV → daily panel
    2. src.features     — panel → tabular (52 fitur)
    3. src.tune         — Optuna → MLflow
    4. src.train        — train final → models/
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def log(msg: str):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def run(cmd: list[str], desc: str, dry_run: bool = False) -> bool:
    """Run a command and return success status."""
    log(f"[STEP] {desc}")
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        print("  [DRY-RUN] skipped")
        return True
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (code {result.returncode})"
    print(f"  [{status}] {elapsed:.1f}s")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="FMCG Demand Forecasting — End-to-End Pipeline"
    )
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials")
    parser.add_argument("--skip-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--skip-train", action="store_true", help="Skip final training")
    parser.add_argument("--dry-run", action="store_true", help="Show steps without executing")
    parser.add_argument("--cleanup", action="store_true", help="Remove temporary artifacts")
    args = parser.parse_args()

    print(f"""
{'█' * 60}
  FMCG Demand Forecasting Pipeline
  Trials: {args.trials}  |  Tune: {'skip' if args.skip_tune else 'yes'}
  Train: {'skip' if args.skip_train else 'yes'}  |  Dry-run: {args.dry_run}
{'█' * 60}
""")

    # ── Step 1: Data Preparation ──
    if not run(
        [sys.executable, "-m", "src.data_prep",
         "--input", str(ROOT / "data/raw/online_retail.csv"),
         "--output", str(ROOT / "data/transform/online_retail_daily_product.csv")],
        "Data Preparation",
        args.dry_run,
    ):
        sys.exit(1)

    # ── Step 2: Feature Engineering ──
    if not run(
        [sys.executable, "-m", "src.features",
         "--input", str(ROOT / "data/raw/online_retail.csv"),
         "--output-tabular", str(ROOT / "data/transform/online_retail_daily_product_tabular.csv")],
        "Feature Engineering (52 fitur)",
        args.dry_run,
    ):
        sys.exit(1)

    # ── Step 3: Hyperparameter Tuning ──
    if not args.skip_tune:
        ok = run(
            [sys.executable, "-m", "src.tune",
             "--trials", str(args.trials),
             "--tabular-csv", str(ROOT / "data/transform/online_retail_daily_product_tabular.csv")],
            "Hyperparameter Tuning (Optuna)",
            args.dry_run,
        )

    # ── Step 4: Production Training ──
    if not args.skip_train:
        # Auto-detect best run from MLflow
        if args.dry_run:
            run_id = "<auto-detected>"
        else:
            run_id = _get_best_mlflow_run()

        if run_id is None:
            print("  [!] No best run found. Training will use default parameters.")
            run_id = ""

        ok = run(
            [sys.executable, "-m", "src.train",
             "--run-id", str(run_id) if run_id else "",
             "--tabular-csv", str(ROOT / "data/transform/online_retail_daily_product_tabular.csv"),
             "--output-dir", str(ROOT / "models/twin_xgb_boosted")],
            f"Production Training (run_id={run_id or 'default'})",
            args.dry_run,
        )

    # ── Cleanup ──
    if args.cleanup:
        log("[STEP] Cleanup temporary artifacts")
        for p in Path(ROOT / "data/transform").glob("*.parquet"):
            p.unlink()
            print(f"  Removed {p.name}")

    print(f"\n{'█' * 60}")
    print("  Pipeline complete!")
    print(f"{'█' * 60}\n")


def _get_best_mlflow_run() -> str | None:
    """Find the best MLflow run by CLS metric."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)

        if experiment is None:
            return None

        # Search runs sorted by metric 'best_cls'
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.best_cls ASC"],
            max_results=1,
        )
        if runs.empty:
            return None
        return runs.iloc[0]["run_id"]
    except Exception:
        return None


if __name__ == "__main__":
    main()
