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
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log(msg: str):
    logger = _setup_logger()
    logger.info("%s", msg)


def run(cmd: list[str], desc: str, dry_run: bool = False) -> bool:
    """Run a command and return success status."""
    logger = _setup_logger()
    log(f"[STEP] {desc}")
    logger.info("$ %s", " ".join(cmd))
    if dry_run:
        logger.info("[DRY-RUN] skipped")
        return True
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (code {result.returncode})"
    logger.info("[%s] %.1fs", status, elapsed)
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

    logger = _setup_logger()
    logger.info("FMCG Demand Forecasting Pipeline")
    logger.info("Trials=%s Tune=%s Train=%s Dry-run=%s", args.trials,
                "skip" if args.skip_tune else "yes",
                "skip" if args.skip_train else "yes",
                args.dry_run)

    # ── Step 1: Bronze → Silver (raw CSV → daily panel) ──
    if not run(
         [sys.executable, "-m", "src.data_prep",
          "--input", str(ROOT / "data/bronze/online_retail.csv"),
          "--output", str(ROOT / "data/silver/online_retail_daily_product.parquet")],
        "Bronze → Silver (data prep)",
        args.dry_run,
    ):
        sys.exit(1)

    # ── Step 2: Silver → Gold (daily panel → 52 features) ──
    if not run(
         [sys.executable, "-m", "src.features",
          "--input", str(ROOT / "data/silver/online_retail_daily_product.parquet"),
          "--output-tabular", str(ROOT / "data/gold/online_retail_daily_product_tabular.parquet")],
        "Silver → Gold (feature engineering)",
        args.dry_run,
    ):
        sys.exit(1)

    # ── Step 3: Hyperparameter Tuning ──
    if not args.skip_tune:
        run(
             [sys.executable, "-m", "src.tune",
              "--trials", str(args.trials),
              "--tabular-parquet", str(ROOT / "data/gold/online_retail_daily_product_tabular.parquet")],
            "Hyperparameter Tuning (Optuna)",
            args.dry_run,
        )

    # ── Step 4: Production Training ──
    if not args.skip_train:
        run(
            [sys.executable, "-m", "src.train",
             "--tabular-parquet", str(ROOT / "data/gold/online_retail_daily_product_tabular.parquet"),
             "--output-dir", str(ROOT / "models/decoupled_actuarial_xgb")],
            "Production Training (best params from experiment)",
            args.dry_run,
        )

    # ── Cleanup ──
    if args.cleanup:
        log("[STEP] Cleanup temporary artifacts (tidak ada — data dikelola DVC)")
        logger.info("Data artifacts dikelola oleh DVC. Tidak ada cleanup otomatis.")

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
