"""Data preparation — chunked cleaning + daily aggregation (stock_code x country).

Pipeline tahap 1: raw CSV → daily panel dengan zero‑sale days.
"""

import argparse
import gc
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CHUNK_SIZE,
    DAILY_PATH,
    DTYPES,
    GROUP_COLS,
    NON_PRODUCT_CODES,
    RAW_PATH,
    TARGET_COL,
    USECOLS,
)


def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan satu chunk transaksi."""
    df = chunk.rename(
        columns={
            "Invoice": "invoice", "StockCode": "stock_code",
            "Quantity": "quantity", "InvoiceDate": "invoice_date",
            "Price": "price", "Country": "country",
        }
    ).copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("float32")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float32")
    df["stock_code"] = df["stock_code"].astype("string").str.strip().str.upper()
    df["invoice"] = df["invoice"].astype("string").str.strip()
    df["country"] = df["country"].astype("string").str.strip()
    df = df.drop_duplicates()
    df = df[df["invoice_date"].notna()]
    df = df[df["stock_code"].notna() & (df["stock_code"] != "")]
    df = df[df["invoice"].notna() & (df["invoice"] != "")]
    df = df[~df["invoice"].str.startswith("C", na=False)]
    df = df[~df["stock_code"].isin(NON_PRODUCT_CODES)]
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    df["date"] = df["invoice_date"].dt.normalize()
    df["revenue"] = df["quantity"] * df["price"]
    df["stock_code"] = df["stock_code"].astype("category")
    df["country"] = df["country"].astype("category")
    df["invoice"] = df["invoice"].astype("category")
    return df[["stock_code", "country", "date", "invoice", "quantity", "price", "revenue"]]


def aggregate_daily_from_chunks(path: Path) -> pd.DataFrame:
    """Baca CSV per‑chunk, agregat harian per (stock_code, country)."""
    parts = []
    max_parts = 25
    for chunk in pd.read_csv(
        path, usecols=USECOLS, dtype=DTYPES,
        chunksize=CHUNK_SIZE, low_memory=False,
    ):
        cleaned = preprocess_chunk(chunk)
        daily = cleaned.groupby(
            GROUP_COLS + ["date"], as_index=False, observed=True
        ).agg(
            demand_qty=("quantity", "sum"),
            revenue=("revenue", "sum"),
            num_invoices=("invoice", "nunique"),
            price_mean=("price", "mean"),
        )
        parts.append(daily)
        if len(parts) >= max_parts:
            partial = pd.concat(parts, ignore_index=True)
            parts = [
                partial.groupby(GROUP_COLS + ["date"], as_index=False, observed=True)
                .agg(demand_qty=("demand_qty", "sum"), revenue=("revenue", "sum"),
                     num_invoices=("num_invoices", "sum"), price_mean=("price_mean", "mean"))
            ]
            del partial
            gc.collect()
        del chunk, cleaned, daily
        gc.collect()
    daily_all = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    daily_all = daily_all.groupby(GROUP_COLS + ["date"], as_index=False, observed=True).agg(
        demand_qty=("demand_qty", "sum"), revenue=("revenue", "sum"),
        num_invoices=("num_invoices", "sum"), price_mean=("price_mean", "mean"),
    )
    daily_all["avg_price"] = np.where(
        daily_all["demand_qty"] > 0,
        daily_all["revenue"] / daily_all["demand_qty"],
        daily_all["price_mean"],
    )
    daily_all = daily_all.drop(columns=["price_mean"])
    for c in GROUP_COLS:
        daily_all[c] = daily_all[c].astype("category")
    for c in [TARGET_COL, "revenue", "avg_price", "num_invoices"]:
        daily_all[c] = daily_all[c].astype("float32")
    return daily_all


def build_full_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Isi zero‑sale days dengan resample harian per item."""
    df = df.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
    def _resample(group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index("date").asfreq("D")
        for c in GROUP_COLS:
            group[c] = group[c].iloc[0]
        return group.reset_index()

    panel = df.groupby(GROUP_COLS, group_keys=False, sort=False).apply(_resample)
    panel = panel.reset_index(drop=True)
    for c in [TARGET_COL, "revenue", "num_invoices"]:
        panel[c] = panel[c].fillna(0)
    panel["avg_price"] = panel["avg_price"].astype("float32")
    panel["avg_price"] = (
        panel.groupby(GROUP_COLS, sort=False)["avg_price"]
        .ffill().bfill().fillna(0)
    )
    for c in GROUP_COLS:
        panel[c] = panel[c].astype("category")
    for c in [TARGET_COL, "revenue", "avg_price", "num_invoices"]:
        panel[c] = panel[c].astype("float32")
    return panel


def process_bronze(input_path: Path = None, output_path: Path = None) -> pd.DataFrame:
    """Bronze layer: raw CSV → daily panel → silver (Parquet).

    Membaca CSV dari data/bronze/, membersihkan, membuat daily panel
    dengan zero-sale days, dan menulis ke data/silver/.
    """
    logger = _setup_logger()
    input_path = input_path or RAW_PATH
    output_path = output_path or DAILY_PATH

    logger.info("Bronze: membaca raw dari %s", input_path)
    daily = aggregate_daily_from_chunks(input_path)
    panel = build_full_panel(daily)
    del daily
    gc.collect()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    logger.info("Silver: panel ditulis ke %s rows=%s cols=%s",
                output_path, panel.shape[0], panel.shape[1])

    manifest = {
        "stage": "bronze",
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(panel.shape[0]),
        "cols": int(panel.shape[1]),
        "date_min": str(panel["date"].min()) if "date" in panel.columns else None,
        "date_max": str(panel["date"].max()) if "date" in panel.columns else None,
        "created_utc": datetime.utcnow().isoformat(),
    }
    _write_manifest(Path("artifacts/manifests"), manifest)
    return panel


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("data_prep")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(manifest_dir: Path, payload: dict) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = manifest_dir / f"data_prep_manifest_{ts}.json"
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    return out


def run_data_prep(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Eksekusi pipeline: bronze → silver (daily panel → parquet)."""
    return process_bronze(input_path=input_path, output_path=output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data preparation: chunked cleaning + daily panel"
    )
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=DAILY_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_data_prep(args.input, args.output)
