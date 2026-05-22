"""Feature engineering — 52 fitur lengkap dari notebook boosted.

Tahap 2: daily panel → feature‑engineered tabular DataFrame.
Semua fitur dirancang leakage‑safe (shift(1) untuk lag/rolling).
"""

import argparse
import gc
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import holidays as hlib
import numpy as np
import pandas as pd

from src.config import (
    COUNTRY_TO_HOLIDAYS,
    DATE_COL,
    FEATURE_COLS,
    GROUP_COLS,
    HOLIDAY_INTENSITY_CAP,
    PEAK_DAYS_PCT,
    PRICE_COL,
    SILVER_DIR,
    TABULAR_PATH,
    TARGET_COL,
)

# ═══════════════════════════════════════════════════════════════
#  A. Calendar + Holiday (14 features)
# ═══════════════════════════════════════════════════════════════

def add_calendar_holiday_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Calendar (9) + holiday binary (2) = 11 features."""
    panel = panel.copy()
    panel["country_code"] = panel["country"].map(COUNTRY_TO_HOLIDAYS).astype("category")

    # Holiday dataframe per country
    years = sorted(panel["date"].dt.year.unique().tolist())
    supported = set(hlib.list_supported_countries())
    holiday_rows = []
    for code in sorted(panel["country_code"].dropna().astype(str).unique()):
        if code not in supported:
            continue
        hset = hlib.country_holidays(code, years=years)
        holiday_rows.append(pd.DataFrame({
            "country_code": code,
            "date": list(hset.keys()),
            "is_hari_besar": 1,
        }))
    hdf = (
        pd.concat(holiday_rows, ignore_index=True)
        if holiday_rows else pd.DataFrame(columns=["country_code", "date", "is_hari_besar"])
    )
    hdf["date"] = pd.to_datetime(hdf["date"], errors="coerce")

    panel = panel.merge(hdf, on=["country_code", "date"], how="left", copy=False)
    if "is_hari_besar" not in panel.columns:
        panel["is_hari_besar"] = 0
    panel["is_hari_besar"] = panel["is_hari_besar"].fillna(0.0).astype("float32").astype("uint8")

    # Pre‑holiday (±1–3 days)
    pre_rows = []
    if not hdf.empty:
        for offset in [1, 2, 3]:
            pre_rows.append(
                hdf.assign(
                    date=hdf["date"] - pd.Timedelta(days=offset), is_pre_hari_besar=1,
                )[["country_code", "date", "is_pre_hari_besar"]]
            )
    pre_df = (
        pd.concat(pre_rows, ignore_index=True)
        if pre_rows else pd.DataFrame(columns=["country_code", "date", "is_pre_hari_besar"])
    )
    pre_df = pre_df.drop_duplicates()
    panel = panel.merge(pre_df, on=["country_code", "date"], how="left", copy=False)
    del pre_df, pre_rows, hdf, holiday_rows
    gc.collect()
    panel["is_pre_hari_besar"] = panel["is_pre_hari_besar"].fillna(0.0).astype("float32").astype("uint8")

    # Calendar
    panel["day_of_week"] = panel["date"].dt.dayofweek.astype("int8")
    panel["week_of_year"] = panel["date"].dt.isocalendar().week.astype("int16")
    panel["month"] = panel["date"].dt.month.astype("int8")
    panel["quarter"] = panel["date"].dt.quarter.astype("int8")
    panel["day_of_month"] = panel["date"].dt.day.astype("int8")
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype("uint8")
    panel["is_month_start"] = panel["date"].dt.is_month_start.astype("uint8")
    panel["is_month_end"] = panel["date"].dt.is_month_end.astype("uint8")
    month_end = panel["date"] + pd.offsets.MonthEnd(0)
    panel["days_to_month_end"] = (month_end - panel["date"]).dt.days.astype("int16")
    panel["week_of_month"] = ((panel["date"].dt.day - 1) // 7 + 1).astype("int8")
    panel["is_month_start_window"] = (panel["date"].dt.day <= 5).astype("uint8")
    panel["is_month_end_window"] = (panel["date"].dt.day >= 25).astype("uint8")
    return panel


# ═══════════════════════════════════════════════════════════════
#  B. Holiday Intensity (4 features)
# ═══════════════════════════════════════════════════════════════

def add_holiday_intensity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Holiday intensity score, days to next holiday, is_holiday_season."""
    panel = panel.copy()

    # 1) Holiday intensity: mean(holiday_demand) / mean(pre_holiday_demand) per country
    # FIX 2: drop_duplicates dengan subset (country_code, date) agar variasi per tanggal terjaga
    intensity_rows = []
    hc = panel[panel["is_hari_besar"] == 1][["country_code", DATE_COL, TARGET_COL]].drop_duplicates(subset=["country_code", DATE_COL])
    if not hc.empty:
        holiday_demand = hc.groupby("country_code")[TARGET_COL].mean().to_dict()
        pre_holiday_demand = {}
        for code in holiday_demand:
            pr = panel[(panel["country_code"] == code) & (panel["is_pre_hari_besar"] == 1)]
            pre_holiday_demand[code] = pr[TARGET_COL].mean() if len(pr) > 0 else 1.0
        for code in holiday_demand:
            base = pre_holiday_demand.get(code, 1.0)
            intensity_rows.append({
                "country_code": code,
                "holiday_intensity": holiday_demand[code] / base if base > 0 else 1.0,
            })
    intensity_df = (
        pd.DataFrame(intensity_rows)
        if intensity_rows else pd.DataFrame(columns=["country_code", "holiday_intensity"])
    )
    panel = panel.merge(intensity_df, on="country_code", how="left", copy=False)
    panel["holiday_intensity"] = panel["holiday_intensity"].fillna(1.0).astype("float32")

    # Filter country dengan <5 hari libur: set intensity=1.0
    hc2 = panel[panel["is_hari_besar"] == 1].groupby("country_code").size()
    low_holiday = hc2[hc2 < 5].index
    if len(low_holiday) > 0:
        panel.loc[panel["country_code"].isin(low_holiday), "holiday_intensity"] = 1.0

    # FIX 2: Cap dinaikkan dari 10.0 ke HOLIDAY_INTENSITY_CAP agar tidak flat
    panel["holiday_intensity"] = panel["holiday_intensity"].clip(upper=HOLIDAY_INTENSITY_CAP)

    del intensity_df, hc, hc2
    gc.collect()

    # 2) days_to_next_holiday (capped 30)
    panel = panel.sort_values(["country", DATE_COL], kind="mergesort").reset_index(drop=True)
    hdates = (
        panel[panel["is_hari_besar"] == 1][["country", DATE_COL]]
        .drop_duplicates()
        .rename(columns={DATE_COL: "next_holiday"})
        .sort_values(["country", "next_holiday"], kind="mergesort")
    )
    if len(hdates) > 0:
        parts = []
        for country, grp in panel.groupby("country", sort=False):
            hgrp = hdates[hdates["country"] == country]
            if hgrp.empty:
                grp["days_to_next_holiday"] = 30
            else:
                m = pd.merge_asof(
                    grp.sort_values(DATE_COL, kind="mergesort"),
                    hgrp.sort_values("next_holiday", kind="mergesort"),
                    left_on=DATE_COL,
                    right_on="next_holiday",
                    direction="forward",
                    allow_exact_matches=True,
                )
                grp["days_to_next_holiday"] = (m["next_holiday"] - m[DATE_COL]).dt.days
            parts.append(grp)
        panel = pd.concat(parts, ignore_index=True)
    else:
        panel["days_to_next_holiday"] = 30
    panel["days_to_next_holiday"] = panel["days_to_next_holiday"].fillna(30).clip(0, 30).astype("int8")

    # 3) is_holiday_season (±3 days)
    panel["is_holiday_season"] = (
        (panel["is_hari_besar"] == 1) | (panel["is_pre_hari_besar"] == 1)
    ).astype("uint8")

    # 4) holiday_x_weekend
    panel["holiday_x_weekend"] = (
        panel["is_hari_besar"].astype("uint8") & panel["is_weekend"].astype("uint8")
    ).astype("uint8")
    return panel


# ═══════════════════════════════════════════════════════════════
#  C. Peak Days (1 feature)
# ═══════════════════════════════════════════════════════════════

def add_peak_days(panel: pd.DataFrame) -> pd.DataFrame:
    """Label top PEAK_DAYS_PCT hari sebagai is_peak_day."""
    daily_total = panel.groupby(DATE_COL)[TARGET_COL].sum().sort_values(ascending=False)
    n_peak = max(1, int(len(daily_total) * PEAK_DAYS_PCT))
    peak_dates = set(daily_total.head(n_peak).index)
    panel["is_peak_day"] = panel[DATE_COL].isin(peak_dates).astype("uint8")
    return panel


# ═══════════════════════════════════════════════════════════════
#  D. Lag / Rolling (33 features)
# ═══════════════════════════════════════════════════════════════

def add_lag_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Semua lag, rolling, spike, momentum — leakage‑safe via shift(1)."""
    panel = panel.copy()
    panel = panel.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)
    s = panel.groupby(GROUP_COLS)[TARGET_COL].shift(1)  # shifted target

    # Lags (9)
    for lag in [1, 2, 7, 14, 21, 28, 35, 56, 84]:
        panel[f"demand_lag_{lag}"] = panel.groupby(GROUP_COLS)[TARGET_COL].shift(lag)

    # Rolling stats (17)
    for w in [7, 28]:
        panel[f"roll_max_{w}"] = s.rolling(w, min_periods=1).max().values
    panel["roll_zero_count_14"] = (s == 0).rolling(14, min_periods=1).sum().values
    for w in [7, 14, 28, 56]:
        panel[f"roll_mean_{w}"] = s.rolling(w, min_periods=1).mean().values
    for w in [7, 14, 28]:
        panel[f"roll_median_{w}"] = s.rolling(w, min_periods=1).median().values
    for w in [7, 14, 28, 56]:
        panel[f"roll_std_{w}"] = s.rolling(w, min_periods=1).std().values
    for w in [56, 84]:
        panel[f"roll_max_{w}"] = s.rolling(w, min_periods=1).max().values

    # Acceleration & spike (3)
    rm3 = s.rolling(3, min_periods=1).mean().values
    rm14 = s.rolling(14, min_periods=1).mean().values
    panel["demand_acceleration_3d"] = rm3 / (rm14 + 1e-8)
    panel["spike_ratio_28"] = panel["roll_max_28"] / (panel["roll_mean_28"] + 1e-8)
    panel["spike_ratio_56"] = panel["roll_max_56"] / (panel["roll_mean_56"] + 1e-8)

    # Percent change (2)
    panel["pct_change_1"] = (panel["demand_lag_1"] - panel["demand_lag_2"]) / (panel["demand_lag_2"] + 1e-8)
    panel["pct_change_7"] = (panel["demand_lag_7"] - panel["demand_lag_14"]) / (panel["demand_lag_14"] + 1e-8)

    # Days since last sale (1)
    last_sale = panel[DATE_COL].where(s > 0)
    last_sale = last_sale.groupby(
        panel[GROUP_COLS].apply(tuple, axis=1)
    ).ffill()
    panel["days_since_last_sale"] = (panel[DATE_COL] - last_sale).dt.days.fillna(9999).astype("int16")

    # Price features (2)
    ps = panel.groupby(GROUP_COLS)[PRICE_COL].shift(1)
    rpm30 = ps.rolling(30, min_periods=1).max().values
    panel["discount_depth_pct"] = (rpm30 - panel[PRICE_COL]) / (rpm30 + 1e-8)
    rpm14 = ps.rolling(14, min_periods=1).mean().values
    panel["price_momentum"] = panel[PRICE_COL] / (rpm14 + 1e-8)

    # Cleanup
    for c in panel.select_dtypes(include=["number"]).columns:
        panel[c] = panel[c].replace([np.inf, -np.inf], 0).fillna(0)
    for c in panel.select_dtypes(include=["float64"]).columns:
        panel[c] = panel[c].astype("float32")
    del s, rm3, rm14, ps, rpm30, rpm14, last_sale
    gc.collect()
    return panel


# ═══════════════════════════════════════════════════════════════
#  E. Pipeline orchestrator
# ═══════════════════════════════════════════════════════════════

def process_silver(input_path: Path = None, output_path: Path = None) -> pd.DataFrame:
    """Silver layer: daily panel → feature‑engineered tabular → gold (Parquet).

    Membaca panel dari data/silver/, menambahkan 52 fitur,
    dan menulis tabular ke data/gold/.
    """
    logger = _setup_logger()
    input_path = input_path or SILVER_DIR / "online_retail_daily_product.parquet"
    output_path = output_path or TABULAR_PATH

    logger.info("Silver: membaca panel dari %s", input_path)
    panel = pd.read_parquet(input_path)
    tabular = build_tabular_dataframe(panel)
    del panel
    gc.collect()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabular.to_parquet(output_path, index=False)
    logger.info("Gold: tabular ditulis ke %s rows=%s cols=%s features=%s",
                output_path, tabular.shape[0], tabular.shape[1], len(FEATURE_COLS))

    manifest = {
        "stage": "silver",
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(tabular.shape[0]),
        "cols": int(tabular.shape[1]),
        "features": len(FEATURE_COLS),
        "created_utc": datetime.utcnow().isoformat(),
    }
    _write_manifest(Path("artifacts/manifests"), manifest)
    return tabular


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("features")
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
    out = manifest_dir / f"features_manifest_{ts}.json"
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    return out


def build_tabular_dataframe(panel: pd.DataFrame) -> pd.DataFrame:
    """Gabung semua feature group jadi satu tabular DataFrame."""
    logger = _setup_logger()
    logger.info("Building tabular from panel: rows=%s cols=%s", panel.shape[0], panel.shape[1])
    panel = add_calendar_holiday_features(panel)
    panel = add_holiday_intensity_features(panel)
    panel = add_peak_days(panel)
    panel = add_lag_rolling_features(panel)
    logger.info("Tabular complete: rows=%s cols=%s features=%s", panel.shape[0], panel.shape[1], len(FEATURE_COLS))
    return panel


def run_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Full pipeline: silver → gold (daily panel → feature tabular)."""
    return process_silver(input_path=input_path, output_path=output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature engineering: 52 features for FMCG demand forecasting"
    )
    parser.add_argument("--input", type=Path, default=SILVER_DIR / "online_retail_daily_product.parquet")
    parser.add_argument("--output-tabular", type=Path, default=TABULAR_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_features(args.input, args.output_tabular)
