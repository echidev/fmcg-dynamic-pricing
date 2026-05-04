import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from online_retail_pipeline import build_daily_product_dataset, clean_online_retail


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features (always safe — no leakage possible)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_lag_rolling_features(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Add lag, rolling, and diff features grouped by product.
    All features use .shift() so no same-day target leakage.
    Also creates lag-1 features for avg_price, revenue, and num_invoices.
    """
    df = df.copy()
    df = df.sort_values([group_col, "date"])

    # ── Demand lag features ──
    for lag in [1, 2, 7, 14, 28]:
        df[f"demand_lag_{lag}"] = df.groupby(group_col)["demand_qty"].shift(lag)

    # ── Price / revenue / invoice lag-1 (daily signals, shifted by 1) ──
    for col, fill_val in [("avg_price", "price_mean"), ("revenue", 0), ("num_invoices", 0)]:
        if col not in df.columns:
            continue
        lag_col = f"{col}_lag_1"
        df[lag_col] = df.groupby(group_col)[col].shift(1)
        if fill_val == "price_mean":
            mean_prices = df.groupby(group_col)[col].transform("mean")
            df[lag_col] = df[lag_col].fillna(mean_prices).fillna(0)
        else:
            df[lag_col] = df[lag_col].fillna(fill_val)

    # ── Rolling mean (shifted by 1 to avoid same-day leakage) ──
    for window in [7, 14, 28]:
        df[f"roll_mean_{window}"] = (
            df.groupby(group_col)["demand_qty"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

    # ── Rolling std ──
    for window in [7, 28]:
        df[f"roll_std_{window}"] = (
            df.groupby(group_col)["demand_qty"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
        )

    # ── Diff features (lag-based, no target leakage) ──
    df["diff_1"] = df["demand_lag_1"] - df["demand_lag_2"]
    df["diff_7"] = df["demand_lag_7"] - df["demand_lag_14"]

    return df


def add_product_features(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Add static product-level aggregation features.
    SAFE VERSION: computed ONLY from the training period (up to cutoff_date).
    Values are mapped to the full dataset so test rows get train-only aggregates.
    Cold-start products (not in train) get 0 for all product features.
    """
    df = df.copy()

    # ── product_popularity: total demand per product ──
    prod_demand = df.groupby(group_col)["demand_qty"].sum()
    df["product_popularity"] = df[group_col].map(prod_demand).fillna(0).astype(int)

    # ── product_revenue_share: share of total revenue ──
    prod_rev = df.groupby(group_col)["revenue"].sum()
    total_rev = prod_rev.sum()
    if total_rev > 0:
        df["product_revenue_share"] = df[group_col].map(prod_rev) / total_rev
    else:
        df["product_revenue_share"] = 0.0
    df["product_revenue_share"] = df["product_revenue_share"].fillna(0)

    # ── product_lifecycle_age: days since first sale ──
    first_sale = df.groupby(group_col)["date"].min()
    df["first_sale_date"] = df[group_col].map(first_sale)
    df["product_lifecycle_age"] = (df["date"] - df["first_sale_date"]).dt.days
    # Cold start: if product never appeared before this row, age = 0
    df["product_lifecycle_age"] = df["product_lifecycle_age"].fillna(0).clip(lower=0)
    df = df.drop(columns=["first_sale_date"])

    return df


def build_tabular_dataframe(df: pd.DataFrame, cutoff_date=None) -> pd.DataFrame:
    """
    Full tabular DataFrame with ALL features (calendar, lag, rolling, product).

    Parameters
    ----------
    cutoff_date : datetime or None
        If provided, product aggregation features are computed ONLY from
        rows with date <= cutoff_date. This prevents leakage from the
        test period into static product-level features.
    """
    df_tab = df.copy()
    df_tab = add_time_features(df_tab)

    # Compute avg_price BEFORE feature engineering (safe: revenue/demand are same-day,
    # but we only use avg_price_lag_1 which shifts it by 1 day)
    df_tab["avg_price"] = df_tab["revenue"] / df_tab["demand_qty"].replace(0, np.nan)
    df_tab["avg_price"] = df_tab.groupby("stock_code")["avg_price"].transform(
        lambda x: x.ffill().bfill().fillna(0)
    )

    # ── Lag & rolling features (shifted, no leakage) ──
    df_tab = add_lag_rolling_features(df_tab, group_col="stock_code")

    # ── Product aggregation features ──
    if cutoff_date is not None:
        df_cutoff = pd.to_datetime(cutoff_date)
        train_mask = df_tab["date"] <= df_cutoff
        df_train_only = df_tab[train_mask].copy()
        df_test_rows = df_tab[~train_mask].copy()

        # Compute product features from train only
        prod_demand = df_train_only.groupby("stock_code")["demand_qty"].sum()
        prod_rev = df_train_only.groupby("stock_code")["revenue"].sum()
        total_rev = prod_rev.sum()
        first_sale = df_train_only.groupby("stock_code")["date"].min()

        # Map to test rows
        df_test_rows["product_popularity"] = (
            df_test_rows["stock_code"].map(prod_demand).fillna(0)
        )
        if total_rev > 0:
            df_test_rows["product_revenue_share"] = (
                df_test_rows["stock_code"].map(prod_rev) / total_rev
            ).fillna(0)
        else:
            df_test_rows["product_revenue_share"] = 0.0

        df_test_rows["first_sale_date"] = df_test_rows["stock_code"].map(first_sale)
        df_test_rows["product_lifecycle_age"] = (
            (df_test_rows["date"] - df_test_rows["first_sale_date"]).dt.days
        ).fillna(0).clip(lower=0)
        df_test_rows = df_test_rows.drop(columns=["first_sale_date"])

        # Compute product features for train rows
        df_train_only = add_product_features(df_train_only, group_col="stock_code")

        # Re-join
        df_tab = pd.concat([df_train_only, df_test_rows], ignore_index=True)
    else:
        # Fallback: no cutoff provided, compute from full data (NOT recommended)
        df_tab = add_product_features(df_tab, group_col="stock_code")

    # Ensure no residual NaN in lag/rolling cols
    lag_rolling_cols = [c for c in df_tab.columns if any(
        c.startswith(prefix) for prefix in
        ["demand_lag", "avg_price_lag", "revenue_lag", "num_invoices_lag",
         "roll_mean", "roll_std", "diff"]
    )]
    df_tab[lag_rolling_cols] = df_tab[lag_rolling_cols].fillna(0)

    return df_tab


def run_preprocess(
    input_path: Path,
    output_tabular_path: Path,
    cutoff_date=None,
) -> pd.DataFrame:
    df_raw = pd.read_csv(input_path)
    df_clean = clean_online_retail(df_raw)
    daily_product = build_daily_product_dataset(df_clean)

    daily_product_tabular = build_tabular_dataframe(daily_product, cutoff_date=cutoff_date)

    output_tabular_path.parent.mkdir(parents=True, exist_ok=True)
    daily_product_tabular.to_csv(output_tabular_path, index=False)

    return daily_product_tabular


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare tabular dataset for XGBoost demand forecasting."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/online_retail.csv"),
        help="Path to raw input CSV",
    )
    parser.add_argument(
        "--output-tabular",
        type=Path,
        default=Path("data/transform/online_retail_daily_product_tabular.csv"),
        help="Path to output tabular CSV",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocess(args.input, args.output_tabular)
