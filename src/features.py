"""Feature engineering for FMCG demand forecasting.

Transforms a cleaned daily product-level dataset into a fully
feature-engineered tabular DataFrame ready for model training.
Includes calendar features, lag/rolling momentum features, and
static product-level aggregations. All feature engineering is
designed to prevent target leakage.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import build_daily_product_dataset, clean_online_retail


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived features to a daily product DataFrame.

    Calendar features are inherently leakage-free because they encode only
    the date of each observation, never future demand.

    Args:
        df: DataFrame with a ``date`` column.

    Returns:
        DataFrame with additional columns: ``day_of_week``, ``week_of_year``,
        ``month``, ``quarter``, ``day_of_month``, ``is_weekend``,
        ``is_month_start``, ``is_month_end``.
    """
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
    """Create lag, rolling-window, and momentum features per product.

    Every feature is shifted by at least one day to guarantee that no
    same-day target value is used as a predictor. Momentum (``diff_*``)
    columns are computed from lag differences only, never from the
    actual target.

    Args:
        df: DataFrame with ``date``, ``demand_qty``, and optionally
            ``avg_price``, ``revenue``, ``num_invoices``.
        group_col: Column name used to group by product (typically
            ``"stock_code"``).

    Returns:
        DataFrame with added columns:
        ``demand_lag_{1,2,7,14,28}``, ``avg_price_lag_1``,
        ``revenue_lag_1``, ``num_invoices_lag_1``,
        ``roll_mean_{7,14,28}``, ``roll_std_{7,28}``, ``diff_1``,
        ``diff_7``.
    """
    df = df.copy()
    df = df.sort_values([group_col, "date"])

    for lag in [1, 2, 7, 14, 28]:
        df[f"demand_lag_{lag}"] = df.groupby(group_col)["demand_qty"].shift(lag)

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

    for window in [7, 14, 28]:
        df[f"roll_mean_{window}"] = (
            df.groupby(group_col)["demand_qty"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

    for window in [7, 28]:
        df[f"roll_std_{window}"] = (
            df.groupby(group_col)["demand_qty"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
        )

    df["diff_1"] = df["demand_lag_1"] - df["demand_lag_2"]
    df["diff_7"] = df["demand_lag_7"] - df["demand_lag_14"]

    return df


def add_product_features(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute static product-level aggregation features.

    Caller is responsible for ensuring ``df`` contains only training-period
    rows. Using the full dataset (including test rows) will leak future
    demand and revenue statistics into the features.

    Args:
        df: Training-only DataFrame with ``date``, ``demand_qty``,
            ``revenue``, and ``group_col``.
        group_col: Column name used to group by product (typically
            ``"stock_code"``).

    Returns:
        DataFrame with added columns: ``product_popularity``,
        ``product_revenue_share``, ``product_lifecycle_age``.
    """
    df = df.copy()

    prod_demand = df.groupby(group_col)["demand_qty"].sum()
    df["product_popularity"] = df[group_col].map(prod_demand).fillna(0).astype(int)

    prod_rev = df.groupby(group_col)["revenue"].sum()
    total_rev = prod_rev.sum()
    if total_rev > 0:
        df["product_revenue_share"] = df[group_col].map(prod_rev) / total_rev
    else:
        df["product_revenue_share"] = 0.0
    df["product_revenue_share"] = df["product_revenue_share"].fillna(0)

    first_sale = df.groupby(group_col)["date"].min()
    df["first_sale_date"] = df[group_col].map(first_sale)
    df["product_lifecycle_age"] = (df["date"] - df["first_sale_date"]).dt.days
    df["product_lifecycle_age"] = df["product_lifecycle_age"].fillna(0).clip(lower=0)
    df = df.drop(columns=["first_sale_date"])

    return df


def build_tabular_dataframe(df: pd.DataFrame, cutoff_date=None) -> pd.DataFrame:
    """Build a fully feature-engineered tabular DataFrame.

    Pipeline order: calendar features -> avg_price -> lag/rolling features
    -> product-level aggregations.

    Parameters
    ----------
    cutoff_date : datetime or None
        Product aggregation features are computed ONLY from rows with
        ``date <= cutoff_date``. Test-period rows receive the mapped
        train-only statistics. If ``None``, a ``ValueError`` is raised
        to prevent accidental leakage from using the full dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with all feature columns ready for model training.
    """
    df_tab = df.copy()
    df_tab = add_time_features(df_tab)

    df_tab["avg_price"] = df_tab["revenue"] / df_tab["demand_qty"].replace(0, np.nan)
    df_tab["avg_price"] = df_tab.groupby("stock_code")["avg_price"].transform(
        lambda x: x.ffill().bfill().fillna(0)
    )

    df_tab = add_lag_rolling_features(df_tab, group_col="stock_code")

    if cutoff_date is None:
        raise ValueError(
            "cutoff_date is required to prevent target leakage in product "
            "aggregation features. Provide a datetime value representing "
            "the end of the training period."
        )

    df_cutoff = pd.to_datetime(cutoff_date)
    train_mask = df_tab["date"] <= df_cutoff
    df_train_only = df_tab[train_mask].copy()
    df_test_rows = df_tab[~train_mask].copy()

    prod_demand = df_train_only.groupby("stock_code")["demand_qty"].sum()
    prod_rev = df_train_only.groupby("stock_code")["revenue"].sum()
    total_rev = prod_rev.sum()
    first_sale = df_train_only.groupby("stock_code")["date"].min()

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

    df_train_only = add_product_features(df_train_only, group_col="stock_code")

    df_tab = pd.concat([df_train_only, df_test_rows], ignore_index=True)

    lag_rolling_cols = [c for c in df_tab.columns if any(
        c.startswith(prefix) for prefix in
        ["demand_lag", "avg_price_lag", "revenue_lag", "num_invoices_lag",
         "roll_mean", "roll_std", "diff"]
    )]
    df_tab[lag_rolling_cols] = df_tab[lag_rolling_cols].fillna(0)

    return df_tab


def run_features(
    input_path: Path,
    output_tabular_path: Path,
    cutoff_date=None,
) -> pd.DataFrame:
    """Execute the full feature engineering pipeline from raw CSV to tabular dataset.

    Args:
        input_path: Path to the raw input CSV file.
        output_tabular_path: Destination path for the tabular CSV output.
        cutoff_date: Training cutoff date to prevent product-feature leakage.

    Returns:
        The fully processed DataFrame.
    """
    df_raw = pd.read_csv(input_path)
    df_clean = clean_online_retail(df_raw)
    daily_product = build_daily_product_dataset(df_clean)

    daily_product_tabular = build_tabular_dataframe(daily_product, cutoff_date=cutoff_date)

    output_tabular_path.parent.mkdir(parents=True, exist_ok=True)
    daily_product_tabular.to_csv(output_tabular_path, index=False)

    return daily_product_tabular


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the feature engineering script.

    Returns:
        Parsed argument namespace with ``input`` and ``output_tabular`` paths.
    """
    parser = argparse.ArgumentParser(
        description="Prepare tabular dataset for FMCG demand forecasting."
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
    run_features(args.input, args.output_tabular)
