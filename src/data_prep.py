"""Data cleaning and daily aggregation pipeline for FMCG demand forecasting.

Takes raw transaction data and produces a clean, daily product-level
dataset. This is the first stage of the pipeline: raw CSV -> cleaned
daily aggregates -> feature engineering (features.py).
"""

import argparse
from pathlib import Path

import pandas as pd


def clean_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate raw Online Retail transaction data.

    Performs column standardization, type coercion, deduplication, removal
    of cancelled invoices (prefix ``C``), filtering of non-product stock
    codes, and exclusion of rows with non-positive quantity or price.

    Args:
        df: Raw DataFrame with columns ``Invoice``, ``StockCode``,
            ``Description``, ``Quantity``, ``InvoiceDate``, ``Price``,
            ``Customer ID``, ``Country``.

    Returns:
        Cleaned DataFrame with standardised column names, a computed
        ``revenue`` column, and only valid product transactions.
    """
    df = df.copy()

    df = df.rename(
        columns={
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country",
        }
    )

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["invoice"] = df["invoice"].astype(str).str.strip()

    df = df.drop_duplicates()

    df = df[df["invoice_date"].notna()]
    df = df[df["stock_code"].notna() & (df["stock_code"] != "")]
    df = df[df["invoice"].notna() & (df["invoice"] != "")]

    df = df[~df["invoice"].str.startswith("C", na=False)]

    non_product_codes = {
        "POST",
        "DOT",
        "C2",
        "M",
        "D",
        "ADJUST",
        "ADJUST2",
        "BANK CHARGES",
        "AMAZONFEE",
        "B",
        "S",
        "PADS",
        "TEST001",
        "TEST002",
        "GIFT_0001_10",
        "GIFT_0001_20",
        "GIFT_0001_30",
        "GIFT_0001_40",
        "GIFT_0001_50",
        "GIFT_0001_70",
        "GIFT_0001_80",
    }
    df = df[~df["stock_code"].isin(non_product_codes)]

    df = df[(df["quantity"] > 0) & (df["price"] > 0)]

    df["revenue"] = df["quantity"] * df["price"]

    return df


def build_daily_product_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactions into a daily product-level dataset.

    Groups by stock code and date, computing total demand quantity, total
    revenue, and the number of unique invoices. A full date range is
    generated per product so that days with zero demand are explicitly
    represented rather than missing.

    Args:
        df: Cleaned DataFrame with ``stock_code``, ``invoice_date``,
            ``quantity``, ``revenue``, and ``invoice`` columns.

    Returns:
        DataFrame indexed by ``(stock_code, date)`` with columns
        ``demand_qty``, ``revenue``, ``num_invoices``. Every product
        has a row for every day in the global date range.
    """
    df = df.copy()
    df["date"] = df["invoice_date"].dt.normalize()

    daily = (
        df.groupby(["stock_code", "date"], as_index=False)
        .agg(
            demand_qty=("quantity", "sum"),
            revenue=("revenue", "sum"),
            num_invoices=("invoice", "nunique"),
        )
    )

    stock_codes = daily["stock_code"].unique()
    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [stock_codes, full_dates], names=["stock_code", "date"]
    )

    daily = (
        daily.set_index(["stock_code", "date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return daily


def run_data_prep(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Execute the cleaning and daily-product aggregation pipeline.

    Args:
        input_path: Path to the raw input CSV file.
        output_path: Destination path for the daily-product CSV output.

    Returns:
        The aggregated daily-product DataFrame.
    """
    df_raw = pd.read_csv(input_path)
    df_clean = clean_online_retail(df_raw)
    daily_product = build_daily_product_dataset(df_clean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_product.to_csv(output_path, index=False)

    return daily_product


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data preparation script.

    Returns:
        Parsed argument namespace with ``input`` and ``output`` paths.
    """
    parser = argparse.ArgumentParser(
        description="Clean Online Retail data and build daily product dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/online_retail.csv"),
        help="Path to raw input CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/transform/online_retail_daily_product.csv"),
        help="Path to output CSV (daily product dataset)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_data_prep(args.input, args.output)
