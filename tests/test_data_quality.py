"""Data quality tests — memastikan data bersih dan siap untuk training."""

import pandas as pd
import pytest
import numpy as np


class TestDataQuality:
    """Test integritas data."""

    def test_no_negative_demand(self, dummy_data):
        """Target variable demand_qty tidak boleh negatif."""
        assert (dummy_data["demand_qty"] >= 0).all(), "Demand tidak boleh negatif!"

    def test_demand_nonnegative_after_sample(self, dummy_data):
        """Demand tidak negatif bahkan setelah sample."""
        sample = dummy_data.sample(n=10, random_state=42)
        assert (sample["demand_qty"] >= 0).all()

    def test_price_completeness(self, dummy_data):
        """Kolom avg_price harus 0% NaN."""
        assert dummy_data["avg_price"].isna().sum() == 0, "Price tidak boleh NaN!"

    def test_date_completeness(self, dummy_data):
        """Kolom date harus 0% NaN."""
        assert dummy_data["date"].isna().sum() == 0, "Date tidak boleh NaN!"

    def test_stock_code_completeness(self, dummy_data):
        """stock_code harus 0% NaN."""
        assert dummy_data["stock_code"].isna().sum() == 0, "Stock code tidak boleh NaN!"

    def test_quantity_positive(self, dummy_data):
        """Semua quantity harus > 0 (setelah cleaning)."""
        from src.data_prep import preprocess_chunk
        chunk = pd.DataFrame({
            "Invoice": ["1", "2"], "StockCode": ["A", "B"],
            "Quantity": [10.0, 5.0], "InvoiceDate": ["2011-01-01", "2011-01-02"],
            "Price": [5.0, 3.0], "Country": ["UK", "UK"],
        })
        cleaned = preprocess_chunk(chunk)
        assert (cleaned["quantity"] > 0).all()

    def test_no_duplicate_invoices(self, dummy_data):
        """Tidak ada duplikat invoice (setelah cleaning)."""
        from src.data_prep import preprocess_chunk
        chunk = pd.DataFrame({
            "Invoice": ["1", "1", "2"], "StockCode": ["A", "A", "B"],
            "Quantity": [10.0, 10.0, 5.0],
            "InvoiceDate": ["2011-01-01", "2011-01-01", "2011-01-02"],
            "Price": [5.0, 5.0, 3.0], "Country": ["UK", "UK", "US"],
        })
        cleaned = preprocess_chunk(chunk)
        assert cleaned["invoice"].duplicated().sum() == 0

    def test_non_product_codes_filtered(self):
        """Kode non‑produk harus difilter."""
        from src.data_prep import preprocess_chunk
        chunk = pd.DataFrame({
            "Invoice": ["1", "2"], "StockCode": ["POST", "A"],
            "Quantity": [10.0, 5.0],
            "InvoiceDate": ["2011-01-01", "2011-01-02"],
            "Price": [5.0, 3.0], "Country": ["UK", "US"],
        })
        cleaned = preprocess_chunk(chunk)
        assert "POST" not in cleaned["stock_code"].values
        assert "A" in cleaned["stock_code"].values
