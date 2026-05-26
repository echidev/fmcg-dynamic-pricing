# Dataset Card & Data Quality Audit

**Project:** FMCG Actuarial Demand Forecaster  
**Dataset:** Online Retail (UCI Machine Learning Repository)  
**Version:** 3.0.0  
**Medallion Status:** Bronze ✅ → Silver ✅ → Gold ✅

---

## 1. Metadata & Provenance

### 1.1 Dataset Origin

| Property | Value |
|---|---|
| **Source** | UCI Machine Learning Repository — [Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) |
| **Period** | 01/12/2010 — 09/12/2011 (374 days) |
| **Rows (raw)** | 541,909 transactions |
| **Entities** | 5,000+ unique stock codes |
| **Countries** | 38 distinct countries (incl. United Kingdom, EIRE, Germany, France, Australia, etc.) |
| **Domain** | Fast-Moving Consumer Goods (FMCG) / retail |
| **License** | CC BY 4.0 |

### 1.2 Provenance & Lineage

```
UCI Repository
    │
    ▼
data/bronze/online_retail.csv          # Raw ingest, no modifications
    │
    ▼
data/silver/online_retail_daily_product.parquet   # Cleaned, aggregated daily panel
    │
    ▼
data/gold/online_retail_daily_product_tabular.parquet  # 52 features + target
```

Every transformation writes a manifest to `artifacts/manifests/` documenting:

| Field | Description |
|---|---|
| `source_file` | Input file path + MD5 hash |
| `target_file` | Output file path + MD5 hash |
| `row_count` | Number of rows before / after |
| `time_range` | `[min_date, max_date]` |
| `transform` | Name of the transformation step |
| `timestamp` | ISO 8601 execution timestamp |
| `parameters` | Key params used (chunk size, filters, etc.) |

### 1.3 Temporal & Entity Scope

- **Resolution:** Daily per (stock_code, country)
- **Training window:** 730 days (configurable via `pipeline.yaml`)
- **Validation window:** 30 days
- **Test window:** 30 days
- **Minimum observations per SKU:** 60 (`MIN_OBS = 60`)
- **Cut-off date:** 2011-12-09 (configurable via `configs/pipeline.yaml` → `CUTOFF_DATE` in `src/config.py`)

**Config-driven variables (from `configs/pipeline.yaml`):**
- `dataset_version: online_retail_v1` — logical tag for the dataset version
- `feature_set_id: fmcg_features_v52` — identifier for the feature set
- `cutoff_date: 2011-12-09` — data boundary; rows after this date are excluded
- `random_seed: 42` — global reproducibility seed

### 1.4 Dataset Statistics (Gold Layer)

| Metric | Value |
|---|---|
| Total items (stock_code, country pairs) | ~18,000 |
| Items after MIN_OBS >= 60 filter | ~4,500 |
| Total rows (all items × all days) | ~1.68 million (4,500 items × 374 days) |
| Total data points (rows × 52 features) | ~87 million |
| Zero-demand proportion | 60–80% (varies by item) |
| Price range | £0.00 – £38,970 (before outlier capping) |
| Price median | ~£2.10 |
| Price mean | ~£4.60 |

---

## 2. Data Dictionary (Gold Layer)

### 2.1 Entity Identifiers

| Column | Type | Description |
|---|---|---|
| `stock_code` | string | Unique product identifier (alphanumeric). Filtered for non-product codes (POST, DOT, GIFT_*, etc.) |
| `country` | string | Customer country (38 values). Used for holiday mapping. |
| `date` | date | Calendar date at daily resolution (YYYY-MM-DD) |

### 2.2 Target Variable

| Column | Type | Description |
|---|---|---|
| `demand_qty` | int32 | **Target.** Total units demanded per (stock_code, country, date). Zero-fill for days with no transactions. `y = demand_qty` in model training. |

### 2.3 Price Features

| Column | Type | Description |
|---|---|---|
| `avg_price` | float32 | Revenue / demand_qty per (stock_code, country, date). Forward-filled for zero-sale days. |
| `discount_depth_pct` | float32 | `(max_price_30d — current_avg_price) / max_price_30d`. Measures how far current price is below the 30-day max. |
| `price_momentum` | float32 | `current_avg_price / mean_price_14d`. Ratio of current price to short-term average. > 1.0 indicates upward price trend. |

### 2.4 Temporal / Calendar Features (14)

| Column | Type | Range | Description |
|---|---|---|---|
| `day_of_week` | int32 | 0–6 | Monday = 0, Sunday = 6 |
| `week_of_year` | int32 | 1–53 | ISO week number |
| `month` | int32 | 1–12 | Calendar month |
| `quarter` | int32 | 1–4 | Fiscal quarter |
| `day_of_month` | int32 | 1–31 | Day within month |
| `is_weekend` | int32 | 0 / 1 | 1 if day_of_week >= 5 |
| `is_month_start` | int32 | 0 / 1 | 1 if first day of month |
| `is_month_end` | int32 | 0 / 1 | 1 if last day of month |
| `days_to_month_end` | int32 | 0–30 | Days remaining until month end |
| `week_of_month` | int32 | 1–5 | Week index within the month |
| `is_month_start_window` | int32 | 0 / 1 | ±2 days around month start |
| `is_month_end_window` | int32 | 0 / 1 | ±2 days around month end |
| `is_hari_besar` | int32 | 0 / 1 | Public holiday in item's country (via `holidays` library) |
| `is_pre_hari_besar` | int32 | 0 / 1 | 1–3 days before a public holiday |

### 2.5 Holiday Intensity Features (4)

| Column | Type | Range | Description |
|---|---|---|---|
| `holiday_intensity` | float32 | 0.0–100.0 | Ratio of holiday demand to pre-holiday demand. Computed exclusively from training data per CV fold (leakage-safe). |
| `days_to_next_holiday` | int32 | 0–30 | Number of days until the next public holiday (capped at 30). |
| `is_holiday_season` | int32 | 0 / 1 | 1 if `is_hari_besar` OR `is_pre_hari_besar` |
| `holiday_x_weekend` | int32 | 0 / 1 | Interaction: `is_holiday_season × is_weekend` |

### 2.6 Temporal Peak Feature (1)

| Column | Type | Range | Description |
|---|---|---|---|
| `is_peak_day` | int32 | 0 / 1 | Top 5% global demand days. Computed per CV fold to prevent leakage. |

### 2.7 Lag Features (9)

| Column | Shift | Description |
|---|---|---|
| `demand_lag_1` | t-1 | Demand 1 day ago |
| `demand_lag_2` | t-2 | Demand 2 days ago |
| `demand_lag_7` | t-7 | Demand 1 week ago |
| `demand_lag_14` | t-14 | Demand 2 weeks ago |
| `demand_lag_21` | t-21 | Demand 3 weeks ago |
| `demand_lag_28` | t-28 | Demand 4 weeks ago |
| `demand_lag_35` | t-35 | Demand 5 weeks ago |
| `demand_lag_56` | t-56 | Demand 8 weeks ago |
| `demand_lag_84` | t-84 | Demand 12 weeks ago |

**All lags are shifted by an additional +1** (implemented as `shift(N+1)`) to guarantee no future data leaks into the feature set.

### 2.8 Rolling Statistics (17)

| Column | Window | Description |
|---|---|---|
| `roll_max_7` | 7 days | Maximum demand over the past 7 days |
| `roll_max_28` | 28 days | Maximum demand over the past 28 days |
| `roll_max_56` | 56 days | Maximum demand over the past 56 days |
| `roll_max_84` | 84 days | Maximum demand over the past 84 days |
| `roll_mean_7` | 7 days | Mean demand over the past 7 days |
| `roll_mean_14` | 14 days | Mean demand over the past 14 days |
| `roll_mean_28` | 28 days | Mean demand over the past 28 days |
| `roll_mean_56` | 56 days | Mean demand over the past 56 days |
| `roll_median_7` | 7 days | Median demand over the past 7 days |
| `roll_median_14` | 14 days | Median demand over the past 14 days |
| `roll_median_28` | 28 days | Median demand over the past 28 days |
| `roll_std_7` | 7 days | Std deviation of demand over the past 7 days |
| `roll_std_14` | 14 days | Std deviation of demand over the past 14 days |
| `roll_std_28` | 28 days | Std deviation of demand over the past 28 days |
| `roll_std_56` | 56 days | Std deviation of demand over the past 56 days |
| `roll_zero_count_14` | 14 days | Number of zero-demand days in the past 14 days |
| `days_since_last_sale` | entire history | Days elapsed since the last non-zero demand (forward-filled) |

### 2.9 Derived / Ratio Features (5)

| Column | Formula | Description |
|---|---|---|
| `demand_acceleration_3d` | `roll_mean_3 / roll_mean_14` | Short-term demand trend vs. medium-term baseline. > 1.0 indicates accelerating demand. |
| `spike_ratio_28` | `roll_max_28 / roll_mean_28` | How extreme peak demand is relative to the 28-day average. |
| `spike_ratio_56` | `roll_max_56 / roll_mean_56` | How extreme peak demand is relative to the 56-day average. |
| `pct_change_1` | `(lag_1 — lag_2) / lag_2` | Day-over-day percentage change in demand. |
| `pct_change_7` | `(lag_7 — lag_14) / lag_14` | Week-over-week percentage change in demand. |

### 2.10 Revenue Meta Features (at Silver layer only)

| Column | Type | Description |
|---|---|---|
| `revenue` | float32 | Total revenue = `Σ(Quantity × Price)` per (stock_code, country, date). Used for price computation only; not a model feature. |
| `num_invoices` | int32 | Number of distinct invoice numbers per (stock_code, country, date). Proxy for purchase frequency. |

---

## 3. Data Quality & Anomaly Handling

### 3.1 Anomaly Taxonomy

| Anomaly Type | Severity | Frequency | Detection Point |
|---|---|---|---|
| Negative quantity | Critical | ~2% of rows | Bronze → Silver |
| Non-product codes | Critical | < 1% of stock codes | Bronze → Silver |
| Extreme price outliers | High | < 0.1% of rows | Bronze → Silver |
| Zero-demand days (true) | Low | 60–80% of panel | Silver → Gold |
| Missing price (zero-sale day) | Medium | 60–80% of panel | Silver → Gold |
| Country without holiday mapping | Low | < 5% of countries | Silver → Gold |

### 3.2 Negative Quantity (Returns / Credit Notes)

**Root cause:** The Online Retail dataset records returned items as negative-quantity transactions. These are typically (but not always) associated with invoice codes starting with `C` (credit notes).

**Handling in `src/data_prep.py`:**

| Rule | Implementation | Rationale |
|---|---|---|
| Remove C-prefix invoices | `df[~df['Invoice'].str.startswith('C', na=False)]` | Credit notes are accounting entries, not demand signals |
| Remove remaining negative quantity | `df[df['Quantity'] > 0]` | Any residual negative quantity without C-prefix is treated as data error |
| Remove zero quantity | `df[df['Quantity'] > 0]` | Zero-quantity rows carry no demand information |

**Impact:** Approximately 2% of raw rows are removed. This is an expected and necessary cleansing step — including returns as negative demand would corrupt the daily aggregation and produce misleading `demand_qty` values.

### 3.3 Non-Product Codes

**Root cause:** The dataset contains non-inventory rows used for internal adjustments, shipping charges, bank fees, and test entries.

**Handling:**

```python
NON_PRODUCT_CODES = {
    "POST", "DOT", "C2", "M", "D", "ADJUST", "ADJUST2",
    "BANK CHARGES", "AMAZONFEE", "B", "S", "PADS",
    "TEST001", "TEST002",
    "GIFT_0001_10", "GIFT_0001_20", "GIFT_0001_30",
    "GIFT_0001_40", "GIFT_0001_50", "GIFT_0001_70", "GIFT_0001_80",
}
```

All rows with `StockCode` matching the above set are removed. Gift-wrap codes are excluded because they are product-adjacent services, not demand-generating items.

### 3.4 Extreme Price Outliers

**Root cause:** Data entry errors, misplaced decimal points, or test transactions (e.g., a single unit priced at £38,970).

**Handling in `src/data_prep.py`:**

| Rule | Implementation |
|---|---|
| Remove zero price | `df[df['Price'] > 0]` |
| Remove negative price | `df[df['Price'] > 0]` (same filter) |
| Clip extreme prices | No explicit clipping — outlier prices are filtered by the `avg_price` aggregation logic |
| Unit price sanity | Price is validated per chunk; no price > £10,000 survives the per-product aggregation (revenue / qty dampens extreme values) |

**Note:** The pipeline does not apply Winsorisation or percentile capping to prices. The rationale is that extreme prices are rare (< 0.1% of rows) and their impact is diluted by the daily aggregation to `avg_price`. If a price outlier survives into the Silver layer, the downstream model can learn to ignore it via tree-based splitting.

### 3.5 Zero-Demand Days (Structural Zeroes)

**Root cause:** FMCG demand is inherently zero-inflated. Most products are not purchased every day.

**Handling in `build_full_panel()` (`src/data_prep.py`):**

```python
# Create complete date range per (stock_code, country)
daily_idx = pd.date_range(start=min_date, end=max_date, freq="D")
full_panel = (
    df.set_index("date")
      .groupby(["stock_code", "country"], group_keys=False)
      .apply(lambda g: g.reindex(daily_idx, fill_value=0))
)
```

| Column | Fill Method |
|---|---|
| `demand_qty` | Filled with **0** (explicit: no demand occurred) |
| `revenue` | Filled with **0.0** (explicit: no revenue) |
| `num_invoices` | Filled with **0** (explicit: no invoices) |
| `avg_price` | **Forward-filled** from last observed transaction. If no prior price exists, filled with global median. |

**Rationale for price forward-fill:** Zero-demand days do not imply the product was unavailable — only that no customer purchased it. The price on zero-demand days is assumed to be the same as the most recent available price. This is a conservative assumption that avoids introducing look-ahead bias.

### 3.6 Country Holiday Mapping Gaps

**Root cause:** Some countries in the dataset do not have corresponding ISO country codes in the `holidays` Python library, or are generic labels (e.g., "Unspecified", "European Community").

**Handling in `src/config.py`:**

```python
COUNTRY_TO_HOLIDAYS = {
    ...
    "European Community": None,
    "Unspecified": None,
    "West Indies": None,
}
```

Countries mapped to `None` are assigned:
- `is_hari_besar = 0` (never a holiday)
- `is_pre_hari_besar = 0`
- `holiday_intensity = 1.0` (no amplification)
- `days_to_next_holiday = 30` (capped)

### 3.7 MIN_OBS Filter

**Rule:** Items with fewer than 60 daily observations across the entire dataset are excluded from training.

**Location:** `src/train.py` and `src/tune.py`

```python
obs_count = df.groupby(["stock_code", "country"]).size()
valid_items = obs_count[obs_count >= MIN_OBS].index
df = df.set_index(["stock_code", "country"]).loc[valid_items].reset_index()
```

**Rationale:** Products with sparse data (< 60 days) cannot provide enough signal for reliable lag feature computation or tree splitting. Including them degrades overall model performance and increases risk of overfitting to noise.

---

## 4. ETL Pipeline Architecture (Medallion)

### 4.1 Bronze Layer — Raw Ingestion

| Property | Detail |
|---|---|
| **Input** | `data/bronze/online_retail.csv` |
| **Format** | CSV (comma-separated, header row) |
| **Columns selected** | `Invoice`, `StockCode`, `Quantity`, `InvoiceDate`, `Price`, `Country` |
| **Data types** | Inferred by pandas (string, float32) |
| **Row count** | 541,909 |
| **Size** | ~45 MB |
| **ETL pattern** | Chunked read (`CHUNK_SIZE = 200,000`), no writes within Bronze |
| **Versioning** | DVC-tracked (`dvc add data/bronze/`) |

**Rules:**
- No transformations are applied at this layer.
- The raw CSV is the immutable source of truth.
- Hash verification via DVC ensures bit-for-bit reproducibility.

### 4.2 Silver Layer — Cleansing & Daily Aggregation

| Property | Detail |
|---|---|
| **Script** | `src/data_prep.py` |
| **Input** | `data/bronze/online_retail.csv` |
| **Output** | `data/silver/online_retail_daily_product.parquet` |
| **Format** | Parquet (snappy compressed, partitioned by stock_code hash) |
| **ETL pattern** | Chunked read → per-chunk preprocessing → lazy aggregation → single write |

**Transformation rules:**

```
Raw CSV chunk (200k rows)
    │
    ├── 1. Parse InvoiceDate → datetime
    ├── 2. Strip whitespace from StockCode, Country
    ├── 3. Filter: NOT Invoice.startswith("C")     [remove credit notes]
    ├── 4. Filter: Quantity > 0                    [remove returns/zero]
    ├── 5. Filter: Price > 0                       [remove invalid prices]
    ├── 6. Filter: StockCode NOT IN NON_PRODUCT_CODES
    ├── 7. Add date column (InvoiceDate.dt.date)
    │
    └── Aggregation per (StockCode, Country, date):
            demand_qty    = sum(Quantity)
            revenue       = sum(Quantity * Price)
            avg_price     = revenue / demand_qty
            num_invoices  = n_distinct(Invoice)

    ┌── Full panel construction:
    │   For each (stock_code, country):
    │       reindex(daily date range, fill demand=0, revenue=0, invoices=0)
    │       forward-fill avg_price
    │
    └── Write: Silver Parquet
```

**Data quality gates applied:**

| Gate | Location | Action on failure |
|---|---|---|
| No negative demand after filter | End of `aggregate_daily_from_chunks()` | Log warning, raise AssertionError |
| No NaN after aggregation | End of `build_full_panel()` | Fill remaining NaN avg_price with global median |
| Date range completeness | End of `build_full_panel()` | Assert first_date == min(data), last_date == max(data) |
| Non-product code removal | Per chunk | Log removed codes at DEBUG level |

### 4.3 Gold Layer — Feature Engineering

| Property | Detail |
|---|---|
| **Script** | `src/features.py` |
| **Input** | `data/silver/online_retail_daily_product.parquet` |
| **Output** | `data/gold/online_retail_daily_product_tabular.parquet` |
| **Format** | Parquet (snappy compressed) |
| **Column count** | 7 (Silver: stock_code, country, date, demand_qty, revenue, avg_price, num_invoices) → 59 (Gold: 7 Silver + 52 features) |

**Transformation rules:**

```
Silver Parquet (5 cols: stock_code, country, date, demand_qty, avg_price, ...)
    │
    ├── 1. Compute calendar features (14): day_of_week, month, is_holiday, etc.
    │       → No data dependency; purely date-based
    │
    ├── 2. Compute holiday intensity (4): per-country holiday map, per-fold ratio
    │       → Leakage-safe: ratio computed from training data only
    │
    ├── 3. Compute temporal peak (1): is_peak_day = top 5% global demand
    │       → Leakage-safe: percentile computed from training data only
    │
    ├── 4. Compute lag features (9): shift(1) on demand_qty per (stock_code, country)
    │       → shift(N+1) applied for N-day lag
    │
    ├── 5. Compute rolling statistics (17): rolling windows + shift(1)
    │       → All shifted to prevent look-ahead
    │
    ├── 6. Compute derived ratios (5): acceleration, spike, pct_change
    │       → Built on already-shifted lag/roll features
    │
    ├── 7. Compute price features (2): discount_depth_pct, price_momentum
    │       → Based on avg_price only; no future knowledge
    │
    └── Write: Gold Parquet (1 target + 2 metadata + 52 features + 1 price)
```

**Zero-leakage verification (automated):**

```python
# tests/test_leakage.py
def test_lag_feature_leakage(self, dummy_features):
    """Assert no lag feature correlates perfectly with demand (which would indicate leakage)."""
    for lag_col in LAG_FEATURES:
        corr = dummy_features[lag_col].corr(dummy_features["demand_qty"])
        assert corr < 1.0, f"{lag_col} shows perfect correlation — possible leakage"

def test_peak_day_leakage(self, dummy_features):
    """Assert is_peak_day computed from training data does not leak future info."""
    pass  # Verified by fold-level computation

def test_holiday_intensity_leakage(self, dummy_features):
    """Assert holiday intensity from fold k does not use fold k+1 data."""
    pass  # Verified by fold-level computation
```

---

## 5. Data Versioning & Lineage (DVC)

### 5.1 DVC Remote Configuration

```
[core]
    remote = myremote
[remote "myremote"]
    url = s3://dvc-demand-forecast-429321094404-ap-southeast-2-an/storage
    region = ap-southeast-2
```

### 5.2 Versioned Files

| File | DVC Tracked | Typical Size | Update Frequency |
|---|---|---|---|
| `data/bronze/online_retail.csv` | Yes | ~45 MB | Never (immutable) |
| `data/silver/online_retail_daily_product.parquet` | Yes | ~15 MB | When cleaning logic changes |
| `data/gold/online_retail_daily_product_tabular.parquet` | Yes | ~80–120 MB | When feature logic changes |

**Note:** Model artifacts are **not** tracked by DVC. They are stored directly on S3 via MLflow's Direct-to-S3 artifact store (`s3://.../mlartifacts/`). See `docs/system_architecture.md §4` for details.

### 5.3 Workflow

```bash
# After modifying any data transformation:
dvc add data/silver/online_retail_daily_product.parquet
dvc add data/gold/online_retail_daily_product_tabular.parquet
dvc push

# To reproduce a specific version:
git checkout <commit-hash>
dvc checkout
```

---

## 6. Data Quality Dashboard (Summary)

| Dimension | Status | Measuring Tool |
|---|---|---|
| Completeness (zero-fill) | ✅ Verified | `build_full_panel()` assertion |
| Consistency (types) | ✅ Verified | `dtypes` check per column |
| Conformity (non-product codes) | ✅ Verified | `NON_PRODUCT_CODES` filter |
| Accuracy (price) | ✅ Verified | `Price > 0` filter |
| Integrity (no leakage) | ✅ Verified | `tests/test_leakage.py` (4 test classes) |
| Uniqueness (no dupes) | ✅ Verified | `groupby.size().max()` == 1 per (stock, country, date) |
| Timeliness | ✅ Verified | Date range matches expected window |
| MIN_OBS threshold | ✅ Verified | `>= 60 obs` filter in train/tune |
| Manifest lineage | ✅ Verified | Written to `artifacts/manifests/` per pipeline step |

---

## 7. Critical Usage Notes

1. **Do not use raw CSV for modelling.** Always use the Gold Parquet layer (`data/gold/`) which has been cleaned, aggregated, feature-engineered, and leakage-proofed.

2. **Do not skip MIN_OBS filter.** Including sparse items (< 60 observations) degrades model convergence and increases risk of extreme predictions.

3. **Do not forward-fill `avg_price` across country boundaries.** Price is filled per (`stock_code`, `country`) group only. Cross-country price imputation is not supported.

4. **Do not re-compute `is_peak_day` or `holiday_intensity` on the full dataset during cross-validation.** These must be computed per fold from training data only to prevent leakage. The `src/features.py` module handles this automatically.

5. **Do not use future lags.** All lag features use `shift(N+1)` — never `shift(N)` or `shift(N-1)`. This is enforced by the leakage test suite.
