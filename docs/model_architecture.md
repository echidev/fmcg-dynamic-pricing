# Model Architecture — DecoupledActuarialXGB

**Model Type:** 3-Pronged Decoupled XGBoost + Actuarial Optimisation  
**File:** `src/model.py`  
**Wrapper:** `src/model_wrapper.py` (MLflow PyFunc)  
**Status:** Production (Active)

---

## 1. Architectural Overview

The DecoupledActuarialXGB separates statistical estimation from business-risk optimisation into three distinct steps:

```
┌────────────────────────────────────────────────────────┐
│              Step 1: Honest Baseline                    │
│  XGBoost Regressor (objective = reg:squarederror)       │
│  Output: E[X] — unbiased conditional mean               │
│  Trained on log1p(y) for variance stabilisation         │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│           Step 2: Risk-Aware Quantile                   │
│  XGBoost Regressor (objective = reg:quantileerror)      │
│  Output: Q_q_target[X] — q-th conditional quantile       │
│  Default: q_target = 0.9799                              │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│         Step 3: Actuarial Optimisation Layer            │
│  Dynamic Critical Fractile per SKU                      │
│  Blends E[X] and Q_q_target[X] via business-cost ratio  │
│  Output: Recommended Stock = max(Q, Final Forecast)     │
└────────────────────────────────────────────────────────┘
```

### Why Decoupled?

| Approach | Limitation |
|---|---|
| Single MSE model | Symmetric loss — ignores asymmetric stockout/overstock costs |
| Single quantile model | Fixed quantile — ignores price elasticity and perishability per SKU |
| **Decoupled (ours)** | Mean captures baseline demand; quantile captures tail risk; actuarial layer blends them dynamically per item based on price, margin, and shelf life |

---

## 2. Step 1: Honest Baseline (Mean Regression)

```
Target Transformation:    y_reg = log1p(y_nz)
Model Objective:          reg:squarederror
Training Set:             y_train > 0 (non-zero rows only)
Inverse Transform:        p_mean = max(expm1(model.predict(X)), 0)
```

The mean regressor is trained exclusively on non-zero demand rows. This prevents the model from learning toward zero under the influence of zero-inflated data. The `log1p` transformation stabilises variance across low- and high-volume SKUs.

**Key properties:**
- Unbiased estimate of conditional mean **given that demand occurs**.
- Output is always non-negative (clamped at 0 via `expm1` + `max`).

---

## 3. Step 2: Risk-Aware Quantile (Quantile Regression)

```
Model Objective:     reg:quantileerror
Quantile Alpha:      q_target (default 0.9799)
Training Set:        y_train > 0 (same as Step 1)
Inverse Transform:   p_quant = max(expm1(model.predict(X)), 0)
```

The quantile regressor uses the pinball loss (quantile error) to estimate the `q_target`-th conditional percentile. The default `q_target = 0.9799` means the model predicts a value that exceeds actual demand approximately 97.99% of the time — a conservative upper bound.

**Pinball loss for quantile alpha = q:**

$$L_q(y, \hat{y}) = \begin{cases} q \cdot (y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1 - q) \cdot (\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

This asymmetric loss penalises under-prediction more heavily when `q > 0.5`, which is the desired behaviour for inventory planning.

---

## 4. Step 3: Actuarial Optimisation Layer

This layer computes a **per-SKU dynamic critical fractile** by modelling the economic trade-off between shortage and overstock costs. It takes `p_mean` and `p_quant` from Steps 1 & 2 and produces the final inventory recommendation.

### 4.1 Margin Ratio

Each item is classified as high-margin or low-margin relative to its cohort median price:

$$
\text{margin\_ratio} =
\begin{cases}
\text{margin\_ratio\_high} = 1.5 & \text{if } \text{price} > \text{median\_price} \\
\text{margin\_ratio\_low} = 0.7 & \text{if } \text{price} \leq \text{median\_price}
\end{cases}
$$

High-price items incur higher shortage cost (lost revenue is greater), so they receive a higher margin ratio.

### 4.2 Perishability Proxy

Items priced below the mean are flagged as perishable (higher spoilage risk):

$$
\text{shelf\_life\_perishable} = \mathbb{1}[\text{price} < \mu_{\text{price}}]
$$

### 4.3 Shortage Penalty (Cost of Understocking)

$$
\text{Cu} = \text{margin\_ratio} \times (1.0 + 0.3 \times \text{shelf\_life\_perishable}) \times \text{shortage\_margin\_multiplier}
$$

Where `shortage_margin_multiplier = 2.2847` is the default amplification factor tuned via Optuna (search range 1.0–3.0).

Perishable items receive a 30% penalty uplift on the shortage cost.

### 4.4 Overstock Cost (Cost of Overstocking)

$$
\text{spoilage\_penalty} = 0.5 \times \text{shelf\_life\_perishable}
$$

$$
\text{Co} = \left(1.0 - \frac{\text{margin\_ratio}}{\max(\text{margin\_ratio})}\right) \times (1.0 + \text{spoilage\_penalty})
$$

Overstock cost is inversely related to margin ratio — high-margin items have lower relative overstock cost (they can be sold through).

### 4.5 Dynamic Critical Fractile

The optimal service level (critical fractile) is the point where marginal cost of understocking equals marginal cost of overstocking:

$$
\text{CF} = \text{clip}\left(\frac{\text{Cu}}{\text{Cu} + \text{Co}},\ 0.20,\ 0.95\right)
$$

The fractile is clamped to [0.20, 0.95] to prevent extreme inventory policies.

### 4.6 Final Forecast

$$
\text{Final Forecast} = \max\left(\mathbb{E}[X] + \left(\mathbb{Q}_{q}[X] - \mathbb{E}[X]\right) \times \text{CF},\ 0\right)
$$

### 4.7 Recommended Stock & Risk Status

$$
\text{Recommended Stock} = \max\left(\mathbb{Q}_{q}[X],\ \text{Final Forecast}\right)
$$

$$
\text{Risk Status} =
\begin{cases}
\text{HIGH} & \text{if } \text{Recommended Stock} > 1.5 \times \mathbb{E}[X] \\
\text{LOW} & \text{otherwise}
\end{cases}
$$

---

## 5. Key Parameters

| Parameter | Default | Optuna Range | Description |
|---|---|---|---|
| `q_target` | 0.9799 | [0.85, 0.98] | Quantile alpha for pinball loss — higher values produce more conservative stock recommendations |
| `shortage_margin_multiplier` | 2.2847 | [1.0, 3.0] | Amplification factor for shortage cost — higher values push OFR upward at the expense of higher inventory |
| `margin_ratio_high` | 1.5 | Fixed | Margin ratio for above-median-price items |
| `margin_ratio_low` | 0.7 | Fixed | Margin ratio for below-median-price items |
| `use_log_target` | `True` | Fixed | Apply log1p transform to target before training |
| `n_estimators` | 400 | [200, 600] | Number of XGBoost boosting rounds |
| `max_depth` | 5 | [3, 7] | Maximum tree depth |
| `learning_rate` | 0.0339 | [0.01, 0.2] | Boosting learning rate |
| `subsample` | 0.82 | [0.7, 1.0] | Row subsampling ratio |
| `colsample_bytree` | 0.8 | Fixed | Column subsampling per tree |
| `min_obs` | 60 | — | Minimum observations per SKU for inclusion in training |

---

## 6. Feature Inventory (52 Total)

### 6.1 Calendar & Holiday Flags (14 features)

| Feature | Type | Description |
|---|---|---|
| `day_of_week` | int [0–6] | Day of the week (0 = Monday) |
| `week_of_year` | int [1–53] | ISO week number |
| `month` | int [1–12] | Calendar month |
| `quarter` | int [1–4] | Fiscal quarter |
| `day_of_month` | int [1–31] | Day within month |
| `is_weekend` | bool | Saturday or Sunday |
| `is_month_start` | bool | First day of month |
| `is_month_end` | bool | Last day of month |
| `days_to_month_end` | int | Days remaining in current month |
| `week_of_month` | int [1–5] | Week number within month |
| `is_month_start_window` | bool | ±2 days around month start |
| `is_month_end_window` | bool | ±2 days around month end |
| `is_hari_besar` | bool | Public holiday in the item's country |
| `is_pre_hari_besar` | bool | 1–3 days before a public holiday |

### 6.2 Holiday Intensity (4 features)

| Feature | Type | Description |
|---|---|---|
| `holiday_intensity` | float | Ratio of holiday demand to pre-holiday demand (computed per fold, leakage-safe) |
| `days_to_next_holiday` | int | Days until the next holiday (capped at 30) |
| `is_holiday_season` | bool | `is_hari_besar` or `is_pre_hari_besar` |
| `holiday_x_weekend` | int | Interaction: `is_holiday_season * is_weekend` |

### 6.3 Temporal Peak (1 feature)

| Feature | Type | Description |
|---|---|---|
| `is_peak_day` | bool | Top 5% global demand days (computed per fold, leakage-safe) |

### 6.4 Lag / Rolling Statistics (33 features)

| Category | Features | Count |
|---|---|---|
| **Demand Lags** | `demand_lag_{1,2,7,14,21,28,35,56,84}` | 9 |
| **Rolling Max** | `roll_max_{7,28,56,84}` | 4 |
| **Rolling Mean** | `roll_mean_{7,14,28,56}` | 4 |
| **Rolling Median** | `roll_median_{7,14,28}` | 3 |
| **Rolling Std** | `roll_std_{7,14,28,56}` | 4 |
| **Zero Count** | `roll_zero_count_14` | 1 |
| **Acceleration** | `demand_acceleration_3d` (roll_mean_3 / roll_mean_14) | 1 |
| **Spike Ratio** | `spike_ratio_{28,56}` (roll_max / roll_mean) | 2 |
| **Percent Change** | `pct_change_{1,7}` | 2 |
| **Days Since Last Sale** | `days_since_last_sale` (forward-filled from last non-zero) | 1 |
| **Discount Depth** | `discount_depth_pct` (max_price_30d — current) / max_price_30d | 1 |
| **Price Momentum** | `price_momentum` (current_price / mean_price_14d) | 1 |

**All lag and rolling features are shifted by +1 observation to prevent data leakage.**

---

## 7. ONNX Export

The mean regressor can be exported to ONNX format for deployment scenarios requiring low-latency inference outside the Python ecosystem.

**Export function:** `src/model.py:export_to_onnx()`

```python
from src.model import DecoupledActuarialXGB, export_to_onnx

export_to_onnx(model, Path("model.onnx"), feature_cols)
```

**Artifacts produced:**
- `model.onnx` — ONNX-format mean regressor (compatible with ONNX Runtime, TensorRT, etc.)

The ONNX export can be skipped during training via `python run_pipeline.py --skip-onnx` or `python -m src.train --skip-onnx`.

**Note:** Only the mean regressor (Step 1) is exported. The quantile and actuarial layers remain in Python for their SKU-specific logic.

---

## 10. MLflow PyFunc Wrapper

The `DecoupledActuarialWrapper` class (`src/model_wrapper.py`) packages the full pipeline as an MLflow PyFunc model.

**Artifacts stored:**
- `mean_model.json` — XGBoost mean regressor (JSON format)
- `quant_model.json` — XGBoost quantile regressor (JSON format)
- `model_config.json` — all hyperparameters, feature list, and training metadata
- `model.onnx` — (optional) ONNX-format mean regressor

**Prediction output schema (DataFrame):**

| Column | Type | Description |
|---|---|---|
| `expected_demand` | float | Mean forecast (Step 1 output, inverse-transformed) |
| `recommended_stock` | float | Actuarially optimised stock level (Step 3 output) |
| `risk_status` | str | `"HIGH"` if recommended_stock exceeds 1.5× expected_demand, else `"LOW"` |

The wrapper is registered in MLflow Model Registry under the name **`FMCG_Actuarial_Demand_Forecaster`** and loaded by the FastAPI inference server at startup.

---

## 11. Training Flow

```
Full Gold Data (52 features)
    │
    ├── Filter: MIN_OBS >= 60 per SKU
    │
    ├── Compute: is_peak_day (top 5% global demand)
    │
    ├── Compute: holiday_intensity (per fold / full data)
    │
    ├── DecoupledActuarialXGB.fit(X_train, y_train)
    │       ├── Filter y_train > 0
    │       ├── log1p transform
    │       ├── model_mean_.fit() → reg:squarederror
    │       └── model_quant_.fit() → reg:quantileerror (alpha=q_target)
    │
    ├── MLflow: log_params, log_metrics (MAE, RMSE, SMAPE, CLS, OFR, OOS, FVA)
    │
    ├── Save artifacts: mean_model.json, quant_model.json, model_config.json
    │
    ├── ONNX export (optional): model.onnx from mean regressor
    │
    └── MLflow Registry: register if OFR >= 0.80 AND CLS < 50,000
```

---

## 12. Hyperparameter Optimisation (Optuna)

Tuning is performed via `src/tune.py` with a **3-fold expanding-window time-series split**:

| Split | Training Dates | Validation Window |
|---|---|---|
| Fold 1 | Days 0–180 | Days 181–210 |
| Fold 2 | Days 0–210 | Days 211–240 |
| Fold 3 | Days 0–240 | Days 241–270 |

**Search strategy:** Tree-Structured Parzen Estimator (TPE) with 100–200 trials.

**Objective function (minimise):** Mean CLS across 3 folds, with a penalty of `1e6` applied if OFR < 0.80 on any fold.

**Promotion gate:** A trial is registered to the MLflow Model Registry only if `OFR >= 0.80` AND `CLS < 50,000` on the full validation set.
