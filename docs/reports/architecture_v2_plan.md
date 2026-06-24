# Architecture V2 Plan — Full Rethink
## Based on 4-fold sequential CV evidence + domain diagnosis

---

## 1. The Core Diagnosis

### What the current 6-segment taxonomy is missing

The SB classification (smooth / erratic / lumpy / intermittent / cold_start / discontinued)
describes DEMAND SHAPE. It says nothing about WHY demand behaves that way.

The root cause of the high WAPE in folds 2 and 4 (holiday, promotional periods) is that
a large fraction of SKUs are NOT structurally erratic or lumpy — they are EXTERNALLY DRIVEN.
Their demand is a direct function of:
  - Discount % and price promotions (Shopify discount_pct is already in the data)
  - Price position relative to recent history (price_vs_roll13 feature exists)
  - Q4 seasonality / holiday calendar (is_q4, hol_christmas, hol_black_friday exist)
  - Catalog newness (sku_age_weeks exists)

A SKU that is "erratic" purely because it gets a 40% discount every 8 weeks and goes
to zero otherwise is NOT the same forecasting problem as a SKU that is genuinely erratic
due to random demand variation. The former needs price/promo features as primary inputs.
The latter needs time-series pattern recognition.

This is the REGIME CHANGE problem:
  - Regime 0: baseline / no promo
  - Regime 1: active promo / holiday
  - Regime 2: post-promo / hangover
  - Regime 3: launch / ramp-up
  - Regime 4: end-of-life / discontinuation trend

Current models have no regime awareness. They see a time series with spikes and call it
"erratic" when the spikes are entirely explained by discount_pct.

---

## 2. New Segmentation — 8 Named Classes

Replace 6-class SB taxonomy with 8 classes that combine demand shape AND causal driver:

| Class | ADI | CV2 | Additional condition | Root cause | Best model family |
|---|---|---|---|---|---|
| promo_driven | any | high | corr(sales, discount_pct) > 0.4 | Demand is promo response | LightGBM (price+promo features dominant) |
| smooth_growing | low | low | YoY growth > 20% | Organic growth | Theta / AutoETS with growth |
| smooth_stable | low | low | YoY growth <= 20% | Mature stable | SeasonalNaive / recent_level |
| erratic | low | high | corr(sales, discount_pct) < 0.4 | True demand volatility | Theta / LightGBM |
| lumpy | high | high | any | Burst demand | SeasonalNaive / CompoundBernoulli |
| intermittent | high | low | any | Sparse but regular | CrostonSBA / TSB |
| short_history | any | any | < 52w AND hierarchy parent active | Borrow from hierarchy | LightGBM (LOO hierarchy features) |
| discontinued | any | any | zero last 26w + zero prior 52w | Dead | ZeroForecast |

Key changes vs current:
- Split smooth into smooth_growing vs smooth_stable (different models needed)
- Add promo_driven as explicit class (currently buried in erratic)
- Rename cold_start -> short_history and give it hierarchy borrowing explicitly
- Tighten discontinued definition (must be truly dead, not seasonal off-peak)

---

## 3. New Features — External Variables

### 3a. Promo / Price features (already partially in data)
- discount_pct_lag1, discount_pct_lag2 (already built)
- is_on_promo: discount_pct > 0.15 (binary)
- promo_intensity: rolling 4w mean of discount_pct
- price_drop_flag: price this week < price_roll13_mean * 0.85
- post_promo_flag: is_on_promo was 1 last week, 0 this week (hangover effect)
- promo_frequency_13w: how many of last 13 weeks had discount > 0.15

### 3b. Price position
- price_vs_catalog_mean: this SKU price / mean price of all SKUs in same product_type
- price_tier: top/mid/bottom tercile within product_type (already as revenue_tier but refine)
- price_elasticity_proxy: corr(sales, -discount_pct) over rolling 26w window

### 3c. Hierarchy borrowing (for short_history SKUs)
- Already have LOO cluster aggregates and product_type aggregates
- Add: product_type_p50_trend (median SKU trend within product type)
- Add: product_type_launch_age (weeks since first SKU in this product_type launched)
- Add: sibling_sku_count (number of active SKUs in same product_type)

### 3d. Demand regime features
- demand_regime: 0=baseline, 1=promo, 2=post_promo, 3=launch, 4=decline
  (derived rule-based from discount_pct, sku_age_weeks, recent trend)
- yoy_growth_13w: mean(last 13w) / mean(same 13w last year) — clipped [-2, 5]
- acceleration: roll4_mean / roll13_mean (is demand accelerating recently?)
- momentum_sign: sign of (roll4_mean - roll13_mean)

### 3e. Calendar / seasonality (add to existing)
- weeks_to_christmas: min(weeks until Dec 25, 52) — decays as holiday approaches
- is_q4_ramp: week_of_year in [40..52] (Oct-Dec build)
- is_post_holiday: week_of_year in [1..8] (Jan-Feb slowdown)

---

## 4. Model Pool — Full Roster

### Statistical
- SeasonalNaive: universal fallback, last-year-same-week
- Theta (DynamicOptimizedTheta): best for smooth_growing + erratic (fold 3 evidence: 0.67)
- AutoETS: smooth_stable
- AutoARIMA: smooth_stable fallback

### Intermittent
- CrostonSBA: intermittent (low CV2, high ADI)
- TSB: intermittent + lumpy (handles zero-inflation)
- CompoundBernoulli: lumpy (fold 1 evidence: best lumpy at 1.01)

### ML
- ClusterPooledLGBM: primary workhorse for promo_driven + short_history
  - 114 features INCLUDING all new promo/price/regime features
  - K=3 revenue clusters
  - Direct multi-step (horizon_step as feature)
  - Sequential hyperparameter tuning across folds (already implemented)

### Foundation Models (zero-shot, for short_history)
- Chronos-T5-tiny: zero-shot, ~0.5s/SKU on CPU
  - Best for: short_history SKUs with no seasonal pattern to borrow
  - Evidence from docs: fold 3 intermittent WAPE 0.990 (better than CompoundBernoulli)
- Moirai-Small: better zero-shot than Chronos for multi-variate (needs Linux)
  - Fallback: Chronos on Windows

### Hierarchy borrowing model (new)
- HierarchyBorrowModel: for short_history SKUs
  - P50 = weighted average of (own mean, product_type P50)
  - Weight on own history increases with sku_age_weeks / 52
  - At 0 weeks: 100% product_type mean
  - At 52 weeks: 50/50 blend
  - At 104 weeks: 100% own history

---

## 5. Model Routing by Segment

| Segment | Primary | Fallback | Rationale |
|---|---|---|---|
| promo_driven | cluster_lgbm | seasonal_naive | Price/promo features critical. LightGBM reads discount_pct, price_drop_flag |
| smooth_growing | theta | auto_ets | Theta captures trend+seasonality. Evidence: fold 3 erratic 0.665 |
| smooth_stable | seasonal_naive | recent_level | Last-year is best for stable mature SKUs |
| erratic | theta | cluster_lgbm | Theta fold3=0.665, LGBM fold2=0.662. Use theta, fallback LGBM |
| lumpy | seasonal_naive | compound_bernoulli | SN best fold2=0.79, fold3=0.59 |
| intermittent | seasonal_naive | croston_sba | SN fold3=0.61, dominant |
| short_history | hierarchy_borrow + lgbm | seasonal_naive | No seasonality to copy. Borrow from hierarchy + LightGBM promo features |
| discontinued | zero_forecast | — | Dead SKUs |

---

## 6. Sequential Learning Protocol (refined)

Each fold teaches the next. Changes that propagate forward:

### After each fold:
1. **Routing update**: if model A beats model B on segment S by >0.02 WAPE, switch routing[S] = A
2. **LightGBM tuning**:
   - If LGBM WAPE > SeasonalNaive: increase num_leaves *1.5, decrease lr *0.7
   - If LGBM WAPE < SeasonalNaive: increase reg_alpha/reg_lambda *1.2 (prevent overfit)
   - If LGBM wins promo_driven segment: lock in and only tune that cluster
3. **Feature pruning**: drop bottom 20% by LightGBM importance (protect core lags/Fourier)
4. **Segmentation adjustment**:
   - After fold 2: recompute promo_driven threshold (corr cutoff) based on fold 1 evidence
   - After fold 3: if smooth_growing has < 5 SKUs, merge back into erratic
5. **Regime detection update**:
   - After each fold: refit the demand_regime rule thresholds using holdout residuals

### What we carry forward fold-to-fold:
- LightGBM hyperparameters (updated)
- Segment routing map (updated per-segment)
- Feature importance ranking (accumulates across folds)
- Dropped features set (grows each fold, never re-added)

---

## 7. WAPE Protocol (locked)

- **Metric**: plain WAPE = sum|actual - p50| / sum(actual)
- **Eligibility**: >= 52 weeks training history at fold origin
- **Exclusions**: discontinued (zero-forecasted, unexpected reactivation excluded)
- **Folds scored**: 2, 3, 4 (fold 1 = warmup only)
- **Target**: WAPE < 0.50 on folds 3 AND 4 individually

---

## 8. Expected WAPE improvement from V2

| Change | Expected delta WAPE |
|---|---|
| promo_driven segment + LightGBM with price features | -0.05 to -0.10 |
| smooth split (growing vs stable) | -0.02 to -0.05 |
| Chronos for short_history | -0.05 (removes cold_start WAPE=1.0) |
| Hierarchy borrowing for short_history | -0.03 to -0.05 |
| Regime features (demand_regime, acceleration) | -0.03 to -0.05 |
| Total expected | -0.15 to -0.25 |

Current best: fold 3 = 0.70, fold 4 TBD
Target: fold 3 + 4 both < 0.50

---

## 9. Implementation order (once fold 4 result is in)

1. Add new features: promo_driven detection, regime features, hierarchy borrow features
2. Implement new 8-class segmentation (extend segment.py)
3. Implement HierarchyBorrowModel (new model class)
4. Wire Chronos for short_history segment
5. Re-run sequential CV with full model pool
6. Commit forecast_26w.csv

*Written after 4-fold sequential CV. Fold 4 result pending.*
