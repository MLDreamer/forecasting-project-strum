# Fontana Candle Co — Forecasting Runbook

> Hand this document, the three CSVs, and the 60%-WAPE-to-beat mandate to a fresh session. It has everything needed to start at Phase 0 and end at Phase 16 without rediscovering any of the below.

---

## A. What You Find on Disk (Starting State)

Three input files in `/mnt/user-data/uploads/`:

- `variants_export.csv` — 393 rows, **deprecated, ignore**
- `processed_data_filtered.csv` — 11,291 rows of weekly sales
- `product_item_master.csv` — 441 variants (the SKU master)

**Key constraints:**

- No documentation, no schema, no README. CSVs have inconsistent column naming (`source_variant_id` vs `variant_id`) — schema-mapping is mandatory.
- The "weekly sales" file is pre-filtered to non-zero weeks. A SKU that sold zero in week 23 has no row. **Reconstructing zeros is mandatory**; without it, demand-pattern classification is impossible.
- The client is **Fontana Candle Co** — a single-channel Shopify DTC brand. No Amazon, no marketplace, no wholesale.
- The downstream consumer is a cost-minimization inventory optimizer ("replenOS") that picks the cost-optimal quantile per SKU. It needs the **full predictive distribution**, not a point forecast.

---

## B. EDA Findings — What the Data Tells You Before Any Modeling

- Sales span **Dec 2020 → May 2026** (5.5 years). Median SKU has ~80 weeks of history. Don't trust averages — distribution matters.
- **73% of in-scope SKUs** (167 of 229) launched 2024 or later. Cold start dominates. Per-SKU classical models (ETS, ARIMA) are infeasible for the majority — they need ≥26 weeks for seasonality identification.
- **Revenue is extremely concentrated**: top 16 SKUs = 50% of revenue, top 57 = 80%, bottom 100 SKUs = 3.6%. Every metric must be revenue-weighted; uniform-weight aggregation optimizes the wrong thing.
- **Seasonality is extreme**: ISO week 47 (Black Friday) averages ~20.8× the August trough. Most retail series have 2–4× swings. This is nearly an order of magnitude more.
- The 26-week forecast horizon sits exactly on the pre-Black-Friday ramp. **Holiday accuracy = entire business case.**
- **96% of SKUs** have `inventory_policy='deny'` — Shopify blocks oversell. Combined with current zero-inventory, this is the strongest available stockout proxy. There is NO inventory history, only a current snapshot.
- **22 SKUs** show negative discount-sales correlation — reactive markdowns (the brand marks down items because they're already weak). Use lagged discount features only; **never contemporaneous**.
- Sales is gross units ordered. Returns are netted in; no separate returns stream. Modeling gross is the only honest choice for v1.
- The product hierarchy is: **6 vendors → 15 product_types → 359 products → 441 variants** (220 in-scope after filtering: archived, Gift Card, return, re:do vendor excluded).
- Within the 220 in-scope: 159 are Candles, 28 Wax Melts, 16 Accessories, with 6 Soap/Bundle/Bath in the long tail. The `product_type` level is Candle-dominated; `product_id` (192 nodes) is the more discriminating borrowing level.
- No marketing spend, no forward promo calendar, no other channels. Model the data you have; document the gaps; don't pretend.
- No cohort/customer-level data. Forecast is order-level only.

---

## C. Locked Business and Technical Decisions

| Decision | Value |
|---|---|
| Forecast horizon | 26 weeks, configurable in YAML |
| Refit cadence | 13 weeks (quarterly), configurable |
| Lead time | 1 week — Week-1 forecast carries highest business weight; use as per-cluster tiebreaker |
| Week boundary | Sunday-start. Internally use `W-SAT` frequency (pandas convention — week ends on Saturday). Verify with client. |
| Quantiles | 19 levels (P05 → P95, step 0.05). Full distribution required. |
| Primary metric | Revenue-weighted WAPE: `Σ(w·\|y-p\|) / Σ(w·y)`. Aggregate-then-divide, NOT element-wise weighted ratio. |
| Calibration guardrail | 80% interval coverage in [0.75, 0.85]. Reject any candidate outside this, regardless of WAPE. |
| Baseline to beat | 60% WAPE. Target <50%, stretch <40%. |
| Hierarchy | 4 levels: total → product_type → product_id → variant |
| Holidays modeled | Thanksgiving, Christmas, Valentine's Day, Mother's Day (year-accurate rules) |
| Dormancy threshold | 26 weeks. A SKU with no sales for 26+ weeks is marked discontinued. |
| Outliers | Leave them in. Handle via robust loss (quantile loss is inherently robust at the median). |
| Channel | Shopify only. No multi-channel for v1. |
| Cold-start SKUs | 36 deferred to v2. They need a sub-model trained on product attributes + analogy. |
| Random seed | 42 everywhere. Document any RNG use; the reproducibility manifest depends on it. |

**Holiday math must be year-accurate.** Thanksgiving 2024-11-28 / 2025-11-27 / 2026-11-26 must all map to ISO week 47–48, `weeks_to_thanksgiving == 0`.

---

## D. Architecture Principles — Non-Negotiable

- **Config-driven core.** Every client value lives in a YAML. The `src/forecasting/` core never names a client or a model.
- **Plugin model registry.** Models register via decorator (`@register_model(name, segments, requires_columns)`). The CV harness iterates over the registry — never names a specific model in pipeline code.
- **Schema-driven features.** Each feature generator declares required + optional columns. A client without `discount_pct` automatically skips promo features. No `if client == ...` in the core.
- **Versioned outputs.** `forecast_v1.parquet` has a declared schema. Bumping the schema means bumping the version.
- **Reproducibility manifest.** Every run writes `manifest.json`: input hashes, config hash, library versions, seed, git commit. Same inputs + config = byte-identical manifest.
- **Phase-gated build.** Every phase has a verification gate. Failure → STOP, report, no proceed. No phase skipped.

---

## E. Phase-by-Phase Execution

### Phase 0 — Repo Scaffold

Create `src/forecasting/` (client-agnostic core), `configs/` (per-client YAMLs + pydantic `_schema.py`), `tests/`, `scripts/` (thin CLI wrappers), `outputs/`. Python 3.11+. Pin everything in `pyproject.toml`.

Set up CI: `ruff`, `ruff format`, `mypy --strict`, `pytest` with coverage ≥70%. Make these gates non-negotiable.

---

### Phases 1–3 — Data Preparation

**Phase 1 (Load + Schema-Map)**
Read CSVs, rename client columns to canonical names (`source_variant_id` → `sku_id`, etc.) via `cfg.data.schema_map`. Compute scope (filter archived/Gift Card/return).

**Phase 2 (Lifecycle)**
Per SKU, compute `active_start = first_sale`, `active_end = last_sale` if archived or dormant ≥26w, else `data_cutoff`. Trims ~82 SKUs as dormant. Allow `keep_active_overrides` for high-velocity SKUs crossing the dormancy boundary near holiday ramp.

**Phase 3 (Densify)**
Zero-fill within active window. Tag stockout-proxy candidates (mid-series zero runs ≥8 weeks + current `inventory_qty=0` under `inventory_policy=deny`). Result: **17,539 dense rows**, ~35.6% zero rate, 89 stockout-flagged SKUs.

**Phase 3.5 (Config Retrofit)**
Create `configs/_schema.py` (pydantic), `configs/fontana_candle.yaml`, `config_loader.py`, `registry.py`. Promote `product_item_master.csv` as primary SKU source. Set `freq` to `W-SAT`.

---

### Phase 4 — Features

Build a schema-driven feature framework: `@register_feature(name, requires, optional)` decorator. Each generator declares dependencies; orchestrator runs only those whose columns are present.

**13 generator categories, 98 columns before context:**

- Lags
- Rolling stats
- Momentum
- Fourier (period-52, harmonics 1–5)
- Calendar
- Holidays
- Promo/price
- Time-since
- Lifecycle-phase
- Demand-pattern statics
- Product-master
- Stockout-proxy plugin

**Leakage discipline is non-negotiable.** Every lag/rolling/time-since is shifted by ≥1 week. Demand-pattern statics are computed at each CV fold's training cutoff. Test it: `feature[t]` depends only on data observed before `t`.

**Plugins live outside the core.** `src/forecasting_plugins/shopify_inventory_stockout.py` registers via `@register_plugin`. Activates only when client config lists it.

---

### Phase 5 — Segmentation + Clustering

Two orthogonal labels per SKU:

- **SB class** (rule-based Syntetos-Boylan from cv² and IDI) drives model-family eligibility.
- **cluster_id** (K-means on 13 SKU-summary features) drives LightGBM pooling.

**SB class distribution:** erratic 56, lumpy 51, intermittent 34, smooth 33, discontinued 32, cold_start 23.

**K-selection rule (deterministic):** Score each K in [3,12] as `0.7·silhouette(full) + 0.3·mean_ARI(CV folds)`. Ties: smaller K. If no K hits `stability_ari_threshold`, fall back to K=8 with loud warning.

**Categoricals in clustering:** one-hot `product_type` and `revenue_tier`, weighted 0.15 (NOT 0.3, NOT default). Higher weight makes clusters recapitulate `product_type`.

**Honest finding:** silhouette 0.22, ARI 0.34 — both below thresholds. Catalog youth (73% < 2 years) prevents stable clusters. Accept K=8 fallback, document it, and let Phase 15 selection decide empirically. Do NOT try to manufacture structure with embeddings or causal graphs — they don't address the root cause (continuous-feature drift in a young catalog).

Anchor `revenue_tier` to the `data_cutoff`, not per-fold. Trailing-26w revenue rank legitimately shifts across folds in a growing catalog; anchoring removes a definitional instability without leaking validation sales.

**Save versioned artifacts:** `cluster_model_v1_kmeans.joblib`, `cluster_model_v1_scaler.joblib`, `cluster_model_v1_encoding.joblib`.

---

### Phase 5b — Cluster Context Features (LOO)

7 leave-one-out features: for each (SKU, week), summarize the cluster excluding self:

- `cluster_mean_sales_lag1`
- `cluster_median_sales_lag1`
- `cluster_std_sales_lag1`
- `cluster_n_active_lag1`
- `share_of_cluster_lag1`
- `cluster_mean_discount_lag1`
- `cluster_momentum_lag1_to_4`

**LOO is non-negotiable.** Including the SKU in its own cluster aggregate is silent leakage. Single-member clusters return NaN for LOO — document and test this edge case.

---

### Phase 6 — Hierarchy Construction

Generic N-level builder, sparse summing matrix from the start. Use `scipy.sparse.csr_matrix`. For Fontana: **420 nodes × 220 bottom = 880 non-zero entries, 2.2 KB.** Don't use dense.

- Validate `is_bottom` anchor at config-validation time. Exactly one level must have `is_bottom: true`.
- Node IDs are level-prefixed for global uniqueness: `total:TOTAL`, `product_type:Candles`, `variant:46606700773604`.
- **Round-trip test is the load-bearing test:** `assert np.allclose(S @ random_bottom_forecast, computed_aggregate)` for 100 random bottom vectors.
- **Critical bug caught:** pandas `groupby` drops NaN groups silently. 5 SKUs with NULL `product_type` would have been forecast at the variant level but missing from the `product_type` rollup. Map NaN to an `_UNKNOWN` sentinel + assert column-sum coherence.

---

### Phase 6b — Hierarchy Context Features (LOO)

8 features borrowing across the hierarchy:

- `product_id` sibling mean/n_active/share (LOO over `product_id`)
- `product_type` mean/YoY/seasonality_strength/share (LOO over `product_type`)
- Total YoY (NOT LOO at the root — leave-one-out on a 220-member denominator is unstable; use the raw portfolio aggregate)

The **cold-start lifesaver**: a 12-week-old SKU inherits `product_type_yoy_ratio_lag52` and `product_id_sibling_mean_sales_lag1` even though it can't compute its own `sales_lag_52`.

---

### Phase 7 — Metrics Module

All metrics revenue-weighted + per-horizon-aware (h=1, 4, 13, 26). The `horizon` param + `horizons: list[int] | None` returns `dict[int, float]` instead of a scalar.

| Metric | Notes |
|---|---|
| **WAPE** | Canonical form. Denominator is `sum(y_true)`. NaN + warn if denom ≤ 0. Aggregate-then-divide, never element-wise. |
| **Pinball** | 9:1 asymmetry at α=0.9. Under-forecasting penalized 9× more than over-forecasting. Test this explicitly. |
| **CRPS** | K-approximation from quantiles — slight underestimate, documented. Consistent across models so comparisons stay fair. |
| **WIS** | Bracher 2021 interval form (not the average-pinball form). Decomposes into sharpness + over- + under-prediction penalties. |
| **MASE** | In-sample t-52 seasonal denominator. NaN if training series < 53 weeks. Aggregate by revenue-weighting per-SKU MASE values, not the components. |

**Test the three revenue-weighting proofs:** equal weights ≡ unweighted; zero weight excludes that SKU; single-nonzero-weight ≡ that-SKU-only metric.

---

### Phase 8 — Baselines + ABC

`ForecastModel` ABC has `fit()` + `predict()`. Returns `ForecastResult(model_name, sku_ids, forecast_weeks, quantile_levels, quantiles)`.

Two `ForecastResult` constructors:
- `from_quantiles` (Moirai, LightGBM, seasonal naive)
- `from_samples` (Chronos: empirical quantiles from MC samples)

Both funnel through `_finalize` which floors at 0 and sorts to enforce monotonicity.

**Two baselines, registered:**
- `seasonal_naive` — lag-52 + residual quantiles, eligible ALL segments
- `zero` — eligible discontinued

Every other model must beat these or it doesn't ship.

---

### Phase 9 — Classical Models

Three statsforecast wrappers behind a shared `_ConformalStatsModel`: `auto_ets` (smooth, erratic), `auto_arima` (smooth), `theta` (smooth, erratic).

**Split-conformal quantiles:** hold out the last 13 weeks per SKU, forecast them, pool calibration residuals into `q_α = point + quantile(residuals, α)`. Floored + sorted by `ForecastResult`.

- Skip SKUs with <26 weeks of history — `statsforecast` can't fit reliably below that; `seasonal_naive` covers them.
- Zero-pad trimmed series to global cutoff so every SKU forecasts the same weeks (preserves the `ForecastResult` cube shape).

---

### Phase 10 — Intermittent Models

Two statsforecast wrappers: `croston_sba` and `tsb`. Eligible: intermittent + lumpy.

Point rate from the statsforecast model, quantile spread from a **vectorized compound-Bernoulli bootstrap** of the empirical demand process. Occurrence `p = n_nonzero/n_obs`; sizes resampled from observed non-zero demand. Rescale the bootstrap so its mean matches the model's rate.

> **Watch out:** statsforecast's intermittent path imports `scipy.sparse.linalg.svds`, which uses PROPACK. OneDrive-synced venvs corrupt the `_propack.pyd` extension. Pin `scipy>=1.13,<1.16` in `pyproject.toml`; statsforecast 2.x requires `<1.16` anyway.

---

### Phase 11 — Cluster-Pooled LightGBM (The Big One)

**Architecture:** one quantile booster per (cluster, level) = **8 × 19 = 152 boosters**. Direct multi-step via `horizon_step` (1..H) as a feature. Categoricals (`sku_id`, `cluster_id`, `product_type`, `status`, `revenue_tier`) use LightGBM's native handling.

- Discontinued SKUs (`cluster_id = -1`) are excluded. They forecast as zero via `ZeroModel`.
- Early stopping on the last 13 weeks. Training rows are direct multi-step: `features-at-t × horizon_step → sales[t+s]`. Excluded base weeks `> cutoff - 13` form the validation set.

**Quantile crossing is real and non-trivial here.** Independent quantile boosters cross 29% per-pair pre-sort on this dataset (range 1.7%–37.5% across clusters). The mandated sort+floor finalization repairs ordering, but the magnitude is real (up to 30% of row scale in high-variance clusters). Log the per-pair rate; row-any is meaningless (it'll be 100%).

**80% PI under-covers: 70.4%**, below the 0.75 floor. This is on the early-stopping window. Phase 15 guardrail will catch it.

---

### Phase 11.5 — A Failed Experiment Worth Knowing About

**Tested:** adding target-week deterministic features (`target_week_iso_week`, `target_week_month`, `target_week_sin/cos_52_1..2`, `target_week_weeks_to_*` for each holiday).

**Result:** it didn't pay off cleanly. Crossing got worse (29→36%), 80% coverage got worse (70.4→68.3%), but point WAPE on the early-stopping window improved (h1: 0.416→0.334, h4: 0.397→0.308). **Reverted to Option A.**

**Lesson:** the as-of-t feature set already includes the full Fourier basis. Adding target-week columns is mostly a deterministic rotation of existing features by `horizon_step` — same information, different angle. The genuinely new signal (holiday-proximity features) is too small to overcome the collinearity cost to independent quantile heads on thin per-cluster data.

**Implication for the AutoML system:** test feature-engineering hypotheses behind a flag, with A/B comparison on the calibration metrics (not just point WAPE). Reverting cleanly when a hypothesis doesn't pay off is a feature, not a setback.

---

### Phase 12 — Tweedie GLM for Lumpy Demand

`TweedieGLMModel` registered as `tweedie`, eligible lumpy. Per-SKU compound Poisson-Gamma GLM (variance power p ∈ (1, 2), log link) on an annual Fourier seasonal design.

Quantiles via simulation: for each future week, simulate (μ, p, φ) compound Poisson-Gamma at `n_samples=2000`, take empirical quantiles via `from_samples`.

**Graceful fallback chain:** seasonal GLM → intercept-only GLM → empirical mean + method-of-moments dispersion. `statsmodels` lazy-imported.

**Smoke check on real data:** 51 lumpy SKUs → 50 seasonal fits, 1 intercept-only, 0 degenerate. P50 median (11.7) < historical mean (14.6) — correct right-skew for sporadic demand.

---

## F. Pitfalls Already Discovered — Don't Repeat

- **Pandas `groupby` drops NaN groups silently.** Anywhere you group by a column that could have NaN (`product_type`, `vendor`, `status`), map NaN to an `_UNKNOWN` sentinel BEFORE grouping. Test for it.
- **OneDrive-synced venvs corrupt compiled extensions.** Pin `scipy <1.16` to dodge PROPACK; expect to occasionally `pip install --force-reinstall` when a `.pyd` mysteriously breaks.
- **`config: dict` parameter shadows the `forecasting.config` module.** Use `from forecasting.config import COL_SALES, COL_SKU_ID` at top-of-file in model classes that accept a config dict.
- **`ForecastResult.values` triggers ruff PD011** (looks like the pandas `.values` accessor). Name the field `quantiles` instead.
- **`pptxgenjs` mutates option objects in place.** Don't reuse a shadow dict across multiple `addShape` calls — use a factory function returning a fresh dict each time.
- **`int(sku)` on a pandas groupby key fails mypy.** Cast: `int(cast("int", sku))`.
- **K-means with a tight `min_cluster_size` constraint can fail silently on weak structure.** Auto-reduce K with a warning rather than silently violating the config.
- **The 60% baseline is the AutoML setup's WAPE, not industry-standard.** Don't bake that number into the code or docs as a generic floor; it's a per-client target.

---

## G. What's Still Ahead

**Phase 13 — Foundation Models**
Chronos-T5-tiny (Amazon, 8M params, MC samples → empirical quantiles) and Moirai-Small (Salesforce, 14M, native quantile head). Apache 2.0, CPU-runnable. They're the cold-start lever. Use authors' defaults; don't tune.

**Phase 14 — Rolling-Origin CV**
4 folds. At each fold, re-derive lifecycle, features, segments using only data ≤ training cutoff. Fit every eligible candidate × every SKU. Score all metrics per-horizon.

**Phase 15 — Per-Cluster Winner Selection**
Filter candidates by calibration guardrail (80% coverage in [0.75, 0.85]). Pick lowest revenue-weighted WAPE on folds 2–4. Tiebreaker: WAPE at h=1.

**Phase 16 — Bottom-Up Probabilistic Reconciliation**
500-path bootstrap from each SKU's quantile CDF. Sum across SKUs at each hierarchy node. Take quantiles of the summed paths. NOT naive sum of bottom quantiles — that systematically over-states uncertainty at upper levels.

---

## H. The Notebook-Config Question

The `lego['forecasting']['weeks_to_avoid_forecasting'] = 26` dict-of-dicts syntax is worse than what's already there. The pipeline uses pydantic Config objects with typed attribute access:

```python
# Good — IDE completes; mypy checks; pydantic validates
cfg.time.forecast_horizon_weeks = 26

# Bad — string keys; typo-prone; no validation
lego['forecasting']['weeks_to_avoid_forecasting'] = 26
```

Stick with pydantic. But yes, support notebook-based config — three small additions:

1. **Make `run()` accept either a YAML path OR a pre-loaded Config object.** Currently it takes a path; one-line change to accept either.
2. **Add a `save_config(cfg, path)` helper** that writes the pydantic model back to YAML. Closes the loop: load → mutate → save → commit.
3. **Add a `notebooks/quickstart.ipynb`** showing the pattern:

```python
from forecasting.config_loader import load_config, save_config
from forecasting.run import run

cfg = load_config('configs/fontana_candle.yaml')
cfg.time.forecast_horizon_weeks = 13          # experiment with shorter horizon
cfg.quantiles.levels = [0.1, 0.5, 0.9]        # fewer quantiles for fast iteration
cfg.models.candidates_per_segment.smooth.remove('auto_arima')  # drop slow candidate

run(cfg=cfg, stop_after='features')           # accepts Config or path
save_config(cfg, 'configs/experiment_short_horizon.yaml')  # commit when satisfied
```

This gets you notebook iteration without losing the YAML-as-canonical-truth contract that the AutoML orchestrator and reproducibility manifest depend on. The notebook user mutates in memory; production YAML is what gets committed and versioned.

> **One thing to forbid:** notebook-driven runs that write to `outputs/` without committing the corresponding YAML. The manifest hashes the config — if the config isn't on disk, the manifest can't reproduce. Either commit the YAML first, or use a `--dry-run` flag for exploration that skips manifest writing.
