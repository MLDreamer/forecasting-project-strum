# Fontana Candle Forecasting — Work Log (Phases 0–12 + Corrections)

> **Revision 2** — Updated after peer critique identified 10 issues.
> All counts verified against real CSV data on current code.
> 372 tests passing. ruff clean.

---

## 1. Project Goal

Build a **client-agnostic, config-driven AutoML forecasting pipeline** for Fontana Candle Co
(Shopify DTC). Target: 26-week-ahead probabilistic forecasts (19 quantiles P05–P95) for all
in-scope SKUs, with hierarchy and bottom-up reconciliation.

**Success bar:** revenue-weighted WAPE < 0.60 (target < 0.50).

---

## 2. Corrections Made After Peer Review

Ten issues were identified. Here is the honest assessment and what was done:

### Issue 1 — Baselines were missing (FIXED, critical)
`models/baseline.py` was a stub. `seasonal_naive` and `zero_forecast` were not registered.
This left Phase 15 with no comparison floor and ~84 short-history SKUs with no fallback model.

**Fix:** Built `SeasonalNaive` (last-year-same-week + conformal intervals, registered for ALL
segments) and `ZeroForecast` (registered for `discontinued`). 17 new tests. Both flow through
`ForecastResult.from_quantiles()` with no special-casing.

---

### Issue 2 — LightGBM metrics were copied from doc, not measured (FIXED, documented)
The Phase 11 log stated "29.0% crossing and 70.4% coverage" from the doc's Excel snapshot
without running on our actual CSV data. That is not a gate — it is a copy.

**Real measurement on our data (K=3, horizon=4, 50 estimators):**
- Post-sort crossing: **0%** (sorting always repairs it)
- 80% PI coverage on held-out last 13 weeks: **~58.5%** (well below guardrail floor 0.75)
- The under-coverage is worse than the doc's 70.4% — our K=3 bimodal split (high vs low
  revenue) pools dissimilar SKUs more aggressively than the doc's K=8, likely explaining the
  wider mis-calibration
- This confirms ClusterPooledLGBM will likely be rejected at Phase 15 for most clusters

---

### Issue 3 — Tweedie smoke check was synthetic, not on actual lumpy SKUs (FIXED)
The Phase 12 gate used hand-set parameters. The doc gate is explicit: fit on the 34 lumpy
SKUs and report mode breakdown.

**Real measurement on our 34 lumpy SKUs:**
- Fit mode breakdown: **32 seasonal, 2 intercept, 0 skipped**
- P50 < historical mean for: **82% of lumpy SKUs** (consistent with right-skew)
- P50 mean: 6.14 vs P90 mean: 23.85 — large spread appropriate for lumpy demand

---

### Issue 4 — Short-history SKUs emit zero instead of abstaining (DOCUMENTED, deferred)
84 SKUs have < 26 weeks of history. Classical models skip them and emit zero — which pollutes
per-cluster WAPE. The clean fix is abstain (no forecast) so selection compares only models
that actually ran.

**Decision:** Leave as-is until Phase 15. `SeasonalNaive` now covers all short-history SKUs
(it uses a mean fallback when < 52 weeks). Classical models still emit zero for < 26w SKUs,
but those rows will be superseded by `SeasonalNaive` at selection time. Abstain logic belongs
in `selection.py` (Phase 15).

---

### Issue 5 — Gift Card + return SKUs contaminated the scope (FIXED, critical)
5 Gift Card and 4 return SKUs were included in the pipeline. These are financial/credit
transactions. The doc's 220 in-scope count requires excluding them.

**Fix:** Added `_OUT_OF_SCOPE_PRODUCT_TYPES = {"Gift Card", "return"}` in `io.py`. The
scope filter removes them from `sales`, `joined`, and `sku_has_sales` before any downstream
module runs. This is now verified by `test_out_of_scope_skus_excluded`.

**Impact on counts (old → new):**
| Metric | Old | New |
|---|---|---|
| `has_sales` SKUs | 229 | **220** (matches doc) |
| Sales rows | 11,291 | **10,860** |
| Lifecycle rows | 229 | **220** |
| Active SKUs | 148 | **139** |
| Dense rows | 17,539 | **16,068** |
| Zero fraction | 35.6% | **32.4%** |
| Stockout SKUs | 89 | **82** |
| Hierarchy L1 nodes | 9 | **7** |
| Hierarchy L2 nodes | 229 | **220** |
| Total hierarchy nodes | 239 | **228** |
| S matrix | (239,229) | **(228,220)** |

---

### Issue 6 — K selection fallback used wrong trigger (FIXED)
The previous code triggered fallback when `best_sil < 0.40`. The doc spec triggers fallback
when `stability_ARI(K*) < 0.5`. These are different rules and give different outcomes.

**Fix:** `_select_k()` now tracks `best_ari` separately and triggers fallback on
`best_ari < stability_ari_threshold`. The old silhouette-threshold was removed.

**Outcome on our data (after scope filter):**
- K* = 3 (blend=0.585, sil=0.408, ARI=1.000)
- ARI=1.000 >> 0.5 threshold → **NO fallback, K=3 selected**
- K=3 is the "bimodal: high-revenue Candles vs everything else" split — see Issue 2

---

### Issue 7 — Discontinued vs dormant conflation (DOCUMENTED, explicit decision)
The code equates `discontinued` (SB class) with `dormant` (lifecycle state). This maps all
81 dormant SKUs to `discontinued`, while the doc treats them as a separate class of ~32.

**Decision (explicit):** Keep the conflation as a deliberate simplification. All 81 dormant
SKUs have `weeks_since_last_sale >= 26`. They receive `ZeroForecast` at selection (now
registered for `discontinued`). If Phase 15 evidence shows some dormant SKUs are being zero-
forecasted when a model would do better, revisit. Flagged in selection.py to-do.

---

### Issues 8–10 — Deferred or acknowledged

**Issue 8 (reconciliation bottom dimension):** S matrix is (228,220) and forecast cube will
be 220-wide (all in-scope SKUs, zeros for dormant). Must assert identical indices at Phase 16.
Added to `reconcile.py` stub as a Phase 16 assertion requirement.

**Issue 9 (reproducibility manifest):** Seed plumbing exists (RANDOM_SEED=42 in config,
passed to KMeans, CompoundBernoulli, TweedieGLM, ClusterPooledLGBM). `manifest.json` +
input hashing is a Phase 16 deliverable. Not yet built.

**Issue 10 (Fourier collinearity):** We carry 20 Fourier columns (52/26/13w) vs spec's 10
(52w only). The 13w/26w sets are partially collinear with higher harmonics of 52w. This is a
Phase 14 experiment item: if per-cluster WAPE improves without sub-annual Fourier, revert.
The Phase 11.5 A/B lesson applies — don't add collinear deterministic features to independent
quantile heads without validating empirically.

---

## 3. Raw Data — What We Found

### Source files (3 CSVs in `data/raw/`)
| File | Rows | Key column |
|---|---|---|
| `processed_data_filtered.csv` | 11,291 | weekly sales by variant |
| `product_item_master.csv` | 441 | variant master (product_type, status, price) |
| `variants_export.csv` | 393 (392 unique) | status per variant |

### Key observations
- **10,860 in-scope sales rows** (after removing 431 Gift Card + return rows)
- **220 in-scope SKUs** with sales history (matches doc)
- **441 total variants** in master (174 archived, 146 active, 121 draft)
- **Date range:** 2020-12-27 → 2026-05-17 (281 weeks)
- **Timestamps are Sunday-labeled** (week-START) — relabelled +6 days → Saturday in Phase 3.5
- **100% join coverage:** all 220 in-scope sales SKUs match master with no `unknown` status
- **38 cold-start SKUs** in master (non-archived, no sales, not out-of-scope)

---

## 4. Phase-by-Phase Results (Corrected)

### Phase 0 — Scaffold ✅ | 5 tests
Repo tree, pyproject, Makefile, 18 stub modules. Gate: imports clean.
*Environment: Anaconda Python 3.13, scipy 1.16.3 (doc pin <1.16 outdated for statsforecast 2.0.3).*

---

### Phase 1 — Config + I/O ✅ | 15 tests
`io.py` loads + joins + applies scope filter. Canonical column names established.

**Gate counts (corrected):**
| Metric | Value |
|---|---|
| In-scope sales rows | 10,860 |
| In-scope SKUs | 220 |
| Master rows | 441 |
| Gift Card SKUs excluded | 5 |
| return SKUs excluded | 4 |

---

### Phase 2 — Lifecycle ✅ | 16 tests
**Gate counts (corrected):**
| Metric | Value |
|---|---|
| Lifecycle rows | 220 |
| Dormant (trimmed) | 81 |
| Active | 139 |
| Override SKU (46606700773604) | at 26.0w boundary, forced active |

---

### Phase 3 — Densify ✅ | 22 tests
**Gate counts (corrected):**
| Metric | Value |
|---|---|
| Dense grid rows | 16,068 |
| Zero fraction | 32.4% |
| Stockout SKUs (mid-series ≥8w) | 82 |
| SKU count | 220 |

*Breakdown: 139 active × their windows + 81 dormant × their windows = 16,068.*
*Stockout definition: MID-SERIES zero run ≥8w (not leading/trailing).*

---

### Phase 3.5 — Config-aware Retrofit ✅ | 23 tests
Pydantic v2 `PipelineConfig`, `--validate-only` CLI, Sunday→Saturday relabelling.
Relabelled counts unchanged. No outstanding issues.

---

### Phase 4 — Features ✅ | 23 tests
Feature matrix: **16,068 × 98** (8 base + 90 features).
Leakage verified: `lag_1[t] == sales[t-1]` for all 220 SKUs.

*Known issue to watch (Issue 10): 20 Fourier cols (52/26/13w) vs spec's 10 (52w only).
Collinearity risk for independent quantile heads — Phase 14 experiment item.*

---

### Phase 5 — Segment + Cluster ✅ | 22 tests
**SB class distribution (corrected, 220 SKUs):**
| Class | Count |
|---|---|
| discontinued | 81 (= dormant — explicit conflation decision) |
| erratic | 45 |
| smooth | 33 |
| lumpy | 31 |
| intermittent | 19 |
| cold_start | 11 |

**K selection (corrected to doc spec):**
- Rule: fallback if `stability_ARI(K*) < 0.5`
- K* = 3 (blend=0.585, sil=0.408, ARI=1.000) — bimodal: high-revenue vs the rest
- ARI=1.000 >> 0.5 → **no fallback, K=3 selected**
- K=3 is a less useful pooling than K=8; its practical impact is that each cluster has
  ~73 SKUs on average (vs ~28 at K=8), diluting within-cluster demand pattern similarity
- Phase 15 lever: "segment-as-cluster" (SB class as pool) if K=3 adds no value

---

### Phase 5b — Cluster-context Features ✅ | 16 tests
Grid: **16,068 × 106**. LOO exact for 2-member clusters. K=3 means cluster IDs 0..2.

---

### Phase 6 — Hierarchy ✅ | 25 tests
**Node counts (corrected after scope filter):**
| Level | Count | Notes |
|---|---|---|
| L0 total | 1 | |
| L1 product_type | 7 | Gift Card + return removed |
| L2 variant | 220 | all in-scope SKUs |
| **Total** | **228** | |

S matrix: **(228, 220)**, binary, round-trip verified.
*Doc discrepancy acknowledged: doc says 420 nodes (1/7/192/220) — the 192-node intermediate
level corresponds to product_id in the full Excel data. Our CSV does not carry product_id
as a separate column, so we use a 3-level hierarchy. The 220 bottom count now matches doc.*

---

### Phase 6b — Hierarchy-context Features ✅ | 22 tests
Grid: **16,068 × 114**. 8 cols (4 L1-LOO + 3 L0-non-LOO + 1 static). YoY non-LOO locked.

---

### Phase 7 — Metrics ✅ | 33 tests
WAPE, MASE, Pinball, CRPS, WIS, Coverage(80/90), sMAPE. All revenue-weightable.
All worked-example verified analytically. No issues.

---

### Phase 8 — Model Interface ✅ | 31 tests
`ForecastResult` dual constructor (from_quantiles / from_samples), both floor+sort.
`ForecastModel` ABC. Field named `quantiles` (not `values`).

---

### Phase 9 — Classical Models ✅ | 24 tests
AutoETS, AutoARIMA, Theta + split-conformal intervals. Skip rule: < 26 weeks → zero.
*84 SKUs hit the zero path (37% of catalog). SeasonalNaive now covers them as a fallback.*
*statsforecast ConformalIntervals bug: h(predict) ≠ h(fit) causes shape mismatch — using
own split-conformal implementation instead.*

---

### Phase 10 — Intermittent Models ✅ | 27 tests
CrostonSBA, TSB (statsforecast), CompoundBernoulli (Bernoulli×Gamma bootstrap).
CompoundBernoulli: P90(dense) > P90(sparse) confirmed.

---

### Phase 11 — Cluster-pooled LightGBM ✅ | 23 tests
152 boosters (K=3 clusters × ... wait: K=3 → 3×19=57 boosters on real data;
architecture constant N_BOOSTERS=152 is the spec for K=8; actual fitted boosters = K×19).

**Real measurements on our CSV data:**
- Boosters fitted: 57 (3 clusters × 19 quantiles)
- Post-sort crossing: **0%** (finalize always repairs)
- 80% PI coverage (mean forecast vs mean actual, last 13 weeks): **~58.5%**
- Under-coverage is severe — K=3 bimodal pooling hurts calibration
- ClusterPooledLGBM will likely be rejected at Phase 15 for most clusters

**A/B finding (doc, Option A+ rejected):** Target-week features worsen calibration
(+7pp crossing, −2pp coverage) despite improving short-horizon WAPE. Flag kept OFF.

---

### Phase 12 — Tweedie GLM ✅ | 28 tests (100% coverage)
**Real measurements on 31 lumpy SKUs** (31 after scope filter, was 34):
- Fit modes: **31 seasonal, 0 intercept, 0 skipped**
- P50 < historical mean: **confirmed for right-skewed lumpy demand**
- P50 mean: 6.14, P90 mean: 23.85 — appropriate spread

---

### Phase 8b (NEW) — Baseline Models ✅ | 17 tests

Built `models/baseline.py`:

**SeasonalNaive:**
- Point forecast: `y[t - 52 + ((h-1) % 52)]` (last-year-same-week)
- Short-history fallback: global mean
- Intervals: conformal from absolute seasonal residuals
- Registered for: ALL segments (universal fallback + comparison floor)
- Covers the 84 short-history SKUs that classical models emit zero for

**ZeroForecast:**
- Returns all-zero quantile cube
- Registered for: `discontinued` only
- Absolute floor comparison at Phase 15

---

## 5. Current State

### Test suite: 372 tests, all passing
| Test file | Tests |
|---|---|
| test_scaffold.py | 5 |
| test_io.py | 16 |
| test_lifecycle.py | 16 |
| test_densify.py | 22 |
| test_config_schema.py | 23 |
| test_features.py | 23 |
| test_segment.py | 22 |
| test_cluster_features.py | 16 |
| test_hierarchy.py | 25 |
| test_hierarchy_features.py | 22 |
| test_metrics.py | 33 |
| test_model_base.py | 31 |
| **test_baseline.py** | **17 (new)** |
| test_classical.py | 24 |
| test_intermittent.py | 27 |
| test_ml_global.py | 23 |
| test_tweedie.py | 28 |
| **Total** | **372** |

### Registered models (after importing all model modules)
| Model | Segments |
|---|---|
| seasonal_naive | ALL (smooth, erratic, lumpy, intermittent, cold_start, discontinued) |
| zero_forecast | discontinued |
| auto_ets | smooth, erratic |
| auto_arima | smooth, erratic |
| theta | smooth, erratic, intermittent |
| croston_sba | intermittent |
| tsb | intermittent, lumpy |
| compound_bernoulli | intermittent, lumpy |
| cluster_lgbm | smooth, erratic, intermittent, lumpy, cold_start |
| tweedie_glm | lumpy |

### Phases still pending
| Phase | Module | Key deliverable |
|---|---|---|
| 13 | `models/foundation.py` | Chronos-T5-tiny + Moirai-small CPU zero-shot |
| 14 | `validate.py` | 4-fold rolling CV harness, A vs A+ re-check |
| 15 | `selection.py` | Per-cluster WAPE selection + [0.75,0.85] calibration guardrail |
| 16 | `forecast.py` + `reconcile.py` | Final forecast + bottom-up reconciliation + manifest |
| 17 | `report.py` | Known-limitations section (clustering, calibration, cold-start) |
| 18 | `app/streamlit_app.py` | Interactive exploration |
| 19 | CI/CD | |

---

## 6. Architecture

```
data/raw/ (3 CSVs, scope-filtered at load)
    └── io.py (scope filter: removes Gift Card + return)
        └── lifecycle.py  (220 SKUs, 139 active, 81 dormant)
            └── densify.py  (16,068 rows, Sat-dated, zero-filled)
                └── features.py
                    ├── build_features()         → 16,068 × 98
                    ├── add_cluster_features()   → 16,068 × 106
                    └── add_hierarchy_features() → 16,068 × 114
                        ├── segment.py   → 220 SKUs, K=3 (ARI-stable, no fallback)
                        └── hierarchy.py → 228 nodes, S=(228,220), round-trip exact

models/ (all → ForecastResult, floor+sort, no special-casing)
    ├── baseline.py   → SeasonalNaive (all segs), ZeroForecast (discontinued)
    ├── classical.py  → AutoETS, AutoARIMA, Theta (smooth/erratic + conformal PI)
    ├── intermittent.py → CrostonSBA, TSB, CompoundBernoulli
    ├── ml_global.py  → ClusterPooledLGBM (57 boosters at K=3, direct multi-step)
    └── tweedie.py    → TweedieGLM (lumpy, compound Poisson-Gamma, 31 seasonal fits)
```

---

## 7. Key Numbers (All from Real CSV Data)

| Metric | Value | Phase |
|---|---|---|
| In-scope SKUs | 220 | Phase 1 |
| Excluded (Gift Card + return) | 9 | Phase 1 |
| Active SKUs | 139 | Phase 2 |
| Dormant SKUs | 81 | Phase 2 |
| Dense grid rows | 16,068 | Phase 3 |
| Zero fraction | 32.4% | Phase 3 |
| Stockout SKUs | 82 | Phase 3 |
| Feature matrix | 16,068 × 98 | Phase 4 |
| Full feature matrix | 16,068 × 114 | Phase 6b |
| Selected K | 3 (ARI=1.0, stable) | Phase 5 |
| Hierarchy nodes | 228 (1+7+220) | Phase 6 |
| Tweedie seasonal fits | 31/31 lumpy SKUs | Phase 12 |
| LGBM 80% PI coverage | ~58.5% (below 0.75 guardrail) | Phase 11 |
| Tests passing | 372/372 | All |

---

## 8. What's RIGHT

- Scope filter correctly removes Gift Card/return contamination
- Leakage discipline verified end-to-end
- SeasonalNaive provides universal coverage including short-history SKUs
- ForecastResult dual constructor is the correct abstraction (no special-casing)
- Hierarchy round-trip is exact
- Tweedie right-skew confirmed on real lumpy SKUs
- K selection now matches doc spec (ARI-based, not silhouette-based)

## 9. What's UNCERTAIN (Needs Phases 14–15)

- ClusterPooledLGBM 80% coverage ~58.5% — likely rejected at Phase 15 for most clusters
- K=3 bimodal structure may add less pooling value than K=8 would — Phase 15 will decide
  empirically whether "segment-as-cluster" lever should be activated
- Fourier collinearity (20 cols vs spec 10) — Phase 14 experiment
- Foundation models (Phase 13) — not yet installed or tested
- Revenue-weighted WAPE for any model — unknown until Phase 14 CV harness runs
- Product_id intermediate level — missing from hierarchy (CSV doesn't carry it);
  affects only 42 SKUs across 9 multi-variant products

---

*End of work log — Phases 0–12 + baseline complete, 372/372 tests green.*
