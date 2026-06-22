# Fontana Candle Co — Forecasting Report

*Generated: forecast origin 2026-05-23  
Pipeline v0.1.0  
Cube hash: `20bea4c3263f1b8e`*


## Executive Summary

- **Forecast horizon:** 26 weeks (2026-05-30 → 2026-11-21)
- **SKUs forecast:** 220 variants
- **Active SKUs in scope:** 139 (discontinued / dormant routed to ZeroForecast)
- **Clusters:** 3
- **Model pool deployed:** seasonal_naive
- **Calibration:** post-hoc conformal scaling applied (alphas 1.27–5.00)

## Model Selection — Per-Cluster Winners

| Cluster | Winner | CRPS | cov80 | Beats Baseline | Fallback | Rejected |
|---|---|---|---|---|---|---|
| 0 | seasonal_naive | 12.9927 | 0.746 | ✗ | ✓ | auto_ets, chronos_tiny, compound_bern, cronston_sba, seasonal_naive, theta, tweedie_glm |
| 1 | seasonal_naive | 65.1205 | 0.678 | ✗ | ✓ | auto_ets, chronos_tiny, compound_bern, cronston_sba, seasonal_naive, theta, tweedie_glm |
| 2 | seasonal_naive | 14.4987 | 0.741 | ✗ | ✓ | auto_ets, chronos_tiny, compound_bern, seasonal_naive, theta, tweedie_glm |

> **V2 lever triggered: segment-as-cluster.**  
> ClusterPooledLGBM won 0/3 clusters. Consider using SB class as the pooling unit instead of K-means clusters in a future iteration.

## Post-Hoc Conformal Calibration

Alpha > 1 means raw prediction intervals were too narrow and were widened.  
Alpha = 1 means no adjustment was needed.

| Cluster | Model | Alpha |
|---|---|---|
| 0 | seasonal_naive | 1.523 |
| 1 | seasonal_naive | 5.000 |
| 2 | seasonal_naive | 1.270 |

## CV Performance (Selection Folds 2–4)

Selection metric: **revenue-weighted CRPS** (lower is better).  
Guardrail: 80% PI coverage ∈ [0.75, 0.85].

| Model | Fold | WAPE | CRPS | cov80 | cov90 | In Selection |
|---|---|---|---|---|---|---|
| auto_ets | 2 | 56.721 | 19.3959 | 0.521 | 0.548 | ✓ |
| auto_ets | 3 | 156.951 | 19.3832 | 0.708 | 0.733 | ✓ |
| auto_ets | 4 | 141.359 | 33.5994 | 0.557 | 0.612 | ✓ |
| chronos_tiny | 2 | 44.614 | 16.5034 | 0.482 | 0.572 | ✓ |
| chronos_tiny | 3 | 151.545 | 18.9383 | 0.558 | 0.641 | ✓ |
| chronos_tiny | 4 | 127.000 | 31.0220 | 0.465 | 0.545 | ✓ |
| compound_bern | 2 | 46.788 | 14.4752 | 0.672 | 0.734 | ✓ |
| compound_bern | 3 | 161.407 | 18.8019 | 0.716 | 0.789 | ✓ |
| compound_bern | 4 | 115.993 | 21.4021 | 0.660 | 0.722 | ✓ |
| cronston_sba | 2 | 42.797 | 15.3695 | 0.668 | 0.707 | ✓ |
| cronston_sba | 3 | 145.942 | 20.0398 | 0.788 | 0.806 | ✓ |
| cronston_sba | 4 | 101.090 | 22.3511 | 0.697 | 0.727 | ✓ |
| seasonal_naive | 2 | 44.103 | 14.5946 | 0.691 | 0.735 | ✓ |
| seasonal_naive | 3 | 117.550 | 16.9280 | 0.779 | 0.818 | ✓ |
| seasonal_naive | 4 | 88.916 | 19.5232 | 0.710 | 0.753 | ✓ |
| theta | 2 | 43.425 | 14.9715 | 0.592 | 0.619 | ✓ |
| theta | 3 | 134.116 | 18.1727 | 0.709 | 0.729 | ✓ |
| theta | 4 | 124.689 | 27.1469 | 0.584 | 0.651 | ✓ |
| tweedie_glm | 2 | 47.120 | 13.2282 | 0.584 | 0.644 | ✓ |
| tweedie_glm | 3 | 153.291 | 151.8202 | 0.598 | 0.665 | ✓ |
| tweedie_glm | 4 | 124.054 | 23.2167 | 0.616 | 0.676 | ✓ |

## Known Limitations — Clustering

**Selected K:** 3  
**Fallback used:** No  
**Best silhouette:** 0.345

**SB class distribution:**

| Class | Count |
|---|---|
| discontinued | 81 |
| erratic | 45 |
| smooth | 33 |
| lumpy | 31 |
| intermittent | 19 |
| cold_start | 11 |

**Root cause:** ~73% of SKUs have < 24 months of history (young catalog). Demand-pattern features (IDI, CV2) are still stabilising, producing weak cluster structure. The K-means solution is not meaningfully stable — the selected K reflects data availability, not true demand heterogeneity.

**Recommendation:** Re-cluster once the median SKU has ≥100 weeks of history. At that point the stability-ARI threshold (0.5) should be achievable at a higher K, enabling more granular pooling.

## Known Limitations — Calibration

**Observed uncalibrated 80% PI coverage (SeasonalNaive, fold-by-fold):**

| Fold | cov80 | Guardrail [0.75, 0.85] |
|---|---|---|
| 1 (origin 2024-05-25) | 0.729 | FAIL |
| 2 (origin 2024-11-23) | 0.691 | FAIL |
| 3 (origin 2025-05-24) | 0.779 | **PASS** |
| 4 (origin 2025-11-22) | 0.710 | FAIL |

**Root causes:**

1. **New-catalog SKUs in holdout** — folds 1, 2, 4 have many SKUs whose first sales occur after the fold cutoff. Any model produces near-zero probability mass at actual demand levels for a SKU it has never seen. These are structurally miss-able and are excluded from guardrail evaluation (cold-start route).

2. **Heavy-tail demand** — CV=1.84, P90/P50=6.2x, 27% of weeks show >2× YoY growth. Standard conformal intervals based on seasonal residuals underestimate the true spread.

**Mitigation applied:** Post-hoc conformal calibration (mean alpha=2.60). Intervals are scaled up so that empirical 80% PI coverage on the CV holdout reaches the guardrail target.

**Residual risk:** Alpha is estimated on the CV holdout and applied to the final forecast. If the final horizon (2026-05-24 → 2026-11-15) differs structurally from the CV holdout, actual coverage may deviate.

## Cold-Start Ablation

**Cold-start SKUs:** 11 (< 4 non-zero observations)

These SKUs have no meaningful seasonal pattern. The baseline (SeasonalNaive) falls back to a constant mean forecast, which may be unreliable. Foundation models (Chronos-T5-tiny, Moirai-small) were evaluated as potential alternatives.

**Cluster winners for clusters containing cold-start SKUs:**

| Cluster | Winner |
|---|---|
| 0 | seasonal_naive |
| 2 | seasonal_naive |

**Chronos-T5-tiny performance:**
- Successfully produces sample-based forecasts for series with as few as 4 observations.
- Inference time: ~0.5s/SKU on CPU (acceptable for batch, borderline for real-time).
- WAPE comparison against SeasonalNaive run per-SKU — see CV performance table.

**Moirai-small status:** Not installable on this deployment environment (requires `numpy~=1.26` + C compiler). Registered as a candidate but skipped. Recommend evaluating on a Linux build environment.

**Recommendation:** If Chronos WAPE < SeasonalNaive WAPE on cold-start SKUs in Phase 14 CV, activate `chronos_tiny` as the cold-start route in Phase 15 selection. This ablation requires running `validate.run_cv` with both models in the candidate pool.

## Known Limitations Summary

| # | Issue | Severity | Mitigation |
|---|---|---|---|
| 1 | Weak cluster structure (young catalog, sil<0.4) | Medium | Accept K=3, re-cluster at 100w median history |
| 2 | 80% PI under-coverage on folds 1/2/4 | High | Post-hoc conformal calibration applied |
| 3 | New-catalog SKUs skipped in guardrail eval | Low | Explicit cold-start route (ZeroForecast / Chronos) |
| 4 | 49% zero weeks in holdout (intermittent demand) | Medium | CompoundBernoulli + Croston/TSB in model pool |
| 5 | Moirai unavailable on this deployment box | Low | Evaluate on Linux; wrapper is production-ready |
| 6 | YoY growth >2× for 27% of SKU-weeks | High | Wide conformal intervals; flag high-growth SKUs for manual review |
| 7 | product_id hierarchy level missing (CSV vs Excel) | Low | 3-level hierarchy (total→product_type→variant) functionally equivalent |
| 8 | Sub-annual Fourier (13/26w) may be collinear with 52w | Low | Phase 14 experiment item; revert to spec (52w only) if WAPE degrades |
