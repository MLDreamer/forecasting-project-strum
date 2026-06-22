# Modeling decisions

The locked decisions below are the project's source of truth and are mirrored in
`src/forecasting/config.py`. Rationale is added as phases land.

## Data handling

| Decision | Choice |
|---|---|
| Missing weeks within active window | Fill with `sales = 0` |
| Active window start | First sale date per SKU |
| Active window end | `status='archived'` OR no sale in last 26 weeks → last sale date; else 2026-05-17 |
| Dormancy boundary | Literal: `weeks_since_last_sale >= 26` ("last 26 weeks" = weeks 0–25; a sale exactly 26 weeks before the cutoff is dormant). Gives 81 trimmed after overrides. |
| Lifecycle overrides | `LIFECYCLE_KEEP_ACTIVE_OVERRIDE` forces reviewed SKUs to stay active. Currently `{46606700773604}` — high recent volume, no seasonal evidence, but zero-forecasting a recently high-velocity SKU is riskier than letting a model try. |
| Mid-series zero runs ≥ 8 weeks | Keep as zeros; add `is_potential_stockout` |
| Outlier treatment | None (no winsorizing) — handled via loss function |
| Unmatched `item_id` | Include with `status = 'unknown'` |
| Archived SKUs with no sales | Excluded from scope |
| SKUs with < 4 weeks history | Global model pool only |
| Duplicate `source_variant_id` | Keep first, log warning |
| Week boundary | Week-ending Sunday (verify on load) |
| Future `avg_unit_price` / `discount_pct` | Carry forward trailing 4-week mean per SKU |

## Forecasting

| Decision | Choice |
|---|---|
| Lowest forecast level | Variant (`item_id`) |
| Hierarchy | Variant → Segment → Total |
| Quantiles | P10, P50, P90 |
| Discount feature | `discount_pct_lag1` only — never contemporaneous |
| Headline metric | Revenue-weighted **WAPE** |
| Probabilistic metric | Revenue-weighted **CRPS** |
| Secondary metrics | MASE, pinball loss per quantile, empirical coverage |
| Holdout | Rolling-origin, 4 folds, 26-week horizon each |
| Baseline | Seasonal naive (last-year-same-week) + zero for cold starts |
| Reconciliation | Bottom-up (default); MinT-shrink optional |

## Model pool

| Segment driver | Models |
|---|---|
| Baseline (must beat) | Seasonal naive, zero |
| Smooth / erratic | AutoETS, AutoARIMA, Theta (conformal intervals) + LightGBM global |
| Intermittent / lumpy | Croston, SBA, TSB (bootstrap quantiles); Tweedie GLM for lumpy |
| Cold-start / short history | LightGBM global quantile model |

Selection rule: lowest revenue-weighted CRPS averaged over folds 2-4 (fold 1
skipped — cold-start data too thin), WAPE tiebreaker.

## Known limitation — weak cluster structure (Phase 5)

Cluster structure is weak for this catalog (silhouette 0.22, stability ARI 0.34)
due to high SKU youth — 73% of SKUs launched within the last 24 months, so the
continuous demand-pattern features are still stabilizing across the CV cut-offs.
K-selection therefore falls back to `fallback_K = 8` (the `stability_ari_threshold`
is relaxed to 0.5 for this client, documented in the config). `revenue_tier` is
anchored to the deployment-time view to remove definitional drift; the residual
instability is the young-catalog effect and is expected to resolve as the catalog
matures (re-cluster recommended once the median SKU has 100+ weeks of history).

Cluster-pooled LightGBM is still included as a candidate; **per-cluster winner
selection (Phase 15) determines deployment empirically** — if clustering adds no
value, classical/foundation models win and `cluster_lgbm` deploys to fewer
clusters. v2 lever if `cluster_lgbm` loses in most clusters: "segment-as-cluster"
(use the SB class itself as the LightGBM pooling unit, no k-means).

**Phase 17 `report.py` must surface this in the report's known-limitations
section.**

## Phase 11 — cluster-pooled LightGBM (design + findings)

`ClusterPooledLGBM` trains one quantile booster per (cluster, level) — 8 clusters
× 19 levels = **152 boosters** — pooling each cluster's SKUs. Multi-step is
**direct**: `horizon_step` (1..H) is a feature and each training row pairs the
as-of-`t` feature vector with target `sales[t+horizon_step]`. Categoricals
(`sku_id`, `cluster_id`, `product_type`, `status`, `revenue_tier`) use LightGBM's
native handling; early stopping holds out the last 13 weeks.

**Feature-timing decision — Option A (ships), Option A+ tested and rejected.**
Features are taken as of `t`; the lag/rolling/momentum/cluster/hierarchy features
*must* stay as-of-`t`. The horizon signal reaches the model through `horizon_step`.

Phase 11.5 tested **Option A+**: also append ~14 *deterministic* target-week
features (`target_week.py` — iso-week, month, annual Fourier, `weeks_to_*` +
relevance per holiday, all functions of `t+s`, hence leakage-safe). Hypothesis:
an explicit "which week am I predicting" signal would cut crossing and improve
calibration. The A/B test (same seed, same data, flag-gated) was **decisive and
negative** — A+ hurt every flagged metric:

| Metric | A (as-of-`t`) | A+ (target-week) | Note |
|---|---|---|---|
| Pre-sort crossing (per-pair) | 29.0% | 36.0% | A+ worse |
| 80% PI coverage (target ~80) | 70.4% | 68.3% | A+ worse |
| 90% PI coverage (target ~90) | 87.5% | 82.2% | A+ worse |
| P50 WAPE h1 (held-out window) | 0.416 | 0.334 | A+ **better** |
| P50 WAPE h4 (held-out window) | 0.397 | 0.308 | A+ **better** |

The picture is mixed, not one-sided. A+ **lowers the median point error ~20% at
short horizons** but **worsens interval calibration** (crossing + 80/90 coverage).
The target-week columns are deterministic rotations of the existing as-of-`t`
Fourier basis by `horizon_step` (a tree must learn that interaction rather than
read it off), so they *do* carry point-forecast signal — but the extra collinear
features also destabilize the 19 independently-trained quantile heads, widening
disagreement and degrading the spread. **Caveat that drives the call:** both WAPE
and coverage above are measured on the early-stopping window, which *is* the
tuning target — so the WAPE gain is likely optimistic (the model can overfit that
window), while the coverage degradation is the reliable signal (it got worse on
the very window the model is tuned to).

**Decision:** v1 ships **Option A** (`target_week_features` defaults OFF). Interval
calibration is the Phase 15 guardrail metric and A+ degrades it; the point-WAPE
gain is unconfirmed out-of-sample. The capability is retained behind the flag.
**Phase 14 follow-up (concrete):** measure A vs A+ revenue-weighted WAPE on true
rolling-origin CV folds (incl. h13/h26, which the early-stopping window cannot
reach). If A+ wins WAPE out-of-sample — WAPE is the selection metric — flip the
flag per-cluster, ideally paired with a v2 calibration layer (post-hoc conformal,
monotone-by-construction quantile, or shared-tree multi-quantile loss) to repair
the intervals. So the flag is a live Phase 14/15 lever, not a closed question.

(Process note: this is the template for hypothesis-driven design choices in this
build — implement behind a flag, A/B against a comparable baseline, accept or
reject on the data, document the result either way. Expect another round at Phase
13, foundation models.)

**Quantile crossing.** The 19 boosters are trained independently, so their raw
outputs are not monotone. Post-processing sorts the 19 quantiles per row and
floors at 0 (`ForecastResult._finalize`). Measured pre-sort crossing on the
deployment forecast is **29.0% of adjacent pairs** (per-cluster 1.7%–37.5%) — far
above the brief's 5% warn threshold, so the warning fires. This is consistent
with the young-catalog weak-structure finding above: adjacent 0.05-spaced
quantiles are barely separable given the data volume. The sort repairs ordering;
the magnitude (up to ~30% of row scale in the high-variance cluster) means the
deployed spread depends on that repair. A monotone-by-construction quantile
composition is a v2 lever if cluster_lgbm is selected widely in Phase 15.

**Calibration.** Held-out (early-stopping window) interval coverage is **P10–P90
= 70.4%** and **P05–P95 = 87.5%**. The 80% PI under-covers and sits *below* the
Phase 15 calibration-guardrail floor [0.75, 0.85] — the inner interval is
over-confident on this data. cluster_lgbm is only a candidate; the **Phase 15
per-cluster WAPE selection with the coverage guardrail** is where this is
enforced, and it may reject cluster_lgbm for clusters where it stays
mis-calibrated. Flagged to watch.

## Interpreter note

The reference machine runs Python 3.13 (a managed 3.11 build is unavailable
through its corporate TLS proxy). Code stays compatible with 3.11+
(`requires-python = ">=3.11"`); CI pins 3.11.
