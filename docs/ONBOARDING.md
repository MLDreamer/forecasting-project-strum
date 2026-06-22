# Forecasting AutoML — Onboarding & Rebuild Playbook

> **Read this first.** It is the single source of truth for rebuilding this project
> from scratch given only the three source Excel files. It records the mission, the
> dataset findings, every non-obvious decision/tweak, the environment landmines, and
> the **phased build plan with a hard verification gate + STOP after every phase**.
> Companion docs: [`docs/modeling_decisions.md`](docs/modeling_decisions.md),
> [`docs/architecture.md`](docs/architecture.md), [`CHANGELOG.md`](CHANGELOG.md).

---

## 1. Mission & success criteria
- Build a **client-agnostic, config-driven AutoML forecasting pipeline**. Instance #1
  is **Fontana Candle Co** (Shopify DTC), but the core in `src/forecasting/` must
  contain **zero client-specific values** — everything lives in a YAML validated by a
  pydantic schema.
- **Output:** 26-week-ahead forecasts, **19 quantiles (P05–P95, step 0.05)**, for the
  ~220 in-scope SKUs, with a 4-level hierarchy and bottom-up reconciliation.
- **Bar to beat:** **revenue-weighted WAPE < 0.60** (target **< 0.50**). WAPE is the
  selection metric; CRPS/WIS/coverage are diagnostics.
- **Selection unit = per cluster** (one winning model per cluster — not one global, not
  per-SKU), chosen on folds 2–4 with an **80%-coverage calibration guardrail [0.75, 0.85]**.
- **Process rule (non-negotiable):** implement ONE numbered phase, run its gate, commit,
  then **STOP and wait for human "go."** Never run ahead.

## 2. The three Excel files → canonical model
The client ships 3 `.xlsx` files. `io.py` is the **only** module that knows their column
names; it maps them to canonical names. They resolve to:
1. **Weekly sales** by variant — **Sunday-dated** (week START), ~11,291 rows.
2. **Product/item master** — 441 variants → `product_id` → `product_type`, `list_price`,
   `status` (active/archived).
3. **Inventory / price detail** — feeds shelf price + stockout signal.

Canonical columns: `sku_id, timestamp, sales, list_price, discount_pct, product_id,
product_type, status` (+ inventory). **Schema mapping is config-driven** (`data.schema_map`).

## 3. Locked architecture principles
1. **Value-free core.** `src/forecasting/` is client-agnostic; client values → YAML +
   `configs/_schema.py` (pydantic). Entry point: `python -m forecasting.run --config
   <yaml> [--stop-after PHASE] [--validate-only]`.
2. **`io.py` owns client column names**; everything downstream uses canonical names.
3. **Registries** (`registry.py`): `@register_model / @register_feature / @register_plugin`.
   The pipeline iterates `MODEL_REGISTRY.candidates_for(segment)` — never names a model.
4. **Leakage discipline:** lags/rolling shifted; cluster & hierarchy aggregates are
   leave-one-out; discount is never contemporaneous; the CV cutoff is applied exactly once.
5. **One uniform model output:** `ForecastResult` quantile cube `(n_sku, h, n_q)` with two
   constructors — `from_quantiles` (quantile-native) and `from_samples` (MC paths). Both
   floor at 0 and sort, so output is always non-negative and non-crossing.
6. **Versioned outputs + `manifest.json`** for reproducibility.

## 4. Dataset findings (ground truth — memorize)
- **Weeks are Sunday-labeled (week START).** A naive `W-SAT` grid yields Saturdays and
  silently breaks joins. **FIX: relabel timestamps +6 days to Saturday week-END (`W-SAT`)
  at load.** This is the #1 landmine.
- Master joins **100%** once `product_item_master` is the SKU master (no `unknown` status).
- **441 variants; 229 have sales history; 220 in-scope to forecast; 36 future cold-starts.**
- **Young catalog: ~73% of SKUs <24 months old** — the root cause of weak clusters and
  thin per-SKU data downstream.
- Dense grid: **17,539 rows; 35.6% zeros; 89 SKUs with stockout-like gaps** (zero runs ≥8w).
- **Dormancy = literal "no sale in last 26 weeks"** ⇒ **81 trimmed** (after 1 override).
- SB-class split: **erratic 56 / lumpy 51 / intermittent 34 / smooth 33 / discontinued 32
  / cold_start 23**. Heterogeneous demand is *why* AutoML (per-cluster winner) is justified.

## 5. Environment & tooling reality (these WILL bite you)
- **Windows + PowerShell 5.1**: no `&&`; check `$LASTEXITCODE`, not `$?` (native-exe stderr
  quirk). Use here-strings for multiline; default encoding is UTF-16.
- **No git, no uv, no make** on the dev box — use the project `.venv` + pip directly.
- **Always** `$env:PYTHONPATH = "src;."` (both `src` and repo root, `;`-separated).
- **Python 3.13** on the box (managed 3.11 blocked by corporate TLS proxy); CI pins 3.11.
  Code stays 3.11+ compatible.
- **OneDrive-synced `.venv` corrupts compiled `.pyd`** (scipy `_propack` once broke
  statsforecast). Fix: `pip install --force-reinstall --no-cache-dir <pkg>`. Suspect this
  first on weird ImportErrors.
- **`scipy<1.16` pinned** (statsforecast 2.x constraint).
- Heavy deps (`statsforecast`, `lightgbm`, `statsmodels`, `torch`, `chronos`, `uni2ts`) are
  **imported lazily inside methods** so importing a model module only registers it.

## 6. Per-phase discipline (every phase, no exceptions)
Run the phase's specific **verification gate (exact counts/shapes)**, then `pytest`
(coverage ≥70%), `ruff check`, `ruff format --check`, `mypy --strict`. Update CHANGELOG +
memory. **Then STOP.** Prefer a **real-data smoke check** that confirms *plausibility*, not
just type-safety (e.g., "median < mean for right-skewed lumpy demand", fit-mode counts).

## 7. The phased rebuild plan (0 → 19)
Each phase: **Goal → key decisions/tweaks → gate → STOP.**

- **0 — Scaffold.** Repo tree, `pyproject.toml`, Makefile, CI templates. Gate: tree exists, imports clean.
- **1 — Config + I/O.** `config.py`, `io.py` loaders, schema validation, join, scope sets.
  Gate: master 441, sales 11,291, scope 220/229/36.
- **2 — Lifecycle.** `infer_lifecycle` (active window; dormancy `>=26w`). Tweak: keep-active
  overrides are a **config list with a required `reason`**. Gate: 229 rows, 81 trimmed.
- **3 — Densify.** Weekly grid, zero-fill, price ffill/bfill, stockout flag. Gate: 17,539
  rows, 35.6% zero, 89 stockout SKUs.
- **3.5 — Config-aware retrofit (the AutoML pivot).** pydantic schema + YAML + `registry.py`
  + `run.py`. **Relabel Sunday→Saturday here.** Gate: counts unchanged + `--validate-only` works.
- **4 — Features.** Registry-driven generators (lags, rolling, momentum, ISO-week Fourier,
  year-accurate holidays, lagged promo/price, statics: Gini/Hurst/IDI/ABC). **Cutoff applied
  once.** Plugin `shopify_inventory_stockout`. Gate: 17,539 × 98; leakage check `lag_1[t]==sales[t-1]`.
- **5 — Segment + cluster.** SB class + K-means; K = blend(0.7·silhouette + 0.3·stability-ARI).
  **Finding/tweak: weak structure (silhouette 0.22, ARI 0.34) → accept K=8 fallback,
  document it**, anchor `revenue_tier`, one-hot weight 0.15, ARI threshold 0.5. Gate: 229 rows, K=8.
- **5b — Cluster-context features.** 7 LOO lag-1 aggregates + `cluster_id`. Gate: 17,539 × 106.
- **6 — Hierarchy.** Generic N-level builder + sparse `S` matrix; level-prefixed IDs; NULL→
  `unknown`; multi-parent guard. Gate: 420 nodes (1/7/192/220); round-trip `S@bottom==agg`.
- **6b — Hierarchy-context features.** 8 LOO/static columns. **Tweak: total-level YoY is
  NON-LOO** (LOO amplified ratio noise). Gate: 17,539 × 114.
- **7 — Metrics.** WAPE (selection), MASE, pinball, CRPS, WIS, coverage(80/90), sMAPE; all
  revenue-weightable + per-horizon. Gate: worked-example unit tests.
- **8 — Model interface.** `ForecastModel` ABC + `ForecastResult` dual constructor. Gate:
  from_quantiles/from_samples both floor+sort.
- **9 — Classical.** AutoETS/AutoARIMA/Theta + split-conformal quantiles; skip <26w. Gate: unit tests.
- **10 — Intermittent.** Croston-SBA + TSB + compound-Bernoulli bootstrap (intermittent+lumpy).
- **11 — Cluster-pooled LightGBM.** 19 quantile boosters × 8 clusters = **152**; direct
  multi-step via `horizon_step`; sort+floor+warn. **Findings: 29% crossing (repaired);
  80% PI under-covers at 70.4% (below guardrail floor).** Gate: 152 boosters, importance plot.
- **11.5 — Target-week features (A/B, REJECTED — keep as a template).** Hypothesis behind a
  flag → A/B → reject on data → document. Result: WORSE intervals (crossing 29→36%, cov
  70.4→68.3) but BETTER short-horizon point error (WAPE h1 0.42→0.33). Diagnosis: deterministic
  rotations of existing Fourier by `horizon_step` ⇒ collinear variance. **Reverted (flag OFF,
  kept). Phase-14 follow-up: re-check A vs A+ WAPE on true CV folds.**
- **12 — Tweedie GLM (lumpy).** Per-SKU compound Poisson-Gamma; mean → quantiles by
  simulation (`from_samples`); fallback chain seasonal→intercept→empirical. Smoke: 51 lumpy
  SKUs, 50 seasonal/1 intercept, P50 < mean (right-skew). Gate: 6 tests, 98% module cov.
- **13 — Foundation models (CURRENT).** Chronos-T5-tiny (`from_samples`) + Moirai-small
  (`from_quantiles`), CPU zero-shot. **Defaults only — no tuning.** Lock pool to these two
  for v1 (heavier models gated behind `use_heavy_models`). **Track: per-horizon WAPE,
  per-SB-class breakdown (esp. cold_start vs seasonal_naive — that's where they must earn
  their slot), inference timing per SKU, and that BOTH flow through `ForecastResult` with no
  special-casing** (the dual-constructor abstraction test).
- **14 — Rolling-origin CV harness.** 4 folds, 26w horizon, per-horizon. *Home of the
  11.5 A-vs-A+ WAPE re-check (incl. h13/h26, unreachable earlier).*
- **15 — Per-cluster WAPE selection + calibration guardrail.** Reject candidates whose 80%
  coverage ∉ [0.75, 0.85]. v2 lever: "segment-as-cluster" if cluster_lgbm loses widely.
- **16 — Final forecast + bottom-up reconciliation** (+ bootstrap). Versioned outputs.
- **17 — Report.** Must surface the clustering + calibration limitations + cold-start ablation.
- **18 — Streamlit app** (+ see §9 notebook-config layer).
- **19 — CI/CD.**

## 8. Judgment-call / tweak log (the non-obvious choices)
1. Sunday→Saturday relabel (`week_relabel_shift_days`). 2. Keep-active override mechanism
(config list + reason). 3. K=8 clustering fallback accepted with documented young-catalog
limitation. 4. Total-level YoY non-LOO. 5. `ForecastResult` field is `quantiles` (not
`values`, avoids a lint rule) — funnels all models. 6. Target-week features rejected after
A/B (flag kept OFF). 7. Tweedie graceful fallback chain. 8. **Discipline:** any "would X
help?" choice → flag → A/B vs baseline → accept/reject on data → document either way.

## 9. Open levers / v2 (decide from Phase-15 evidence)
- "Segment-as-cluster" (SB class as the LightGBM pool) if k-means adds no value.
- Calibration repair for cluster_lgbm if it wins clusters but under-covers: post-hoc
  conformal, monotone-by-construction quantiles, or shared-tree multi-quantile loss.
- Target-week features re-enabled per-cluster if Phase-14 CV shows A+ WAPE wins out-of-sample.
- Heavier foundation models (Chronos-small, TimesFM, Lag-Llama) behind `use_heavy_models`.

## 10. Should users configure via a notebook? (YES — as a validated layer)
Offer a notebook/dict entry point, but it must **build/override the same pydantic `Config`,
never bypass validation** (the schema is the safety gate that keeps the core client-agnostic).
Design: a thin `ConfigBuilder` that deep-merges dict overrides onto the YAML and re-validates,
e.g. `cfg = ConfigBuilder.from_yaml(...).set("lifecycle.dormancy_threshold_weeks", 26).build()`.
Support **business-term aliases** mapping to canonical config paths (your "weeks to avoid
forecasting" = `lifecycle.dormancy_threshold_weeks`) so non-engineers get readable knobs while
the schema stays authoritative. Keep `--config <yaml>` as the headless/CI path; both funnel
through pydantic. Slot this alongside Phase 18.
