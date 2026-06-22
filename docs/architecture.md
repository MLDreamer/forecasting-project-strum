# Architecture

> Living document. The high-level design is fixed; per-phase detail is filled in
> as each phase lands.

## Goal

Produce probabilistic weekly unit-sales forecasts (P10 / P50 / P90) for every
in-scope SKU over a 26-week horizon (week-ending 2026-05-24 → 2026-11-15),
reconciled across a Variant → Segment → Total hierarchy, plus an interactive
Streamlit app for exploration.

## Pipeline (data flow)

```
data/raw/*.{csv,xlsx}
   │  01_load        io.py            → data/interim/joined.parquet
   │  02_lifecycle   lifecycle.py     → data/interim/lifecycle.parquet
   │  03_densify     densify.py       → data/interim/dense_weekly.parquet
   │  04_features    features.py      → data/processed/features.parquet
   │  05_segment     segment.py       → data/processed/segments.parquet
   │  06_hierarchy   hierarchy.py     → data/processed/hierarchy.parquet, S_matrix.npz
   │  07_cv          validate.py      → outputs/cv_results.parquet, cv_summary.parquet
   │  08_forecast    selection.py     → outputs/winning_models.parquet
   │                 forecast.py      → outputs/forecast_final.csv, forecast_hierarchy.parquet
   │                 reconcile.py     (bottom-up bootstrap / MinT)
   │  09_report      report.py        → outputs/forecast_report.md
   ▼
app/streamlit_app.py  (reads pre-computed artifacts only)
```

## Module map

| Module | Responsibility | Phase |
|---|---|---|
| `config.py` | Paths, constants, locked decisions, seeds | 1 |
| `io.py` | Load + validate inputs, join sales↔master | 1 |
| `lifecycle.py` | Per-SKU active-window inference | 2 |
| `densify.py` | Zero-fill weekly grid, stockout flag | 3 |
| `features.py` | Lags, rolling, calendar, promo/price (leakage-guarded) | 4 |
| `segment.py` | Syntetos-Boylan + cold-start, revenue tiers | 5 |
| `hierarchy.py` | Hierarchy nodes + summing matrix | 6 |
| `metrics.py` | WAPE, CRPS, MASE, pinball, coverage | 7 |
| `models/` | baseline, classical, intermittent, ml_global, tweedie | 8-12 |
| `validate.py` | Rolling-origin CV harness (4 folds) | 13 |
| `selection.py` | Winning model per segment (rev-weighted CRPS) | 14 |
| `reconcile.py` | Bottom-up bootstrap + MinT | 15 |
| `forecast.py` | Refit + predict + reconcile + write | 16 |
| `report.py` | Executive markdown report | 17 |

## Design principles

- All constants/hyperparameters in `config.py`; no magic numbers in logic.
- Pure functions; DataFrames passed explicitly; no global state.
- Parquet for intermediate artifacts; CSV only for the final deliverable.
- Reproducible: seeds in `config.py`, pinned deps, no runtime network calls.
- `mypy --strict` on `src/forecasting`; Google-style docstrings on public APIs.

See [data_schema.md](data_schema.md), [modeling_decisions.md](modeling_decisions.md),
and [hierarchy.md](hierarchy.md) for detail.
