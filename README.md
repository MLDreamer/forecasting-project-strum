# Fontana Candle Co — AutoML Demand Forecasting

> AutoML demand-forecasting pipeline for a Shopify DTC candle brand.
> Produces 26-week probabilistic forecasts (19 quantiles) for 220 SKUs
> across a product hierarchy, with per-named-segment model selection,
> revenue-weighted WAPE evaluation, and a live Streamlit dashboard.

---

## What this does

| Step | What happens |
|---|---|
| **Scope** | Load 3 Excel/CSV files → filter out Gift Card / return SKUs → 220 in-scope SKUs |
| **Lifecycle** | Identify 139 active / 81 dormant SKUs |
| **Densify** | Zero-fill weekly grid (16,068 rows, Saturday-dated) |
| **Features** | 114-column feature matrix (lags, Fourier, holidays, LOO cluster/hierarchy aggregates) |
| **SB Segment** | Classify each SKU: erratic · lumpy · smooth · intermittent · cold_start · discontinued |
| **Bake-off** | 7 models × 4 rolling-origin CV folds, scored by **revenue-weighted WAPE** |
| **Select** | One winner per named SB segment; conformal calibration guardrail [0.75, 0.85] |
| **Forecast** | 220 × 26 weeks × 19 quantiles; bottom-up bootstrap reconciliation |
| **Publish** | 8 flat contract tables → `outputs/latest/` → Streamlit dashboard auto-refreshes |

**Current result:** RW-WAPE = **0.788** (fixed price, eligibility filter, pooled folds 2–4).
Fold 3 alone (steady-state period): **0.653** — below the 0.60 target.

---

## Segment winners

| Segment | Model | Logic |
|---|---|---|
| erratic | `TrendSeasonalModel` | Seasonal naive × clipped YoY growth [0.5, 3.0] |
| lumpy | `SeasonalNaive` | Last-year-same-week + conformal PI |
| smooth | `RecentLevelModel` | 8-week mean (handles stocked-out SKUs as near-zero) |
| intermittent | `SeasonalNaive` | Same |
| cold_start | `SeasonalNaive` | Short-history fallback |
| discontinued | `ZeroForecast` | Dormant ≥ 26 weeks |

---

## Quick start

```bash
# Install
pip install -r requirements.txt

# Run the pipeline (trains + publishes)
python -m forecasting.run --config configs/fontana_candle.yaml

# Launch the dashboard
streamlit run app/streamlit_app.py

# Or use the notebook interface
python notebooks/configure_and_run.py
```

### Notebook (business-term knobs)
```python
from configs.config_builder import ConfigBuilder

cfg = (ConfigBuilder.from_yaml("configs/fontana_candle.yaml")
       .set("forecast weeks", 26)
       .set("weeks to avoid forecasting", 26)
       .build())          # ← pydantic validates; raises on bad values
```

---

## Repository structure

```
data/raw/          3 input files (Excel/CSV)
configs/           fontana_candle.yaml  ·  _schema.py  ·  config_builder.py
src/forecasting/   pipeline core (value-free; all client values in YAML)
app/               Streamlit dashboard (4 pages)
notebooks/         configure_and_run.py
outputs/latest/    8 contract tables read by the dashboard
runs/              per-run archives (manifest, logs, tables)
.github/workflows/ monthly_forecast.yml  ·  ci.yml
tests/             478 tests (pytest)
```

---

## Dashboard pages

| Page | What it shows |
|---|---|
| **Overview** | Portfolio 26-week forecast · RW-WAPE vs 0.60 · segment donut |
| **Drilldown** | Total → product type → SKU with breadcrumb navigation |
| **AutoML** | Architecture flow · per-segment winners · candidate pool |
| **Run log** | Run history · manifest hashes · WAPE trend · raw log |

---

## Monthly deploy loop

1. **Trigger** — cron 1st of month, manual "Train now", or on Excel push
2. **Train** — GitHub Action trains on Ubuntu 3.11, writes `runs/<YYYY-MM>/`
3. **Promote** — gate: RW-WAPE ≤ 0.95, all tables present → copies to `outputs/latest/`
4. **Commit** — results pushed to `results` branch
5. **Redeploy** — Streamlit Cloud watches `results`, auto-redeploys on push
6. **Fail-safe** — bad run keeps last good forecast live; GitHub Issue opened

---

## Metric definition (locked)

```
RW-WAPE = Σ(price_i × |y_i,h − f_i,h|)  /  Σ(price_i × y_i,h)
```

- `price_i` = fixed per-unit price from product master (training-time, not holdout actuals)
- Pooled across all eligible (SKU, week) rows in selection folds 2–4
- Eligibility: `first_sale ≤ fold_origin − 26 weeks` (excludes brand-new SKUs)
- CRPS and 80% PI coverage are diagnostics, not the selection metric

---

## Known limitations

| Issue | Status |
|---|---|
| RW-WAPE 0.788 (target 0.60) | Fold 2 holiday surge + young catalog; fold 3 alone = 0.653 |
| Moirai unavailable on Windows | Needs Linux + gcc; wrapper ready |
| K=3 coarse clustering | Re-cluster at catalog maturity (≥100w median history) |
| Fourier 20 cols vs spec 10 | Phase 14 A/B pending |

---

## Tests

```bash
pytest tests/                  # 478 tests
ruff check src/ tests/ configs/ app/
ruff format --check src/ tests/ configs/ app/
```

---

## License

Private. Fontana Candle Co. All rights reserved.
