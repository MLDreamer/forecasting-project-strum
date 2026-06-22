"""Page 3 — AutoML

Shows:
- Architecture flow diagram (V4)
- Per-segment winners table (from selection.parquet)
- Registry candidate pool
- Blended RW-WAPE vs 0.60 bar
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).parent.parent.parent
_src = _root / "src"
for _p in (_root, _src):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import streamlit as st

from app.lib.charts import SEGMENT_COLORS, architecture_flow
from app.lib.load import AppData

st.set_page_config(page_title="AutoML", layout="wide")

if "app_data" not in st.session_state:
    st.warning("Please open the main page first.")
    st.stop()

app_data: AppData = st.session_state["app_data"]
sel = app_data.selection
m = app_data.manifest

st.markdown("## AutoML — Model Selection")

# ── Blended WAPE KPI ──────────────────────────────────────────────────────────
blended_wape = m.get("blended_wape", None)
col1, col2, col3 = st.columns(3)
col1.metric(
    "Blended RW-WAPE",
    f"{blended_wape:.3f}" if isinstance(blended_wape, float) else "—",
    delta="vs bar 0.60",
    delta_color="inverse" if isinstance(blended_wape, float) and blended_wape > 0.60 else "normal",
    help="Revenue-weighted WAPE, pooled selection folds 2-4, fixed price weights, eligibility filter.",
)
col2.metric("Horizon", f"{m.get('horizon_weeks', 26)} weeks")
col3.metric("SKUs", m.get("n_sku", "—"))

st.divider()

# ── Architecture flow ─────────────────────────────────────────────────────────
st.subheader("Pipeline architecture")
fig_arch = architecture_flow()
st.plotly_chart(fig_arch, use_container_width=True)
st.caption(
    "Every run: Raw → Scope + I/O → Lifecycle → Densify → Features (98→114 cols) → "
    "SB Segment → Per-segment bake-off → Select by RW-WAPE → Reconcile + Forecast"
)

st.divider()

# ── Per-segment winners ───────────────────────────────────────────────────────
st.subheader("Per-segment model selection")
st.caption(
    "Selection metric: **revenue-weighted WAPE** (fixed per-unit price, "
    "pooled selection folds 2–4, eligibility filter: first_sale ≤ origin−26w). "
    "CRPS and 80% coverage are diagnostics."
)

if not sel.empty:
    import pandas as pd

    display_cols = ["segment", "winning_model", "wape", "crps", "cov80", "guardrail_pass"]
    display_cols = [c for c in display_cols if c in sel.columns]
    styled = sel[display_cols].copy() if display_cols else sel.copy()

    # Colour code segments
    def _seg_color(row):
        seg = row.get("segment", "")
        color = SEGMENT_COLORS.get(seg, "")
        bg = f"background-color: {color}22" if color else ""
        return [bg] * len(row)

    styled_df = styled.style.apply(_seg_color, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=min(300, (len(sel) + 1) * 38))
else:
    # Fallback: show the SEGMENT_MODEL_MAP from forecast.py
    st.info("Selection table not available — showing default routing.")
    try:
        import pandas as pd

        from forecasting.forecast import SEGMENT_MODEL_MAP

        rows = [
            {"segment": sc, "winning_model": m_name} for sc, m_name in SEGMENT_MODEL_MAP.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except ImportError:
        st.warning("Could not load SEGMENT_MODEL_MAP.")

st.divider()

# ── Candidate pool ────────────────────────────────────────────────────────────
st.subheader("Model candidate pool")
st.caption("All models registered in the pipeline, by segment eligibility.")

try:
    import pandas as pd

    import forecasting.models.baseline  # noqa: F401
    import forecasting.models.classical  # noqa: F401
    import forecasting.models.foundation  # noqa: F401
    import forecasting.models.intermittent  # noqa: F401
    import forecasting.models.ml_global  # noqa: F401
    import forecasting.models.tweedie  # noqa: F401
    from forecasting.registry import _MODEL_ENTRIES

    rows = []
    for name, entry in sorted(_MODEL_ENTRIES.items()):
        rows.append(
            {
                "model": name,
                "segments": ", ".join(sorted(entry.get("segments", []))),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
except Exception as e:
    st.warning(f"Could not load model registry: {e}")

st.divider()

# ── Segment descriptions ──────────────────────────────────────────────────────
st.subheader("Named demand segments (Syntetos-Boylan)")
st.markdown("""
| Segment | IDI | CV² | Revenue share | Best model |
|---|---|---|---|---|
| **erratic** | < 1.32 | ≥ 0.49 | ~38–51% | `trend_seasonal` |
| **lumpy** | ≥ 1.32 | ≥ 0.49 | ~27–35% | `seasonal_naive` |
| **smooth** | < 1.32 | < 0.49 | ~3–20% | `recent_level` |
| **intermittent** | ≥ 1.32 | < 0.49 | ~7–14% | `seasonal_naive` |
| **cold_start** | < 4 non-zero obs | — | ~0% | `seasonal_naive` |
| **discontinued** | dormant ≥ 26w | — | 0% | `zero_forecast` |

*IDI = inter-demand interval, CV² = squared coefficient of variation of non-zero demands.*
""")
