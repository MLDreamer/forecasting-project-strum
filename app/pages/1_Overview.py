"""Page 1 — Overview (L0 total portfolio)

Shows the overall portfolio forecast with:
- KPI strip: next-26w P50 total, RW-WAPE vs 0.60, run date, # SKUs
- Three-region time series for the total node
- Segment-mix donut for the whole portfolio
- "Drill in →" link to Page 2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.lib.charts import segment_donut, three_region_chart
from app.lib.load import AppData

st.set_page_config(page_title="Overview", layout="wide")

# ── Get app data from session_state ─────────────────────────────────────────
if "app_data" not in st.session_state:
    st.warning("Please open the main page first to load data.")
    st.stop()

app_data: AppData = st.session_state["app_data"]
m = app_data.manifest
hn = app_data.hierarchy_nodes
fl = app_data.forecast_long
al = app_data.actuals_long
bl = app_data.backtest_long
sm = app_data.segment_mix

# ── KPI strip ────────────────────────────────────────────────────────────────
st.markdown("## Overview — Portfolio Forecast")

# Compute KPIs from data
total_forecast_p50 = 0.0
if not fl.empty:
    total_rows = (
        fl[fl["node_id"] == "L0_total"]
        if "L0_total" in fl["node_id"].values
        else fl[fl.get("level", fl["node_id"].str.startswith("L0")) == 0]
        if "level" in fl.columns
        else fl.head(0)
    )
    if not total_rows.empty and "q50" in total_rows.columns:
        total_forecast_p50 = float(total_rows["q50"].sum())

blended_wape = m.get("blended_wape", None)
wape_str = f"{blended_wape:.3f}" if isinstance(blended_wape, float) else "—"
wape_delta = "vs bar 0.60" if isinstance(blended_wape, float) else None
wape_delta_color = (
    "inverse" if isinstance(blended_wape, float) and blended_wape > 0.60 else "normal"
)

n_sku = m.get("n_sku", "—")
run_id = m.get("run_id", "—")
origin = m.get("forecast_origin", "—")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Run", run_id)
col2.metric("Forecast origin", origin)
col3.metric(
    "RW-WAPE (locked)",
    wape_str,
    delta=wape_delta,
    delta_color=wape_delta_color,
    help="Revenue-weighted WAPE, pooled selection folds 2-4, fixed price weight.",
)
col4.metric("SKUs forecast", n_sku)
if total_forecast_p50 > 0:
    col5.metric("26w total P50", f"{total_forecast_p50:,.0f} units")

st.divider()

# ── Time series + donut ───────────────────────────────────────────────────────
col_ts, col_donut = st.columns([2, 1])

with col_ts:
    st.subheader("Total portfolio — history · holdout · forecast")

    # Filter to L0 node
    root_id = "L0_total"
    if not hn.empty and "node_id" in hn.columns:
        l0_nodes = hn[hn["level"] == 0]["node_id"].tolist()
        root_id = l0_nodes[0] if l0_nodes else "L0_total"

    act_df = al[al["node_id"] == root_id] if not al.empty and "node_id" in al.columns else al
    bt_df = (
        bl[bl["node_id"] == root_id].rename(columns={"q50": "q50"})
        if not bl.empty and "node_id" in bl.columns
        else bl
    )
    fc_df = fl[fl["node_id"] == root_id] if not fl.empty and "node_id" in fl.columns else fl

    fig = three_region_chart(act_df, bt_df, fc_df, node_name="L0 — Total Portfolio")
    st.plotly_chart(fig, use_container_width=True)

with col_donut:
    st.subheader("Segment mix (revenue)")
    if not sm.empty:
        fig_d = segment_donut(sm, root_id)
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.info("No segment mix data available.")

# ── Drill-in link ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("🔍 **Want to explore by product type or individual SKU?**")
st.page_link("pages/2_Drilldown.py", label="Drill down into the hierarchy →", icon="🔎")
