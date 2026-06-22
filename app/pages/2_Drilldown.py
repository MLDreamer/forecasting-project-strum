"""Page 2 — Drilldown (total → product_type → SKU)

Recursive drill-down with breadcrumb navigation.
State: st.session_state["path"] = list of node_ids from root to current.

Non-leaf: time series + segment donut + children list (click to drill)
Leaf (SKU): time series + segment badge + chosen model + demand drivers
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.lib.charts import SEGMENT_COLORS, driver_bars, segment_donut, three_region_chart
from app.lib.load import AppData

st.set_page_config(page_title="Drilldown", layout="wide")

if "app_data" not in st.session_state:
    st.warning("Please open the main page first.")
    st.stop()

app_data: AppData = st.session_state["app_data"]
hn = app_data.hierarchy_nodes
fl = app_data.forecast_long
al = app_data.actuals_long
bl = app_data.backtest_long
sm = app_data.segment_mix
li = app_data.leaf_importance
sel = app_data.selection

# ── State: current drill path ─────────────────────────────────────────────────
if "path" not in st.session_state or not st.session_state["path"]:
    if not hn.empty:
        l0 = hn[hn["level"] == 0]["node_id"].tolist()
        st.session_state["path"] = [l0[0]] if l0 else ["L0_total"]
    else:
        st.session_state["path"] = ["L0_total"]

path: list[str] = st.session_state["path"]
current_id = path[-1]

# ── Breadcrumb ────────────────────────────────────────────────────────────────
st.markdown("## Hierarchy Drilldown")
if not hn.empty:
    crumb_cols = st.columns(min(len(path) + 1, 8))
    for i, nid in enumerate(path):
        label = hn[hn["node_id"] == nid]["name"].values[0] if nid in hn["node_id"].values else nid
        if crumb_cols[i].button(
            f"{'📦' if i == 0 else '📁' if i < len(path) - 1 else '🏷️'} {label}", key=f"crumb_{i}"
        ):
            st.session_state["path"] = path[: i + 1]
            st.rerun()

st.divider()

# ── Get current node info ─────────────────────────────────────────────────────
if hn.empty:
    st.info("No hierarchy data available. Run the pipeline first.")
    st.stop()

node_row = hn[hn["node_id"] == current_id]
if node_row.empty:
    st.error(f"Node {current_id} not found.")
    st.stop()

node_info = node_row.iloc[0]
is_leaf = int(node_info["level"]) == 2
node_name = str(node_info["name"])
node_level = int(node_info["level"])

# ── Time series ───────────────────────────────────────────────────────────────
act_df = al[al["node_id"] == current_id] if not al.empty and "node_id" in al.columns else al.head(0)
bt_df = bl[bl["node_id"] == current_id] if not bl.empty and "node_id" in bl.columns else bl.head(0)
fc_df = fl[fl["node_id"] == current_id] if not fl.empty and "node_id" in fl.columns else fl.head(0)

level_label = {0: "Portfolio", 1: "Product type", 2: "SKU"}.get(node_level, "Node")

if is_leaf:
    col_ts, col_info = st.columns([2, 1])
else:
    col_ts, col_donut = st.columns([2, 1])

with col_ts:
    st.subheader(f"{level_label}: {node_name}")
    fig = three_region_chart(act_df, bt_df, fc_df, node_name=f"{level_label} — {node_name}")
    st.plotly_chart(fig, use_container_width=True)

if is_leaf:
    with col_info:
        # Segment badge + model + WAPE
        segment = str(node_info.get("segment", ""))
        color = SEGMENT_COLORS.get(segment, "#94A3B8")
        st.markdown(
            f"**Segment:** :{color.replace('#', '')}[{segment}]" if segment else "**Segment:** —"
        )

        # Winning model for this segment
        if not sel.empty and "segment" in sel.columns:
            sel_row = sel[sel["segment"] == segment]
            if not sel_row.empty:
                model = sel_row.iloc[0].get("winning_model", "seasonal_naive")
                wape_s = sel_row.iloc[0].get("wape", None)
                st.markdown(f"**Model:** `{model}`")
                if wape_s and isinstance(wape_s, float):
                    st.markdown(f"**Segment WAPE:** {wape_s:.3f}")

        st.divider()
        # Demand drivers
        st.subheader("Demand drivers")
        if not li.empty:
            # Use segment-level importance if available
            key_col = "sb_class" if "sb_class" in li.columns else None
            if key_col and segment in li[key_col].values:
                fig_d = driver_bars(li, segment)
            else:
                fig_d = driver_bars(
                    li.head(8), li.iloc[0].get("sb_class", "") if len(li) > 0 else ""
                )
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No feature importance data.")
else:
    with col_donut:
        st.subheader("Segment mix")
        if not sm.empty:
            fig_d = segment_donut(sm, current_id)
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No segment data.")

# ── Children list (non-leaf only) ────────────────────────────────────────────
if not is_leaf:
    st.divider()
    children = hn[hn["parent_id"] == current_id].copy() if "parent_id" in hn.columns else hn.head(0)

    if children.empty:
        st.info("No children.")
    else:
        st.subheader(f"Children ({len(children)})")

        # Sort by forecast volume if available
        if not fc_df.empty and "q50" in fc_df.columns:
            child_forecast = (
                (
                    fl[fl["node_id"].isin(children["node_id"].tolist())]
                    .groupby("node_id")["q50"]
                    .sum()
                    .reset_index()
                    .rename(columns={"q50": "forecast_total"})
                )
                if "q50" in fl.columns
                else __import__("pandas").DataFrame()
            )
            if not child_forecast.empty:
                children = children.merge(child_forecast, on="node_id", how="left").fillna(
                    {"forecast_total": 0}
                )
                children = children.sort_values("forecast_total", ascending=False)

        # Display as clickable rows
        for _, child in children.iterrows():
            cid = child["node_id"]
            cname = child["name"]
            cn_skus = child.get("n_skus", "")
            cseg = child.get("segment", "")
            clevel = {0: "Portfolio", 1: "Product type", 2: "SKU"}.get(
                int(child.get("level", 2)), ""
            )
            fc_vol = child.get("forecast_total", "")
            fc_str = (
                f"  |  26w P50: {fc_vol:,.0f}" if isinstance(fc_vol, float) and fc_vol > 0 else ""
            )
            seg_str = f"  |  {cseg}" if cseg else ""
            col_btn, col_meta = st.columns([1, 3])
            with col_btn:
                if st.button(f"🔍 {cname}", key=f"drill_{cid}"):
                    st.session_state["path"] = path + [cid]
                    st.rerun()
            with col_meta:
                st.caption(f"{clevel}  |  {cn_skus} SKUs{seg_str}{fc_str}")
