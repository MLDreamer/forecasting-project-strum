"""Page 4 — Run log

Shows:
- Table of all runs from runs/index.json (date, status, WAPE, winners)
- Selected run manifest (input hashes, lib versions, seed, git SHA)
- Month-over-month WAPE trend
- Raw run.log if available
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.lib.load import AppData

st.set_page_config(page_title="Run Log", layout="wide")

if "app_data" not in st.session_state:
    st.warning("Please open the main page first.")
    st.stop()

app_data: AppData = st.session_state["app_data"]
runs_index = app_data.runs_index
m = app_data.manifest

st.markdown("## Run Log")
st.caption("Every monthly training run is versioned. Failed runs never update the dashboard.")

# ── Run history table ─────────────────────────────────────────────────────────
if runs_index:
    df = pd.DataFrame(runs_index)
    for col in ["blended_wape"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    st.subheader("All runs")
    st.dataframe(df, use_container_width=True, height=min(400, (len(df) + 1) * 38))

    # WAPE trend chart
    if "blended_wape" in df.columns and "date" in df.columns:
        st.subheader("RW-WAPE trend")
        df_trend = df.dropna(subset=["blended_wape"]).sort_values("date")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_trend["date"],
                y=df_trend["blended_wape"],
                mode="lines+markers",
                name="Blended RW-WAPE",
                line={"color": "#F59E0B", "width": 2.5},
                marker={"size": 8},
            )
        )
        fig.add_hline(y=0.60, line_color="#EF4444", line_dash="dash", annotation_text="Target 0.60")
        fig.update_layout(
            height=260,
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis={"title": "RW-WAPE", "showgrid": True, "gridcolor": "#F1F5F9"},
            xaxis={"showgrid": False},
            margin={"l": 40, "r": 20, "t": 20, "b": 30},
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No run history found in `runs/index.json`. Run the pipeline to generate runs.")

st.divider()

# ── Current manifest ──────────────────────────────────────────────────────────
st.subheader("Current run manifest")
if m:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Run ID:** `{m.get('run_id', '—')}`")
        st.markdown(f"**Git SHA:** `{m.get('git_sha', '—')}`")
        st.markdown(f"**Forecast origin:** {m.get('forecast_origin', '—')}")
        st.markdown(f"**Horizon:** {m.get('horizon_weeks', 26)} weeks")
        st.markdown(f"**Pipeline version:** `{m.get('pipeline_version', '—')}`")
        st.markdown(f"**Cube hash:** `{m.get('forecast_cube_hash', '—')}`")
    with col2:
        st.markdown(f"**Blended RW-WAPE:** `{m.get('blended_wape', '—')}`")
        st.markdown(f"**Status:** `{m.get('status', '—')}`")

        # Input hashes
        input_hashes = m.get("input_hashes", {})
        if input_hashes:
            st.markdown("**Input file hashes:**")
            for fname, fhash in input_hashes.items():
                st.markdown(f"  - `{fname}`: `{fhash}`")

        # Winners
        winners = m.get("winners", {})
        if winners:
            st.markdown("**Segment winners:**")
            for seg, model in sorted(winners.items()):
                st.markdown(f"  - {seg}: `{model}`")
else:
    st.info("No manifest found.")

st.divider()

# ── Raw run.log ───────────────────────────────────────────────────────────────
st.subheader("Run log (latest)")
run_id = m.get("run_id", "")
runs_dir = Path(__file__).parent.parent.parent / "runs"
log_path = runs_dir / run_id / "run.log" if run_id else None

if log_path and log_path.exists():
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    # Show last 200 lines to keep the page snappy
    lines = log_text.splitlines()
    if len(lines) > 200:
        st.caption(f"Showing last 200 of {len(lines)} lines.")
        lines = lines[-200:]
    st.code("\n".join(lines), language="text")
else:
    if run_id:
        st.info(
            f"No run.log found at `runs/{run_id}/run.log`. Log is written by the GitHub Action."
        )
    else:
        st.info("No run ID in manifest.")
