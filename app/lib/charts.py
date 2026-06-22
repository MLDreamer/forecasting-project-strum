"""lib/charts.py — The four reusable Plotly figures.

V1: Three-region time series (history | holdout | forecast)
V2: Segment-mix donut
V3: Demand-driver horizontal bars
V4: Architecture flow diagram (HTML/Plotly)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Fixed segment colours (consistent across all pages)
SEGMENT_COLORS: dict[str, str] = {
    "erratic": "#F59E0B",  # amber
    "lumpy": "#EF4444",  # red
    "smooth": "#10B981",  # green
    "intermittent": "#6366F1",  # indigo
    "cold_start": "#8B5CF6",  # violet
    "discontinued": "#6B7280",  # grey
}

AMBER = "#F59E0B"
MUTED = "#94A3B8"
BAND_FILL = "rgba(245, 158, 11, 0.15)"


def three_region_chart(
    actuals_df: pd.DataFrame,  # rows: week, units, region (history/holdout)
    backtest_df: pd.DataFrame,  # rows: week, q50, fold_id
    forecast_df: pd.DataFrame,  # rows: week, q10, q50, q90
    node_name: str = "",
    height: int = 380,
) -> go.Figure:
    """V1 — Three-region time series: history · holdout · forecast."""
    fig = go.Figure()

    if not actuals_df.empty:
        hist = actuals_df[actuals_df["region"] == "history"].sort_values("week")
        hold = actuals_df[actuals_df["region"] == "holdout"].sort_values("week")

        # History line
        if not hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist["week"],
                    y=hist["units"],
                    mode="lines",
                    name="History",
                    line={"color": MUTED, "width": 1.5},
                )
            )

        # Holdout actual
        if not hold.empty:
            fig.add_trace(
                go.Scatter(
                    x=hold["week"],
                    y=hold["units"],
                    mode="lines",
                    name="Holdout (actual)",
                    line={"color": MUTED, "width": 1.5, "dash": "dot"},
                )
            )

        # Back-test P50
        if not backtest_df.empty:
            bt = backtest_df.sort_values("week")
            fig.add_trace(
                go.Scatter(
                    x=bt["week"],
                    y=bt["q50"],
                    mode="lines",
                    name="Back-test P50",
                    line={"color": AMBER, "width": 1.5, "dash": "dot"},
                    opacity=0.7,
                )
            )

    # Forecast band
    if not forecast_df.empty:
        fc = forecast_df.sort_values("week")
        fig.add_trace(
            go.Scatter(
                x=list(fc["week"]) + list(reversed(fc["week"])),
                y=list(fc.get("q90", fc.get("q10", fc["q50"])))
                + list(reversed(fc.get("q10", fc["q50"]))),
                fill="toself",
                fillcolor=BAND_FILL,
                line={"color": "rgba(0,0,0,0)"},
                name="80% PI",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc["week"],
                y=fc["q50"],
                mode="lines",
                name="Forecast P50",
                line={"color": AMBER, "width": 2.5},
            )
        )

    # Now line (last history week) — convert to numeric timestamp for add_vline
    if not actuals_df.empty:
        hist_weeks = actuals_df[actuals_df["region"] == "history"]["week"]
        if not hist_weeks.empty:
            # add_vline requires a numeric x when xaxis is a string categorical axis;
            # convert to ISO string which Plotly handles correctly via shapes instead
            now_week_str = str(hist_weeks.max())
            fig.add_shape(
                type="line",
                x0=now_week_str,
                x1=now_week_str,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line={"color": AMBER, "width": 2},
            )
            fig.add_annotation(
                x=now_week_str,
                y=1.02,
                yref="paper",
                text="Now",
                showarrow=False,
                font_size=11,
                font_color=AMBER,
                xanchor="left",
            )
            # dummy statement to keep indentation block
    fig.update_layout(
        height=height,
        title={"text": node_name, "font_size": 14},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
        legend={"orientation": "h", "y": -0.15, "font_size": 11},
        xaxis={"showgrid": False, "title": ""},
        yaxis={"showgrid": True, "gridcolor": "#F1F5F9", "title": "Units"},
    )
    return fig


def segment_donut(segment_mix_df: pd.DataFrame, node_id: str, height: int = 280) -> go.Figure:
    """V2 — Segment-mix donut."""
    df = (
        segment_mix_df[segment_mix_df["node_id"] == node_id].copy()
        if not segment_mix_df.empty
        else pd.DataFrame()
    )

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No segment data", showarrow=False, font_size=13)
        fig.update_layout(height=height, paper_bgcolor="white")
        return fig

    labels = df["segment"].tolist()
    values = df["revenue"].tolist()
    colors = [SEGMENT_COLORS.get(s, "#94A3B8") for s in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker_colors=colors,
            textinfo="label+percent",
            hovertemplate="%{label}<br>Revenue: $%{value:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="white",
    )
    return fig


def driver_bars(importance_df: pd.DataFrame, sb_class: str, top_n: int = 8) -> go.Figure:
    """V3 — Demand-driver horizontal bar chart (LightGBM diagnostic)."""
    df = (
        importance_df[
            importance_df.get("sb_class", importance_df.get("sku_id", pd.Series(dtype=str)))
            == sb_class
        ]
        if "sb_class" in importance_df.columns
        else importance_df
    )
    if df.empty or "feature" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No feature importance data", showarrow=False, font_size=12)
        fig.update_layout(height=260, paper_bgcolor="white")
        return fig

    top = df.nlargest(top_n, "importance") if "importance" in df.columns else df.head(top_n)
    colors = [AMBER if i == 0 else "#CBD5E1" for i in range(len(top))]

    fig = go.Figure(
        go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker_color=list(reversed(colors)),
        )
    )
    fig.update_layout(
        height=max(200, top_n * 30 + 60),
        title={
            "text": "Demand drivers (explains, does not forecast)",
            "font_size": 12,
            "font_color": "#64748B",
        },
        yaxis={"autorange": "reversed"},
        xaxis={"showgrid": False, "title": "Feature importance"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 10, "r": 20, "t": 40, "b": 20},
    )
    return fig


def architecture_flow() -> go.Figure:
    """V4 — AutoML pipeline architecture as a styled flow diagram."""
    steps = [
        ("Raw Excel", "#F1F5F9"),
        ("Scope + I/O", "#F1F5F9"),
        ("Lifecycle", "#F1F5F9"),
        ("Densify", "#F1F5F9"),
        ("Features (98→114 cols)", "#FEF3C7"),
        ("SB Segment", "#FEF3C7"),
        ("Per-segment bake-off", "#FEF9C3"),
        ("Select · RW-WAPE", "#FEF3C7"),
        ("Reconcile + Forecast", BAND_FILL),
    ]

    x_pos = [i / (len(steps) - 1) for i in range(len(steps))]

    shapes, annotations = [], []
    for i, (name, bg) in enumerate(steps):
        x = x_pos[i]
        shapes.append(
            {
                "type": "rect",
                "x0": x - 0.05,
                "x1": x + 0.05,
                "y0": 0.3,
                "y1": 0.7,
                "fillcolor": bg,
                "line_color": "#CBD5E1",
                "line_width": 1,
            }
        )
        annotations.append(
            {
                "x": x,
                "y": 0.5,
                "text": name.replace(" ", "<br>"),
                "showarrow": False,
                "font_size": 9,
                "align": "center",
            }
        )
        if i < len(steps) - 1:
            annotations.append(
                {
                    "x": (x_pos[i] + x_pos[i + 1]) / 2,
                    "y": 0.5,
                    "text": "→",
                    "showarrow": False,
                    "font_size": 14,
                    "font_color": "#94A3B8",
                }
            )

    fig = go.Figure()
    fig.update_layout(
        height=140,
        shapes=shapes,
        annotations=annotations,
        xaxis={"visible": False, "range": [-0.07, 1.07]},
        yaxis={"visible": False, "range": [0, 1]},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 0, "r": 0, "t": 10, "b": 10},
    )
    return fig
