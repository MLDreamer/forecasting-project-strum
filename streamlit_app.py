"""Root-level entry point for Streamlit Cloud.

Streamlit Cloud works most reliably when the main file is at repo root.
This simply delegates to app/streamlit_app.py with the correct sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).parent
_src = _root / "src"
for _p in (_root, _src):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import streamlit as st

from app.lib.load import get_manifest_hash, has_data, load_app_data

st.set_page_config(
    page_title="Fontana Candle · Forecasting",
    page_icon="🕯️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading forecast data…")
def _load(manifest_hash: str):  # noqa: ARG001
    return load_app_data()


if not has_data():
    st.warning(
        "No forecast data found in `outputs/latest/`. Run the pipeline first.",
        icon="⚠️",
    )
    st.stop()

manifest_hash = get_manifest_hash()
app_data = _load(manifest_hash)
st.session_state["app_data"] = app_data

with st.sidebar:
    st.markdown("### 🕯️ Fontana Candle")
    st.caption(
        f"Run: **{app_data.manifest.get('run_id', '—')}**  \n"
        f"Origin: {app_data.manifest.get('forecast_origin', '—')}  \n"
        f"WAPE: **{app_data.manifest.get('blended_wape', '—')}**"
    )
    st.divider()
    st.caption("Navigate using the pages above ↑")

st.markdown("## 🕯️ Fontana Candle Forecasting Dashboard")
st.markdown(
    "Use the **pages** in the left sidebar to navigate:\n\n"
    "- **Overview** — total portfolio 26-week forecast\n"
    "- **Drilldown** — explore by product type → SKU\n"
    "- **AutoML** — which model won each segment and why\n"
    "- **Run log** — run history, manifest, audit trail"
)

m = app_data.manifest
col1, col2, col3, col4 = st.columns(4)
col1.metric("Run ID", m.get("run_id", "—"))
col2.metric("Forecast origin", m.get("forecast_origin", "—"))
col3.metric("SKUs forecast", m.get("n_sku", "—"))
col4.metric(
    "RW-WAPE",
    f"{m.get('blended_wape', '—'):.3f}" if isinstance(m.get("blended_wape"), float) else "—",
)
