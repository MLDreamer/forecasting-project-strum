"""lib/load.py — Load the 8 contract tables from outputs/latest/.

Cached on the manifest hash so a new monthly commit invalidates the cache
and the dashboard shows fresh data automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import pandas as pd


class AppData(NamedTuple):
    hierarchy_nodes: pd.DataFrame
    forecast_long: pd.DataFrame
    actuals_long: pd.DataFrame
    backtest_long: pd.DataFrame
    segment_mix: pd.DataFrame
    leaf_importance: pd.DataFrame
    selection: pd.DataFrame
    manifest: dict
    runs_index: list[dict]


def _latest_dir() -> Path:
    base = Path(__file__).parent.parent.parent
    return base / "outputs" / "latest"


def _runs_dir() -> Path:
    return Path(__file__).parent.parent.parent / "runs"


def get_manifest_hash() -> str:
    p = _latest_dir() / "run_manifest.json"
    if not p.exists():
        return "no-data"
    try:
        m = json.loads(p.read_text())
        return str(m.get("forecast_cube_hash", "unknown"))
    except Exception:
        return "error"


def load_app_data() -> AppData:
    """Load all contract tables.  Call this inside @st.cache_data keyed on manifest hash."""
    d = _latest_dir()

    def _pq(name: str) -> pd.DataFrame:
        p = d / name
        if p.exists():
            return pd.read_parquet(p)
        return pd.DataFrame()

    manifest: dict = {}
    mp = d / "run_manifest.json"
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text())
        except Exception:
            manifest = {}

    runs_index: list[dict] = []
    ri = _runs_dir() / "index.json"
    if ri.exists():
        try:
            runs_index = json.loads(ri.read_text())
        except Exception:
            runs_index = []

    return AppData(
        hierarchy_nodes=_pq("hierarchy_nodes.parquet"),
        forecast_long=_pq("forecast_long.parquet"),
        actuals_long=_pq("actuals_long.parquet"),
        backtest_long=_pq("backtest_long.parquet"),
        segment_mix=_pq("segment_mix.parquet"),
        leaf_importance=_pq("leaf_importance.parquet"),
        selection=_pq("selection.parquet"),
        manifest=manifest,
        runs_index=runs_index,
    )


def has_data() -> bool:
    return (_latest_dir() / "run_manifest.json").exists()
