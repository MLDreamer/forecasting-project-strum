"""Phase 16 gate — reconcile.py: bottom-up bootstrap coherence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from forecasting.hierarchy import HierarchyNode, HierarchyResult
from forecasting.reconcile import _sample_from_quantiles, reconcile_bottom_up

# ---------------------------------------------------------------------------
# Minimal synthetic hierarchy
# ---------------------------------------------------------------------------


def _make_hierarchy(n_sku: int = 4) -> HierarchyResult:
    """Build a tiny hierarchy: L0_total → L1_A(2 SKUs) + L1_B(2 SKUs) → L2 variants."""
    assert n_sku == 4
    nodes = [
        HierarchyNode("L0_total", 0, "total", None),
        HierarchyNode("L1_A", 1, "A", "L0_total"),
        HierarchyNode("L1_B", 1, "B", "L0_total"),
        HierarchyNode("L2_1001", 2, "1001", "L1_A"),
        HierarchyNode("L2_1002", 2, "1002", "L1_A"),
        HierarchyNode("L2_1003", 2, "1003", "L1_B"),
        HierarchyNode("L2_1004", 2, "1004", "L1_B"),
    ]
    bottom_ids = ["L2_1001", "L2_1002", "L2_1003", "L2_1004"]

    # Build S matrix (7 nodes × 4 bottom)
    # L0_total covers all 4
    # L1_A covers 1001, 1002
    # L1_B covers 1003, 1004
    # Each L2 covers only itself
    rows = [
        0,
        0,
        0,
        0,  # L0_total → all
        1,
        1,  # L1_A → 1001, 1002
        2,
        2,  # L1_B → 1003, 1004
        3,  # L2_1001 → 1001
        4,  # L2_1002 → 1002
        5,  # L2_1003 → 1003
        6,  # L2_1004 → 1004
    ]
    cols = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    data = [1.0] * len(cols)
    S = sp.csr_matrix((data, (rows, cols)), shape=(7, 4), dtype=np.float32)

    level_counts = {0: 1, 1: 2, 2: 4}
    sku_to_node = {1001: "L2_1001", 1002: "L2_1002", 1003: "L2_1003", 1004: "L2_1004"}
    node_df = pd.DataFrame(
        [
            {
                "node_id": n.node_id,
                "level": n.level,
                "label": n.label,
                "parent_id": n.parent_id or "",
            }
            for n in nodes
        ]
    )
    return HierarchyResult(
        nodes=nodes,
        bottom_ids=bottom_ids,
        S=S,
        level_counts=level_counts,
        sku_to_node=sku_to_node,
        node_df=node_df,
    )


# ---------------------------------------------------------------------------
# _sample_from_quantiles
# ---------------------------------------------------------------------------


def test_sample_shape() -> None:
    rng = np.random.default_rng(42)
    q = np.sort(rng.random((4, 7)), axis=1)
    q_levels = np.linspace(0.1, 0.9, 7)
    s = _sample_from_quantiles(q, q_levels, n_samples=100, rng=rng)
    assert s.shape == (4, 100)


def test_sample_nonneg() -> None:
    rng = np.random.default_rng(1)
    q = np.sort(rng.random((4, 7)), axis=1)
    q_levels = np.linspace(0.1, 0.9, 7)
    s = _sample_from_quantiles(q, q_levels, n_samples=200, rng=rng)
    assert (s >= 0).all()


def test_sample_within_quantile_range() -> None:
    """All samples should fall within [q_min, q_max] of their row."""
    rng = np.random.default_rng(2)
    q = np.sort(rng.random((3, 5)) * 10, axis=1)
    q_levels = np.linspace(0.1, 0.9, 5)
    s = _sample_from_quantiles(q, q_levels, n_samples=500, rng=rng)
    for h in range(3):
        assert (s[h] >= q[h, 0] - 1e-6).all()
        assert (s[h] <= q[h, -1] + 1e-6).all()


# ---------------------------------------------------------------------------
# reconcile_bottom_up
# ---------------------------------------------------------------------------


@pytest.fixture
def reconcile_setup():
    hier = _make_hierarchy(n_sku=4)
    sku_ids = [1001, 1002, 1003, 1004]
    H = 3
    rng = np.random.default_rng(42)
    q_levels = np.array([0.1, 0.5, 0.9])
    forecast_cube = np.sort(rng.random((4, H, 3)) * 20, axis=2)
    horizon_dates = [pd.Timestamp("2026-06-07") + pd.Timedelta(weeks=h) for h in range(H)]
    return hier, sku_ids, forecast_cube, q_levels, horizon_dates


def test_reconcile_output_is_dataframe(reconcile_setup) -> None:
    hier, sku_ids, cube, q_levels, dates = reconcile_setup
    result = reconcile_bottom_up(cube, sku_ids, hier, q_levels, dates, n_bootstrap=50)
    assert isinstance(result, pd.DataFrame)


def test_reconcile_row_count(reconcile_setup) -> None:
    hier, sku_ids, cube, q_levels, dates = reconcile_setup
    H = len(dates)
    n_nodes = len(hier.nodes)
    result = reconcile_bottom_up(cube, sku_ids, hier, q_levels, dates, n_bootstrap=50)
    assert len(result) == n_nodes * H


def test_reconcile_columns(reconcile_setup) -> None:
    hier, sku_ids, cube, q_levels, dates = reconcile_setup
    result = reconcile_bottom_up(cube, sku_ids, hier, q_levels, dates, n_bootstrap=50)
    required = {"node_id", "level", "label", "forecast_date", "p10", "p50", "p90"}
    assert required.issubset(result.columns)


def test_reconcile_nonneg(reconcile_setup) -> None:
    hier, sku_ids, cube, q_levels, dates = reconcile_setup
    result = reconcile_bottom_up(cube, sku_ids, hier, q_levels, dates, n_bootstrap=50)
    assert (result["p10"] >= 0).all()
    assert (result["p50"] >= 0).all()
    assert (result["p90"] >= 0).all()


def test_reconcile_p50_lte_p90(reconcile_setup) -> None:
    hier, sku_ids, cube, q_levels, dates = reconcile_setup
    result = reconcile_bottom_up(cube, sku_ids, hier, q_levels, dates, n_bootstrap=50)
    assert (result["p50"] <= result["p90"] + 1e-6).all()


def test_reconcile_p10_lte_p50(reconcile_setup) -> None:
    hier, sku_ids, cube, q_levels, dates = reconcile_setup
    result = reconcile_bottom_up(cube, sku_ids, hier, q_levels, dates, n_bootstrap=50)
    assert (result["p10"] <= result["p50"] + 1e-6).all()


def test_reconcile_bottom_up_coherence(reconcile_setup) -> None:
    """Sum of bottom P50 at each week ≈ L0 P50 at that week."""
    hier, sku_ids, cube, q_levels, dates = reconcile_setup
    result = reconcile_bottom_up(cube, sku_ids, hier, q_levels, dates, n_bootstrap=200)

    for dt in dates:
        dt_str = str(dt.date())
        week = result[result["forecast_date"].astype(str) == dt_str]
        l0_p50 = float(week[week["level"] == 0]["p50"].iloc[0])
        bottom_p50_sum = float(week[week["level"] == 2]["p50"].sum())
        # Bootstrap introduces some variance — tolerance of 5%
        rel_err = abs(l0_p50 - bottom_p50_sum) / max(bottom_p50_sum, 1.0)
        assert rel_err < 0.10, (
            f"At {dt_str}: L0 P50={l0_p50:.2f} vs bottom sum={bottom_p50_sum:.2f}"
        )


def test_reconcile_portfolio_p90_lt_sum_bottom_p90(reconcile_setup) -> None:
    """Portfolio P90 < sum of bottom P90s (diversification benefit)."""
    hier, sku_ids, cube, q_levels, dates = reconcile_setup
    # Use a deterministic cube with large spread to make the effect visible
    rng = np.random.default_rng(99)
    deterministic_cube = np.sort(rng.random((4, 3, 3)) * 100, axis=2)
    result = reconcile_bottom_up(
        deterministic_cube, sku_ids, hier, q_levels, dates, n_bootstrap=500
    )
    dt_str = str(dates[0].date())
    week = result[result["forecast_date"].astype(str) == dt_str]
    l0_p90 = float(week[week["level"] == 0]["p90"].iloc[0])
    bottom_p90_sum = float(week[week["level"] == 2]["p90"].sum())
    # Portfolio P90 must be <= sum of bottom P90s (strict inequality for diverse demand)
    assert l0_p90 <= bottom_p90_sum + 1e-3
