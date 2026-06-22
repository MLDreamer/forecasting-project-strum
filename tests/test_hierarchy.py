"""Phase 6 gate: hierarchy structure, S matrix correctness, round-trip check.

Note on node counts: the doc specifies 420 nodes (1/7/192/220) from the original
Excel data.  Our CSV export gives 228 nodes (1/7/220) — after Gift Card + return
filter removes 9 SKUs.  The structural invariants (level-prefixed IDs,
multi-parent guard, round-trip S@bottom==agg) are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from forecasting import config
from forecasting.densify import densify
from forecasting.features import build_features
from forecasting.hierarchy import (
    HierarchyResult,
    _make_node_id,
    build_hierarchy,
    verify_roundtrip,
)
from forecasting.io import load_all
from forecasting.lifecycle import infer_lifecycle
from forecasting.segment import segment_and_cluster


@pytest.fixture(scope="module")
def hierarchy() -> HierarchyResult:
    d = load_all()
    lc = infer_lifecycle(d.sales, d.master)
    dense = densify(d.sales, lc, d.joined, week_relabel_shift_days=6)
    feat = build_features(dense.dense, lc)
    seg = segment_and_cluster(feat.features, lc)
    return build_hierarchy(seg.segments, d.master)


# ---------------------------------------------------------------------------
# Gate: node counts
# ---------------------------------------------------------------------------


def test_total_node_count(hierarchy: HierarchyResult) -> None:
    """228 nodes from our CSV data: 1 total + 7 product_types + 220 variants."""
    assert len(hierarchy.nodes) == 228


def test_level_0_count(hierarchy: HierarchyResult) -> None:
    assert hierarchy.level_counts[0] == 1


def test_level_1_count(hierarchy: HierarchyResult) -> None:
    """7 product_type nodes."""
    assert hierarchy.level_counts[1] == 7


def test_level_2_count(hierarchy: HierarchyResult) -> None:
    """220 variant (bottom) nodes."""
    assert hierarchy.level_counts[2] == 220


def test_bottom_ids_count(hierarchy: HierarchyResult) -> None:
    assert len(hierarchy.bottom_ids) == 220


# ---------------------------------------------------------------------------
# S matrix shape and structure
# ---------------------------------------------------------------------------


def test_s_shape(hierarchy: HierarchyResult) -> None:
    assert hierarchy.S.shape == (228, 220)


def test_s_is_sparse(hierarchy: HierarchyResult) -> None:
    assert sp.issparse(hierarchy.S)


def test_s_is_binary(hierarchy: HierarchyResult) -> None:
    """Every non-zero entry in S must be exactly 1.0."""
    vals = np.unique(hierarchy.S.data)
    assert set(vals.tolist()) == {1.0}


def test_s_l0_row_sums_to_n_bottom(hierarchy: HierarchyResult) -> None:
    """The L0 (total) row of S sums to n_bottom (all variants are under total)."""
    root_id = _make_node_id(0, "total")
    node_idx = {n.node_id: i for i, n in enumerate(hierarchy.nodes)}
    l0_row = np.asarray(hierarchy.S[node_idx[root_id], :].todense()).ravel()
    assert l0_row.sum() == 220


def test_s_bottom_rows_are_identity(hierarchy: HierarchyResult) -> None:
    """Each bottom-level row of S has exactly one 1 (itself)."""
    node_idx = {n.node_id: i for i, n in enumerate(hierarchy.nodes)}
    for j, bid in enumerate(hierarchy.bottom_ids):
        row = np.asarray(hierarchy.S[node_idx[bid], :].todense()).ravel()
        assert row.sum() == 1.0
        assert row[j] == 1.0


def test_s_l1_row_sums_match_child_counts(hierarchy: HierarchyResult) -> None:
    """Each L1 row sum = number of variants under that product_type."""
    from collections import Counter

    node_idx = {n.node_id: i for i, n in enumerate(hierarchy.nodes)}
    parent_counts = Counter(n.parent_id for n in hierarchy.nodes if n.level == 2 and n.parent_id)
    for n in hierarchy.nodes:
        if n.level != 1:
            continue
        expected = parent_counts[n.node_id]
        actual = int(np.asarray(hierarchy.S[node_idx[n.node_id], :].todense()).ravel().sum())
        assert actual == expected, (
            f"L1 node {n.node_id}: expected {expected} children, got {actual}"
        )


# ---------------------------------------------------------------------------
# Round-trip: S @ bottom == aggregated
# ---------------------------------------------------------------------------


def test_roundtrip_random_vector(hierarchy: HierarchyResult) -> None:
    """S @ random_bottom_vector must equal itself at bottom rows."""
    rng = np.random.default_rng(42)
    v = rng.random(220).astype(np.float32)
    assert verify_roundtrip(hierarchy, v)


def test_roundtrip_all_ones(hierarchy: HierarchyResult) -> None:
    v = np.ones(220, dtype=np.float32)
    assert verify_roundtrip(hierarchy, v)


def test_roundtrip_all_zeros(hierarchy: HierarchyResult) -> None:
    v = np.zeros(220, dtype=np.float32)
    assert verify_roundtrip(hierarchy, v)


def test_l0_agg_equals_bottom_sum(hierarchy: HierarchyResult) -> None:
    """L0 aggregate must equal sum of all bottom values."""
    rng = np.random.default_rng(99)
    v = rng.random(220).astype(np.float32)
    agg = np.asarray(hierarchy.S @ v).ravel()
    root_id = _make_node_id(0, "total")
    node_idx = {n.node_id: i for i, n in enumerate(hierarchy.nodes)}
    l0_val = agg[node_idx[root_id]]
    assert abs(l0_val - v.sum()) < 1e-3


def test_l1_aggs_sum_to_l0(hierarchy: HierarchyResult) -> None:
    """Sum of all L1 aggregates must equal the L0 aggregate."""
    rng = np.random.default_rng(7)
    v = rng.random(220).astype(np.float32)
    agg = np.asarray(hierarchy.S @ v).ravel()
    node_idx = {n.node_id: i for i, n in enumerate(hierarchy.nodes)}
    root_id = _make_node_id(0, "total")
    l0_val = agg[node_idx[root_id]]
    l1_sum = sum(agg[node_idx[n.node_id]] for n in hierarchy.nodes if n.level == 1)
    assert abs(l1_sum - l0_val) < 1e-3


# ---------------------------------------------------------------------------
# Node ID uniqueness and level-prefix invariant
# ---------------------------------------------------------------------------


def test_node_ids_globally_unique(hierarchy: HierarchyResult) -> None:
    ids = [n.node_id for n in hierarchy.nodes]
    assert len(ids) == len(set(ids))


def test_node_ids_level_prefixed(hierarchy: HierarchyResult) -> None:
    """Every node_id must start with 'L{level}_'."""
    for n in hierarchy.nodes:
        assert n.node_id.startswith(f"L{n.level}_"), (
            f"node {n.node_id} does not match prefix L{n.level}_"
        )


def test_root_has_no_parent(hierarchy: HierarchyResult) -> None:
    roots = [n for n in hierarchy.nodes if n.level == 0]
    assert len(roots) == 1
    assert roots[0].parent_id is None


def test_all_non_root_have_parent(hierarchy: HierarchyResult) -> None:
    for n in hierarchy.nodes:
        if n.level > 0:
            assert n.parent_id is not None, f"Non-root node {n.node_id} has no parent"


def test_all_parent_ids_exist(hierarchy: HierarchyResult) -> None:
    """Every parent_id must reference a valid node in the hierarchy."""
    all_ids = {n.node_id for n in hierarchy.nodes}
    for n in hierarchy.nodes:
        if n.parent_id is not None:
            assert n.parent_id in all_ids, f"Parent {n.parent_id} of {n.node_id} not found"


# ---------------------------------------------------------------------------
# sku_to_node mapping
# ---------------------------------------------------------------------------


def test_sku_to_node_all_covered(hierarchy: HierarchyResult) -> None:
    d = load_all()
    lc = infer_lifecycle(d.sales, d.master)
    dense = densify(d.sales, lc, d.joined, week_relabel_shift_days=6)
    feat = build_features(dense.dense, lc)
    seg = segment_and_cluster(feat.features, lc)
    for sku in seg.segments[config.COL_SKU_ID]:
        assert int(sku) in hierarchy.sku_to_node, f"SKU {sku} missing from sku_to_node"


def test_sku_to_node_values_in_bottom_ids(hierarchy: HierarchyResult) -> None:
    bottom_set = set(hierarchy.bottom_ids)
    for sku, nid in hierarchy.sku_to_node.items():
        assert nid in bottom_set, f"sku_to_node[{sku}] = {nid} not in bottom_ids"


# ---------------------------------------------------------------------------
# node_df schema
# ---------------------------------------------------------------------------


def test_node_df_columns(hierarchy: HierarchyResult) -> None:
    required = {"node_id", "level", "label", "parent_id"}
    assert required.issubset(hierarchy.node_df.columns)


def test_node_df_row_count(hierarchy: HierarchyResult) -> None:
    assert len(hierarchy.node_df) == len(hierarchy.nodes)
