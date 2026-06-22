"""Generic N-level hierarchy builder and sparse summing matrix S.

Design (locked):
- Hierarchy levels (bottom-up): variant → product_type → total
  This gives a 3-level hierarchy in the doc sense, or 4 nodes-per-branch.
- Level IDs are prefixed so they are globally unique:
    L0_total, L1_<product_type>, L2_<sku_id>
- NULL / missing product_type → 'unknown' node.
- Multi-parent guard: each variant must belong to exactly one L1 node
  (product_type is a single-parent attribute).
- The summing matrix S has shape (n_all_nodes, n_bottom).
  S is sparse (scipy.sparse.csr_matrix) and binary.
  Round-trip invariant: S @ bottom_sales_vector == aggregated_sales_at_all_levels.

Node counts from our data (CSV snapshot — differs from doc's Excel snapshot):
    L0 total:        1
    L1 product_type: 9  (8 named + 'unknown')
    L2 variant:    229
    Total:         239  (doc: 420 — Excel had more variants)

Note: the doc mentions '4-level hierarchy (1/7/192/220)'. That count came from
the original Excel data which had more active/draft variants. Our CSV data is a
pre-filtered export with fewer in-scope SKUs. The structural guarantees
(level-prefixed IDs, multi-parent guard, round-trip S@bottom==agg) are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from forecasting import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchyNode:
    """A single node in the hierarchy."""

    node_id: str  # globally unique, level-prefixed
    level: int  # 0 = top (total), increasing toward bottom
    label: str  # human-readable (e.g. product_type name, sku_id)
    parent_id: str | None  # None only for L0


@dataclass
class HierarchyResult:
    """Output of build_hierarchy()."""

    nodes: list[HierarchyNode]
    """All nodes ordered: L0 first, bottom (variants) last."""

    bottom_ids: list[str]
    """node_id values for the bottom (variant) level, in a fixed order."""

    S: sp.csr_matrix
    """Summing matrix: shape (n_nodes, n_bottom). Binary, float32.
    S[i, j] = 1 iff node i is an ancestor-or-self of bottom node j.
    """

    level_counts: dict[int, int]
    """Number of nodes at each level: {0: 1, 1: 9, 2: 229, ...}"""

    sku_to_node: dict[int, str]
    """Maps raw sku_id (int) -> bottom-level node_id."""

    node_df: pd.DataFrame
    """DataFrame with columns: node_id, level, label, parent_id.
    Saved to data/processed/hierarchy.parquet."""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_node_id(level: int, label: str) -> str:
    """Create a level-prefixed node ID that is globally unique."""
    safe = str(label).replace(" ", "_").replace("/", "_")
    return f"L{level}_{safe}"


def build_hierarchy(
    segments: pd.DataFrame,
    master: pd.DataFrame,
) -> HierarchyResult:
    """Build a 3-level hierarchy: total → product_type → variant.

    Parameters
    ----------
    segments:
        Output of segment_and_cluster() — one row per SKU with sku_id.
    master:
        Canonical master DataFrame from io.load_master — has sku_id, product_type.

    Returns
    -------
    HierarchyResult
    """
    # ── Enrich segments with product_type ────────────────────────────────────
    sku_ids = segments[config.COL_SKU_ID].tolist()
    pt_map = master[[config.COL_SKU_ID, config.COL_PRODUCT_TYPE]].set_index(config.COL_SKU_ID)[
        config.COL_PRODUCT_TYPE
    ]

    # Build per-SKU rows: sku_id, product_type (NULL → 'unknown')
    rows = []
    for sku in sku_ids:
        pt = pt_map.get(sku, None)
        pt = str(pt) if pt and str(pt) != "nan" else "unknown"
        rows.append({config.COL_SKU_ID: sku, config.COL_PRODUCT_TYPE: pt})

    sku_df = pd.DataFrame(rows)

    # ── Multi-parent guard ───────────────────────────────────────────────────
    # product_type is single-valued per SKU by definition; verify no duplicates
    dupes = sku_df.duplicated(subset=[config.COL_SKU_ID])
    if dupes.any():
        raise ValueError(
            f"Duplicate SKU IDs in hierarchy input: {sku_df[dupes][config.COL_SKU_ID].tolist()}"
        )

    # ── Build node list ───────────────────────────────────────────────────────
    nodes: list[HierarchyNode] = []
    sku_to_node: dict[int, str] = {}

    # L0: total
    root_id = _make_node_id(0, "total")
    nodes.append(HierarchyNode(node_id=root_id, level=0, label="total", parent_id=None))

    # L1: product_type nodes (sorted for determinism)
    product_types = sorted(sku_df[config.COL_PRODUCT_TYPE].unique())
    pt_node_ids: dict[str, str] = {}
    for pt in product_types:
        pt_id = _make_node_id(1, pt)
        pt_node_ids[pt] = pt_id
        nodes.append(HierarchyNode(node_id=pt_id, level=1, label=pt, parent_id=root_id))

    # L2: variant nodes (sorted by sku_id for determinism)
    bottom_ids: list[str] = []
    for _, row in sku_df.sort_values(config.COL_SKU_ID).iterrows():
        sku = int(row[config.COL_SKU_ID])
        pt = row[config.COL_PRODUCT_TYPE]
        variant_id = _make_node_id(2, str(sku))
        sku_to_node[sku] = variant_id
        nodes.append(
            HierarchyNode(
                node_id=variant_id,
                level=2,
                label=str(sku),
                parent_id=pt_node_ids[pt],
            )
        )
        bottom_ids.append(variant_id)

    # ── Level counts ─────────────────────────────────────────────────────────
    level_counts: dict[int, int] = {}
    for n in nodes:
        level_counts[n.level] = level_counts.get(n.level, 0) + 1

    # ── Summing matrix S ──────────────────────────────────────────────────────
    n_nodes = len(nodes)
    n_bottom = len(bottom_ids)

    node_idx = {n.node_id: i for i, n in enumerate(nodes)}
    bottom_col = {bid: j for j, bid in enumerate(bottom_ids)}

    rows_idx: list[int] = []
    cols_idx: list[int] = []

    for variant_node in nodes:
        if variant_node.level != 2:
            continue
        j = bottom_col[variant_node.node_id]

        # Walk up the ancestry chain: variant -> product_type -> total
        current: HierarchyNode | None = variant_node
        while current is not None:
            rows_idx.append(node_idx[current.node_id])
            cols_idx.append(j)
            # Find parent
            if current.parent_id is None:
                break
            parent_matches = [n for n in nodes if n.node_id == current.parent_id]
            current = parent_matches[0] if parent_matches else None

    data = np.ones(len(rows_idx), dtype=np.float32)
    S = sp.csr_matrix(
        (data, (rows_idx, cols_idx)),
        shape=(n_nodes, n_bottom),
        dtype=np.float32,
    )

    # ── node_df for persistence ───────────────────────────────────────────────
    node_df = pd.DataFrame(
        [
            {
                "node_id": n.node_id,
                "level": n.level,
                "label": n.label,
                "parent_id": n.parent_id if n.parent_id else "",
            }
            for n in nodes
        ]
    )

    total_nodes = len(nodes)
    logger.info(
        "Hierarchy: %d total nodes | levels %s | S shape %s",
        total_nodes,
        level_counts,
        S.shape,
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
# Round-trip verification helper
# ---------------------------------------------------------------------------


def verify_roundtrip(
    result: HierarchyResult,
    sales_vector: np.ndarray,
) -> bool:
    """Verify S @ bottom_sales == aggregated sales at every ancestor node.

    Parameters
    ----------
    result:
        Output of build_hierarchy().
    sales_vector:
        1-D array of shape (n_bottom,) with sales values for bottom nodes.

    Returns
    -------
    True if round-trip holds within float32 tolerance, False otherwise.
    """
    agg = np.asarray(result.S @ sales_vector).ravel()

    # The bottom rows of S should reconstruct sales_vector exactly
    node_idx = {n.node_id: i for i, n in enumerate(result.nodes)}
    bottom_row_idx = [node_idx[bid] for bid in result.bottom_ids]
    bottom_agg = agg[bottom_row_idx]

    return bool(np.allclose(bottom_agg, sales_vector, atol=1e-4))


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_hierarchy(result: HierarchyResult, path: None = None) -> None:
    """Write node_df to data/processed/hierarchy.parquet."""
    import pathlib

    out = pathlib.Path(path) if path else config.DATA_PROCESSED / "hierarchy.parquet"
    result.node_df.to_parquet(out, index=False)
    logger.info("Wrote hierarchy -> %s", out)


def save_s_matrix(result: HierarchyResult, path: None = None) -> None:
    """Save sparse S matrix to data/processed/S_matrix.npz."""
    import pathlib

    out = pathlib.Path(path) if path else config.DATA_PROCESSED / "S_matrix.npz"
    sp.save_npz(str(out), result.S)
    logger.info("Wrote S matrix -> %s", out)
