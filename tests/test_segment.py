"""Phase 5 gate: 220 rows, K=3 clusters, SB class distribution, revenue tiers."""

from __future__ import annotations

import pytest

from forecasting import config
from forecasting.densify import densify
from forecasting.features import build_features
from forecasting.io import load_all
from forecasting.lifecycle import infer_lifecycle
from forecasting.segment import SegmentResult, _sb_class, segment_and_cluster


@pytest.fixture(scope="module")
def result() -> SegmentResult:
    d = load_all()
    lc = infer_lifecycle(d.sales, d.master)
    dense = densify(d.sales, lc, d.joined, week_relabel_shift_days=6)
    feat = build_features(dense.dense, lc)
    return segment_and_cluster(feat.features, lc)


# ---------------------------------------------------------------------------
# Gate counts
# ---------------------------------------------------------------------------


def test_segment_row_count(result: SegmentResult) -> None:
    """One row per in-scope SKU (220 after Gift Card + return filter)."""
    assert len(result.segments) == 220


def test_selected_k(result: SegmentResult) -> None:
    """K is data-driven (ARI-based fallback); K=3 wins blend on our data."""
    assert result.selected_k == 3


def test_used_fallback(result: SegmentResult) -> None:
    """No fallback fired."""
    assert result.used_fallback is False


def test_cluster_ids_range(result: SegmentResult) -> None:
    """Cluster IDs must be 0..K-1."""
    assert set(result.segments["cluster_id"].unique()) == set(range(result.selected_k))


# ---------------------------------------------------------------------------
# SB class distribution
# ---------------------------------------------------------------------------


def test_all_sb_classes_valid(result: SegmentResult) -> None:
    valid = {"smooth", "erratic", "intermittent", "lumpy", "cold_start", "discontinued"}
    assert set(result.segments["sb_class"].unique()).issubset(valid)


def test_discontinued_equals_dormant(result: SegmentResult) -> None:
    """Discontinued count must match dormant SKU count from lifecycle."""
    d = load_all()
    lc = infer_lifecycle(d.sales, d.master)
    n_disc = (result.segments["sb_class"] == "discontinued").sum()
    assert n_disc == len(lc.sku_dormant)


def test_all_classes_present(result: SegmentResult) -> None:
    """At least smooth, erratic, lumpy, intermittent must be present."""
    classes = set(result.segments["sb_class"].unique())
    for cls in ("smooth", "erratic", "lumpy", "intermittent"):
        assert cls in classes, f"Missing SB class: {cls}"


def test_sb_class_counts_sum_to_220(result: SegmentResult) -> None:
    assert result.segments["sb_class"].notna().sum() == 220


# ---------------------------------------------------------------------------
# Revenue tier
# ---------------------------------------------------------------------------


def test_revenue_tier_values(result: SegmentResult) -> None:
    assert set(result.segments["revenue_tier"].unique()).issubset({"A", "B", "C"})


def test_revenue_tier_all_assigned(result: SegmentResult) -> None:
    assert result.segments["revenue_tier"].notna().all()


def test_revenue_tier_a_is_top_20pct(result: SegmentResult) -> None:
    n_a = (result.segments["revenue_tier"] == "A").sum()
    assert n_a <= int(0.22 * 220 + 2)  # ~20% ± slack


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_required_columns(result: SegmentResult) -> None:
    required = {
        config.COL_SKU_ID,
        "sb_class",
        "revenue_tier",
        "cluster_id",
        "idi",
        "cv2",
        "zero_rate",
        "silhouette_score",
        "best_k",
        "used_fallback_k",
    }
    assert required.issubset(result.segments.columns)


def test_no_null_cluster_id(result: SegmentResult) -> None:
    assert result.segments["cluster_id"].notna().all()


def test_cluster_id_integer(result: SegmentResult) -> None:
    assert result.segments["cluster_id"].dtype in ("int32", "int64", int)


def test_best_k_column_matches_selected_k(result: SegmentResult) -> None:
    assert (result.segments["best_k"] == result.selected_k).all()


# ---------------------------------------------------------------------------
# Unit tests for _sb_class helper
# ---------------------------------------------------------------------------


def test_smooth_sku() -> None:
    import numpy as np

    # Low IDI (regular), low CV2 (stable amounts)
    sales = np.array([10.0, 12.0, 11.0, 10.0, 9.0, 11.0, 10.0, 12.0])
    assert _sb_class(sales, is_dormant=False) == "smooth"


def test_lumpy_sku() -> None:
    import numpy as np

    # High IDI (gaps >=1.32), high CV2 (variable amounts >=0.49), >=4 non-zero obs
    sales = np.array(
        [
            0,
            0,
            50,
            0,
            0,
            0,
            0,
            0,
            0,
            100,
            0,
            0,
            0,
            0,
            0,
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            200,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            10,
        ]
    )
    assert _sb_class(sales, is_dormant=False) == "lumpy"


def test_discontinued_sku() -> None:
    import numpy as np

    sales = np.array([5.0, 3.0, 0.0, 0.0, 0.0])
    assert _sb_class(sales, is_dormant=True) == "discontinued"


def test_cold_start_sku() -> None:
    import numpy as np

    # Only 3 non-zero observations
    sales = np.array([0, 0, 5, 0, 3, 0, 2, 0, 0, 0, 0])
    assert _sb_class(sales, is_dormant=False) == "cold_start"


# ---------------------------------------------------------------------------
# Plausibility smoke
# ---------------------------------------------------------------------------


def test_idi_positive(result: SegmentResult) -> None:
    assert (result.segments["idi"] > 0).all()


def test_silhouette_in_valid_range(result: SegmentResult) -> None:
    assert result.segments["silhouette_score"].between(-1.0, 1.0).all()


def test_unique_skus(result: SegmentResult) -> None:
    assert result.segments[config.COL_SKU_ID].nunique() == 220
