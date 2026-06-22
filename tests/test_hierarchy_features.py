"""Phase 6b gate: 8 hierarchy-context features -> 16,068 x 114."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting import config
from forecasting.densify import densify
from forecasting.features import (
    HIERARCHY_FEATURE_COLS,
    add_cluster_features,
    add_hierarchy_features,
    build_features,
)
from forecasting.io import load_all
from forecasting.lifecycle import infer_lifecycle
from forecasting.segment import segment_and_cluster


@pytest.fixture(scope="module")
def df114():  # type: ignore[return]
    d = load_all()
    lc = infer_lifecycle(d.sales, d.master)
    dense = densify(d.sales, lc, d.joined, week_relabel_shift_days=6)
    feat = build_features(dense.dense, lc)
    seg = segment_and_cluster(feat.features, lc)
    df106 = add_cluster_features(feat.features, seg.segments)
    return add_hierarchy_features(df106)


# ---------------------------------------------------------------------------
# Gate: shape
# ---------------------------------------------------------------------------


def test_row_count(df114) -> None:  # type: ignore[return]
    assert len(df114) == 16_068


def test_column_count(df114) -> None:  # type: ignore[return]
    assert len(df114.columns) == 114, f"Expected 114, got {len(df114.columns)}"


def test_hierarchy_col_names_present(df114) -> None:  # type: ignore[return]
    for col in HIERARCHY_FEATURE_COLS:
        assert col in df114.columns, f"Missing: {col}"


def test_hierarchy_col_count(df114) -> None:  # type: ignore[return]
    assert len(HIERARCHY_FEATURE_COLS) == 8


# ---------------------------------------------------------------------------
# No nulls / finite values
# ---------------------------------------------------------------------------


def test_no_nulls_in_hier_cols(df114) -> None:  # type: ignore[return]
    for col in HIERARCHY_FEATURE_COLS:
        n = df114[col].isna().sum()
        assert n == 0, f"{col} has {n} NaN values"


def test_hier_cols_finite(df114) -> None:  # type: ignore[return]
    for col in HIERARCHY_FEATURE_COLS:
        assert np.isfinite(df114[col].values).all(), f"{col} has non-finite values"


# ---------------------------------------------------------------------------
# LOO correctness: L1 (product_type) aggregates
# ---------------------------------------------------------------------------


def test_pt_loo_lag1_mean_nonneg(df114) -> None:  # type: ignore[return]
    assert (df114["hier_pt_loo_lag1_mean"] >= 0).all()


def test_pt_loo_roll4_mean_nonneg(df114) -> None:  # type: ignore[return]
    assert (df114["hier_pt_loo_roll4_mean"] >= 0).all()


def test_pt_loo_roll13_mean_nonneg(df114) -> None:  # type: ignore[return]
    assert (df114["hier_pt_loo_roll13_mean"] >= 0).all()


def test_pt_loo_nonzero_rate_unit_interval(df114) -> None:  # type: ignore[return]
    assert df114["hier_pt_loo_nonzero_rate"].between(0.0, 1.0).all()


def test_pt_loo_single_variant_gets_zero_mean(df114) -> None:
    """A product_type with only 1 variant at a timestamp has no others to average — should be 0."""
    # Find timestamps where a product_type has count=1
    pt_ts_counts = df114.groupby(["product_type", config.COL_TIMESTAMP]).size()
    solo = pt_ts_counts[pt_ts_counts == 1]
    if len(solo) == 0:
        pytest.skip("No solo product_type/timestamp combinations found")
    # Get one such (pt, ts) pair
    pt, ts = solo.index[0]
    row = df114[(df114["product_type"] == pt) & (df114[config.COL_TIMESTAMP] == ts)]
    # LOO mean must be 0 (no other members)
    assert row["hier_pt_loo_lag1_mean"].values[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Non-LOO total level: YoY and portfolio stats
# ---------------------------------------------------------------------------


def test_total_lag1_mean_nonneg(df114) -> None:  # type: ignore[return]
    assert (df114["hier_total_lag1_mean"] >= 0).all()


def test_total_roll52_mean_nonneg(df114) -> None:  # type: ignore[return]
    assert (df114["hier_total_roll52_mean"] >= 0).all()


def test_total_yoy_clipped(df114) -> None:  # type: ignore[return]
    """YoY ratio must be clipped to [0.1, 10.0]."""
    assert (df114["hier_total_yoy"] >= 0.1).all()
    assert (df114["hier_total_yoy"] <= 10.0).all()


def test_total_yoy_constant_per_timestamp(df114) -> None:
    """YoY is non-LOO — same value for ALL SKUs at a given timestamp."""
    nunique = df114.groupby(config.COL_TIMESTAMP)["hier_total_yoy"].nunique()
    bad = nunique[nunique > 1]
    assert len(bad) == 0, f"hier_total_yoy varies within timestamp: {bad.index[:3].tolist()}"


def test_total_lag1_mean_constant_per_timestamp(df114) -> None:
    """Portfolio mean is non-LOO — same for all SKUs at a timestamp."""
    nunique = df114.groupby(config.COL_TIMESTAMP)["hier_total_lag1_mean"].nunique()
    bad = nunique[nunique > 1]
    assert len(bad) == 0, f"hier_total_lag1_mean varies within timestamp: {bad}"


# ---------------------------------------------------------------------------
# Static hierarchy feature
# ---------------------------------------------------------------------------


def test_pt_n_variants_positive(df114) -> None:  # type: ignore[return]
    assert (df114["hier_pt_n_variants"] >= 1).all()


def test_pt_n_variants_constant_per_product_type(df114) -> None:
    """Number of variants in each product_type must be constant across all rows."""
    nunique = df114.groupby("product_type")["hier_pt_n_variants"].nunique()
    bad = nunique[nunique > 1]
    assert len(bad) == 0, f"hier_pt_n_variants varies within product_type: {bad}"


def test_pt_n_variants_max_is_candles(df114) -> None:
    """Candles is the largest product_type — should have the highest variant count."""
    candles_n = df114[df114["product_type"] == "Candles"]["hier_pt_n_variants"].iloc[0]
    assert candles_n == df114["hier_pt_n_variants"].max()


# ---------------------------------------------------------------------------
# Prior-phase columns preserved
# ---------------------------------------------------------------------------


def test_cluster_cols_preserved(df114) -> None:  # type: ignore[return]
    for col in ["cluster_id", "cluster_loo_lag1_mean", "cluster_loo_nonzero_rate"]:
        assert col in df114.columns, f"Cluster col {col} missing after hier join"


def test_phase4_cols_preserved(df114) -> None:  # type: ignore[return]
    for col in ["lag_1", "roll4_mean", "fourier_52w_sin_k1", "hol_christmas"]:
        assert col in df114.columns, f"Phase4 col {col} missing"


def test_base_cols_preserved(df114) -> None:  # type: ignore[return]
    for col in [config.COL_SKU_ID, config.COL_TIMESTAMP, config.COL_SALES]:
        assert col in df114.columns
