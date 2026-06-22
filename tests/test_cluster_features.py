"""Phase 5b gate: cluster_id + 7 LOO lag-1 aggregates -> 16,068 x 106."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting import config
from forecasting.densify import densify
from forecasting.features import CLUSTER_FEATURE_COLS, add_cluster_features, build_features
from forecasting.io import load_all
from forecasting.lifecycle import infer_lifecycle
from forecasting.segment import segment_and_cluster


@pytest.fixture(scope="module")
def cluster_df():  # type: ignore[return]
    d = load_all()
    lc = infer_lifecycle(d.sales, d.master)
    dense = densify(d.sales, lc, d.joined, week_relabel_shift_days=6)
    feat = build_features(dense.dense, lc)
    seg = segment_and_cluster(feat.features, lc)
    return add_cluster_features(feat.features, seg.segments), seg


# ---------------------------------------------------------------------------
# Gate: shape
# ---------------------------------------------------------------------------


def test_row_count(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    assert len(df) == 16_068


def test_column_count(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    assert len(df.columns) == 106, f"Expected 106 cols, got {len(df.columns)}"


def test_cluster_feature_col_names(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    for col in CLUSTER_FEATURE_COLS:
        assert col in df.columns, f"Missing cluster feature: {col}"


def test_cluster_id_present(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    assert "cluster_id" in df.columns


# ---------------------------------------------------------------------------
# cluster_id integrity
# ---------------------------------------------------------------------------


def test_no_null_cluster_id(cluster_df) -> None:  # type: ignore[return]
    df, seg = cluster_df
    assert df["cluster_id"].notna().all()


def test_cluster_id_range(cluster_df) -> None:  # type: ignore[return]
    df, seg = cluster_df
    assert set(df["cluster_id"].unique()) == set(range(seg.selected_k))


def test_cluster_id_consistent_per_sku(cluster_df) -> None:  # type: ignore[return]
    """Every row for a given SKU must have the same cluster_id."""
    df, _ = cluster_df
    nunique = df.groupby(config.COL_SKU_ID)["cluster_id"].nunique()
    assert (nunique == 1).all(), (
        f"cluster_id varies within SKU: {nunique[nunique > 1].index.tolist()}"
    )


# ---------------------------------------------------------------------------
# LOO correctness
# ---------------------------------------------------------------------------


def test_loo_exact_two_member_cluster(cluster_df) -> None:  # type: ignore[return]
    """For a 2-member cluster, loo_mean of A must equal lag_1 of B at shared timestamps."""
    df, seg = cluster_df
    # Find the 2-member cluster (cluster 2 in our data)
    cluster_sizes = seg.segments.groupby("cluster_id").size()
    two_member = cluster_sizes[cluster_sizes == 2]
    if len(two_member) == 0:
        pytest.skip("No 2-member cluster found")

    c = two_member.index[0]
    skus = seg.segments[seg.segments["cluster_id"] == c]["sku_id"].tolist()
    assert len(skus) == 2

    skuA, skuB = skus
    ts_A = set(df[df[config.COL_SKU_ID] == skuA]["timestamp"])
    ts_B = set(df[df[config.COL_SKU_ID] == skuB]["timestamp"])
    shared = sorted(ts_A & ts_B)
    assert len(shared) > 0, "No shared timestamps for 2-member cluster"

    for t in shared[:5]:
        rowA = df[(df[config.COL_SKU_ID] == skuA) & (df["timestamp"] == t)]
        rowB = df[(df[config.COL_SKU_ID] == skuB) & (df["timestamp"] == t)]
        # A's LOO mean = B's lag_1, and B's LOO mean = A's lag_1
        assert abs(rowA["cluster_loo_lag1_mean"].values[0] - rowB["lag_1"].values[0]) < 1e-6, (
            f"LOO mean mismatch at t={t}: A.loo={rowA['cluster_loo_lag1_mean'].values[0]}, B.lag1={rowB['lag_1'].values[0]}"
        )
        assert abs(rowB["cluster_loo_lag1_mean"].values[0] - rowA["lag_1"].values[0]) < 1e-6


def test_loo_sum_nonneg(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    assert (df["cluster_loo_lag1_sum"] >= 0).all()


def test_loo_std_nonneg(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    assert (df["cluster_loo_lag1_std"] >= 0).all()


def test_loo_nonzero_rate_in_unit_interval(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    assert df["cluster_loo_nonzero_rate"].between(0.0, 1.0).all()


def test_loo_mean_leq_loo_sum(cluster_df) -> None:  # type: ignore[return]
    """LOO mean must be <= LOO sum (mean is average, sum is total)."""
    df, _ = cluster_df
    # loo_mean = loo_sum / (count-1) so loo_mean <= loo_sum when count >= 2
    mask = df["cluster_loo_lag1_sum"] > 0
    assert (
        df.loc[mask, "cluster_loo_lag1_mean"] <= df.loc[mask, "cluster_loo_lag1_sum"] + 1e-6
    ).all()


def test_no_nulls_in_cluster_cols(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    for col in CLUSTER_FEATURE_COLS + ["cluster_id"]:
        null_count = df[col].isna().sum()
        assert null_count == 0, f"{col} has {null_count} NaN values"


def test_loo_finite(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    for col in CLUSTER_FEATURE_COLS:
        assert np.isfinite(df[col].values).all(), f"{col} has non-finite values"


# ---------------------------------------------------------------------------
# Base columns from Phase 4 still present
# ---------------------------------------------------------------------------


def test_phase4_feature_cols_preserved(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    for col in ["lag_1", "roll4_mean", "fourier_52w_sin_k1", "hol_christmas", "idi"]:
        assert col in df.columns, f"Base feature col {col} missing after cluster join"


# ---------------------------------------------------------------------------
# Roll LOO plausibility: cluster_loo_roll4 ~ cluster_loo_lag1_mean order
# ---------------------------------------------------------------------------


def test_roll4_loo_nonneg(cluster_df) -> None:  # type: ignore[return]
    df, _ = cluster_df
    # Roll4 LOO can be negative if cluster has only 1 other member with negative roll? No —
    # roll4_mean is always >= 0 (sales >= 0). So loo_roll4 >= 0.
    assert (df["cluster_loo_roll4_mean"] >= -1e-9).all()
