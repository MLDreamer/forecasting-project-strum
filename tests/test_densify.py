"""Phase 3 gate: dense grid counts, zero fraction, stockout flag."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting import config
from forecasting.densify import DenseResult, _max_mid_series_zero_run, densify
from forecasting.io import load_all
from forecasting.lifecycle import infer_lifecycle


@pytest.fixture(scope="module")
def result() -> DenseResult:
    d = load_all()
    lc = infer_lifecycle(d.sales, d.master)
    return densify(d.sales, lc, d.joined)


# ---------------------------------------------------------------------------
# Gate counts
# ---------------------------------------------------------------------------
def test_dense_row_count(result: DenseResult) -> None:
    assert len(result.dense) == 16_068


def test_zero_fraction(result: DenseResult) -> None:
    """32.4% of rows must be zero-sales (±0.2pp tolerance)."""
    assert abs(result.zero_fraction - 0.324) < 0.002


def test_stockout_sku_count(result: DenseResult) -> None:
    """82 SKUs have a mid-series zero run >= 8 weeks."""
    assert len(result.stockout_skus) == 82


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_required_columns(result: DenseResult) -> None:
    required = {
        config.COL_SKU_ID,
        config.COL_TIMESTAMP,
        config.COL_SALES,
        config.COL_LIST_PRICE,
        config.COL_DISCOUNT_PCT,
        config.COL_PRODUCT_TYPE,
        config.COL_STATUS,
        "is_potential_stockout",
    }
    assert required.issubset(result.dense.columns)


def test_no_null_sales(result: DenseResult) -> None:
    assert result.dense[config.COL_SALES].notna().all()


def test_sales_non_negative(result: DenseResult) -> None:
    assert (result.dense[config.COL_SALES] >= 0).all()


def test_no_null_price_after_fill(result: DenseResult) -> None:
    assert result.dense[config.COL_LIST_PRICE].notna().all()


def test_no_null_discount_after_fill(result: DenseResult) -> None:
    assert result.dense[config.COL_DISCOUNT_PCT].notna().all()


# ---------------------------------------------------------------------------
# Grid integrity
# ---------------------------------------------------------------------------
def test_timestamps_are_sunday(result: DenseResult) -> None:
    dow = result.dense[config.COL_TIMESTAMP].dt.dayofweek
    assert (dow == 6).all(), "All dense timestamps must be Sunday"


def test_unique_sku_timestamp_pairs(result: DenseResult) -> None:
    dupes = result.dense.duplicated(subset=[config.COL_SKU_ID, config.COL_TIMESTAMP]).sum()
    assert dupes == 0


def test_sku_count(result: DenseResult) -> None:
    """Dense grid covers all 220 SKUs (active + dormant)."""
    assert result.dense[config.COL_SKU_ID].nunique() == 220


def test_consecutive_weekly_grid_per_sku(result: DenseResult) -> None:
    """Every SKU's timestamps must form a gapless weekly sequence."""
    for sku, grp in result.dense.groupby(config.COL_SKU_ID):
        ts = grp[config.COL_TIMESTAMP].sort_values().reset_index(drop=True)
        if len(ts) < 2:
            continue
        diffs = ts.diff().dropna().dt.days
        assert (diffs == 7).all(), f"SKU {sku} has non-weekly gap in dense grid"


def test_stockout_flag_consistent(result: DenseResult) -> None:
    """is_potential_stockout must be constant per SKU (it's a SKU-level flag)."""
    for sku, grp in result.dense.groupby(config.COL_SKU_ID):
        vals = grp["is_potential_stockout"].unique()
        assert len(vals) == 1, f"SKU {sku} has mixed stockout flag"


def test_stockout_skus_match_flag(result: DenseResult) -> None:
    """stockout_skus frozenset must equal the set of SKUs where flag is True."""
    flagged = frozenset(
        result.dense.loc[result.dense["is_potential_stockout"], config.COL_SKU_ID].unique()
    )
    assert flagged == result.stockout_skus


# ---------------------------------------------------------------------------
# Zero-fill correctness: no non-zero sales row should be zeroed out
# ---------------------------------------------------------------------------
def test_original_sales_preserved(result: DenseResult) -> None:
    """Every originally observed non-zero sale must appear unchanged in dense grid."""
    from forecasting.io import load_all as _load

    d = _load()
    nonzero = d.sales[d.sales[config.COL_SALES] > 0][
        [config.COL_SKU_ID, config.COL_TIMESTAMP, config.COL_SALES]
    ]
    merged = nonzero.merge(
        result.dense[[config.COL_SKU_ID, config.COL_TIMESTAMP, config.COL_SALES]],
        on=[config.COL_SKU_ID, config.COL_TIMESTAMP],
        suffixes=("_orig", "_dense"),
    )
    assert len(merged) == len(nonzero), "Some original sales rows lost in dense grid"
    assert (merged["sales_orig"] == merged["sales_dense"]).all()


# ---------------------------------------------------------------------------
# Unit tests for _max_mid_series_zero_run helper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        ([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 8),  # exactly 8
        ([1, 0, 0, 0, 0, 0, 0, 0, 1], 7),  # 7 — below threshold
        ([0, 0, 0, 1, 0, 0, 1], 2),  # leading zeros excluded; mid run = 2
        ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # trailing zeros not counted
        ([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 9),  # two runs, pick longest
        ([1], 0),
        ([0, 0, 0], 0),
    ],
)
def test_max_mid_series_zero_run(arr: list[int], expected: int) -> None:
    assert _max_mid_series_zero_run(np.array(arr, dtype=float)) == expected
