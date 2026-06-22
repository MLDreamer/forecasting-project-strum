"""Phase 1 gate: load, join, and scope-set counts."""

import pandas as pd
import pytest

from forecasting import config
from forecasting.io import LoadedData, load_all


@pytest.fixture(scope="module")
def data() -> LoadedData:
    return load_all()


# ---------------------------------------------------------------------------
# Raw counts
# ---------------------------------------------------------------------------
def test_sales_row_count(data: LoadedData) -> None:
    assert len(data.sales) == 10_860


def test_sales_unique_skus(data: LoadedData) -> None:
    # 220 in-scope after Gift Card + return filter
    assert data.sales[config.COL_SKU_ID].nunique() == 220


def test_master_row_count(data: LoadedData) -> None:
    assert len(data.master) == 441


# ---------------------------------------------------------------------------
# Canonical columns present
# ---------------------------------------------------------------------------
def test_sales_canonical_columns(data: LoadedData) -> None:
    required = {
        config.COL_SKU_ID,
        config.COL_TIMESTAMP,
        config.COL_SALES,
        config.COL_LIST_PRICE,
        config.COL_DISCOUNT_PCT,
    }
    assert required.issubset(data.sales.columns)


def test_master_canonical_columns(data: LoadedData) -> None:
    required = {config.COL_SKU_ID, config.COL_STATUS, config.COL_LIST_PRICE}
    assert required.issubset(data.master.columns)


# ---------------------------------------------------------------------------
# Join integrity
# ---------------------------------------------------------------------------
def test_joined_row_count(data: LoadedData) -> None:
    """Left-join must not inflate or shrink the sales row count."""
    assert len(data.joined) == 10_860


def test_no_null_status_after_join(data: LoadedData) -> None:
    assert data.joined[config.COL_STATUS].notna().all()


def test_full_join_coverage(data: LoadedData) -> None:
    """All in-scope sales SKUs must resolve to a known status (no 'unknown')."""
    unknown = (data.joined[config.COL_STATUS] == "unknown").sum()
    assert unknown == 0, f"{unknown} rows have status='unknown'"


# ---------------------------------------------------------------------------
# Scope sets
# ---------------------------------------------------------------------------
def test_has_sales_count(data: LoadedData) -> None:
    # 229 raw - 5 Gift Card - 4 return = 220 in-scope SKUs
    assert len(data.sku_has_sales) == 220


def test_cold_start_non_empty(data: LoadedData) -> None:
    # Phase 2 will pin this exactly; for Phase 1 just verify it's plausible
    assert 20 <= len(data.sku_cold_start) <= 50


def test_cold_start_not_in_has_sales(data: LoadedData) -> None:
    assert data.sku_cold_start.isdisjoint(data.sku_has_sales)


def test_cold_start_not_archived(data: LoadedData) -> None:
    """Cold-start SKUs must come from non-archived master rows."""
    cold_status = data.master.loc[
        data.master[config.COL_SKU_ID].isin(data.sku_cold_start),
        config.COL_STATUS,
    ]
    assert (cold_status != "archived").all()


def test_out_of_scope_skus_excluded(data: LoadedData) -> None:
    out_of_scope = {"Gift Card", "return"}
    if config.COL_PRODUCT_TYPE in data.joined.columns:
        assert not data.joined[config.COL_PRODUCT_TYPE].isin(out_of_scope).any()


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
def test_timestamps_are_sunday(data: LoadedData) -> None:
    dow = data.sales[config.COL_TIMESTAMP].dt.dayofweek
    assert (dow == 6).all(), "Expected all timestamps to be Sunday (day 6)"


def test_timestamp_range(data: LoadedData) -> None:
    assert data.sales[config.COL_TIMESTAMP].min() == pd.Timestamp("2020-12-27")
    assert data.sales[config.COL_TIMESTAMP].max() == pd.Timestamp("2026-05-17")


# ---------------------------------------------------------------------------
# Smoke: LoadedData is immutable (frozen dataclass)
# ---------------------------------------------------------------------------
def test_loaded_data_frozen(data: LoadedData) -> None:
    with pytest.raises((AttributeError, TypeError)):
        data.sales = pd.DataFrame()  # type: ignore[misc]
