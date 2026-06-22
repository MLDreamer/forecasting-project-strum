"""Phase 2 gate: lifecycle inference counts and correctness."""

from __future__ import annotations

import pytest

from forecasting import config
from forecasting.io import load_all
from forecasting.lifecycle import LifecycleResult, infer_lifecycle


@pytest.fixture(scope="module")
def result() -> LifecycleResult:
    d = load_all()
    return infer_lifecycle(d.sales, d.master)


# ---------------------------------------------------------------------------
# Gate counts (locked by onboarding doc)
# ---------------------------------------------------------------------------
def test_lifecycle_row_count(result: LifecycleResult) -> None:
    """One row per SKU that has sales history."""
    assert len(result.lifecycle) == 220


def test_dormant_count(result: LifecycleResult) -> None:
    """81 SKUs trimmed as dormant (after keep-active overrides applied)."""
    assert result.lifecycle["is_dormant"].sum() == 81


def test_active_count(result: LifecycleResult) -> None:
    """139 SKUs in-scope to forecast."""
    assert result.lifecycle["is_active"].sum() == 139


def test_active_plus_dormant_equals_total(result: LifecycleResult) -> None:
    """Active + dormant must account for all SKUs (no archived-with-sales)."""
    total = len(result.lifecycle)
    assert result.lifecycle["is_active"].sum() + result.lifecycle["is_dormant"].sum() == total


# ---------------------------------------------------------------------------
# Override SKU behaviour
# ---------------------------------------------------------------------------
def test_override_sku_is_active(result: LifecycleResult) -> None:
    override_sku = 46606700773604
    row = result.lifecycle[result.lifecycle[config.COL_SKU_ID] == override_sku]
    assert len(row) == 1, "Override SKU not found in lifecycle output"
    assert row["is_active"].iloc[0], "Override SKU must be active"
    assert row["is_dormant"].iloc[0] is False or not row["is_dormant"].iloc[0]
    assert row["keep_active_override"].iloc[0], "Override flag must be set"


def test_override_sku_weeks_since(result: LifecycleResult) -> None:
    """Override SKU sits exactly at the dormancy boundary (26.0 weeks)."""
    override_sku = 46606700773604
    row = result.lifecycle[result.lifecycle[config.COL_SKU_ID] == override_sku]
    assert abs(row["weeks_since_last_sale"].iloc[0] - 26.0) < 0.1


# ---------------------------------------------------------------------------
# Scope set consistency
# ---------------------------------------------------------------------------
def test_sku_active_frozenset(result: LifecycleResult) -> None:
    assert len(result.sku_active) == 139


def test_sku_dormant_frozenset(result: LifecycleResult) -> None:
    assert len(result.sku_dormant) == 81


def test_active_dormant_disjoint(result: LifecycleResult) -> None:
    assert result.sku_active.isdisjoint(result.sku_dormant)


def test_union_equals_all_skus(result: LifecycleResult) -> None:
    all_skus = frozenset(result.lifecycle[config.COL_SKU_ID])
    assert result.sku_active | result.sku_dormant == all_skus


# ---------------------------------------------------------------------------
# Active window logic
# ---------------------------------------------------------------------------
def test_dormant_active_window_end_equals_last_sale(result: LifecycleResult) -> None:
    dormant = result.lifecycle[result.lifecycle["is_dormant"]]
    assert (dormant["active_window_end"] == dormant["last_sale"]).all()


def test_active_window_end_equals_cutoff_for_active(result: LifecycleResult) -> None:
    cutoff = result.lifecycle["active_window_end"].max()
    active = result.lifecycle[result.lifecycle["is_active"]]
    # All active SKUs (non-archived, non-dormant) should have window_end = cutoff
    assert (active["active_window_end"] == cutoff).all()


def test_first_sale_lte_last_sale(result: LifecycleResult) -> None:
    assert (result.lifecycle["first_sale"] <= result.lifecycle["last_sale"]).all()


# ---------------------------------------------------------------------------
# Dormancy boundary: literal >=26 weeks
# ---------------------------------------------------------------------------
def test_dormancy_is_literal_gte_26_weeks(result: LifecycleResult) -> None:
    lc = result.lifecycle
    # After override correction, dormant flag must match the rule for non-overrides
    non_override = lc[~lc["keep_active_override"]]
    assert (non_override["is_dormant"] == (non_override["weeks_since_last_sale"] >= 26)).all()


# ---------------------------------------------------------------------------
# Plausibility smoke check
# ---------------------------------------------------------------------------
def test_weeks_since_non_negative(result: LifecycleResult) -> None:
    assert (result.lifecycle["weeks_since_last_sale"] >= 0).all()


def test_active_skus_weeks_since_lt_26(result: LifecycleResult) -> None:
    """No non-override active SKU can have weeks_since >= 26."""
    active_no_override = result.lifecycle[
        result.lifecycle["is_active"] & ~result.lifecycle["keep_active_override"]
    ]
    assert (active_no_override["weeks_since_last_sale"] < 26).all()
