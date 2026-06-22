"""Weekly grid densification, zero-fill, price forward/back-fill, stockout flag.

Rules (locked):
- Grid spans first_sale → active_window_end per SKU (Sunday-dated, matching raw data).
- Missing weeks within the active window → sales = 0.
- list_price / discount_pct: forward-fill then backward-fill per SKU
  (so every week has a price; new SKUs with no preceding price get the first known price).
- is_potential_stockout: True for SKUs whose MID-SERIES zero run is >= 8 consecutive weeks.
  Mid-series = after the first non-zero week and before (or at) the last non-zero week.
  Leading / trailing zero tails are excluded from the stockout detection window.
- Phase 3.5 will relabel timestamps Sunday → Saturday. This module stays Sunday-dated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting import config
from forecasting.lifecycle import LifecycleResult

logger = logging.getLogger(__name__)

_STOCKOUT_MIN_RUN: int = 8  # consecutive zero weeks that flag a potential stockout


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DenseResult:
    """Output of densify()."""

    dense: pd.DataFrame
    """Dense weekly grid with columns:
       sku_id, timestamp, sales, list_price, discount_pct,
       product_type, status, is_potential_stockout.
    """

    zero_fraction: float
    """Fraction of rows where sales == 0."""

    stockout_skus: frozenset[int]
    """SKUs flagged with a mid-series stockout-like gap."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _max_mid_series_zero_run(sales: np.ndarray) -> int:
    """Return the longest zero run that occurs WITHIN the active-sale window.

    Leading and trailing zeros (before the first sale / after the last sale)
    are excluded so we only catch genuine mid-series gaps.
    """
    n = len(sales)
    # Find first and last non-zero index
    first_nz = next((i for i in range(n) if sales[i] > 0), n)
    last_nz = next((i for i in range(n - 1, -1, -1) if sales[i] > 0), -1)

    if first_nz >= last_nz:
        # All zeros or only one non-zero point — no mid-series window
        return 0

    mid = sales[first_nz : last_nz + 1]  # inclusive of both endpoints
    max_run = run = 0
    for v in mid:
        if v == 0:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def densify(
    sales: pd.DataFrame,
    lifecycle: LifecycleResult,
    joined: pd.DataFrame,
    week_relabel_shift_days: int = 0,
) -> DenseResult:
    """Build a zero-filled weekly grid for all 229 SKUs.

    Parameters
    ----------
    sales:
        Canonical sales DataFrame from io.load_sales (11,291 rows).
    lifecycle:
        Output of lifecycle.infer_lifecycle.
    joined:
        Output of io.join_and_scope — carries product_type, status per row.
    week_relabel_shift_days:
        Number of days to shift timestamps after grid construction.
        Default 0 (no relabelling, Sunday-dated as in raw data).
        Set to 6 (config.WEEK_RELABEL_SHIFT_DAYS) to relabel Sunday week-start
        to Saturday week-end.  Applied as the LAST step so join logic stays
        on the original Sunday dates.

    Returns
    -------
    DenseResult
    """
    lc_df = lifecycle.lifecycle.set_index(config.COL_SKU_ID)

    # Static per-SKU metadata (product_type, status) — take first occurrence
    meta = (
        joined.groupby(config.COL_SKU_ID)[[config.COL_PRODUCT_TYPE, config.COL_STATUS]]
        .first()
        .reset_index()
    )

    # Sales lookup: (sku_id, timestamp) → (sales, list_price, discount_pct)
    sales_cols = [
        config.COL_SKU_ID,
        config.COL_TIMESTAMP,
        config.COL_SALES,
        config.COL_LIST_PRICE,
        config.COL_DISCOUNT_PCT,
    ]
    # Build grid rows
    grid_rows: list[tuple[int, pd.Timestamp]] = []
    for sku in lc_df.index:
        first: pd.Timestamp = lc_df.loc[sku, "first_sale"]
        end: pd.Timestamp = lc_df.loc[sku, "active_window_end"]
        for week in pd.date_range(start=first, end=end, freq="W-SUN"):
            grid_rows.append((sku, week))

    dense = pd.DataFrame(grid_rows, columns=[config.COL_SKU_ID, config.COL_TIMESTAMP])

    # Merge in sales (left join — missing weeks stay NaN then get zero-filled)
    dense = dense.merge(
        sales[sales_cols],
        on=[config.COL_SKU_ID, config.COL_TIMESTAMP],
        how="left",
    )
    dense[config.COL_SALES] = dense[config.COL_SALES].fillna(0.0)

    # Forward-fill then backward-fill price/discount per SKU.
    # SKUs with no observed discount default to 0.0 (no discount).
    dense = dense.sort_values([config.COL_SKU_ID, config.COL_TIMESTAMP])
    for col in (config.COL_LIST_PRICE, config.COL_DISCOUNT_PCT):
        filled = dense.groupby(config.COL_SKU_ID)[col].transform(lambda s: s.ffill().bfill())
        dense[col] = filled.fillna(0.0)

    # Merge static metadata
    dense = dense.merge(meta, on=config.COL_SKU_ID, how="left")

    # Stockout flag (mid-series zero run >= 8 weeks)
    stockout_map: dict[int, bool] = {}
    for sku, grp in dense.groupby(config.COL_SKU_ID):
        arr = grp.sort_values(config.COL_TIMESTAMP)[config.COL_SALES].to_numpy()
        stockout_map[int(sku)] = _max_mid_series_zero_run(arr) >= _STOCKOUT_MIN_RUN

    dense["is_potential_stockout"] = dense[config.COL_SKU_ID].map(stockout_map)

    # Sunday → Saturday relabelling (Phase 3.5).
    # Applied after all join/fill logic so internal indices stay consistent.
    if week_relabel_shift_days:
        dense[config.COL_TIMESTAMP] = dense[config.COL_TIMESTAMP] + pd.Timedelta(
            days=week_relabel_shift_days
        )

    zero_fraction = float((dense[config.COL_SALES] == 0).mean())
    stockout_skus: frozenset[int] = frozenset(k for k, v in stockout_map.items() if v)

    logger.info(
        "Densify: %d rows | %.1f%% zeros | %d stockout SKUs",
        len(dense),
        zero_fraction * 100,
        len(stockout_skus),
    )

    return DenseResult(
        dense=dense.reset_index(drop=True),
        zero_fraction=zero_fraction,
        stockout_skus=stockout_skus,
    )


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------
def save_dense(result: DenseResult, path: None = None) -> None:
    """Write dense grid to data/interim/dense_weekly.parquet."""
    import pathlib

    out = pathlib.Path(path) if path else config.DATA_INTERIM / "dense_weekly.parquet"
    result.dense.to_parquet(out, index=False)
    logger.info("Wrote dense grid → %s", out)
