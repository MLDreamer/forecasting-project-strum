"""Per-SKU active-window inference and dormancy trimming.

Lifecycle rules (locked):
- Active window start  : first sale date per SKU.
- Active window end    : last sale date for archived SKUs OR dormant SKUs;
                         else the data cutoff (max timestamp in sales).
- Dormancy             : weeks_since_last_sale >= DORMANCY_THRESHOLD_WEEKS
                         (literal — a sale exactly 26 weeks before cutoff IS dormant).
- Keep-active overrides: config.LIFECYCLE_KEEP_ACTIVE_OVERRIDES forces specific
                         SKUs to stay active regardless of dormancy.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import pandas as pd

from forecasting import config

logger = logging.getLogger(__name__)

_OVERRIDE_SKUS: frozenset[int] = frozenset(sku for sku, _ in config.LIFECYCLE_KEEP_ACTIVE_OVERRIDES)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LifecycleResult:
    """Output of infer_lifecycle()."""

    lifecycle: pd.DataFrame
    """One row per SKU with columns:
       sku_id, first_sale, last_sale, active_window_end,
       weeks_since_last_sale, is_dormant, is_active, keep_active_override.
    """

    sku_active: frozenset[int]
    """SKUs flagged is_active=True (in-scope to forecast)."""

    sku_dormant: frozenset[int]
    """SKUs flagged is_dormant=True (trimmed from forecast scope)."""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def infer_lifecycle(
    sales: pd.DataFrame,
    master: pd.DataFrame,
    cutoff: pd.Timestamp | None = None,
) -> LifecycleResult:
    """Infer per-SKU lifecycle from sales history.

    Parameters
    ----------
    sales:
        Canonical sales DataFrame (output of io.load_sales).
    master:
        Canonical master DataFrame (output of io.load_master).
    cutoff:
        The reference date for dormancy calculation.  Defaults to the
        maximum timestamp in *sales* (i.e. the last observed week).

    Returns
    -------
    LifecycleResult
    """
    if cutoff is None:
        cutoff = sales[config.COL_TIMESTAMP].max()

    # --- Per-SKU first / last sale -------------------------------------------
    grp = sales.groupby(config.COL_SKU_ID)[config.COL_TIMESTAMP]
    first_sale: pd.Series = grp.min()
    last_sale: pd.Series = grp.max()

    lc = pd.DataFrame({"first_sale": first_sale, "last_sale": last_sale}).reset_index()
    lc = lc.rename(columns={config.COL_SKU_ID: config.COL_SKU_ID})

    # --- Weeks since last sale (relative to cutoff) --------------------------
    lc["weeks_since_last_sale"] = (cutoff - lc["last_sale"]).dt.days / 7.0

    # --- Dormancy flag -------------------------------------------------------
    lc["is_dormant"] = lc["weeks_since_last_sale"] >= config.DORMANCY_THRESHOLD_WEEKS

    # --- Keep-active overrides -----------------------------------------------
    lc["keep_active_override"] = lc[config.COL_SKU_ID].isin(_OVERRIDE_SKUS)

    overridden_dormant = lc["is_dormant"] & lc["keep_active_override"]
    if overridden_dormant.any():
        n = overridden_dormant.sum()
        skus = lc.loc[overridden_dormant, config.COL_SKU_ID].tolist()
        warnings.warn(
            f"{n} SKU(s) were dormant but forced active by override: {skus}",
            stacklevel=2,
        )
        lc.loc[overridden_dormant, "is_dormant"] = False

    # --- Active flag ---------------------------------------------------------
    # Also mark archived SKUs as not active (they are discontinued)
    archived_skus: set[int] = set(
        master.loc[master[config.COL_STATUS] == "archived", config.COL_SKU_ID]
    )
    lc["is_active"] = ~lc["is_dormant"] & ~lc[config.COL_SKU_ID].isin(archived_skus)

    # --- Active window end ---------------------------------------------------
    # Dormant or archived → last_sale; otherwise → cutoff
    lc["active_window_end"] = lc["last_sale"].where(
        lc["is_dormant"] | lc[config.COL_SKU_ID].isin(archived_skus),
        other=cutoff,
    )

    # --- Logging summary -----------------------------------------------------
    n_dormant = lc["is_dormant"].sum()
    n_active = lc["is_active"].sum()
    n_override = lc["keep_active_override"].sum()
    logger.info(
        "Lifecycle: %d SKUs total | %d active | %d dormant (trimmed) | %d override(s)",
        len(lc),
        n_active,
        n_dormant,
        n_override,
    )

    sku_active: frozenset[int] = frozenset(lc.loc[lc["is_active"], config.COL_SKU_ID])
    sku_dormant: frozenset[int] = frozenset(lc.loc[lc["is_dormant"], config.COL_SKU_ID])

    return LifecycleResult(lifecycle=lc, sku_active=sku_active, sku_dormant=sku_dormant)


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------
def save_lifecycle(result: LifecycleResult, path: None = None) -> None:
    """Write lifecycle parquet to data/interim/lifecycle.parquet."""
    import pathlib

    out = pathlib.Path(path) if path else config.DATA_INTERIM / "lifecycle.parquet"
    result.lifecycle.to_parquet(out, index=False)
    logger.info("Wrote lifecycle → %s", out)
