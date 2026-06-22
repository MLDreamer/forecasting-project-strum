"""Load and validate raw input files; join sales ↔ master.

This is the ONLY module that knows client-specific column names.
Everything downstream uses canonical names defined in config.py.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from forecasting import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client-specific column names (Fontana Candle / Shopify export)
# ---------------------------------------------------------------------------
_SALES_RENAME: dict[str, str] = {
    "item_id": config.COL_SKU_ID,
    "timestamp": config.COL_TIMESTAMP,
    "sales": config.COL_SALES,
    "avg_unit_price": config.COL_LIST_PRICE,
    "discount_pct": config.COL_DISCOUNT_PCT,
}

_MASTER_KEEP: list[str] = [
    "source_variant_id",
    "product_type",
    "status",
    "price",
]

# product_types that are out-of-scope for demand forecasting.
# Gift Cards are financial instruments (not physical goods); return rows are
# credit transactions.  Including them contaminates product_type rollups,
# the grand total, and every hier_total_* feature.
# Excluding them aligns the in-scope count with the doc's 220.
_OUT_OF_SCOPE_PRODUCT_TYPES: frozenset[str] = frozenset({"Gift Card", "return"})


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedData:
    """Output of load_all() — canonical DataFrames + scope sets."""

    sales: pd.DataFrame
    """Raw sales rows with canonical columns; 11,291 rows."""

    master: pd.DataFrame
    """SKU master with canonical columns; 441 rows."""

    joined: pd.DataFrame
    """Sales left-joined with master; 11,291 rows, canonical columns."""

    sku_has_sales: frozenset[int]
    """SKUs that have at least one sales row (229)."""

    sku_cold_start: frozenset[int]
    """Active/draft SKUs in master with zero sales history."""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _find_file(directory: Path, stem: str) -> Path:
    """Return first match for ``stem`` with any of .csv/.xlsx/.xls."""
    for ext in (".csv", ".xlsx", ".xls"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {stem}{{.csv,.xlsx,.xls}} in {directory}")


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_excel(path)


def load_sales(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load weekly sales, rename to canonical columns, parse timestamps."""
    raw_dir = raw_dir or config.DATA_RAW
    stem = config.RAW_SALES_FILE.removesuffix(".csv")
    path = _find_file(raw_dir, stem)
    df = _read(path)

    df = df.rename(columns=_SALES_RENAME)
    df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP])

    # Verify week boundary is Sunday
    dow = df[config.COL_TIMESTAMP].dt.dayofweek  # 0=Mon, 6=Sun
    non_sunday = (dow != 6).sum()
    if non_sunday:
        warnings.warn(
            f"{non_sunday} rows have non-Sunday timestamps — expected Sunday week-start.",
            stacklevel=2,
        )

    logger.info("Loaded sales: %d rows, %d unique SKUs", len(df), df[config.COL_SKU_ID].nunique())
    return df


def load_master(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load product/item master, keep relevant columns."""
    raw_dir = raw_dir or config.DATA_RAW
    stem = config.RAW_MASTER_FILE.removesuffix(".csv")
    path = _find_file(raw_dir, stem)
    df = _read(path)

    # Deduplicate on source_variant_id (keep first, warn)
    dupes = df["source_variant_id"].duplicated()
    if dupes.any():
        warnings.warn(
            f"Dropping {dupes.sum()} duplicate source_variant_id rows from master.",
            stacklevel=2,
        )
        df = df[~dupes].copy()

    # Drop rows with null source_variant_id
    null_ids = df["source_variant_id"].isna()
    if null_ids.any():
        warnings.warn(
            f"Dropping {null_ids.sum()} rows with NaN source_variant_id from master.",
            stacklevel=2,
        )
        df = df[~null_ids].copy()

    df = df[_MASTER_KEEP].rename(
        columns={
            "source_variant_id": config.COL_SKU_ID,
            "price": config.COL_LIST_PRICE,
        }
    )
    logger.info("Loaded master: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Join + scope sets
# ---------------------------------------------------------------------------
def join_and_scope(
    sales: pd.DataFrame,
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, frozenset[int], frozenset[int]]:
    """Left-join sales onto master; return joined df + scope frozensets.

    Returns
    -------
    joined
        Sales rows enriched with product_type, status, list_price from master.
        Rows unmatched in master get status='unknown'.
    sku_has_sales
        SKU IDs present in the sales file.
    sku_cold_start
        SKUs in master that are active/draft but have no sales history.
    """
    joined = sales.merge(
        master,
        on=config.COL_SKU_ID,
        how="left",
        suffixes=("", "_master"),
    )

    # Unmatched rows → status = 'unknown'
    unmatched = joined[config.COL_STATUS].isna().sum()
    if unmatched:
        warnings.warn(
            f"{unmatched} sales rows unmatched in master — setting status='unknown'.",
            stacklevel=2,
        )
        joined[config.COL_STATUS] = joined[config.COL_STATUS].fillna("unknown")

    # Resolve list_price: use master price; fall back to avg_unit_price from sales
    # (master column already named list_price after load_master rename;
    #  sales has its own list_price from _SALES_RENAME)
    if "list_price_master" in joined.columns:
        joined[config.COL_LIST_PRICE] = joined["list_price_master"].combine_first(
            joined[config.COL_LIST_PRICE]
        )
        joined = joined.drop(columns=["list_price_master"])

    # Apply scope filter: exclude out-of-scope product types (Gift Card, return).
    # These are financial/credit transactions, not demand.  Filtering here keeps
    # all downstream modules (features, segment, hierarchy) free of contamination.
    out_of_scope_skus: frozenset[int] = frozenset(
        master.loc[
            master[config.COL_PRODUCT_TYPE].isin(_OUT_OF_SCOPE_PRODUCT_TYPES),
            config.COL_SKU_ID,
        ]
    )
    if out_of_scope_skus:
        n_before = len(joined)
        joined = joined[~joined[config.COL_SKU_ID].isin(out_of_scope_skus)].copy()
        logger.info(
            "Scope filter: removed %d rows (%d SKUs) with out-of-scope product_type",
            n_before - len(joined),
            len(out_of_scope_skus),
        )

    sku_has_sales: frozenset[int] = frozenset(joined[config.COL_SKU_ID].unique())

    not_archived = master[
        ~master[config.COL_STATUS].isin(["archived"])
        & ~master[config.COL_SKU_ID].isin(out_of_scope_skus)
    ]
    sku_cold_start: frozenset[int] = frozenset(
        not_archived.loc[
            ~not_archived[config.COL_SKU_ID].isin(sku_has_sales),
            config.COL_SKU_ID,
        ]
    )

    logger.info(
        "Joined: %d rows | has_sales=%d | cold_start=%d | out_of_scope=%d",
        len(joined),
        len(sku_has_sales),
        len(sku_cold_start),
        len(out_of_scope_skus),
    )
    return joined, sku_has_sales, sku_cold_start


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def load_all(raw_dir: Path | None = None) -> LoadedData:
    """Load all raw files, join, and return a LoadedData bundle.

    The returned `sales` DataFrame is filtered to in-scope SKUs only
    (Gift Card and return SKUs removed).  All downstream modules —
    lifecycle, densify, features — see only the cleaned 220-SKU set.
    """
    raw_dir = raw_dir or config.DATA_RAW
    sales = load_sales(raw_dir)
    master = load_master(raw_dir)
    joined, sku_has_sales, sku_cold_start = join_and_scope(sales, master)

    # Filter the raw sales DataFrame to match the scoped sku_has_sales set.
    # This ensures lifecycle / densify / features see no Gift Card or return rows.
    sales_filtered = sales[sales[config.COL_SKU_ID].isin(sku_has_sales)].copy()

    return LoadedData(
        sales=sales_filtered,
        master=master,
        joined=joined,
        sku_has_sales=sku_has_sales,
        sku_cold_start=sku_cold_start,
    )
