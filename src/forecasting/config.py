"""Paths, constants, and locked decisions for the forecasting pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
OUTPUTS = REPO_ROOT / "outputs"
CONFIGS_DIR = REPO_ROOT / "configs"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Raw file names (detected by io.py; client-specific names stay here)
# ---------------------------------------------------------------------------
RAW_SALES_FILE = "processed_data_filtered.csv"
RAW_MASTER_FILE = "product_item_master.csv"
RAW_VARIANTS_FILE = "variants_export.csv"

# ---------------------------------------------------------------------------
# Canonical column names (io.py maps client names → these)
# ---------------------------------------------------------------------------
COL_SKU_ID = "sku_id"
COL_TIMESTAMP = "timestamp"
COL_SALES = "sales"
COL_LIST_PRICE = "list_price"
COL_DISCOUNT_PCT = "discount_pct"
COL_PRODUCT_ID = "product_id"
COL_PRODUCT_TYPE = "product_type"
COL_STATUS = "status"

# ---------------------------------------------------------------------------
# Scope / lifecycle (Phase 2 uses these)
# ---------------------------------------------------------------------------
DORMANCY_THRESHOLD_WEEKS: int = 26
FORECAST_HORIZON_WEEKS: int = 26

# SKUs forced to stay active despite dormancy criterion.
# Each entry: (sku_id, reason_string)
LIFECYCLE_KEEP_ACTIVE_OVERRIDES: list[tuple[int, str]] = [
    (46606700773604, "High recent velocity; zero-forecasting riskier than model attempt"),
]

# ---------------------------------------------------------------------------
# Modelling constants (Phase 11+)
# ---------------------------------------------------------------------------
QUANTILES: list[float] = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]
N_CV_FOLDS: int = 4
SELECTION_FOLDS: list[int] = [2, 3, 4]  # fold 1 skipped (cold-start too thin)
CALIBRATION_GUARDRAIL: tuple[float, float] = (0.75, 0.85)
FALLBACK_K: int = 8
STABILITY_ARI_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Week relabelling (Phase 3.5)
# Sunday-labelled (week-start) → Saturday-labelled (week-end)
# ---------------------------------------------------------------------------
WEEK_RELABEL_SHIFT_DAYS: int = 6
