"""configure_and_run.py — The notebook-style entry point for the pipeline.

Run this as a script or open as a Jupyter notebook (rename to .ipynb if needed).
This is the "code feel" described in the architecture docs: set business-term
knobs via the ConfigBuilder, then trigger run.run() — identical to the CLI.

Usage:
    python notebooks/configure_and_run.py
    or: python -m forecasting.run --config configs/fontana_candle.yaml

Both paths produce the same outputs/latest/ tables and manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the notebooks/ dir
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# CELL 1 — Set knobs (business terms)
# ---------------------------------------------------------------------------
from configs.config_builder import ALIASES, ConfigBuilder

# Print all available aliases so the client knows what they can set
print("Available business-term aliases:")
for alias, canonical in sorted(ALIASES.items()):
    print(f"  '{alias}'  →  {canonical}")
print()

cfg = (
    ConfigBuilder.from_yaml("configs/fontana_candle.yaml")
    # Uncomment and change any of these to override defaults:
    # .set("weeks of history required", 26)
    # .set("weeks to avoid forecasting", 26)
    # .set("scope statuses", ["active", "draft"])
    # .set("forecast weeks", 26)
    # .set("use heavy models", False)
    .build()  # ← pydantic validates here; raises on bad values
)

print(f"Config loaded: client={cfg.client}")
print(f"  horizon_weeks:              {cfg.model.horizon_weeks}")
print(f"  dormancy_threshold_weeks:   {cfg.lifecycle.dormancy_threshold_weeks}")
print(f"  scope_statuses:             active + draft (hardcoded in io.py)")
print(f"  n_quantiles:                {len(cfg.model.quantiles)}")
print(f"  calibration_guardrail:      {cfg.model.calibration_guardrail}")
print()

# ---------------------------------------------------------------------------
# CELL 2 — (Optional) remap Excel column names if they changed this month
# ---------------------------------------------------------------------------
# If the client's Excel columns were renamed, edit data.schema_map here.
# This is the ONLY place client column names should ever appear.
# Example:
#   cfg_dict = cfg.model_dump()
#   cfg_dict["data"]["schema_map"]["new_column_name"] = "canonical_name"
#   cfg = ConfigBuilder.from_dict(cfg_dict).build()

# ---------------------------------------------------------------------------
# CELL 3 — Run the pipeline
# ---------------------------------------------------------------------------
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from forecasting import run as _run

RUN_ID = time.strftime("%Y-%m")
print(f"Starting pipeline run: {RUN_ID}")
print("This will produce:")
print(f"  runs/{RUN_ID}/  — all 8 contract tables + manifest + log")
print(f"  outputs/  — final forecast CSV + parquet + manifest")
print()

# Uncomment to actually run:
# _run.run(["--config", "configs/fontana_candle.yaml", "--stop-after", "forecast"])

print("(Dry-run mode: uncomment the _run.run() call above to execute)")
print()

# ---------------------------------------------------------------------------
# CELL 4 — Launch the dashboard (if Streamlit is available)
# ---------------------------------------------------------------------------
print("To view results in the dashboard:")
print("  streamlit run app/streamlit_app.py")
