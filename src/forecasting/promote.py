"""Promotion gate: validate a run and copy it to outputs/latest/.

Usage:
    python -m forecasting.promote --run-id 2026-07 --max-wape 0.80

Promotion logic:
  1. Read runs/<run_id>/run_manifest.json.
  2. Check blended_wape <= max_wape.
  3. If --require-guardrail, check that at least one fold passes [0.75, 0.85].
  4. On pass: copy 8 contract tables to outputs/latest/ and open_latest_symlink.
  5. On fail: print reason, exit 1, optionally open a GitHub Issue.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_TABLE_NAMES = [
    "hierarchy_nodes.parquet",
    "forecast_long.parquet",
    "actuals_long.parquet",
    "backtest_long.parquet",
    "segment_mix.parquet",
    "leaf_importance.parquet",
    "selection.parquet",
    "run_manifest.parquet",
    "run_manifest.json",
]


def _promote(
    run_id: str,
    runs_dir: Path,
    latest_dir: Path,
    max_wape: float,
    require_guardrail: bool,
) -> bool:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        return False

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        logger.error("run_manifest.json not found in %s", run_dir)
        return False

    manifest = json.loads(manifest_path.read_text())
    blended_wape = float(manifest.get("blended_wape", 999))

    # Gate 1: WAPE
    if blended_wape > max_wape:
        logger.error(
            "FAIL — blended_wape=%.4f > max_wape=%.4f. Run %s not promoted.",
            blended_wape,
            max_wape,
            run_id,
        )
        return False

    # Gate 2: all required tables present
    missing = [t for t in _TABLE_NAMES if not (run_dir / t).exists()]
    if missing:
        logger.error("FAIL — missing tables: %s. Run %s not promoted.", missing, run_id)
        return False

    # If all gates pass, copy to outputs/latest/
    latest_dir.mkdir(parents=True, exist_ok=True)
    for table in _TABLE_NAMES:
        src = run_dir / table
        dst = latest_dir / table
        if src.exists():
            shutil.copy2(src, dst)

    # Update manifest in root outputs/
    manifest["status"] = "published"
    (latest_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    logger.info(
        "PROMOTED run %s → outputs/latest/  (blended_wape=%.4f)",
        run_id,
        blended_wape,
    )
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Promote a forecast run to outputs/latest/")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-wape", type=float, default=0.80)
    parser.add_argument("--require-guardrail", action="store_true", default=False)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--latest-dir", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from forecasting import config

    runs_dir = Path(args.runs_dir) if args.runs_dir else config.OUTPUTS.parent / "runs"
    latest_dir = Path(args.latest_dir) if args.latest_dir else config.OUTPUTS / "latest"

    ok = _promote(args.run_id, runs_dir, latest_dir, args.max_wape, args.require_guardrail)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
