"""Pipeline entry point.

Usage:
    python -m forecasting.run --config configs/fontana_candle.yaml
    python -m forecasting.run --config configs/fontana_candle.yaml --validate-only
    python -m forecasting.run --config configs/fontana_candle.yaml --stop-after densify
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Ordered list of phase names that --stop-after accepts
PHASES: list[str] = [
    "io",
    "lifecycle",
    "densify",
    "features",
    "segment",
    "hierarchy",
    "cv",
    "forecast",
    "report",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="forecasting.run",
        description="AutoML forecasting pipeline — Fontana Candle Co",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML pipeline config.",
    )
    parser.add_argument(
        "--stop-after",
        metavar="PHASE",
        choices=PHASES,
        default=None,
        help=f"Stop after this phase. One of: {', '.join(PHASES)}",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Load and validate the config, then exit without running any phase.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level),
        stream=sys.stderr,
    )


def _should_stop(phase: str, stop_after: str | None) -> bool:
    if stop_after is None:
        return False
    return PHASES.index(phase) >= PHASES.index(stop_after)


def run(argv: list[str] | None = None) -> None:
    """Main pipeline runner — called by __main__ or tests."""
    args = _parse_args(argv)
    _setup_logging(args.log_level)

    # --- Config load + validation -------------------------------------------
    from configs._schema import PipelineConfig

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    cfg = PipelineConfig.from_yaml(cfg_path)
    logger.info("Config validated: client=%s", cfg.client)

    if args.validate_only:
        logger.info("--validate-only: config OK, exiting.")
        return

    # --- Phase: io -----------------------------------------------------------
    from forecasting.io import load_all

    raw_dir = Path(cfg.data.raw_dir)
    data = load_all(raw_dir)
    logger.info(
        "IO: sales=%d rows | master=%d rows | has_sales=%d | cold_start=%d",
        len(data.sales),
        len(data.master),
        len(data.sku_has_sales),
        len(data.sku_cold_start),
    )
    if _should_stop("io", args.stop_after):
        return

    # --- Phase: lifecycle ----------------------------------------------------
    from forecasting.lifecycle import infer_lifecycle, save_lifecycle

    lc_result = infer_lifecycle(data.sales, data.master)
    save_lifecycle(lc_result)
    logger.info(
        "Lifecycle: active=%d | dormant=%d",
        len(lc_result.sku_active),
        len(lc_result.sku_dormant),
    )
    if _should_stop("lifecycle", args.stop_after):
        return

    # --- Phase: densify ------------------------------------------------------
    from forecasting.densify import densify, save_dense

    dense_result = densify(
        data.sales,
        lc_result,
        data.joined,
        week_relabel_shift_days=cfg.densify.week_relabel_shift_days,
    )
    save_dense(dense_result)
    logger.info(
        "Densify: %d rows | %.1f%% zeros | %d stockout SKUs",
        len(dense_result.dense),
        dense_result.zero_fraction * 100,
        len(dense_result.stockout_skus),
    )
    if _should_stop("densify", args.stop_after):
        return

    # --- Phases 4–19: stubs (implemented in later phases) --------------------
    for phase in PHASES[PHASES.index("features") :]:
        logger.info("Phase '%s' not yet implemented — skipping.", phase)
        if _should_stop(phase, args.stop_after):
            return


if __name__ == "__main__":
    run()
