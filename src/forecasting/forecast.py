"""Refit winning models on full data, apply calibration, produce final forecast.

Phase 16 flow:
1. For each cluster, instantiate the winning model from SelectionResult.
2. Refit on the FULL training data (all history up to the forecast origin).
3. Apply post-hoc conformal calibration alpha from SelectionResult.
4. Produce a single quantile cube (n_sku, H, n_q) aligned to bottom SKU order.
5. Write forecast_final.csv (P10/P50/P90 per SKU per week) and
   forecast_hierarchy.parquet (reconciled hierarchy).
6. Write manifest.json for reproducibility.

Forecast horizon: 2026-05-24 → 2026-11-15 (26 weeks, Saturday-dated).
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting import config
from forecasting.hierarchy import HierarchyResult
from forecasting.reconcile import reconcile_bottom_up
from forecasting.segment import SegmentResult
from forecasting.selection import SelectionResult, apply_calibration

logger = logging.getLogger(__name__)

# Model name → class mapping (populated by importing model modules)
_MODEL_CLASS_MAP: dict[str, type] = {}

# Per-segment routing (S1.3 — locked, do not override without a CV-validated reason)
# Maps named SB class → model name to use in the final forecast.
# These match the CV-measured per-segment winners:
#   erratic  → trend_seasonal  (seasonal × clipped YoY, beats SN in fold 3 erratic)
#   smooth   → recent_level    (8-week mean, handles stocked-out SKUs as near-zero)
#   lumpy    → seasonal_naive  (nothing beat it for lumpy)
#   intermittent → seasonal_naive
#   cold_start   → seasonal_naive (no history; will be Chronos when wired)
#   discontinued → zero_forecast  (dormant)
SEGMENT_MODEL_MAP: dict[str, str] = {
    "erratic": "trend_seasonal",
    "smooth": "recent_level",
    "lumpy": "seasonal_naive",
    "intermittent": "seasonal_naive",
    "cold_start": "seasonal_naive",
    "discontinued": "zero_forecast",
}


def _get_model_class(model_name: str) -> type | None:
    """Return the model class for a registered model name."""
    # Lazy-import all model modules to populate the registry
    if not _MODEL_CLASS_MAP:
        import forecasting.models.baseline  # noqa: F401
        import forecasting.models.classical  # noqa: F401
        import forecasting.models.foundation  # noqa: F401
        import forecasting.models.intermittent  # noqa: F401
        import forecasting.models.ml_global  # noqa: F401
        import forecasting.models.tweedie  # noqa: F401
        from forecasting.registry import MODEL_REGISTRY

        _MODEL_CLASS_MAP.update(MODEL_REGISTRY)

    return _MODEL_CLASS_MAP.get(model_name)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class ForecastArtifacts:
    """All outputs produced by generate_forecast()."""

    forecast_cube: np.ndarray  # (n_sku, H, n_q) — bottom-level, calibrated
    sku_order: list[int]  # SKU ids aligned to axis 0
    q_levels: np.ndarray  # quantile levels aligned to axis 2
    horizon_dates: list[pd.Timestamp]  # Saturday dates for each horizon step
    reconciled: pd.DataFrame  # hierarchy-reconciled P10/P50/P90 per node per week
    manifest: dict  # reproducibility metadata


# ---------------------------------------------------------------------------
# Core forecast generator
# ---------------------------------------------------------------------------


def generate_forecast(
    full_features: pd.DataFrame,
    full_dense: pd.DataFrame,
    selection: SelectionResult,
    segments: SegmentResult,
    hierarchy: HierarchyResult,
    q_levels: np.ndarray | None = None,
    horizon: int = config.FORECAST_HORIZON_WEEKS,
    forecast_origin: pd.Timestamp | None = None,
    bootstrap_samples: int = 500,
    output_dir: Path | None = None,
) -> ForecastArtifacts:
    """Refit winning models and produce the final forecast.

    Parameters
    ----------
    full_features : DataFrame (16,068 × 114)
    full_dense : DataFrame (16,068 rows)
    selection : SelectionResult from Phase 15
    segments : SegmentResult from Phase 5
    hierarchy : HierarchyResult from Phase 6
    q_levels : quantile levels; defaults to config.QUANTILES
    horizon : forecast horizon in weeks (default 26)
    forecast_origin : last observed Saturday; defaults to max(timestamp)
    bootstrap_samples : number of bootstrap paths for reconciliation
    output_dir : where to write outputs; defaults to config.OUTPUTS

    Returns
    -------
    ForecastArtifacts
    """
    if q_levels is None:
        q_levels = np.array(config.QUANTILES)
    if output_dir is None:
        output_dir = config.OUTPUTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if forecast_origin is None:
        forecast_origin = full_features[config.COL_TIMESTAMP].max()

    # Build horizon date index (Saturday-dated)
    horizon_dates = [forecast_origin + pd.Timedelta(weeks=h) for h in range(1, horizon + 1)]

    # All bottom-level SKU ids (same order as hierarchy)
    sku_order_str = hierarchy.bottom_ids  # L2_<sku_id> strings
    sku_ids = [int(nid.split("_", 1)[1]) for nid in sku_order_str]
    n_sku = len(sku_ids)
    n_q = len(q_levels)

    # Final quantile cube — accumulate per-cluster
    forecast_cube = np.zeros((n_sku, horizon, n_q))
    sku_to_idx = {sku: i for i, sku in enumerate(sku_ids)}

    # Build cluster → SKU map (exclude discontinued)
    active_seg = segments.segments[segments.segments["sb_class"] != "discontinued"]
    cluster_sku_map: dict[int, list[int]] = {}
    for cluster_id, grp in active_seg.groupby("cluster_id"):
        cluster_sku_map[int(cluster_id)] = list(grp[config.COL_SKU_ID].astype(int))

    # Per-segment routing (S1.3): group SKUs by their named SB class, not by cluster.
    # Each SB class gets its own fitted model.  This overrides the cluster-level winner
    # from Phase 15 with the CV-validated per-segment defaults from SEGMENT_MODEL_MAP.
    seg_map_df = segments.segments[[config.COL_SKU_ID, "sb_class"]].copy()
    seg_map = dict(
        zip(seg_map_df[config.COL_SKU_ID].astype(int), seg_map_df["sb_class"], strict=False)
    )

    # Group all in-scope SKUs by their SB class
    sb_class_skus: dict[str, list[int]] = {}
    for sku in sku_ids:
        sb = seg_map.get(sku, "seasonal_naive")
        sb_class_skus.setdefault(sb, []).append(sku)

    train_data = full_dense[full_dense[config.COL_TIMESTAMP] <= forecast_origin]

    for sb_class, sku_ids_in_segment in sb_class_skus.items():
        segment_model_name = SEGMENT_MODEL_MAP.get(sb_class, "seasonal_naive")
        # Calibration alpha: use the cluster-level alpha for the first cluster these SKUs belong to
        # (blended because segment spans clusters; use mean alpha across relevant clusters)
        seg_alphas = []
        for sku in sku_ids_in_segment:
            for cluster_id, c_skus in cluster_sku_map.items():
                if sku in c_skus:
                    winner = selection.winner_for(cluster_id)
                    a = selection.calibration_alphas.get((cluster_id, winner), 1.0)
                    seg_alphas.append(a)
                    break
        alpha = float(np.mean(seg_alphas)) if seg_alphas else 1.0

        ModelClass = _get_model_class(segment_model_name)
        if ModelClass is None:
            logger.warning(
                "Segment %s: model '%s' not found — using SeasonalNaive",
                sb_class,
                segment_model_name,
            )
            from forecasting.models.baseline import SeasonalNaive

            ModelClass = SeasonalNaive

        series_dict = {
            str(sku): train_data[train_data[config.COL_SKU_ID] == sku][config.COL_SALES].values
            for sku in sku_ids_in_segment
            if len(train_data[train_data[config.COL_SKU_ID] == sku]) > 0
        }

        if not series_dict:
            logger.warning("Segment %s: no training data — skipping.", sb_class)
            continue

        # Retain legacy variable names so rest of loop body is unchanged
        cluster_id = sb_class  # used only for logging
        winner_name = segment_model_name

        try:
            model = ModelClass(q_levels=q_levels)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit_series(series_dict)
                raw_result = model.predict(np.empty(0), horizon)

            # Apply post-hoc conformal calibration
            if alpha != 1.0:
                raw_result = apply_calibration(raw_result, alpha, q_levels)

            # Align into forecast_cube
            fitted_order = sorted(series_dict.keys())
            for j, sku_str in enumerate(fitted_order):
                sku = int(sku_str)
                idx = sku_to_idx.get(sku)
                if idx is not None and j < raw_result.n_sku:
                    forecast_cube[idx] = raw_result.quantiles[j]

            logger.info(
                "Segment %s model=%s alpha=%.3f n_sku=%d",
                sb_class,
                winner_name,
                alpha,
                len(series_dict),
            )

        except Exception as exc:
            logger.error("Segment %s %s FAILED: %s", sb_class, winner_name, exc)
            # Fallback: zero forecast (safer than leaving noise)
            continue

    # Reconcile bottom-up
    reconciled_df = reconcile_bottom_up(
        forecast_cube=forecast_cube,
        sku_ids=sku_ids,
        hierarchy=hierarchy,
        q_levels=q_levels,
        horizon_dates=horizon_dates,
        n_bootstrap=bootstrap_samples,
    )

    # Write forecast_final.csv (variant-level P10/P50/P90)
    p10_idx = int(np.argmin(np.abs(q_levels - 0.10)))
    p50_idx = int(np.argmin(np.abs(q_levels - 0.50)))
    p90_idx = int(np.argmin(np.abs(q_levels - 0.90)))

    final_rows = []
    for i, sku in enumerate(sku_ids):
        for h, dt in enumerate(horizon_dates):
            final_rows.append(
                {
                    "sku_id": sku,
                    "forecast_date": dt.date(),
                    "p10": float(forecast_cube[i, h, p10_idx]),
                    "p50": float(forecast_cube[i, h, p50_idx]),
                    "p90": float(forecast_cube[i, h, p90_idx]),
                }
            )

    forecast_df = pd.DataFrame(final_rows)
    forecast_path = output_dir / "forecast_final.csv"
    forecast_df.to_csv(forecast_path, index=False)
    logger.info("Wrote forecast_final.csv → %s (%d rows)", forecast_path, len(forecast_df))

    # Write forecast_hierarchy.parquet
    hier_path = output_dir / "forecast_hierarchy.parquet"
    reconciled_df.to_parquet(hier_path, index=False)
    logger.info("Wrote forecast_hierarchy.parquet → %s", hier_path)

    # Write manifest.json
    manifest = _build_manifest(
        forecast_origin=forecast_origin,
        horizon=horizon,
        n_sku=n_sku,
        q_levels=q_levels,
        selection=selection,
        forecast_cube=forecast_cube,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Wrote manifest.json → %s", manifest_path)

    return ForecastArtifacts(
        forecast_cube=forecast_cube,
        sku_order=sku_ids,
        q_levels=q_levels,
        horizon_dates=horizon_dates,
        reconciled=reconciled_df,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def _build_manifest(
    forecast_origin: pd.Timestamp,
    horizon: int,
    n_sku: int,
    q_levels: np.ndarray,
    selection: SelectionResult,
    forecast_cube: np.ndarray,
) -> dict:
    """Build reproducibility manifest."""
    import forecasting

    # Hash of the forecast cube for byte-level reproducibility check
    cube_hash = hashlib.sha256(forecast_cube.tobytes()).hexdigest()[:16]

    winners = {str(cid): w.winner_model for cid, w in selection.cluster_winners.items()}
    alphas = {
        f"{cid}:{mname}": round(a, 4) for (cid, mname), a in selection.calibration_alphas.items()
    }

    return {
        "pipeline_version": getattr(forecasting, "__version__", "0.1.0"),
        "random_seed": config.RANDOM_SEED,
        "forecast_origin": str(forecast_origin.date()),
        "horizon_weeks": horizon,
        "n_sku": n_sku,
        "n_quantiles": len(q_levels),
        "q_levels": q_levels.tolist(),
        "cluster_winners": winners,
        "calibration_alphas": alphas,
        "forecast_cube_hash": cube_hash,
        "guardrail": [config.CALIBRATION_GUARDRAIL[0], config.CALIBRATION_GUARDRAIL[1]],
    }
