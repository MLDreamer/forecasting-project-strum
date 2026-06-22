"""Emit the eight pipeline→app contract tables (S2).

These flat tables are the ONLY data the Streamlit dashboard reads.
The pipeline writes them; the app never recomputes anything.

Tables emitted to runs/<run_id>/ (and on pass, copied to outputs/latest/):

1.  hierarchy_nodes      node_id, parent_id, level, name, n_skus, segment (leaf)
2.  forecast_long        node_id × future_week × 19 quantiles
3.  actuals_long         node_id × historical_week (units, revenue, region)
4.  backtest_long        node_id × holdout_week (q10/q50/q90, fold_id)
5.  segment_mix          node_id × sb_class (revenue, units, n_skus)
6.  leaf_importance      sku_id × feature (importance, rank) — LightGBM diagnostic
7.  selection            sb_class × (winning_model, wape, crps, cov80, guardrail_pass)
8.  run_manifest         one row (run_id, git_sha, run_ts, input_hashes, blended_wape …)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting import config
from forecasting.forecast import ForecastArtifacts
from forecasting.hierarchy import HierarchyResult
from forecasting.segment import SegmentResult
from forecasting.selection import SelectionResult
from forecasting.validate import CVResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: file hash for reproducibility manifest
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    """SHA-256 first 16 hex chars of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except FileNotFoundError:
        return "missing"


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _hierarchy_nodes(hierarchy: HierarchyResult, segments: SegmentResult) -> pd.DataFrame:
    seg_map = dict(
        zip(
            segments.segments[config.COL_SKU_ID].astype(int),
            segments.segments["sb_class"],
            strict=False,
        )
    )
    rows = []
    for node in hierarchy.nodes:
        n_skus = int(hierarchy.S[list(hierarchy.nodes).index(node), :].sum())
        segment = seg_map.get(int(node.label), "") if node.level == 2 else ""
        rows.append(
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id or "",
                "level": node.level,
                "name": node.label,
                "n_skus": n_skus,
                "segment": segment,
            }
        )
    return pd.DataFrame(rows)


def _forecast_long(artifacts: ForecastArtifacts, hierarchy: HierarchyResult) -> pd.DataFrame:
    """Reconciled node-level quantile forecasts as tidy long format."""
    recon = artifacts.reconciled.copy()
    # recon has: node_id, level, label, forecast_date, p10, p50, p90
    # Expand to all 19 quantile levels using the forecast cube for leaf nodes
    q_levels = artifacts.q_levels
    sku_ids = artifacts.sku_order
    sku_to_idx = {sku: i for i, sku in enumerate(sku_ids)}
    horizon_dates = artifacts.horizon_dates

    # For leaf nodes use cube directly; for upper nodes use reconciled P10/P50/P90
    rows = []
    for _, row in recon.iterrows():
        base = {
            "node_id": row["node_id"],
            "week": str(row["forecast_date"]),
            "level": int(row["level"]),
        }
        if row["level"] == 2:
            # leaf — full 19-quantile from cube
            try:
                sku = int(row["label"])
                i = sku_to_idx.get(sku)
                h_idx = [
                    j
                    for j, d in enumerate(horizon_dates)
                    if str(d.date()) == str(row["forecast_date"])
                ]
                if i is not None and h_idx:
                    h = h_idx[0]
                    for qi, q in enumerate(q_levels):
                        base[f"q{int(round(q * 100)):02d}"] = float(
                            artifacts.forecast_cube[i, h, qi]
                        )
                else:
                    base["q10"] = float(row["p10"])
                    base["q50"] = float(row["p50"])
                    base["q90"] = float(row["p90"])
            except (ValueError, KeyError):
                base["q10"] = float(row["p10"])
                base["q50"] = float(row["p50"])
                base["q90"] = float(row["p90"])
        else:
            base["q10"] = float(row["p10"])
            base["q50"] = float(row["p50"])
            base["q90"] = float(row["p90"])
        rows.append(base)
    return pd.DataFrame(rows)


def _actuals_long(
    full_dense: pd.DataFrame,
    hierarchy: HierarchyResult,
    price_map: pd.Series,
    forecast_origin: pd.Timestamp,
) -> pd.DataFrame:
    """Historical + holdout actuals aggregated to hierarchy nodes."""
    # Compute node-level actuals using S matrix
    sku_bottom_ids = hierarchy.bottom_ids
    sku_ids = [int(nid.split("_", 1)[1]) for nid in sku_bottom_ids]
    sku_to_col = {sku: j for j, sku in enumerate(sku_ids)}

    # Build weekly SKU matrix
    full_dense[config.COL_TIMESTAMP].max()
    all_ts = sorted(full_dense[config.COL_TIMESTAMP].unique())

    rows = []
    for ts in all_ts:
        region = "history" if ts <= forecast_origin else "holdout"
        week_df = full_dense[full_dense[config.COL_TIMESTAMP] == ts]
        sku_units = np.zeros(len(sku_ids))
        for _, r in week_df.iterrows():
            j = sku_to_col.get(int(r[config.COL_SKU_ID]))
            if j is not None:
                sku_units[j] = float(r[config.COL_SALES])

        # Aggregate using S matrix
        node_units = np.asarray(hierarchy.S @ sku_units).ravel()
        for k, node in enumerate(hierarchy.nodes):
            rows.append(
                {
                    "node_id": node.node_id,
                    "week": str(ts.date()),
                    "units": float(node_units[k]),
                    "region": region,
                }
            )

    return pd.DataFrame(rows)


def _backtest_long(
    cv_result: CVResult, hierarchy: HierarchyResult, price_map: pd.Series
) -> pd.DataFrame:
    """Holdout back-test quantiles aggregated to hierarchy nodes."""
    if cv_result is None:
        return pd.DataFrame(columns=["node_id", "week", "q10", "q50", "q90", "fold_id"])

    sku_bottom_ids = hierarchy.bottom_ids
    sku_ids = [int(nid.split("_", 1)[1]) for nid in sku_bottom_ids]
    sku_to_col = {sku: j for j, sku in enumerate(sku_ids)}
    n_bottom = len(sku_ids)

    rows = []
    for fold, actuals in cv_result.fold_actuals.items():
        sku_order = cv_result.sku_order.get(fold, [])
        # Use the seasonal_naive (winning model) predictions
        key = ("seasonal_naive", fold)
        if key not in cv_result.fold_predictions:
            continue
        result = cv_result.fold_predictions[key]
        H = result.horizon
        # We don't have exact dates here — use relative week index
        n = min(len(sku_order), result.n_sku, actuals.shape[0])

        # Build full node-level forecasts via S matrix
        for h in range(min(H, 26)):
            # Build (n_bottom, ) quantile vectors for this horizon step
            q10_vec = np.zeros(n_bottom)
            q50_vec = np.zeros(n_bottom)
            q90_vec = np.zeros(n_bottom)
            q10_idx = int(np.argmin(np.abs(result.q_levels - 0.10)))
            q50_idx = int(np.argmin(np.abs(result.q_levels - 0.50)))
            q90_idx = int(np.argmin(np.abs(result.q_levels - 0.90)))
            for i in range(n):
                sku = sku_order[i]
                j = sku_to_col.get(int(sku))
                if j is not None:
                    q10_vec[j] = float(result.quantiles[i, h, q10_idx])
                    q50_vec[j] = float(result.quantiles[i, h, q50_idx])
                    q90_vec[j] = float(result.quantiles[i, h, q90_idx])
            node_q10 = np.asarray(hierarchy.S @ q10_vec).ravel()
            node_q50 = np.asarray(hierarchy.S @ q50_vec).ravel()
            node_q90 = np.asarray(hierarchy.S @ q90_vec).ravel()
            for k, node in enumerate(hierarchy.nodes):
                rows.append(
                    {
                        "node_id": node.node_id,
                        "week": f"fold{fold}_h{h + 1}",
                        "q10": float(node_q10[k]),
                        "q50": float(node_q50[k]),
                        "q90": float(node_q90[k]),
                        "fold_id": fold,
                    }
                )

    return pd.DataFrame(rows)


def _segment_mix(
    full_dense: pd.DataFrame, segments: SegmentResult, hierarchy: HierarchyResult
) -> pd.DataFrame:
    """Revenue and unit mix by named SB segment per hierarchy node."""
    seg_map = dict(
        zip(
            segments.segments[config.COL_SKU_ID].astype(int),
            segments.segments["sb_class"],
            strict=False,
        )
    )
    full_dense.groupby(config.COL_SKU_ID)[
        "list_price"
    ].last() if "list_price" in full_dense.columns else pd.Series(dtype=float)

    sku_bottom_ids = hierarchy.bottom_ids
    sku_ids = [int(nid.split("_", 1)[1]) for nid in sku_bottom_ids]
    {sku: j for j, sku in enumerate(sku_ids)}
    len(sku_ids)

    sb_classes = list(set(seg_map.values()))
    rows = []
    for node_idx, node in enumerate(hierarchy.nodes):
        bottom_cols = hierarchy.S[node_idx, :].nonzero()[1].tolist()
        node_skus = [sku_ids[j] for j in bottom_cols]
        for sc in sb_classes:
            sc_skus = [s for s in node_skus if seg_map.get(s, "") == sc]
            if not sc_skus:
                continue
            sku_rows = full_dense[full_dense[config.COL_SKU_ID].isin(sc_skus)]
            units = float(sku_rows[config.COL_SALES].sum())
            revenue = (
                float(
                    (
                        sku_rows[config.COL_SALES]
                        * sku_rows.get("list_price", pd.Series(20.0, index=sku_rows.index))
                    ).sum()
                )
                if "list_price" in full_dense.columns
                else units * 20.0
            )
            rows.append(
                {
                    "node_id": node.node_id,
                    "segment": sc,
                    "revenue": revenue,
                    "units": units,
                    "n_skus": len(sc_skus),
                }
            )
    return pd.DataFrame(rows)


def _leaf_importance(segments: SegmentResult, full_features: pd.DataFrame) -> pd.DataFrame:
    """Fit a diagnostic LightGBM per segment and emit feature importances."""
    try:
        import lightgbm as lgb
    except ImportError:
        return pd.DataFrame(columns=["sku_id", "feature", "importance", "rank"])

    from forecasting.models.ml_global import build_training_rows

    seg_map = dict(
        zip(
            segments.segments[config.COL_SKU_ID].astype(int),
            segments.segments["sb_class"],
            strict=False,
        )
    )
    rows = []

    for sb_class in ["erratic", "lumpy", "smooth", "intermittent"]:
        sc_skus = [s for s, c in seg_map.items() if c == sb_class]
        if len(sc_skus) < 3:
            continue
        sc_df = full_features[full_features[config.COL_SKU_ID].isin(sc_skus)]
        if len(sc_df) < 50:
            continue
        try:
            train_df = build_training_rows(sc_df, segments.segments, horizon=4)
            base_excl = {
                config.COL_TIMESTAMP,
                config.COL_SALES,
                "target",
                "list_price",
                "discount_pct",
                "is_potential_stockout",
            }
            feat_cols = [c for c in train_df.columns if c not in base_excl | {"target"}]
            X = train_df[feat_cols].fillna(0).values
            y = train_df["target"].values
            m = lgb.LGBMRegressor(
                objective="regression", n_estimators=50, verbose=-1, random_state=42
            )
            m.fit(X, y)
            imp = m.feature_importances_
            for feat, importance, rank in zip(
                feat_cols, imp, (-imp).argsort().argsort() + 1, strict=False
            ):
                rows.append(
                    {
                        "sb_class": sb_class,
                        "feature": feat,
                        "importance": int(importance),
                        "rank": int(rank),
                    }
                )
        except Exception as e:
            logger.warning("leaf_importance for %s failed: %s", sb_class, e)

    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["sb_class", "feature", "importance", "rank"])
    )


def _selection_table(
    cv_result: CVResult | None, selection: SelectionResult, segments: SegmentResult
) -> pd.DataFrame:
    """Per-segment winner table for the AutoML page."""
    dict(
        zip(
            segments.segments[config.COL_SKU_ID].astype(int),
            segments.segments["sb_class"],
            strict=False,
        )
    )
    rows = []
    sb_classes = ["smooth", "erratic", "lumpy", "intermittent", "cold_start", "discontinued"]
    from forecasting.forecast import SEGMENT_MODEL_MAP

    for sc in sb_classes:
        model = SEGMENT_MODEL_MAP.get(sc, "seasonal_naive")
        rows.append(
            {
                "segment": sc,
                "winning_model": model,
                "wape": float("nan"),
                "crps": float("nan"),
                "cov80": float("nan"),
                "guardrail_pass": False,
            }
        )
    return pd.DataFrame(rows)


def _run_manifest(
    run_id: str,
    artifacts: ForecastArtifacts,
    selection: SelectionResult,
    blended_wape: float,
) -> dict[str, Any]:
    """Build reproducibility manifest dict."""
    import forecasting

    raw_dir = config.DATA_RAW
    input_hashes = {
        f.name: _file_hash(raw_dir / f.name)
        for f in raw_dir.iterdir()
        if f.suffix in (".csv", ".xlsx", ".xls")
    }

    git_sha = "unknown"
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
            timeout=5,
        )
        git_sha = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        pass

    return {
        "run_id": run_id,
        "git_sha": git_sha,
        "forecast_origin": artifacts.manifest.get("forecast_origin", ""),
        "horizon_weeks": artifacts.manifest.get("horizon_weeks", 26),
        "n_sku": artifacts.manifest.get("n_sku", 0),
        "blended_wape": round(blended_wape, 4),
        "winners": {
            sc: SEGMENT_MODEL_MAP.get(sc, "seasonal_naive")
            for sc in ["smooth", "erratic", "lumpy", "intermittent"]
        },
        "input_hashes": input_hashes,
        "forecast_cube_hash": artifacts.manifest.get("forecast_cube_hash", ""),
        "pipeline_version": getattr(forecasting, "__version__", "0.1.0"),
        "status": "published",
    }


# ---------------------------------------------------------------------------
# Main publisher
# ---------------------------------------------------------------------------

# Import SEGMENT_MODEL_MAP at module level for _selection_table
from forecasting.forecast import SEGMENT_MODEL_MAP  # noqa: E402


def publish_tables(
    run_id: str,
    artifacts: ForecastArtifacts,
    selection: SelectionResult,
    segments: SegmentResult,
    hierarchy: HierarchyResult,
    full_dense: pd.DataFrame,
    full_features: pd.DataFrame,
    cv_result: CVResult | None = None,
    blended_wape: float = float("nan"),
    output_dir: Path | None = None,
) -> Path:
    """Write all 8 contract tables to runs/<run_id>/ and return the run dir.

    Parameters
    ----------
    run_id : str, e.g. "2026-07"
    artifacts : ForecastArtifacts from generate_forecast()
    selection : SelectionResult from Phase 15
    segments : SegmentResult from Phase 5
    hierarchy : HierarchyResult from Phase 6
    full_dense : the full 16,068-row dense grid
    full_features : the full 16,068 × 114 feature matrix
    cv_result : CVResult from Phase 14 (optional)
    blended_wape : blended revenue-weighted WAPE for the manifest
    output_dir : defaults to config.OUTPUTS parent / "runs"
    """
    if output_dir is None:
        output_dir = config.OUTPUTS.parent / "runs"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    price_map = (
        full_dense.groupby(config.COL_SKU_ID)["list_price"].last()
        if "list_price" in full_dense.columns
        else pd.Series(dtype=float)
    )
    forecast_origin = artifacts.horizon_dates[0] - pd.Timedelta(weeks=1)

    logger.info("Publishing %s contract tables to %s", run_id, run_dir)

    # 1. hierarchy_nodes
    hn = _hierarchy_nodes(hierarchy, segments)
    hn.to_parquet(run_dir / "hierarchy_nodes.parquet", index=False)

    # 2. forecast_long
    fl = _forecast_long(artifacts, hierarchy)
    fl.to_parquet(run_dir / "forecast_long.parquet", index=False)

    # 3. actuals_long
    al = _actuals_long(full_dense, hierarchy, price_map, forecast_origin)
    al.to_parquet(run_dir / "actuals_long.parquet", index=False)

    # 4. backtest_long
    bl = _backtest_long(cv_result, hierarchy, price_map)
    bl.to_parquet(run_dir / "backtest_long.parquet", index=False)

    # 5. segment_mix
    sm = _segment_mix(full_dense, segments, hierarchy)
    sm.to_parquet(run_dir / "segment_mix.parquet", index=False)

    # 6. leaf_importance (diagnostic LightGBM)
    li = _leaf_importance(segments, full_features)
    li.to_parquet(run_dir / "leaf_importance.parquet", index=False)

    # 7. selection
    sel_df = _selection_table(cv_result, selection, segments)
    sel_df.to_parquet(run_dir / "selection.parquet", index=False)

    # 8. run_manifest (json + parquet)
    rm = _run_manifest(run_id, artifacts, selection, blended_wape)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(rm, indent=2, default=str))
    pd.DataFrame([rm]).to_parquet(run_dir / "run_manifest.parquet", index=False)

    # Also update runs/index.json
    index_path = Path(output_dir) / "index.json"
    index = []
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except Exception:
            index = []
    # Remove old entry for same run_id if re-running
    index = [e for e in index if e.get("run_id") != run_id]
    index.append(
        {
            "run_id": run_id,
            "date": rm["forecast_origin"],
            "blended_wape": rm["blended_wape"],
            "status": rm["status"],
            "winners": rm["winners"],
        }
    )
    index_path.write_text(json.dumps(index, indent=2, default=str))

    logger.info("Published 8 tables + manifest to %s", run_dir)
    return run_dir
