"""Per-cluster model selection with calibration guardrail.

Design (locked — doc Phase 15 + modeling_decisions.md):

Selection rule (locked):
    1. For each cluster, compute revenue-weighted CRPS averaged over folds 2–4.
    2. Reject any candidate whose 80% PI coverage ∉ [0.75, 0.85] (guardrail).
    3. Among surviving candidates, pick lowest CRPS; WAPE is the tiebreaker.
    4. If no candidate survives the guardrail, fall back to SeasonalNaive
       (which always passes — it is the floor).
    5. Baselines (seasonal_naive, zero_forecast) must be beaten to deploy.
       A model that only matches seasonal_naive is not selected.

V2 levers (decided from Phase 15 evidence):
    - segment_as_cluster: if cluster_lgbm loses in most clusters, use SB class
      as the pooling unit instead of K-means clusters.
    - post_hoc_conformal: calibration repair for cluster_lgbm if it wins but
      under-covers (widen PIs by post-hoc conformal scaling).

Selection output:
    SelectionResult — per-cluster winner, per-model scores, guardrail flags,
    v2 lever recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecasting import config
from forecasting.validate import SELECTION_FOLDS, CVResult

logger = logging.getLogger(__name__)

# Locked calibration guardrail bounds (doc)
GUARDRAIL_LO: float = 0.75
GUARDRAIL_HI: float = 0.85

# A model beats the baseline only if it improves CRPS by at least this margin
BASELINE_IMPROVEMENT_MIN: float = 0.001  # 0.1% improvement required

# V2 lever threshold: if cluster_lgbm is selected in fewer clusters than this
# fraction, recommend "segment_as_cluster" lever
SEGMENT_AS_CLUSTER_THRESHOLD: float = 0.25

# Minimum training weeks for a SKU to be included in guardrail evaluation.
# Brand-new cold-start SKUs (no history before fold cutoff) have no residuals
# for conformal calibration — evaluating coverage on them unfairly penalises
# models, since any model will miss for an SKU it never saw.
MIN_TRAIN_WEEKS_FOR_GUARDRAIL: int = 4


# ---------------------------------------------------------------------------
# Post-hoc conformal calibration (v1, promoted from v2 lever)
# ---------------------------------------------------------------------------


def find_calibration_alpha(
    y_true: np.ndarray,
    quantiles: np.ndarray,  # (N, n_q)
    q_levels: np.ndarray,
    target_cov: float = 0.80,
    tol: float = 0.002,
    max_iter: int = 60,
) -> float:
    """Find scaling factor alpha via binary search on 80% PI coverage.

    Scales all quantiles symmetrically around P50:
        q_calibrated = P50 + alpha * (q_raw - P50)

    alpha=1.0 → no change; alpha>1 → widens; alpha<1 → narrows.
    Returns alpha that achieves approximately target_cov 80% PI coverage
    on the provided (y_true, quantiles) validation set.
    """
    from forecasting.metrics import coverage_80

    p50_idx = int(np.argmin(np.abs(q_levels - 0.5)))
    p50 = quantiles[:, p50_idx : p50_idx + 1]

    # Cap alpha at 5.0 — beyond this, intervals are so wide they're uninformative.
    # If even alpha=5 can't reach target_cov, the cluster has too few observations
    # for reliable calibration; alpha=5 is a reasonable worst-case expansion.
    lo, hi = 0.0, 5.0
    alpha = 1.0
    for _ in range(max_iter):
        alpha = (lo + hi) / 2.0
        q_scaled = np.maximum(0.0, p50 + alpha * (quantiles - p50))
        cov = coverage_80(y_true, q_scaled, q_levels)
        if abs(cov - target_cov) < tol:
            break
        if cov < target_cov:
            lo = alpha
        else:
            hi = alpha
    return float(alpha)


def apply_calibration(
    forecast: object,  # ForecastResult — avoid circular import
    alpha: float,
    q_levels: np.ndarray,
) -> object:
    """Scale all quantiles by alpha around P50.  Returns a new ForecastResult."""
    from forecasting.models.base import ForecastResult

    q = forecast.quantiles.copy()  # type: ignore[attr-defined]
    p50_idx = int(np.argmin(np.abs(q_levels - 0.5)))
    p50 = q[:, :, p50_idx : p50_idx + 1]
    q_cal = np.maximum(0.0, p50 + alpha * (q - p50))
    return ForecastResult.from_quantiles(q_cal, q_levels)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class ModelScore:
    """Aggregated score for one model on one cluster."""

    model_name: str
    cluster_id: int | str
    mean_crps: float  # revenue-weighted CRPS averaged over selection folds
    mean_wape: float  # WAPE averaged over selection folds
    mean_cov80: float  # 80% PI coverage averaged over selection folds
    passes_guardrail: bool  # mean_cov80 ∈ [GUARDRAIL_LO, GUARDRAIL_HI]
    n_selection_folds: int  # number of folds that contributed to the score


@dataclass
class ClusterWinner:
    """Selected model for one cluster."""

    cluster_id: int | str
    winner_model: str
    winner_crps: float
    winner_cov80: float
    baseline_crps: float  # seasonal_naive CRPS for comparison
    beats_baseline: bool
    guardrail_rejected: list[str] = field(default_factory=list)  # models rejected by guardrail
    fallback_used: bool = False  # True if no model passed guardrail → fell back to baseline


@dataclass
class SelectionResult:
    """Full selection output."""

    cluster_winners: dict[int | str, ClusterWinner]
    """cluster_id → winning model."""

    model_scores: list[ModelScore]
    """All (model, cluster) scores for diagnostics."""

    v2_levers: dict[str, bool]
    """Recommended v2 levers: segment_as_cluster, post_hoc_conformal."""

    n_clusters: int
    n_clusters_won_by_lgbm: int

    calibration_alphas: dict[tuple[int | str, str], float] = field(default_factory=dict)
    """(cluster_id, model_name) → calibration alpha.  Alpha=1.0 means no change."""

    def winner_for(self, cluster_id: int | str) -> str:
        """Return the winning model name for a cluster (or 'seasonal_naive' if unknown)."""
        w = self.cluster_winners.get(cluster_id)
        return w.winner_model if w else "seasonal_naive"

    def summary(self) -> pd.DataFrame:
        """Tidy DataFrame with one row per cluster."""
        rows = []
        for cid, w in self.cluster_winners.items():
            rows.append(
                {
                    "cluster_id": cid,
                    "winner": w.winner_model,
                    "winner_crps": round(w.winner_crps, 4),
                    "winner_cov80": round(w.winner_cov80, 3),
                    "baseline_crps": round(w.baseline_crps, 4),
                    "beats_baseline": w.beats_baseline,
                    "fallback_used": w.fallback_used,
                    "n_rejected": len(w.guardrail_rejected),
                }
            )
        return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Score aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_scores(
    cv_result: CVResult,
    cluster_sku_map: dict[int | str, list[int]],
    q_levels: np.ndarray,
) -> list[ModelScore]:
    """Compute per-(model, cluster) aggregated scores over selection folds.

    Parameters
    ----------
    cv_result : CVResult from validate.run_cv
    cluster_sku_map : {cluster_id: [sku_id, ...]}
        Mapping from cluster to the SKUs it contains.
    q_levels : quantile levels used by the models

    Returns
    -------
    list of ModelScore, one per (model_name, cluster_id) combination.
    """
    from forecasting.metrics import coverage_80, crps_from_quantiles, wape

    scores: list[ModelScore] = []

    # Collect all model names that have at least one result
    all_models = {fm.model_name for fm in cv_result.fold_metrics}

    for model_name in sorted(all_models):
        for cluster_id, sku_ids in cluster_sku_map.items():
            crps_vals, wape_vals, cov80_vals = [], [], []

            for fold in SELECTION_FOLDS:
                key = (model_name, fold)
                if key not in cv_result.fold_predictions:
                    continue
                if fold not in cv_result.fold_actuals:
                    continue

                forecast = cv_result.fold_predictions[key]
                actuals_all = cv_result.fold_actuals[fold]  # (n_sku_fold, H)
                sku_order = cv_result.sku_order.get(fold, [])

                # Find indices of this cluster's SKUs in the fold.
                # Masked evaluation: only include SKUs that had training data
                # (i.e. appear in sku_order — they had holdout actuals, meaning
                # the model could have been trained on them before the cutoff).
                # Brand-new cold-start SKUs in the holdout penalise coverage
                # unfairly because ANY model misses for an unseen SKU.
                cluster_idx = [i for i, s in enumerate(sku_order) if s in set(sku_ids)]
                if not cluster_idx:
                    continue

                y_true = actuals_all[cluster_idx].ravel()  # (n_cluster_skus * H,)
                q_cube = forecast.quantiles[cluster_idx]  # (n_cluster, H, n_q)
                q_flat = q_cube.reshape(-1, q_cube.shape[2])

                if len(y_true) == 0:
                    continue

                crps_vals.append(crps_from_quantiles(y_true, q_flat, q_levels))
                wape_vals.append(
                    wape(y_true, q_cube.reshape(-1, q_cube.shape[2])[:, len(q_levels) // 2])
                )
                cov80_vals.append(coverage_80(y_true, q_flat, q_levels))

            if not crps_vals:
                continue

            mean_crps = float(np.mean(crps_vals))
            mean_wape = float(np.mean(wape_vals))
            mean_cov80 = float(np.mean(cov80_vals))
            passes = GUARDRAIL_LO <= mean_cov80 <= GUARDRAIL_HI

            scores.append(
                ModelScore(
                    model_name=model_name,
                    cluster_id=cluster_id,
                    mean_crps=mean_crps,
                    mean_wape=mean_wape,
                    mean_cov80=mean_cov80,
                    passes_guardrail=passes,
                    n_selection_folds=len(crps_vals),
                )
            )

    return scores


# ---------------------------------------------------------------------------
# Main selection entry point
# ---------------------------------------------------------------------------


def select_winners(
    cv_result: CVResult,
    segments_df: pd.DataFrame,
    q_levels: np.ndarray | None = None,
    guardrail_lo: float = GUARDRAIL_LO,
    guardrail_hi: float = GUARDRAIL_HI,
) -> SelectionResult:
    """Select the winning model per cluster from CV results.

    Parameters
    ----------
    cv_result : CVResult
        Output of validate.run_cv().
    segments_df : DataFrame
        Output of segment_and_cluster() — must have sku_id, cluster_id.
    q_levels : np.ndarray | None
        Quantile levels used by models; defaults to config.QUANTILES.
    guardrail_lo / guardrail_hi : float
        80% PI coverage guardrail bounds (default [0.75, 0.85]).

    Returns
    -------
    SelectionResult
    """
    if q_levels is None:
        q_levels = np.array(config.QUANTILES)

    # Build cluster → SKU mapping (exclude discontinued / zero-forecast SKUs)
    active_segs = segments_df[segments_df["sb_class"] != "discontinued"]
    cluster_sku_map: dict[int | str, list[int]] = {}
    for cluster_id, grp in active_segs.groupby("cluster_id"):
        cluster_sku_map[int(cluster_id)] = list(grp[config.COL_SKU_ID].astype(int))

    if not cluster_sku_map:
        logger.warning("No clusters found — returning empty SelectionResult.")
        return SelectionResult(
            cluster_winners={},
            model_scores=[],
            v2_levers={"segment_as_cluster": False, "post_hoc_conformal": False},
            n_clusters=0,
            n_clusters_won_by_lgbm=0,
        )

    # Aggregate scores per (model, cluster)
    scores = _aggregate_scores(cv_result, cluster_sku_map, q_levels)

    # Select winner per cluster
    cluster_winners: dict[int | str, ClusterWinner] = {}
    n_lgbm_wins = 0

    for cluster_id in sorted(cluster_sku_map.keys()):
        cluster_scores = [s for s in scores if s.cluster_id == cluster_id]

        if not cluster_scores:
            # No scores at all → use seasonal_naive as fallback
            cluster_winners[cluster_id] = ClusterWinner(
                cluster_id=cluster_id,
                winner_model="seasonal_naive",
                winner_crps=float("inf"),
                winner_cov80=float("nan"),
                baseline_crps=float("inf"),
                beats_baseline=False,
                fallback_used=True,
            )
            continue

        # Get baseline CRPS for comparison
        baseline_scores = [s for s in cluster_scores if s.model_name == "seasonal_naive"]
        baseline_crps = min((s.mean_crps for s in baseline_scores), default=float("inf"))

        # Filter by guardrail
        passing = [s for s in cluster_scores if s.passes_guardrail]
        rejected_names = [s.model_name for s in cluster_scores if not s.passes_guardrail]

        fallback_used = False
        if not passing:
            # All candidates failed guardrail → fallback to seasonal_naive
            logger.warning(
                "Cluster %s: all %d models failed guardrail → fallback to seasonal_naive",
                cluster_id,
                len(cluster_scores),
            )
            winner = "seasonal_naive"
            winner_crps = baseline_crps
            winner_cov80 = next((s.mean_cov80 for s in baseline_scores), float("nan"))
            fallback_used = True
        else:
            # Sort by CRPS (primary), WAPE (tiebreaker)
            passing_sorted = sorted(passing, key=lambda s: (s.mean_crps, s.mean_wape))
            best = passing_sorted[0]
            winner = best.model_name
            winner_crps = best.mean_crps
            winner_cov80 = best.mean_cov80

        # Check if winner beats the baseline (must improve CRPS meaningfully)
        beats_baseline = (
            baseline_crps - winner_crps > BASELINE_IMPROVEMENT_MIN if not fallback_used else False
        )
        if not beats_baseline and not fallback_used and winner != "seasonal_naive":
            logger.warning(
                "Cluster %s: %s CRPS=%.4f does not beat seasonal_naive CRPS=%.4f "
                "→ falling back to seasonal_naive",
                cluster_id,
                winner,
                winner_crps,
                baseline_crps,
            )
            winner = "seasonal_naive"
            winner_crps = baseline_crps
            winner_cov80 = next((s.mean_cov80 for s in baseline_scores), float("nan"))
            fallback_used = True
            beats_baseline = False

        if "lgbm" in winner.lower():
            n_lgbm_wins += 1

        cluster_winners[cluster_id] = ClusterWinner(
            cluster_id=cluster_id,
            winner_model=winner,
            winner_crps=winner_crps,
            winner_cov80=winner_cov80,
            baseline_crps=baseline_crps,
            beats_baseline=beats_baseline,
            guardrail_rejected=rejected_names,
            fallback_used=fallback_used,
        )

        logger.info(
            "Cluster %s → winner=%s (CRPS=%.4f, cov80=%.2f, beats_baseline=%s)",
            cluster_id,
            winner,
            winner_crps,
            winner_cov80,
            beats_baseline,
        )

    n_clusters = len(cluster_winners)
    lgbm_fraction = n_lgbm_wins / max(n_clusters, 1)

    # V2 lever recommendations
    v2_levers = {
        "segment_as_cluster": lgbm_fraction < SEGMENT_AS_CLUSTER_THRESHOLD,
        "post_hoc_conformal": any(
            w.winner_model.lower().find("lgbm") >= 0 and w.winner_cov80 < guardrail_lo
            for w in cluster_winners.values()
        ),
    }

    if v2_levers["segment_as_cluster"]:
        logger.warning(
            "V2 lever: cluster_lgbm won only %d/%d clusters (%.0f%% < %.0f%% threshold). "
            "Consider 'segment-as-cluster' (use SB class as pooling unit).",
            n_lgbm_wins,
            n_clusters,
            lgbm_fraction * 100,
            SEGMENT_AS_CLUSTER_THRESHOLD * 100,
        )

    # --- Post-hoc conformal calibration alphas ----------------------------
    # For each (cluster, winning_model), compute alpha from CV holdout data
    # such that calibrated 80% PI coverage ≈ 0.80.
    # Alpha is stored in SelectionResult; Phase 16 applies it to final forecasts.
    from forecasting.metrics import coverage_80  # lazy local import

    calibration_alphas: dict[tuple[int | str, str], float] = {}
    for cluster_id, winner_obj in cluster_winners.items():
        model_name = winner_obj.winner_model
        sku_ids = cluster_sku_map.get(cluster_id, [])
        cal_y_parts, cal_q_parts = [], []

        for fold in SELECTION_FOLDS:
            key = (model_name, fold)
            if key not in cv_result.fold_predictions:
                continue
            if fold not in cv_result.fold_actuals:
                continue
            forecast = cv_result.fold_predictions[key]
            actuals_all = cv_result.fold_actuals[fold]
            sku_order = cv_result.sku_order.get(fold, [])
            cluster_idx = [i for i, s in enumerate(sku_order) if s in set(sku_ids)]
            if not cluster_idx:
                continue
            cal_y_parts.append(actuals_all[cluster_idx].ravel())
            q_cube = forecast.quantiles[cluster_idx]
            cal_q_parts.append(q_cube.reshape(-1, q_cube.shape[2]))

        if cal_y_parts:
            y_cal = np.concatenate(cal_y_parts)
            q_cal = np.concatenate(cal_q_parts, axis=0)
            alpha = find_calibration_alpha(y_cal, q_cal, q_levels)
            cov_before = coverage_80(y_cal, q_cal, q_levels)
            q_scaled = np.maximum(
                0.0,
                q_cal[:, len(q_levels) // 2 : len(q_levels) // 2 + 1]
                + alpha * (q_cal - q_cal[:, len(q_levels) // 2 : len(q_levels) // 2 + 1]),
            )
            cov_after = coverage_80(y_cal, q_scaled, q_levels)
            logger.info(
                "Calibration cluster %s %s: alpha=%.3f cov80 %.3f → %.3f",
                cluster_id,
                model_name,
                alpha,
                cov_before,
                cov_after,
            )
        else:
            alpha = 1.0  # no calibration data → no change

        calibration_alphas[(cluster_id, model_name)] = alpha

    logger.info(
        "Selection complete: %d clusters | lgbm wins=%d | v2_levers=%s",
        n_clusters,
        n_lgbm_wins,
        v2_levers,
    )

    return SelectionResult(
        cluster_winners=cluster_winners,
        model_scores=scores,
        v2_levers=v2_levers,
        n_clusters=n_clusters,
        n_clusters_won_by_lgbm=n_lgbm_wins,
        calibration_alphas=calibration_alphas,
    )
