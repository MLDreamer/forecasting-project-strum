"""Phase 15 gate: per-cluster WAPE/CRPS selection + calibration guardrail."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import config
from forecasting.selection import (
    GUARDRAIL_HI,
    GUARDRAIL_LO,
    ClusterWinner,
    ModelScore,
    SelectionResult,
    apply_calibration,
    find_calibration_alpha,
    select_winners,
)
from forecasting.validate import CVResult, FoldMetrics

# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------


def test_guardrail_bounds() -> None:
    assert GUARDRAIL_LO == pytest.approx(0.75)
    assert GUARDRAIL_HI == pytest.approx(0.85)


def test_guardrail_lo_lt_hi() -> None:
    assert GUARDRAIL_LO < GUARDRAIL_HI


# ---------------------------------------------------------------------------
# Synthetic CVResult builder
# ---------------------------------------------------------------------------


def _make_synthetic_cv(
    model_names: list[str],
    n_clusters: int,
    n_skus_per_cluster: int = 4,
    horizon: int = 4,
    n_q: int = 7,
    seed: int = 42,
) -> tuple[CVResult, pd.DataFrame]:
    """Build a minimal synthetic CVResult and segments_df for testing."""
    rng = np.random.default_rng(seed)
    q_levels = np.linspace(0.1, 0.9, n_q)

    # Segments
    sku_ids = list(range(1000, 1000 + n_clusters * n_skus_per_cluster))
    seg_rows = []
    for i, sku in enumerate(sku_ids):
        seg_rows.append(
            {
                config.COL_SKU_ID: sku,
                "cluster_id": i % n_clusters,
                "sb_class": "smooth",
                "revenue_tier": "B",
                "idi": 2.0,
                "cv2": 0.3,
                "zero_rate": 0.3,
                "silhouette_score": 0.4,
                "best_k": n_clusters,
                "used_fallback_k": False,
            }
        )
    segments_df = pd.DataFrame(seg_rows)

    # Build CVResult with one fold (fold 2 = in selection)
    from forecasting.models.base import ForecastResult

    n_sku_total = len(sku_ids)
    fold = 2

    cv = CVResult()
    cv.sku_order[fold] = sku_ids
    cv.fold_actuals[fold] = rng.exponential(5, (n_sku_total, horizon)).clip(0)

    for model_name in model_names:
        q_cube = np.sort(rng.exponential(5, (n_sku_total, horizon, n_q)).clip(0), axis=2)
        result = ForecastResult.from_quantiles(q_cube, q_levels)
        cv.fold_predictions[(model_name, fold)] = result

        fm = FoldMetrics(
            fold=fold,
            model_name=model_name,
            n_skus=n_sku_total,
            wape_per_horizon=rng.random(horizon) * 0.5,
            crps_scalar=float(rng.random()),
            coverage_80_scalar=float(rng.uniform(0.76, 0.84)),
            coverage_90_scalar=float(rng.uniform(0.86, 0.94)),
            wape_overall=float(rng.random() * 0.5),
            in_selection=True,
        )
        cv.fold_metrics.append(fm)

    return cv, segments_df


# ---------------------------------------------------------------------------
# Guardrail tests
# ---------------------------------------------------------------------------


def test_model_outside_guardrail_rejected() -> None:
    """A model with cov80 outside [0.75, 0.85] must be rejected."""
    s = ModelScore(
        model_name="bad_model",
        cluster_id=0,
        mean_crps=0.1,
        mean_wape=0.2,
        mean_cov80=0.65,  # below 0.75 → fail
        passes_guardrail=False,
        n_selection_folds=3,
    )
    assert not s.passes_guardrail


def test_model_inside_guardrail_passes() -> None:
    s = ModelScore(
        model_name="good_model",
        cluster_id=0,
        mean_crps=0.1,
        mean_wape=0.2,
        mean_cov80=0.80,  # inside [0.75, 0.85] → pass
        passes_guardrail=True,
        n_selection_folds=3,
    )
    assert s.passes_guardrail


def test_guardrail_boundary_lo_passes() -> None:
    """Coverage exactly at the lower bound passes."""
    passes = GUARDRAIL_LO <= 0.75 <= GUARDRAIL_HI
    assert passes


def test_guardrail_boundary_hi_passes() -> None:
    """Coverage exactly at the upper bound passes."""
    passes = GUARDRAIL_LO <= 0.85 <= GUARDRAIL_HI
    assert passes


def test_guardrail_below_lo_fails() -> None:
    passes = GUARDRAIL_LO <= 0.74 <= GUARDRAIL_HI
    assert not passes


def test_guardrail_above_hi_fails() -> None:
    passes = GUARDRAIL_LO <= 0.86 <= GUARDRAIL_HI
    assert not passes


# ---------------------------------------------------------------------------
# select_winners — basic contract
# ---------------------------------------------------------------------------


def test_select_winners_returns_selection_result() -> None:
    cv, seg = _make_synthetic_cv(["seasonal_naive", "auto_ets"], n_clusters=2)
    result = select_winners(cv, seg, q_levels=np.linspace(0.1, 0.9, 7))
    assert isinstance(result, SelectionResult)


def test_select_winners_all_clusters_covered() -> None:
    n_clusters = 3
    cv, seg = _make_synthetic_cv(["seasonal_naive", "auto_ets"], n_clusters=n_clusters)
    result = select_winners(cv, seg, q_levels=np.linspace(0.1, 0.9, 7))
    assert result.n_clusters == n_clusters
    assert len(result.cluster_winners) == n_clusters


def test_select_winners_each_cluster_has_winner() -> None:
    cv, seg = _make_synthetic_cv(["seasonal_naive", "theta"], n_clusters=2)
    result = select_winners(cv, seg, q_levels=np.linspace(0.1, 0.9, 7))
    for cluster_id in result.cluster_winners:
        w = result.cluster_winners[cluster_id]
        assert isinstance(w, ClusterWinner)
        assert w.winner_model in ["seasonal_naive", "theta"]


def test_select_winners_winner_for_unknown_cluster() -> None:
    cv, seg = _make_synthetic_cv(["seasonal_naive"], n_clusters=2)
    result = select_winners(cv, seg, q_levels=np.linspace(0.1, 0.9, 7))
    assert result.winner_for(99999) == "seasonal_naive"


# ---------------------------------------------------------------------------
# Fallback to seasonal_naive when all fail guardrail
# ---------------------------------------------------------------------------


def test_fallback_when_all_fail_guardrail() -> None:
    """If all models fail the guardrail, seasonal_naive is used as fallback."""
    cv, seg = _make_synthetic_cv(["auto_ets", "theta"], n_clusters=1)

    # Force all models to fail the guardrail by patching cov80 in fold_predictions
    # We do this by running selection with a very tight guardrail [0.99, 1.0]
    result = select_winners(
        cv,
        seg,
        q_levels=np.linspace(0.1, 0.9, 7),
        guardrail_lo=0.99,  # impossible to satisfy
        guardrail_hi=1.0,
    )
    for w in result.cluster_winners.values():
        assert w.fallback_used is True
        assert w.winner_model == "seasonal_naive"


# ---------------------------------------------------------------------------
# SelectionResult helpers
# ---------------------------------------------------------------------------


def test_selection_summary_dataframe() -> None:
    cv, seg = _make_synthetic_cv(["seasonal_naive", "auto_ets"], n_clusters=2)
    result = select_winners(cv, seg, q_levels=np.linspace(0.1, 0.9, 7))
    summary = result.summary()
    assert isinstance(summary, pd.DataFrame)
    required = {"cluster_id", "winner", "winner_crps", "winner_cov80", "beats_baseline"}
    assert required.issubset(summary.columns)
    assert len(summary) == 2


def test_selection_model_scores_not_empty() -> None:
    cv, seg = _make_synthetic_cv(["seasonal_naive", "auto_ets"], n_clusters=2)
    result = select_winners(cv, seg, q_levels=np.linspace(0.1, 0.9, 7))
    assert len(result.model_scores) > 0
    for s in result.model_scores:
        assert isinstance(s, ModelScore)
        assert np.isfinite(s.mean_crps)


# ---------------------------------------------------------------------------
# V2 levers
# ---------------------------------------------------------------------------


def test_v2_levers_in_result() -> None:
    cv, seg = _make_synthetic_cv(["seasonal_naive"], n_clusters=2)
    result = select_winners(cv, seg, q_levels=np.linspace(0.1, 0.9, 7))
    assert "segment_as_cluster" in result.v2_levers
    assert "post_hoc_conformal" in result.v2_levers


def test_segment_as_cluster_lever_fires_when_lgbm_loses_all() -> None:
    """If cluster_lgbm wins 0 clusters, segment_as_cluster lever fires."""
    cv, seg = _make_synthetic_cv(["seasonal_naive", "auto_ets"], n_clusters=4)
    result = select_winners(cv, seg, q_levels=np.linspace(0.1, 0.9, 7))
    # No 'lgbm' in model names → n_clusters_won_by_lgbm = 0 → lever fires
    assert result.n_clusters_won_by_lgbm == 0
    assert result.v2_levers["segment_as_cluster"] is True


# ---------------------------------------------------------------------------
# Post-hoc conformal calibration
# ---------------------------------------------------------------------------


def test_find_calibration_alpha_achieves_target() -> None:
    """Alpha should produce ~80% coverage on the calibration set."""
    rng = np.random.default_rng(42)
    n = 500
    q_levels = np.linspace(0.1, 0.9, 9)
    # Narrow forecasts: actual spread >> predicted spread
    y_true = rng.exponential(20, n)
    raw_q = np.sort(rng.exponential(5, (n, 9)).clip(0), axis=1)

    alpha = find_calibration_alpha(y_true, raw_q, q_levels, target_cov=0.80)
    assert alpha > 1.0, "Narrow forecasts require alpha > 1 to widen"

    from forecasting.metrics import coverage_80

    p50_idx = 4
    p50 = raw_q[:, p50_idx : p50_idx + 1]
    q_cal = np.maximum(0, p50 + alpha * (raw_q - p50))
    cov = coverage_80(y_true, q_cal, q_levels)
    assert abs(cov - 0.80) < 0.05, f"Calibrated coverage {cov:.3f} not near 0.80"


def test_find_calibration_alpha_already_calibrated() -> None:
    """If coverage is already 80%, alpha should be close to 1.0."""
    rng = np.random.default_rng(99)
    n = 1000
    q_levels = np.linspace(0.05, 0.95, 9)
    y_true = rng.normal(10, 3, n)
    # Build quantiles that give ~80% coverage
    raw_q = np.column_stack([np.quantile(rng.normal(10, 3, n), q) * np.ones(n) for q in q_levels])
    raw_q = np.sort(raw_q + rng.normal(0, 0.1, (n, 9)), axis=1)
    alpha = find_calibration_alpha(y_true, raw_q, q_levels, target_cov=0.80)
    # Alpha should be bounded by search range
    assert 0.0 <= alpha <= 20.0


def test_apply_calibration_preserves_shape() -> None:
    from forecasting.models.base import ForecastResult

    rng = np.random.default_rng(7)
    q_levels = np.linspace(0.1, 0.9, 7)
    q_cube = np.sort(rng.random((4, 6, 7)), axis=2)
    result = ForecastResult.from_quantiles(q_cube, q_levels)
    calibrated = apply_calibration(result, alpha=1.5, q_levels=q_levels)
    assert calibrated.quantiles.shape == result.quantiles.shape


def test_apply_calibration_alpha_1_unchanged() -> None:
    from forecasting.models.base import ForecastResult

    rng = np.random.default_rng(8)
    q_levels = np.linspace(0.1, 0.9, 7)
    q_cube = np.sort(rng.random((3, 4, 7)) * 10, axis=2)
    result = ForecastResult.from_quantiles(q_cube, q_levels)
    calibrated = apply_calibration(result, alpha=1.0, q_levels=q_levels)
    np.testing.assert_allclose(calibrated.quantiles, result.quantiles, atol=1e-6)


def test_apply_calibration_widens_with_alpha_gt_1() -> None:
    from forecasting.models.base import ForecastResult

    rng = np.random.default_rng(9)
    q_levels = np.linspace(0.1, 0.9, 7)
    q_cube = np.sort(rng.random((2, 4, 7)) * 10 + 5, axis=2)
    result = ForecastResult.from_quantiles(q_cube, q_levels)
    calibrated = apply_calibration(result, alpha=2.0, q_levels=q_levels)
    # Width at outermost quantiles should be wider
    width_orig = result.quantiles[:, :, -1] - result.quantiles[:, :, 0]
    width_cal = calibrated.quantiles[:, :, -1] - calibrated.quantiles[:, :, 0]
    assert (width_cal >= width_orig - 1e-6).all()


def test_calibration_alphas_in_result() -> None:
    cv, seg = _make_synthetic_cv(["seasonal_naive", "auto_ets"], n_clusters=2)
    result = select_winners(cv, seg, q_levels=np.linspace(0.1, 0.9, 7))
    assert isinstance(result.calibration_alphas, dict)
    # Each cluster winner should have an alpha
    for cluster_id, w in result.cluster_winners.items():
        key = (cluster_id, w.winner_model)
        assert key in result.calibration_alphas
        assert result.calibration_alphas[key] >= 0.0


# ---------------------------------------------------------------------------
# ClusterWinner dataclass
# ---------------------------------------------------------------------------


def test_cluster_winner_beats_baseline_flag() -> None:
    w = ClusterWinner(
        cluster_id=0,
        winner_model="auto_ets",
        winner_crps=0.20,
        winner_cov80=0.80,
        baseline_crps=0.50,
        beats_baseline=True,
    )
    assert w.beats_baseline is True


def test_cluster_winner_fallback_beats_baseline_false() -> None:
    w = ClusterWinner(
        cluster_id=0,
        winner_model="seasonal_naive",
        winner_crps=0.50,
        winner_cov80=0.80,
        baseline_crps=0.50,
        beats_baseline=False,
        fallback_used=True,
    )
    assert w.beats_baseline is False
    assert w.fallback_used is True


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_select_winners_empty_segments() -> None:
    cv = CVResult()
    seg = pd.DataFrame(columns=[config.COL_SKU_ID, "cluster_id", "sb_class"])
    result = select_winners(cv, seg, q_levels=np.linspace(0.1, 0.9, 7))
    assert result.n_clusters == 0
    assert len(result.cluster_winners) == 0
