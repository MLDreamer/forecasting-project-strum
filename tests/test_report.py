"""Phase 17 gate: report.py — executive markdown report generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting import config
from forecasting.forecast import ForecastArtifacts
from forecasting.report import (
    _section_calibration_alphas,
    _section_calibration_limitations,
    _section_clustering_limitations,
    _section_cold_start_ablation,
    _section_cv_performance,
    _section_executive_summary,
    _section_known_limitations,
    generate_report,
)
from forecasting.segment import SegmentResult
from forecasting.selection import ClusterWinner, SelectionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_segments(n_sku: int = 6) -> SegmentResult:
    seg_df = pd.DataFrame(
        {
            config.COL_SKU_ID: list(range(1001, 1001 + n_sku)),
            "cluster_id": [i % 2 for i in range(n_sku)],
            "sb_class": [
                "smooth",
                "erratic",
                "lumpy",
                "intermittent",
                "cold_start",
                "discontinued",
            ],
            "revenue_tier": ["A", "B", "B", "C", "C", "C"],
            "idi": [1.5] * n_sku,
            "cv2": [0.3] * n_sku,
            "zero_rate": [0.3] * n_sku,
            "silhouette_score": [0.36] * n_sku,
            "best_k": [2] * n_sku,
            "used_fallback_k": [False] * n_sku,
        }
    )
    return SegmentResult(segments=seg_df, selected_k=2, used_fallback=False, best_silhouette=0.361)


def _make_selection() -> SelectionResult:
    winners = {
        0: ClusterWinner(0, "seasonal_naive", 0.35, 0.78, 0.40, True),
        1: ClusterWinner(
            1, "auto_ets", 0.28, 0.81, 0.45, True, guardrail_rejected=["cluster_lgbm"]
        ),
    }
    return SelectionResult(
        cluster_winners=winners,
        model_scores=[],
        v2_levers={"segment_as_cluster": True, "post_hoc_conformal": False},
        n_clusters=2,
        n_clusters_won_by_lgbm=0,
        calibration_alphas={(0, "seasonal_naive"): 1.3, (1, "auto_ets"): 1.1},
    )


def _make_artifacts(n_sku: int = 6) -> ForecastArtifacts:
    return ForecastArtifacts(
        forecast_cube=np.zeros((n_sku, 4, 3)),
        sku_order=list(range(1001, 1001 + n_sku)),
        q_levels=np.array([0.1, 0.5, 0.9]),
        horizon_dates=[pd.Timestamp("2026-06-07") + pd.Timedelta(weeks=h) for h in range(4)],
        reconciled=pd.DataFrame(),
        manifest={
            "pipeline_version": "0.1.0",
            "forecast_origin": "2026-05-23",
            "forecast_cube_hash": "abc123",
            "cluster_winners": {"0": "seasonal_naive", "1": "auto_ets"},
            "calibration_alphas": {"0:seasonal_naive": 1.3, "1:auto_ets": 1.1},
        },
    )


# ---------------------------------------------------------------------------
# Section tests
# ---------------------------------------------------------------------------


def test_section_executive_summary_contains_sku_count() -> None:
    arts = _make_artifacts()
    sel = _make_selection()
    segs = _make_segments()
    text = _section_executive_summary(arts, sel, segs)
    assert "6" in text  # n_sku


def test_section_executive_summary_contains_horizon() -> None:
    arts = _make_artifacts()
    sel = _make_selection()
    segs = _make_segments()
    text = _section_executive_summary(arts, sel, segs)
    assert "4 weeks" in text or "4" in text


def test_section_cluster_winners_contains_all_clusters() -> None:
    from forecasting.report import _section_cluster_winners

    sel = _make_selection()
    text = _section_cluster_winners(sel)
    assert "seasonal_naive" in text
    assert "auto_ets" in text


def test_section_cluster_winners_v2_lever_mentioned() -> None:
    from forecasting.report import _section_cluster_winners

    sel = _make_selection()
    text = _section_cluster_winners(sel)
    # segment_as_cluster lever should be mentioned
    assert "segment" in text.lower()


def test_section_calibration_alphas_table() -> None:
    sel = _make_selection()
    text = _section_calibration_alphas(sel)
    assert "1.300" in text or "1.3" in text
    assert "seasonal_naive" in text


def test_section_calibration_alphas_empty_when_no_alphas() -> None:
    sel = _make_selection()
    sel.calibration_alphas.clear()
    text = _section_calibration_alphas(sel)
    assert text == ""


def test_section_cv_performance_no_cv() -> None:
    text = _section_cv_performance(None)
    assert "not available" in text.lower()


def test_section_clustering_limitations_contains_k() -> None:
    segs = _make_segments()
    text = _section_clustering_limitations(segs)
    assert "K" in text or "cluster" in text.lower()
    assert "young catalog" in text.lower()


def test_section_clustering_limitations_contains_sb_classes() -> None:
    segs = _make_segments()
    text = _section_clustering_limitations(segs)
    for cls in ["smooth", "erratic", "lumpy"]:
        assert cls in text


def test_section_calibration_limitations_contains_root_causes() -> None:
    sel = _make_selection()
    text = _section_calibration_limitations(sel)
    assert "heavy-tail" in text.lower() or "heavy tail" in text.lower()
    assert "cold-start" in text.lower() or "cold start" in text.lower()


def test_section_cold_start_ablation_contains_chronos() -> None:
    sel = _make_selection()
    segs = _make_segments()
    text = _section_cold_start_ablation(sel, segs, None)
    assert "Chronos" in text
    assert "cold_start" in text or "cold-start" in text.lower()


def test_section_known_limitations_has_table() -> None:
    text = _section_known_limitations()
    assert "|" in text  # markdown table
    assert "cluster" in text.lower()
    assert "calibrat" in text.lower()


# ---------------------------------------------------------------------------
# generate_report integration tests
# ---------------------------------------------------------------------------


def test_generate_report_returns_string() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        arts = _make_artifacts()
        sel = _make_selection()
        segs = _make_segments()
        text = generate_report(arts, sel, segs, output_dir=Path(tmpdir))
    assert isinstance(text, str)
    assert len(text) > 100


def test_generate_report_writes_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        arts = _make_artifacts()
        sel = _make_selection()
        segs = _make_segments()
        generate_report(arts, sel, segs, output_dir=Path(tmpdir))
        report_path = Path(tmpdir) / "forecast_report.md"
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 100


def test_generate_report_contains_required_sections() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        arts = _make_artifacts()
        sel = _make_selection()
        segs = _make_segments()
        text = generate_report(arts, sel, segs, output_dir=Path(tmpdir))

    required_sections = [
        "Executive Summary",
        "Model Selection",
        "Calibration",
        "Clustering",
        "Cold-Start",
        "Known Limitations",
    ]
    for section in required_sections:
        assert section in text, f"Missing section: {section}"


def test_generate_report_contains_cube_hash() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        arts = _make_artifacts()
        sel = _make_selection()
        segs = _make_segments()
        text = generate_report(arts, sel, segs, output_dir=Path(tmpdir))
    assert "abc123" in text


def test_generate_report_valid_markdown() -> None:
    """Report must be non-empty and contain header markers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        arts = _make_artifacts()
        sel = _make_selection()
        segs = _make_segments()
        text = generate_report(arts, sel, segs, output_dir=Path(tmpdir))
    assert text.startswith("# ")
    assert "##" in text
