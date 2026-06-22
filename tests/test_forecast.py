"""Phase 16 gate — forecast.py: final forecast + calibration + manifest."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from forecasting import config
from forecasting.forecast import ForecastArtifacts, _build_manifest, generate_forecast
from forecasting.hierarchy import HierarchyNode, HierarchyResult
from forecasting.segment import SegmentResult
from forecasting.selection import ClusterWinner, SelectionResult

# ---------------------------------------------------------------------------
# Minimal synthetic fixtures
# ---------------------------------------------------------------------------


def _make_tiny_hierarchy() -> HierarchyResult:
    nodes = [
        HierarchyNode("L0_total", 0, "total", None),
        HierarchyNode("L1_Candles", 1, "Candles", "L0_total"),
        HierarchyNode("L2_1001", 2, "1001", "L1_Candles"),
        HierarchyNode("L2_1002", 2, "1002", "L1_Candles"),
    ]
    bottom_ids = ["L2_1001", "L2_1002"]
    rows = [0, 0, 1, 1, 2, 3]
    cols = [0, 1, 0, 1, 0, 1]
    data = [1.0] * 6
    S = sp.csr_matrix((data, (rows, cols)), shape=(4, 2), dtype=np.float32)
    return HierarchyResult(
        nodes=nodes,
        bottom_ids=bottom_ids,
        S=S,
        level_counts={0: 1, 1: 1, 2: 2},
        sku_to_node={1001: "L2_1001", 1002: "L2_1002"},
        node_df=pd.DataFrame(
            [
                {
                    "node_id": n.node_id,
                    "level": n.level,
                    "label": n.label,
                    "parent_id": n.parent_id or "",
                }
                for n in nodes
            ]
        ),
    )


def _make_tiny_segments() -> SegmentResult:
    seg_df = pd.DataFrame(
        {
            config.COL_SKU_ID: [1001, 1002],
            "cluster_id": [0, 0],
            "sb_class": ["smooth", "smooth"],
            "revenue_tier": ["B", "B"],
            "idi": [2.0, 2.0],
            "cv2": [0.3, 0.3],
            "zero_rate": [0.3, 0.3],
            "silhouette_score": [0.4, 0.4],
            "best_k": [1, 1],
            "used_fallback_k": [False, False],
        }
    )
    return SegmentResult(segments=seg_df, selected_k=1, used_fallback=False, best_silhouette=0.4)


def _make_tiny_selection() -> SelectionResult:
    winners = {
        0: ClusterWinner(
            cluster_id=0,
            winner_model="seasonal_naive",
            winner_crps=0.3,
            winner_cov80=0.80,
            baseline_crps=0.4,
            beats_baseline=True,
        )
    }
    return SelectionResult(
        cluster_winners=winners,
        model_scores=[],
        v2_levers={"segment_as_cluster": False, "post_hoc_conformal": False},
        n_clusters=1,
        n_clusters_won_by_lgbm=0,
        calibration_alphas={(0, "seasonal_naive"): 1.2},
    )


def _make_tiny_dense(n_weeks: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    ts = pd.date_range("2024-01-06", periods=n_weeks, freq="W-SAT")
    for sku in [1001, 1002]:
        for dt in ts:
            rows.append(
                {
                    config.COL_SKU_ID: sku,
                    config.COL_TIMESTAMP: dt,
                    config.COL_SALES: float(rng.exponential(10) * rng.binomial(1, 0.6)),
                }
            )
    return pd.DataFrame(rows)


def _make_tiny_features(n_weeks: int = 60) -> pd.DataFrame:
    """Minimal feature df — only needs sku_id and timestamp for forecast.py."""
    dense = _make_tiny_dense(n_weeks)
    return dense


# ---------------------------------------------------------------------------
# _build_manifest
# ---------------------------------------------------------------------------


def test_build_manifest_keys() -> None:
    sel = _make_tiny_selection()
    cube = np.random.default_rng(0).random((2, 4, 19))
    m = _build_manifest(
        forecast_origin=pd.Timestamp("2026-05-23"),
        horizon=4,
        n_sku=2,
        q_levels=np.array(config.QUANTILES),
        selection=sel,
        forecast_cube=cube,
    )
    required = {
        "pipeline_version",
        "random_seed",
        "forecast_origin",
        "horizon_weeks",
        "n_sku",
        "forecast_cube_hash",
        "cluster_winners",
        "calibration_alphas",
    }
    assert required.issubset(m.keys())


def test_build_manifest_cube_hash_deterministic() -> None:
    sel = _make_tiny_selection()
    cube = np.ones((2, 4, 19))
    m1 = _build_manifest(pd.Timestamp("2026-05-23"), 4, 2, np.array(config.QUANTILES), sel, cube)
    m2 = _build_manifest(pd.Timestamp("2026-05-23"), 4, 2, np.array(config.QUANTILES), sel, cube)
    assert m1["forecast_cube_hash"] == m2["forecast_cube_hash"]


def test_build_manifest_different_cubes_different_hash() -> None:
    sel = _make_tiny_selection()
    cube1 = np.ones((2, 4, 19))
    cube2 = np.zeros((2, 4, 19))
    m1 = _build_manifest(pd.Timestamp("2026-05-23"), 4, 2, np.array(config.QUANTILES), sel, cube1)
    m2 = _build_manifest(pd.Timestamp("2026-05-23"), 4, 2, np.array(config.QUANTILES), sel, cube2)
    assert m1["forecast_cube_hash"] != m2["forecast_cube_hash"]


# ---------------------------------------------------------------------------
# generate_forecast
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def forecast_artifacts() -> ForecastArtifacts:
    """Run generate_forecast on minimal synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hier = _make_tiny_hierarchy()
        segs = _make_tiny_segments()
        sel = _make_tiny_selection()
        dense = _make_tiny_dense(60)
        features = _make_tiny_features(60)

        arts = generate_forecast(
            full_features=features,
            full_dense=dense,
            selection=sel,
            segments=segs,
            hierarchy=hier,
            q_levels=np.array([0.1, 0.5, 0.9]),
            horizon=4,
            forecast_origin=dense[config.COL_TIMESTAMP].max(),
            bootstrap_samples=50,
            output_dir=Path(tmpdir),
        )
    return arts


def test_forecast_cube_shape(forecast_artifacts: ForecastArtifacts) -> None:
    assert forecast_artifacts.forecast_cube.shape == (2, 4, 3)


def test_forecast_cube_nonneg(forecast_artifacts: ForecastArtifacts) -> None:
    assert (forecast_artifacts.forecast_cube >= 0).all()


def test_forecast_sku_order(forecast_artifacts: ForecastArtifacts) -> None:
    assert len(forecast_artifacts.sku_order) == 2
    assert set(forecast_artifacts.sku_order) == {1001, 1002}


def test_forecast_horizon_dates_length(forecast_artifacts: ForecastArtifacts) -> None:
    assert len(forecast_artifacts.horizon_dates) == 4


def test_forecast_horizon_dates_are_saturday(forecast_artifacts: ForecastArtifacts) -> None:
    for dt in forecast_artifacts.horizon_dates:
        assert dt.day_of_week == 5, f"{dt} is not Saturday"


def test_forecast_reconciled_is_dataframe(forecast_artifacts: ForecastArtifacts) -> None:
    assert isinstance(forecast_artifacts.reconciled, pd.DataFrame)
    assert len(forecast_artifacts.reconciled) > 0


def test_forecast_manifest_present(forecast_artifacts: ForecastArtifacts) -> None:
    assert isinstance(forecast_artifacts.manifest, dict)
    assert "forecast_cube_hash" in forecast_artifacts.manifest
    assert "cluster_winners" in forecast_artifacts.manifest


def test_forecast_output_files_written() -> None:
    """generate_forecast must write forecast_final.csv and manifest.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hier = _make_tiny_hierarchy()
        segs = _make_tiny_segments()
        sel = _make_tiny_selection()
        dense = _make_tiny_dense(60)

        generate_forecast(
            full_features=dense,
            full_dense=dense,
            selection=sel,
            segments=segs,
            hierarchy=hier,
            q_levels=np.array([0.1, 0.5, 0.9]),
            horizon=4,
            output_dir=Path(tmpdir),
            bootstrap_samples=30,
        )

        assert (Path(tmpdir) / "forecast_final.csv").exists()
        assert (Path(tmpdir) / "manifest.json").exists()
        assert (Path(tmpdir) / "forecast_hierarchy.parquet").exists()


def test_manifest_json_valid() -> None:
    """manifest.json must be valid JSON with required keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hier = _make_tiny_hierarchy()
        segs = _make_tiny_segments()
        sel = _make_tiny_selection()
        dense = _make_tiny_dense(60)

        generate_forecast(
            full_features=dense,
            full_dense=dense,
            selection=sel,
            segments=segs,
            hierarchy=hier,
            q_levels=np.array([0.1, 0.5, 0.9]),
            horizon=4,
            output_dir=Path(tmpdir),
            bootstrap_samples=30,
        )

        manifest_path = Path(tmpdir) / "manifest.json"
        with open(manifest_path) as f:
            m = json.load(f)

        assert "forecast_origin" in m
        assert "pipeline_version" in m
        assert "random_seed" in m


def test_forecast_csv_columns() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        hier = _make_tiny_hierarchy()
        segs = _make_tiny_segments()
        sel = _make_tiny_selection()
        dense = _make_tiny_dense(60)

        generate_forecast(
            full_features=dense,
            full_dense=dense,
            selection=sel,
            segments=segs,
            hierarchy=hier,
            q_levels=np.array([0.1, 0.5, 0.9]),
            horizon=4,
            output_dir=Path(tmpdir),
            bootstrap_samples=30,
        )

        csv_df = pd.read_csv(Path(tmpdir) / "forecast_final.csv")
        required = {"sku_id", "forecast_date", "p10", "p50", "p90"}
        assert required.issubset(csv_df.columns)
        assert len(csv_df) == 2 * 4  # 2 SKUs × 4 horizon weeks
        assert (csv_df["p10"] >= 0).all()
        assert (csv_df["p50"] >= csv_df["p10"] - 1e-6).all()
        assert (csv_df["p90"] >= csv_df["p50"] - 1e-6).all()
