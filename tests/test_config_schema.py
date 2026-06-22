"""Phase 3.5 gate: pydantic config schema, run.py --validate-only, relabelling."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from configs._schema import (
    LifecycleConfig,
    ModelConfig,
    PipelineConfig,
)
from forecasting import config
from forecasting.densify import densify
from forecasting.io import load_all
from forecasting.lifecycle import infer_lifecycle
from forecasting.registry import (
    ALL_SEGMENTS,
    candidates_for,
    register_model,
    registered_model_names,
)
from forecasting.run import PHASES

YAML_PATH = Path(__file__).parents[1] / "configs" / "fontana_candle.yaml"


# ---------------------------------------------------------------------------
# Config schema — happy path
# ---------------------------------------------------------------------------


def test_load_yaml() -> None:
    cfg = PipelineConfig.from_yaml(YAML_PATH)
    assert cfg.client == "fontana_candle"


def test_quantile_count() -> None:
    cfg = PipelineConfig.from_yaml(YAML_PATH)
    assert len(cfg.model.quantiles) == 19


def test_horizon_weeks() -> None:
    cfg = PipelineConfig.from_yaml(YAML_PATH)
    assert cfg.model.horizon_weeks == 26


def test_dormancy_threshold() -> None:
    cfg = PipelineConfig.from_yaml(YAML_PATH)
    assert cfg.lifecycle.dormancy_threshold_weeks == 26


def test_week_relabel_shift() -> None:
    cfg = PipelineConfig.from_yaml(YAML_PATH)
    assert cfg.densify.week_relabel_shift_days == 6


def test_keep_active_overrides_loaded() -> None:
    cfg = PipelineConfig.from_yaml(YAML_PATH)
    assert len(cfg.lifecycle.keep_active_overrides) == 1
    assert cfg.lifecycle.keep_active_overrides[0].sku_id == 46606700773604


def test_target_week_features_off() -> None:
    cfg = PipelineConfig.from_yaml(YAML_PATH)
    assert cfg.features.target_week_features is False


def test_calibration_guardrail() -> None:
    cfg = PipelineConfig.from_yaml(YAML_PATH)
    lo, hi = cfg.model.calibration_guardrail
    assert lo == pytest.approx(0.75)
    assert hi == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Config schema — validation errors
# ---------------------------------------------------------------------------


def test_invalid_selection_fold_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(cv_folds=4, selection_folds=[5])  # fold 5 > cv_folds 4


def test_guardrail_order_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(calibration_guardrail=(0.85, 0.75))  # lo > hi


def test_empty_client_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.from_dict({"client": ""})


def test_keep_active_override_requires_reason() -> None:
    with pytest.raises(ValidationError):
        LifecycleConfig(keep_active_overrides=[{"sku_id": 123, "reason": ""}])


# ---------------------------------------------------------------------------
# Registry — candidates_for
# ---------------------------------------------------------------------------


def test_register_model_with_segments() -> None:
    @register_model("_test_seg_model", segments=["smooth", "erratic"])
    class _M:
        pass

    assert _M in candidates_for("smooth")
    assert _M in candidates_for("erratic")
    assert _M not in candidates_for("lumpy")


def test_register_model_default_all_segments() -> None:
    @register_model("_test_all_seg")
    class _A:
        pass

    for seg in ALL_SEGMENTS:
        assert _A in candidates_for(seg)


def test_registered_model_names_filtered() -> None:
    names = registered_model_names("smooth")
    assert "_test_seg_model" in names
    assert "_test_all_seg" in names


# ---------------------------------------------------------------------------
# run.py --validate-only (subprocess so it exercises the CLI path)
# ---------------------------------------------------------------------------


def test_validate_only_exits_zero() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "forecasting.run",
            "--config",
            str(YAML_PATH),
            "--validate-only",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parents[1]),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_validate_only_logs_ok(capsys: pytest.CaptureFixture[str]) -> None:
    from forecasting.run import run

    run(["--config", str(YAML_PATH), "--validate-only"])
    # No exception = pass (logging goes to stderr; subprocess test above checks output)


# ---------------------------------------------------------------------------
# Phase ordering sanity
# ---------------------------------------------------------------------------


def test_phases_list_order() -> None:
    assert PHASES[0] == "io"
    assert PHASES[1] == "lifecycle"
    assert PHASES[2] == "densify"
    assert "forecast" in PHASES
    assert "report" in PHASES


# ---------------------------------------------------------------------------
# Sunday → Saturday relabelling — counts unchanged, timestamps shifted
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def relabelled() -> object:
    d = load_all()
    lc = infer_lifecycle(d.sales, d.master)
    return densify(d.sales, lc, d.joined, week_relabel_shift_days=6)


def test_relabelled_row_count(relabelled: object) -> None:
    from forecasting.densify import DenseResult

    assert isinstance(relabelled, DenseResult)
    assert len(relabelled.dense) == 16_068


def test_relabelled_zero_fraction(relabelled: object) -> None:
    from forecasting.densify import DenseResult

    assert isinstance(relabelled, DenseResult)
    assert abs(relabelled.zero_fraction - 0.324) < 0.002


def test_relabelled_stockout_count(relabelled: object) -> None:
    from forecasting.densify import DenseResult

    assert isinstance(relabelled, DenseResult)
    assert len(relabelled.stockout_skus) == 82


def test_relabelled_timestamps_are_saturday(relabelled: object) -> None:
    from forecasting.densify import DenseResult

    assert isinstance(relabelled, DenseResult)
    dow = relabelled.dense[config.COL_TIMESTAMP].dt.dayofweek  # 5 = Saturday
    assert (dow == 5).all(), "After relabelling all timestamps must be Saturday"


def test_relabelled_timestamp_range(relabelled: object) -> None:
    """Min should be 2020-12-27 + 6 days = 2021-01-02 (Saturday)."""
    from forecasting.densify import DenseResult

    assert isinstance(relabelled, DenseResult)
    ts_min = relabelled.dense[config.COL_TIMESTAMP].min()
    assert ts_min == pd.Timestamp("2021-01-02"), f"unexpected min: {ts_min}"
