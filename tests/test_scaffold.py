"""Phase 0 gate: tree exists and core imports are clean."""

import importlib
from pathlib import Path

MODULES = [
    "forecasting",
    "forecasting.config",
    "forecasting.registry",
    "forecasting.io",
    "forecasting.lifecycle",
    "forecasting.densify",
    "forecasting.features",
    "forecasting.segment",
    "forecasting.hierarchy",
    "forecasting.metrics",
    "forecasting.validate",
    "forecasting.selection",
    "forecasting.reconcile",
    "forecasting.forecast",
    "forecasting.report",
    "forecasting.run",
    "forecasting.models",
    "forecasting.models.base",
    "forecasting.models.baseline",
    "forecasting.models.classical",
    "forecasting.models.intermittent",
    "forecasting.models.ml_global",
    "forecasting.models.tweedie",
    "forecasting.models.foundation",
]


def test_all_modules_importable() -> None:
    for mod in MODULES:
        importlib.import_module(mod)


def test_required_dirs_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "src/forecasting",
        "src/forecasting/models",
        "data/raw",
        "data/interim",
        "data/processed",
        "outputs",
        "configs",
        "tests",
        "app",
    ]
    for d in required:
        assert (root / d).is_dir(), f"Missing directory: {d}"


def test_raw_data_files_present() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    for fname in [
        "processed_data_filtered.csv",
        "product_item_master.csv",
        "variants_export.csv",
    ]:
        assert (raw / fname).exists(), f"Missing raw file: {fname}"


def test_config_paths() -> None:
    from forecasting.config import DATA_INTERIM, DATA_PROCESSED, DATA_RAW, OUTPUTS

    assert DATA_RAW.exists()
    assert DATA_INTERIM.exists()
    assert DATA_PROCESSED.exists()
    assert OUTPUTS.exists()


def test_registry_decorators() -> None:
    from forecasting.registry import (
        FEATURE_REGISTRY,
        MODEL_REGISTRY,
        PLUGIN_REGISTRY,
        register_feature,
        register_model,
        register_plugin,
    )

    @register_model("_test_model")
    class _M:
        pass

    @register_feature("_test_feat")
    def _f() -> None:
        pass

    @register_plugin("_test_plugin")
    class _P:
        pass

    assert "_test_model" in MODEL_REGISTRY
    assert "_test_feat" in FEATURE_REGISTRY
    assert "_test_plugin" in PLUGIN_REGISTRY
