"""Foundation model wrappers: Chronos-T5-tiny and Moirai-small.

Design (doc Phase 13):
- Both models are CPU zero-shot (defaults only, no tuning).
- Chronos uses from_samples (generates MC paths → ForecastResult).
- Moirai uses from_quantiles (produces quantile outputs directly → ForecastResult).
- BOTH flow through ForecastResult with no special-casing — this is the
  dual-constructor abstraction test from Phase 8.
- Heavier variants (Chronos-small, TimesFM, Lag-Llama) gated behind
  `use_heavy_models` flag (config default: False).
- Track per-SKU inference timing and report total wall-clock.

Availability:
- Chronos: requires `chronos-forecasting` + `torch`. Installable on this box.
- Moirai:  requires `uni2ts` which needs numpy~=1.26 (needs C compiler, not
  available on this Windows box). Registered but raises ImportError gracefully.

Lazy imports: torch / chronos / uni2ts imported inside methods so registering
this module does not require the heavy deps for unrelated tests.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from forecasting.models.base import ForecastModel, ForecastResult
from forecasting.registry import register_model

logger = logging.getLogger(__name__)

# Fix Windows OpenMP duplicate-library warning from torch + numpy
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Chronos model identifiers
_CHRONOS_TINY = "amazon/chronos-t5-tiny"
_CHRONOS_SMALL = "amazon/chronos-t5-small"  # use_heavy_models=True

# Moirai model identifiers
_MOIRAI_SMALL = "Salesforce/moirai-1.0-R-small"

# Default number of sample paths for Chronos (from_samples)
_CHRONOS_N_SAMPLES: int = 20


# ---------------------------------------------------------------------------
# Timing tracker
# ---------------------------------------------------------------------------


@dataclass
class InferenceTiming:
    """Per-SKU inference timing summary."""

    total_skus: int = 0
    total_seconds: float = 0.0
    per_sku_seconds: list[float] = field(default_factory=list)

    @property
    def mean_seconds_per_sku(self) -> float:
        if not self.per_sku_seconds:
            return 0.0
        return float(np.mean(self.per_sku_seconds))

    @property
    def p95_seconds_per_sku(self) -> float:
        if not self.per_sku_seconds:
            return 0.0
        return float(np.quantile(self.per_sku_seconds, 0.95))

    def log_summary(self, model_name: str) -> None:
        logger.info(
            "%s timing: %d SKUs | total=%.1fs | mean/SKU=%.3fs | P95/SKU=%.3fs",
            model_name,
            self.total_skus,
            self.total_seconds,
            self.mean_seconds_per_sku,
            self.p95_seconds_per_sku,
        )


# ---------------------------------------------------------------------------
# Chronos-T5-tiny  (from_samples)
# ---------------------------------------------------------------------------


@register_model(
    "chronos_tiny",
    segments=["smooth", "erratic", "intermittent", "lumpy", "cold_start"],
)
class ChronosTiny(ForecastModel):
    """Chronos-T5-tiny zero-shot forecaster.

    Generates MC sample paths → ForecastResult.from_samples().
    The Chronos output tensor has shape (batch, n_samples, horizon);
    we transpose to (batch, horizon, n_samples) before feeding from_samples().
    """

    def __init__(
        self,
        q_levels: np.ndarray | None = None,
        n_samples: int = _CHRONOS_N_SAMPLES,
        use_heavy_models: bool = False,
        batch_size: int = 8,
    ) -> None:
        super().__init__(q_levels)
        self._model_id = _CHRONOS_SMALL if use_heavy_models else _CHRONOS_TINY
        self.n_samples = n_samples
        self.batch_size = batch_size
        self._pipeline = None
        self._sku_series: dict[str, np.ndarray] = {}
        self.timing = InferenceTiming()

    def fit(self, X: np.ndarray, y: np.ndarray) -> ChronosTiny:  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> ChronosTiny:
        """Store training series; Chronos is zero-shot so no fitting occurs."""
        try:
            import torch  # lazy  # noqa: F401
            from chronos import ChronosPipeline  # lazy
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting and torch are required for ChronosTiny. "
                "Install with: pip install chronos-forecasting"
            ) from exc

        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        if self._pipeline is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._pipeline = ChronosPipeline.from_pretrained(
                    self._model_id,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                )
        self._fitted = True
        logger.info("ChronosTiny: loaded %s, %d SKUs", self._model_id, len(series_dict))
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:  # noqa: ARG002
        """Run Chronos zero-shot inference in batches."""
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("Call fit_series() before predict().")

        import torch  # lazy

        uid_order = sorted(self._sku_series.keys())
        n_sku = len(uid_order)
        # Accumulate samples: (n_sku, horizon, n_samples)
        all_samples = np.zeros((n_sku, horizon, self.n_samples))

        t_total_start = time.perf_counter()

        for batch_start in range(0, n_sku, self.batch_size):
            batch_ids = uid_order[batch_start : batch_start + self.batch_size]
            contexts = [
                torch.tensor(self._sku_series[uid], dtype=torch.float32) for uid in batch_ids
            ]
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # samples: (batch, n_samples, horizon)
                raw = self._pipeline.predict(
                    contexts,
                    prediction_length=horizon,
                    num_samples=self.n_samples,
                    limit_prediction_length=False,
                )
            t1 = time.perf_counter()
            batch_time = t1 - t0
            per_sku = batch_time / len(batch_ids)
            self.timing.per_sku_seconds.extend([per_sku] * len(batch_ids))

            # Transpose → (batch, horizon, n_samples)
            batch_samples = np.transpose(raw.numpy(), (0, 2, 1))
            for j, uid in enumerate(batch_ids):
                i = uid_order.index(uid)
                all_samples[i] = batch_samples[j]

        self.timing.total_seconds = time.perf_counter() - t_total_start
        self.timing.total_skus = n_sku
        self.timing.log_summary("ChronosTiny")

        return ForecastResult.from_samples(all_samples, self.q_levels)


# ---------------------------------------------------------------------------
# Moirai-small  (from_quantiles)
# ---------------------------------------------------------------------------


@register_model(
    "moirai_small",
    segments=["smooth", "erratic", "intermittent", "lumpy", "cold_start"],
)
class MoiraiSmall(ForecastModel):
    """Moirai-small zero-shot forecaster.

    Produces quantile outputs directly → ForecastResult.from_quantiles().
    Requires uni2ts (Salesforce) which needs numpy~=1.26 + C compiler.
    On systems where uni2ts is unavailable, fit_series() raises ImportError
    with a clear install message — the model is registered but not runnable.
    """

    def __init__(
        self,
        q_levels: np.ndarray | None = None,
        use_heavy_models: bool = False,
        patch_size: int = 32,
    ) -> None:
        super().__init__(q_levels)
        self._model_id = _MOIRAI_SMALL  # only small in v1; heavy gated by flag
        self.patch_size = patch_size
        self._sku_series: dict[str, np.ndarray] = {}
        self.timing = InferenceTiming()

    def fit(self, X: np.ndarray, y: np.ndarray) -> MoiraiSmall:  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> MoiraiSmall:
        """Store training series; Moirai is zero-shot so no fitting occurs."""
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # lazy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "uni2ts is required for MoiraiSmall but is not installed on this system. "
                "It requires numpy~=1.26 and a C compiler. "
                "Install with: pip install git+https://github.com/SalesforceAIResearch/uni2ts.git"
            ) from exc

        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        self._fitted = True
        logger.info("MoiraiSmall: stored %d SKUs (zero-shot)", len(series_dict))
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:  # noqa: ARG002
        """Run Moirai zero-shot inference."""
        if not self._fitted:
            raise RuntimeError("Call fit_series() before predict().")

        try:
            import torch  # lazy
            from uni2ts.model.moirai import MoiraiForecast  # lazy
        except ImportError as exc:
            raise ImportError("uni2ts / torch required for MoiraiSmall inference.") from exc

        uid_order = sorted(self._sku_series.keys())
        n_sku = len(uid_order)
        n_q = len(self.q_levels)
        q_cube = np.zeros((n_sku, horizon, n_q))

        t_total_start = time.perf_counter()

        model = MoiraiForecast.load_from_checkpoint(
            prediction_length=horizon,
            context_length=max(len(v) for v in self._sku_series.values()),
            patch_size=self.patch_size,
            num_samples=len(self.q_levels),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            checkpoint_path=self._model_id,
        )

        for i, uid in enumerate(uid_order):
            y = torch.tensor(self._sku_series[uid], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            t0 = time.perf_counter()
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = model(past_target=y)
            t1 = time.perf_counter()
            self.timing.per_sku_seconds.append(t1 - t0)

            # forecast: (1, n_samples, horizon) → take as quantiles
            preds = forecast.squeeze(0).numpy()  # (n_samples, horizon)
            # Sort sample axis → treat as empirical quantile levels
            preds_sorted = np.sort(preds, axis=0)  # (n_q, horizon)
            # Map to our q_levels by interpolating at evenly-spaced sample quantiles
            sample_quantiles = np.linspace(0, 1, preds.shape[0])
            for qi, q in enumerate(self.q_levels):
                idx = int(np.searchsorted(sample_quantiles, q))
                idx = min(idx, preds.shape[0] - 1)
                q_cube[i, :, qi] = preds_sorted[idx]

        self.timing.total_seconds = time.perf_counter() - t_total_start
        self.timing.total_skus = n_sku
        self.timing.log_summary("MoiraiSmall")

        return ForecastResult.from_quantiles(q_cube, self.q_levels)
