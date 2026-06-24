"""PatchTST model for trend-extrapolation forecasting.

PatchTST (Nie et al. 2022) splits the time series into patches and uses
a Transformer encoder to learn patch-level representations. It is especially
good at capturing medium-term trends and structural breaks — exactly what
SKU 34778233372834 needs (demand ramp from 91 to 1097 over 26 weeks).

Design:
- Patch length: 4 weeks (monthly granularity)
- Stride: 2 weeks (50% overlap)
- Context length: min(104, available_history) weeks
- Prediction length: horizon (26 weeks)
- Model trained from scratch per fold on ALL eligible SKUs jointly
  (global model — shares weights across SKUs)
- Quantiles: bootstrap residuals from in-sample fit

Registered for: erratic, smooth_growing, smooth_stable, promo_driven, cold_start
(not lumpy/intermittent — PatchTST needs reasonably dense series)

CPU training time: ~2-5 min for 50 SKUs × 200 weeks context
"""

from __future__ import annotations

import os
import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from forecasting.models.base import ForecastModel, ForecastResult
from forecasting.registry import register_model

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ── Hyperparameters ───────────────────────────────────────────────────────────
PATCH_LEN   = 4      # weeks per patch
STRIDE      = 2      # patch stride
CONTEXT_LEN = 104    # max context (2 years)
D_MODEL     = 64     # transformer hidden dim (small = fast on CPU)
N_HEADS     = 4
N_LAYERS    = 2
D_FF        = 128
DROPOUT     = 0.1
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 1e-3
MIN_SERIES  = 26     # minimum weeks to include in training


@register_model(
    "patchtst",
    segments=["erratic", "smooth_growing", "smooth_stable", "smooth",
              "promo_driven", "cold_start", "intermittent"],
)
class PatchTSTModel(ForecastModel):
    """Global PatchTST: trains one Transformer on all eligible SKUs jointly.

    Training is done on the full dict of series passed to fit_series().
    Prediction generates point forecasts + residual-based quantile intervals.
    """

    def __init__(
        self,
        q_levels: np.ndarray | None = None,
        horizon: int = 26,
        context_len: int = CONTEXT_LEN,
        patch_len: int = PATCH_LEN,
        stride: int = STRIDE,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LR,
    ) -> None:
        super().__init__(q_levels)
        self.horizon     = horizon
        self.context_len = context_len
        self.patch_len   = patch_len
        self.stride      = stride
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_layers    = n_layers
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr

        self._model      = None
        self._scaler_mean: dict[str, float] = {}
        self._scaler_std:  dict[str, float] = {}
        self._residuals:   dict[str, np.ndarray] = {}
        self._sku_series:  dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PatchTSTModel":  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> "PatchTSTModel":
        """Train a global PatchTST on all provided series."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._sku_series = {k: v.copy() for k, v in series_dict.items()}

        # Filter: only series with enough history
        eligible = {
            k: v for k, v in series_dict.items()
            if len(v) >= MIN_SERIES and v.sum() > 0
        }
        if not eligible:
            logger.warning("PatchTST: no eligible series — skipping training")
            self._fitted = True
            return self

        # Normalize per-SKU (zero-mean unit-variance on training window)
        normed = {}
        for uid, y in eligible.items():
            mu  = float(y.mean())
            std = float(y.std()) if y.std() > 1e-6 else 1.0
            self._scaler_mean[uid] = mu
            self._scaler_std[uid]  = std
            normed[uid] = (y - mu) / std

        # Build (X_ctx, y_tgt) pairs for all SKUs
        ctx  = min(self.context_len, min(len(v) for v in normed.values()))
        ctx  = max(ctx, self.patch_len * 2)

        X_list, y_list = [], []
        for uid, y_n in normed.items():
            T = len(y_n)
            max_start = T - ctx - self.horizon
            if max_start < 0:
                # Short series: pad left with zeros
                y_padded = np.concatenate([np.zeros(ctx + self.horizon - T), y_n])
                X_list.append(y_padded[:ctx])
                y_list.append(y_padded[ctx:ctx + self.horizon])
            else:
                # Sample up to 3 windows per SKU
                starts = np.linspace(0, max_start, min(3, max_start + 1), dtype=int)
                for s in starts:
                    X_list.append(y_n[s:s + ctx])
                    y_list.append(y_n[s + ctx:s + ctx + self.horizon])

        if not X_list:
            self._fitted = True
            return self

        X_arr = torch.tensor(np.stack(X_list), dtype=torch.float32).unsqueeze(-1)
        y_arr = torch.tensor(np.stack(y_list), dtype=torch.float32)

        dataset = TensorDataset(X_arr, y_arr)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build PatchTST via transformers library
        try:
            self._model = self._build_model(ctx)
            optimizer   = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )

            self._model.train()
            for epoch in range(self.epochs):
                total_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    # PatchTST expects (batch, seq_len, n_vars)
                    out = self._model(past_values=xb)
                    pred = out.prediction_outputs  # (batch, horizon, n_vars)
                    loss = nn.MSELoss()(pred.squeeze(-1), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()
                if (epoch + 1) % 5 == 0:
                    logger.info(
                        "PatchTST epoch %d/%d loss=%.4f",
                        epoch + 1, self.epochs, total_loss / len(loader)
                    )

            # Compute in-sample residuals for conformal intervals
            self._model.eval()
            with torch.no_grad():
                for uid, y_n in normed.items():
                    T = len(y_n)
                    if T < ctx + self.horizon:
                        self._residuals[uid] = np.array([0.1])
                        continue
                    x_in = torch.tensor(
                        y_n[-(ctx + self.horizon):-self.horizon],
                        dtype=torch.float32
                    ).reshape(1, ctx, 1)
                    out   = self._model(past_values=x_in)
                    pred  = out.prediction_outputs.squeeze().numpy()
                    actual= y_n[-self.horizon:]
                    self._residuals[uid] = np.abs(actual - pred)

        except Exception as e:
            logger.warning("PatchTST training failed: %s — falling back to SeasonalNaive", e)
            self._model = None

        self._fitted = True
        return self

    def _build_model(self, context_len: int):
        """Build PatchTST model using HuggingFace transformers."""
        from transformers import PatchTSTConfig, PatchTSTForPrediction

        n_patches = (context_len - self.patch_len) // self.stride + 1

        config = PatchTSTConfig(
            num_input_channels=1,
            context_length=context_len,
            patch_length=self.patch_len,
            stride=self.stride,
            prediction_length=self.horizon,
            d_model=self.d_model,
            num_attention_heads=self.n_heads,
            num_hidden_layers=self.n_layers,
            ffn_dim=D_FF,
            dropout=DROPOUT,
            head_dropout=DROPOUT,
            pooling_type="mean",
            channel_attention=False,
            scaling="std",
            loss="mse",
        )
        return PatchTSTForPrediction(config)

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:  # noqa: ARG002
        import torch

        uid_order = sorted(self._sku_series.keys())
        n_sku = len(uid_order)
        n_q   = len(self.q_levels)
        q_cube = np.zeros((n_sku, horizon, n_q))

        ctx = min(self.context_len, min(
            len(v) for v in self._sku_series.values() if len(v) > 0
        ) if self._sku_series else self.context_len)
        ctx = max(ctx, self.patch_len * 2)

        for i, uid in enumerate(uid_order):
            y = self._sku_series[uid]
            if len(y) == 0:
                continue

            mu  = self._scaler_mean.get(uid, float(y.mean()))
            std = self._scaler_std.get(uid, max(float(y.std()), 1e-6))
            res = self._residuals.get(uid, np.array([std * 0.5]))

            if self._model is not None:
                try:
                    y_n = (y - mu) / std
                    T   = len(y_n)
                    if T >= ctx:
                        x_in = torch.tensor(
                            y_n[-ctx:], dtype=torch.float32
                        ).reshape(1, ctx, 1)
                    else:
                        pad  = np.zeros(ctx - T)
                        x_in = torch.tensor(
                            np.concatenate([pad, y_n]),
                            dtype=torch.float32
                        ).reshape(1, ctx, 1)

                    self._model.eval()
                    with torch.no_grad():
                        out  = self._model(past_values=x_in)
                        pred = out.prediction_outputs.squeeze().numpy()

                    # Denormalize
                    point = np.maximum(0.0, pred * std + mu)
                except Exception:
                    # Fallback to SeasonalNaive-style
                    point = self._sn_fallback(y, horizon)
            else:
                point = self._sn_fallback(y, horizon)

            # Conformal quantile intervals
            for qi, q in enumerate(self.q_levels):
                if abs(q - 0.5) < 1e-9:
                    q_cube[i, :, qi] = point[:horizon]
                elif q > 0.5:
                    thresh = float(np.quantile(res, q))
                    q_cube[i, :, qi] = point[:horizon] + thresh * std
                else:
                    thresh = float(np.quantile(res, 1.0 - q))
                    q_cube[i, :, qi] = np.maximum(
                        0.0, point[:horizon] - thresh * std
                    )

        return ForecastResult.from_quantiles(q_cube, self.q_levels)

    @staticmethod
    def _sn_fallback(y: np.ndarray, horizon: int) -> np.ndarray:
        """SeasonalNaive fallback when model not available."""
        if len(y) >= 52:
            t = y[-52:]
            return np.array([max(0.0, t[h % 52]) for h in range(horizon)])
        return np.full(horizon, max(0.0, float(y.mean())))
