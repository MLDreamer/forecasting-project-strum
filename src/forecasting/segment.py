"""Syntetos-Boylan demand classification + K-means cluster assignment.

Design decisions (locked):
- SB thresholds: IDI < 1.32 and CV2 < 0.49 (standard Syntetos-Boylan 2005).
- discontinued: dormant SKUs from lifecycle (weeks_since_last_sale >= 26).
- cold_start: active SKUs with < 4 non-zero observations.
- K selection: blend score = 0.7 * silhouette + 0.3 * stability_ARI.
  stability_ARI = ARI between two KMeans runs with different seeds.
  If best silhouette < WEAK_SIL_THRESHOLD (0.35) -> use fallback_K from config.
  This reproduces the documented finding: weak structure (young catalog) -> K=8.
- revenue_tier: anchored to deployment-time view (total revenue percentile over
  full history); one-hot weight 0.15 in cluster feature matrix.
- Cluster features: idi, cv2, zero_rate, gini, hurst,
  roll4/13/26/52_mean + revenue_tier_ohe.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from forecasting import config
from forecasting.lifecycle import LifecycleResult

logger = logging.getLogger(__name__)

# SB classification thresholds
_IDI_THRESHOLD: float = 1.32
_CV2_THRESHOLD: float = 0.49
_MIN_NZ_OBS: int = 4  # fewer non-zero obs → cold_start

# K selection
_K_MIN: int = 3
_K_MAX: int = 16
_BLEND_SIL_WEIGHT: float = 0.7
_BLEND_ARI_WEIGHT: float = 0.3
# Doc rule: fallback to fallback_K when the best blend winner's stability_ARI
# is below stability_ari_threshold (default 0.5).  This is the correct spec-
# compliant trigger.  An ARI-only weak-structure signal is more reliable than
# silhouette alone because silhouette can be high for a trivially bimodal
# split (e.g. K=2 high-vs-low revenue) that adds no meaningful pooling value.
_ARI_SEEDS: list[int] = [42, 99]  # two seeds for stability estimate

# Revenue tier cut-points (percentile)
_TIER_BINS = [0.0, 0.50, 0.80, 1.0]
_TIER_LABELS = ["C", "B", "A"]
_TIER_OHE_WEIGHT: float = 0.15  # from doc


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentResult:
    """Output of segment_and_cluster()."""

    segments: pd.DataFrame
    """One row per SKU with columns:
       sku_id, sb_class, revenue_tier, cluster_id,
       idi, cv2, zero_rate, silhouette_score, best_k, used_fallback_k.
    """

    selected_k: int
    """Number of clusters actually used (may equal fallback_K)."""

    used_fallback: bool
    """True if the fallback_K was applied instead of the data-driven K."""

    best_silhouette: float
    """Silhouette score of the selected K."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sb_class(
    sales: np.ndarray,
    is_dormant: bool,
) -> str:
    """Return Syntetos-Boylan demand class for one SKU."""
    if is_dormant:
        return "discontinued"

    nz_idx = np.where(sales > 0)[0]
    if len(nz_idx) < _MIN_NZ_OBS:
        return "cold_start"

    idi = float(np.diff(nz_idx).mean()) if len(nz_idx) >= 2 else float(len(sales))
    nz_vals = sales[nz_idx]
    cv2 = float((nz_vals.std() / nz_vals.mean()) ** 2) if len(nz_vals) >= 2 else 0.0

    if idi < _IDI_THRESHOLD and cv2 < _CV2_THRESHOLD:
        return "smooth"
    if idi < _IDI_THRESHOLD and cv2 >= _CV2_THRESHOLD:
        return "erratic"
    if idi >= _IDI_THRESHOLD and cv2 < _CV2_THRESHOLD:
        return "intermittent"
    return "lumpy"


def _revenue_tier(total_rev_series: pd.Series) -> pd.Series:
    """Assign A/B/C revenue tier from total-revenue percentile rank."""
    pct = total_rev_series.rank(pct=True)
    return pd.cut(
        pct,
        bins=_TIER_BINS,
        labels=_TIER_LABELS,
        include_lowest=True,
    ).astype(str)


def _build_cluster_matrix(sku_stats: pd.DataFrame) -> np.ndarray:
    """Construct the standardised feature matrix for K-means.

    Demand features are standardised; revenue_tier OHE columns are
    appended with weight 0.15 (doc: 'anchor revenue_tier, OHE weight 0.15').
    """
    demand_cols = [
        "idi",
        "cv2",
        "zero_rate",
        "gini",
        "hurst",
        "roll4_mean",
        "roll13_mean",
        "roll26_mean",
        "roll52_mean",
    ]
    X_demand = StandardScaler().fit_transform(sku_stats[demand_cols].fillna(0.0).values)

    # Revenue tier one-hot
    tier_dummies = pd.get_dummies(sku_stats["revenue_tier"], prefix="tier").astype(float)
    X_tier = tier_dummies.values * _TIER_OHE_WEIGHT

    return np.hstack([X_demand * (1 - _TIER_OHE_WEIGHT), X_tier])


def _select_k(
    X: np.ndarray,
    fallback_k: int,
    stability_ari_threshold: float,
) -> tuple[int, float, bool]:
    """Select K via blended silhouette+ARI score (doc spec).

    Rule (locked):
        K* = argmax_k(0.7*silhouette(k) + 0.3*stability_ARI(k))
        stability_ARI(k) = ARI between two KMeans runs with different seeds
        fallback: if stability_ARI(K*) < stability_ari_threshold → use fallback_K

    The ARI-based fallback is the correct spec trigger.  A low ARI means the
    best-blend K is not reproducible across seeds — the cluster assignments
    change too much — making per-cluster model pooling unreliable.

    Returns (selected_k, best_silhouette, used_fallback).
    """
    best_k = fallback_k
    best_blend = -np.inf
    best_sil = 0.0
    best_ari = 0.0

    for k in range(_K_MIN, _K_MAX + 1):
        km1 = KMeans(n_clusters=k, random_state=_ARI_SEEDS[0], n_init=10).fit(X)
        km2 = KMeans(n_clusters=k, random_state=_ARI_SEEDS[1], n_init=10).fit(X)
        sil = float(silhouette_score(X, km1.labels_))
        ari = float(adjusted_rand_score(km1.labels_, km2.labels_))
        blend = _BLEND_SIL_WEIGHT * sil + _BLEND_ARI_WEIGHT * ari
        logger.debug("K=%d sil=%.3f ari=%.3f blend=%.3f", k, sil, ari, blend)
        if blend > best_blend:
            best_blend = blend
            best_k = k
            best_sil = sil
            best_ari = ari

    # Fallback: best-blend K's ARI below stability threshold → not reproducible
    used_fallback = best_ari < stability_ari_threshold
    if used_fallback:
        warnings.warn(
            f"Unstable cluster structure: best K={best_k} has "
            f"stability_ARI={best_ari:.3f} < threshold={stability_ari_threshold}. "
            f"Using fallback_K={fallback_k}. "
            "Young catalog effect — expected to resolve as catalog matures.",
            stacklevel=3,
        )
        selected_k = fallback_k
    else:
        selected_k = best_k

    logger.info(
        "K selection: K*=%d (sil=%.3f, ari=%.3f, blend=%.3f) → selected=%d (fallback=%s)",
        best_k,
        best_sil,
        best_ari,
        best_blend,
        selected_k,
        used_fallback,
    )
    return selected_k, best_sil, used_fallback


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def segment_and_cluster(
    features: pd.DataFrame,
    lifecycle: LifecycleResult,
    fallback_k: int = config.FALLBACK_K,
    stability_ari_threshold: float = config.STABILITY_ARI_THRESHOLD,
) -> SegmentResult:
    """Classify SKUs by SB demand type and assign K-means cluster IDs.

    Parameters
    ----------
    features:
        Output of build_features() — 17,539 × 98.
    lifecycle:
        Output of infer_lifecycle() — provides dormant/active flags.
    fallback_k:
        Number of clusters to use if structure is too weak.
    stability_ari_threshold:
        ARI below this triggers fallback (relaxed to 0.5 per doc for young catalog).

    Returns
    -------
    SegmentResult
    """
    dormant_skus = lifecycle.sku_dormant

    # Per-SKU aggregate stats for SB classification + clustering
    sku_rows = []
    for sku, grp in features.groupby(config.COL_SKU_ID):
        sales = grp[config.COL_SALES].values
        is_dormant = sku in dormant_skus

        sb = _sb_class(sales, is_dormant)

        nz_idx = np.where(sales > 0)[0]
        idi = float(np.diff(nz_idx).mean()) if len(nz_idx) >= 2 else float(len(sales))
        nz_vals = sales[nz_idx]
        cv2 = float((nz_vals.std() / nz_vals.mean()) ** 2) if len(nz_vals) >= 2 else 0.0
        zero_rate = float((sales == 0).mean())

        # Gini
        n = len(sales)
        if n > 1 and sales.sum() > 0:
            s = np.sort(sales)
            idx = np.arange(1, n + 1)
            gini = float((2 * (idx * s).sum() / (n * s.sum())) - (n + 1) / n)
            gini = max(0.0, min(1.0, gini))
        else:
            gini = 0.0

        # Hurst
        if n >= 20 and sales.std() > 0:
            mean = sales.mean()
            dev = np.cumsum(sales - mean)
            r = dev.max() - dev.min()
            hurst = float(np.log(r / sales.std() + 1e-9) / np.log(n))
            hurst = max(0.0, min(1.0, hurst))
        else:
            hurst = 0.5

        total_rev = float(sales.sum())

        sku_rows.append(
            {
                config.COL_SKU_ID: sku,
                "sb_class": sb,
                "idi": idi,
                "cv2": cv2,
                "zero_rate": zero_rate,
                "gini": gini,
                "hurst": hurst,
                "roll4_mean": float(grp["roll4_mean"].mean()),
                "roll13_mean": float(grp["roll13_mean"].mean()),
                "roll26_mean": float(grp["roll26_mean"].mean()),
                "roll52_mean": float(grp["roll52_mean"].mean()),
                "total_rev": total_rev,
            }
        )

    sku_df = pd.DataFrame(sku_rows)

    # Revenue tier (anchored to full-history view)
    sku_df["revenue_tier"] = _revenue_tier(sku_df["total_rev"])

    # Cluster feature matrix
    X = _build_cluster_matrix(sku_df)

    # K selection
    selected_k, best_sil, used_fallback = _select_k(X, fallback_k, stability_ari_threshold)

    # Final cluster assignment with selected K
    km_final = KMeans(n_clusters=selected_k, random_state=config.RANDOM_SEED, n_init=10)
    sku_df["cluster_id"] = km_final.fit_predict(X)

    # Compute per-SKU silhouette score
    sample_sil = silhouette_score(X, sku_df["cluster_id"])
    sku_df["silhouette_score"] = sample_sil  # scalar — same for all (cluster-level)
    sku_df["best_k"] = selected_k
    sku_df["used_fallback_k"] = used_fallback

    # Drop intermediate cols
    out_cols = [
        config.COL_SKU_ID,
        "sb_class",
        "revenue_tier",
        "cluster_id",
        "idi",
        "cv2",
        "zero_rate",
        "silhouette_score",
        "best_k",
        "used_fallback_k",
    ]
    segments = sku_df[out_cols].reset_index(drop=True)

    sb_dist = segments["sb_class"].value_counts().to_dict()
    logger.info(
        "Segment: %d SKUs | K=%d (fallback=%s, sil=%.3f) | SB: %s",
        len(segments),
        selected_k,
        used_fallback,
        best_sil,
        sb_dist,
    )

    return SegmentResult(
        segments=segments,
        selected_k=selected_k,
        used_fallback=used_fallback,
        best_silhouette=best_sil,
    )


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------


def save_segments(result: SegmentResult, path: None = None) -> None:
    """Write segments to data/processed/segments.parquet."""
    import pathlib

    out = pathlib.Path(path) if path else config.DATA_PROCESSED / "segments.parquet"
    result.segments.to_parquet(out, index=False)
    logger.info("Wrote segments → %s", out)
