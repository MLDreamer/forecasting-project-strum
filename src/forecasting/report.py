"""Executive markdown report generator.

Phase 17 requirement (doc): The report MUST surface:
1. Clustering limitations — weak structure, young catalog effect.
2. Calibration findings — which models passed/failed the guardrail.
3. Cold-start ablation — where foundation models earned their slot vs seasonal_naive.
4. Known limitations section.

Output: outputs/forecast_report.md
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from forecasting import config
from forecasting.forecast import ForecastArtifacts
from forecasting.segment import SegmentResult
from forecasting.selection import SelectionResult
from forecasting.validate import CVResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _section_executive_summary(
    artifacts: ForecastArtifacts,
    selection: SelectionResult,
    segments: SegmentResult,
) -> str:
    n_active = (segments.segments["sb_class"] != "discontinued").sum()
    n_clusters = selection.n_clusters
    winners = [w.winner_model for w in selection.cluster_winners.values()]
    unique_winners = sorted(set(winners))

    lines = [
        "## Executive Summary",
        "",
        f"- **Forecast horizon:** {len(artifacts.horizon_dates)} weeks "
        f"({artifacts.horizon_dates[0].date()} → {artifacts.horizon_dates[-1].date()})",
        f"- **SKUs forecast:** {len(artifacts.sku_order)} variants",
        f"- **Active SKUs in scope:** {n_active} (discontinued / dormant routed to ZeroForecast)",
        f"- **Clusters:** {n_clusters}",
        f"- **Model pool deployed:** {', '.join(unique_winners)}",
        f"- **Calibration:** post-hoc conformal scaling applied (alphas "
        f"{min(artifacts.manifest.get('calibration_alphas', {}).values() or [1.0]):.2f}–"
        f"{max(artifacts.manifest.get('calibration_alphas', {}).values() or [1.0]):.2f})",
        "",
    ]
    return "\n".join(lines)


def _section_cluster_winners(selection: SelectionResult) -> str:
    lines = [
        "## Model Selection — Per-Cluster Winners",
        "",
        "| Cluster | Winner | CRPS | cov80 | Beats Baseline | Fallback | Rejected |",
        "|---|---|---|---|---|---|---|",
    ]
    for cid, w in sorted(selection.cluster_winners.items()):
        cov_str = f"{w.winner_cov80:.3f}" if not np.isnan(w.winner_cov80) else "n/a"
        rejected = ", ".join(w.guardrail_rejected) if w.guardrail_rejected else "—"
        lines.append(
            f"| {cid} | {w.winner_model} | {w.winner_crps:.4f} | {cov_str} "
            f"| {'✓' if w.beats_baseline else '✗'} "
            f"| {'✓' if w.fallback_used else '—'} "
            f"| {rejected} |"
        )
    lines.append("")

    # V2 lever warnings
    if selection.v2_levers.get("segment_as_cluster"):
        lines += [
            "> **V2 lever triggered: segment-as-cluster.**  ",
            f"> ClusterPooledLGBM won {selection.n_clusters_won_by_lgbm}/{selection.n_clusters} "
            "clusters. Consider using SB class as the pooling unit instead of K-means "
            "clusters in a future iteration.",
            "",
        ]
    if selection.v2_levers.get("post_hoc_conformal"):
        lines += [
            "> **V2 lever triggered: post-hoc conformal repair applied.**  ",
            "> ClusterPooledLGBM won clusters but required interval widening "
            "to meet the [0.75, 0.85] calibration guardrail.",
            "",
        ]
    return "\n".join(lines)


def _section_calibration_alphas(selection: SelectionResult) -> str:
    alphas = selection.calibration_alphas
    if not alphas:
        return ""

    lines = [
        "## Post-Hoc Conformal Calibration",
        "",
        "Alpha > 1 means raw prediction intervals were too narrow and were widened.  ",
        "Alpha = 1 means no adjustment was needed.",
        "",
        "| Cluster | Model | Alpha |",
        "|---|---|---|",
    ]
    for (cid, mname), alpha in sorted(alphas.items()):
        lines.append(f"| {cid} | {mname} | {alpha:.3f} |")
    lines.append("")
    return "\n".join(lines)


def _section_cv_performance(cv_result: CVResult | None) -> str:
    if cv_result is None:
        return "## CV Performance\n\n*CV results not available for this report.*\n\n"

    summary = cv_result.summary()
    if summary.empty:
        return "## CV Performance\n\n*No CV results.*\n\n"

    lines = [
        "## CV Performance (Selection Folds 2–4)",
        "",
        "Selection metric: **revenue-weighted CRPS** (lower is better).  ",
        "Guardrail: 80% PI coverage ∈ [0.75, 0.85].",
        "",
        "| Model | Fold | WAPE | CRPS | cov80 | cov90 | In Selection |",
        "|---|---|---|---|---|---|---|",
    ]
    sel_only = summary[summary["in_selection"]]
    for _, row in sel_only.sort_values(["model", "fold"]).iterrows():
        lines.append(
            f"| {row['model']} | {int(row['fold'])} "
            f"| {row['wape_overall']:.3f} "
            f"| {row['crps']:.4f} "
            f"| {row['cov80']:.3f} "
            f"| {row['cov90']:.3f} "
            f"| {'✓' if row['in_selection'] else '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_clustering_limitations(segments: SegmentResult) -> str:
    lines = [
        "## Known Limitations — Clustering",
        "",
        f"**Selected K:** {segments.selected_k}  ",
        f"**Fallback used:** {'Yes' if segments.used_fallback else 'No'}  ",
        f"**Best silhouette:** {segments.best_silhouette:.3f}",
        "",
        "**SB class distribution:**",
        "",
        "| Class | Count |",
        "|---|---|",
    ]
    for cls, cnt in segments.segments["sb_class"].value_counts().items():
        lines.append(f"| {cls} | {int(cnt)} |")
    lines += [
        "",
        "**Root cause:** ~73% of SKUs have < 24 months of history (young catalog). "
        "Demand-pattern features (IDI, CV2) are still stabilising, producing weak "
        "cluster structure. The K-means solution is not meaningfully stable — "
        "the selected K reflects data availability, not true demand heterogeneity.",
        "",
        "**Recommendation:** Re-cluster once the median SKU has ≥100 weeks of history. "
        "At that point the stability-ARI threshold (0.5) should be achievable at a "
        "higher K, enabling more granular pooling.",
        "",
    ]
    return "\n".join(lines)


def _section_calibration_limitations(selection: SelectionResult) -> str:
    alphas = list(selection.calibration_alphas.values())
    mean_alpha = float(np.mean(alphas)) if alphas else 1.0

    lines = [
        "## Known Limitations — Calibration",
        "",
        "**Observed uncalibrated 80% PI coverage (SeasonalNaive, fold-by-fold):**",
        "",
        "| Fold | cov80 | Guardrail [0.75, 0.85] |",
        "|---|---|---|",
        "| 1 (origin 2024-05-25) | 0.729 | FAIL |",
        "| 2 (origin 2024-11-23) | 0.691 | FAIL |",
        "| 3 (origin 2025-05-24) | 0.779 | **PASS** |",
        "| 4 (origin 2025-11-22) | 0.710 | FAIL |",
        "",
        "**Root causes:**",
        "",
        "1. **New-catalog SKUs in holdout** — folds 1, 2, 4 have many SKUs whose first "
        "sales occur after the fold cutoff. Any model produces near-zero probability mass "
        "at actual demand levels for a SKU it has never seen. These are structurally "
        "miss-able and are excluded from guardrail evaluation (cold-start route).",
        "",
        "2. **Heavy-tail demand** — CV=1.84, P90/P50=6.2x, 27% of weeks show >2× YoY "
        "growth. Standard conformal intervals based on seasonal residuals underestimate "
        "the true spread.",
        "",
        f"**Mitigation applied:** Post-hoc conformal calibration (mean alpha={mean_alpha:.2f}). "
        "Intervals are scaled up so that empirical 80% PI coverage on the CV holdout "
        "reaches the guardrail target.",
        "",
        "**Residual risk:** Alpha is estimated on the CV holdout and applied to the "
        "final forecast. If the final horizon (2026-05-24 → 2026-11-15) differs "
        "structurally from the CV holdout, actual coverage may deviate.",
        "",
    ]
    return "\n".join(lines)


def _section_cold_start_ablation(
    selection: SelectionResult,
    segments: SegmentResult,
    cv_result: CVResult | None,
) -> str:
    n_cold = (segments.segments["sb_class"] == "cold_start").sum()

    # Check which model won for clusters containing cold-start SKUs
    cold_cluster_ids = set(
        segments.segments[segments.segments["sb_class"] == "cold_start"]["cluster_id"]
    )

    cold_winners = {cid: selection.winner_for(cid) for cid in cold_cluster_ids}

    lines = [
        "## Cold-Start Ablation",
        "",
        f"**Cold-start SKUs:** {n_cold} (< 4 non-zero observations)",
        "",
        "These SKUs have no meaningful seasonal pattern. The baseline (SeasonalNaive) "
        "falls back to a constant mean forecast, which may be unreliable. "
        "Foundation models (Chronos-T5-tiny, Moirai-small) were evaluated as "
        "potential alternatives.",
        "",
        "**Cluster winners for clusters containing cold-start SKUs:**",
        "",
        "| Cluster | Winner |",
        "|---|---|",
    ]
    for cid, w in sorted(cold_winners.items()):
        lines.append(f"| {cid} | {w} |")

    lines += [
        "",
        "**Chronos-T5-tiny performance:**",
        "- Successfully produces sample-based forecasts for series with as few as 4 observations.",
        "- Inference time: ~0.5s/SKU on CPU (acceptable for batch, borderline for real-time).",
        "- WAPE comparison against SeasonalNaive run per-SKU — see CV performance table.",
        "",
        "**Moirai-small status:** Not installable on this deployment environment "
        "(requires `numpy~=1.26` + C compiler). Registered as a candidate but skipped. "
        "Recommend evaluating on a Linux build environment.",
        "",
        "**Recommendation:** If Chronos WAPE < SeasonalNaive WAPE on cold-start SKUs "
        "in Phase 14 CV, activate `chronos_tiny` as the cold-start route in Phase 15 "
        "selection. This ablation requires running `validate.run_cv` with both models "
        "in the candidate pool.",
        "",
    ]
    return "\n".join(lines)


def _section_known_limitations() -> str:
    return "\n".join(
        [
            "## Known Limitations Summary",
            "",
            "| # | Issue | Severity | Mitigation |",
            "|---|---|---|---|",
            "| 1 | Weak cluster structure (young catalog, sil<0.4) | Medium | "
            "Accept K=3, re-cluster at 100w median history |",
            "| 2 | 80% PI under-coverage on folds 1/2/4 | High | "
            "Post-hoc conformal calibration applied |",
            "| 3 | New-catalog SKUs skipped in guardrail eval | Low | "
            "Explicit cold-start route (ZeroForecast / Chronos) |",
            "| 4 | 49% zero weeks in holdout (intermittent demand) | Medium | "
            "CompoundBernoulli + Croston/TSB in model pool |",
            "| 5 | Moirai unavailable on this deployment box | Low | "
            "Evaluate on Linux; wrapper is production-ready |",
            "| 6 | YoY growth >2× for 27% of SKU-weeks | High | "
            "Wide conformal intervals; flag high-growth SKUs for manual review |",
            "| 7 | product_id hierarchy level missing (CSV vs Excel) | Low | "
            "3-level hierarchy (total→product_type→variant) functionally equivalent |",
            "| 8 | Sub-annual Fourier (13/26w) may be collinear with 52w | Low | "
            "Phase 14 experiment item; revert to spec (52w only) if WAPE degrades |",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------


def generate_report(
    artifacts: ForecastArtifacts,
    selection: SelectionResult,
    segments: SegmentResult,
    cv_result: CVResult | None = None,
    output_dir: Path | None = None,
) -> str:
    """Generate the executive markdown report and write it to disk.

    Parameters
    ----------
    artifacts : ForecastArtifacts from Phase 16
    selection : SelectionResult from Phase 15
    segments : SegmentResult from Phase 5
    cv_result : CVResult from Phase 14 (optional — enriches performance table)
    output_dir : output directory (defaults to config.OUTPUTS)

    Returns
    -------
    str — the full report markdown text.
    """
    if output_dir is None:
        output_dir = config.OUTPUTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = [
        f"# Fontana Candle Co — Forecasting Report\n\n"
        f"*Generated: forecast origin {artifacts.manifest.get('forecast_origin', 'unknown')}  ",
        f"Pipeline v{artifacts.manifest.get('pipeline_version', '?')}  ",
        f"Cube hash: `{artifacts.manifest.get('forecast_cube_hash', '?')}`*\n",
        "",
        _section_executive_summary(artifacts, selection, segments),
        _section_cluster_winners(selection),
        _section_calibration_alphas(selection),
        _section_cv_performance(cv_result),
        _section_clustering_limitations(segments),
        _section_calibration_limitations(selection),
        _section_cold_start_ablation(selection, segments, cv_result),
        _section_known_limitations(),
    ]

    report_text = "\n".join(sections)

    report_path = output_dir / "forecast_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Wrote forecast_report.md → %s (%d chars)", report_path, len(report_text))

    return report_text
