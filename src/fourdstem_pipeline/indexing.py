"""Stage 2B phase/orientation indexing.

This module consumes accepted Stage 2A ROI Bragg outputs and candidate CIF
metadata. It generates a simple kinematic template stack from CIF lattice
parameters and matches accepted ROI mean diffraction patterns against those
templates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .contracts import is_roi_ready_for_indexing
from .export import (
    apply_ipf_colors,
    mask_center_for_display,
    save_bar_png,
    save_cluster_phase_map,
    save_ipf_legend,
    save_phase_match_map,
    save_png,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingCandidate:
    """Candidate phase/template metadata for Stage 2B."""

    name: str
    path: Path
    phase: str | None = None
    reference_peaks: tuple[Any, ...] = ()
    sha256: str | None = None
    cell: dict[str, float] | None = None
    template_stack_path: Path | None = None
    template_metadata_path: Path | None = None
    template_count: int = 0
    scoring_mode: str = "not_scored"
    space_group: int | None = None


@dataclass(frozen=True)
class ROIIndexingResult:
    """Indexing contract result for one Stage 2A ROI.

    Conservative naming: fields use ``candidate`` / ``match`` / ``orientation_candidate``
    to reflect that template correlation is a screening signal, not a crystallographic
    phase identification.  Only full multi-condition validation (future Stage 2C) can
    assign a definitive phase.
    """

    name: str
    status: str
    stage2a_bragg_summary_path: str | None
    n_bragg_peaks: int
    candidate_phase: str | None = None
    match_score: float | None = None
    match_quality: str = "not_scored"
    orientation_candidate_deg: float | None = None
    scoring_mode: str = "not_scored"
    best_zone_axis: list[float] | None = None
    score_margin: float | None = None
    phase_confidence: str = "not_scored"
    second_best_candidate: str | None = None
    second_best_zone_axis: list[float] | None = None
    second_best_score: float | None = None
    # Peak-position residual metrics (v3)
    matched_peak_count: int | None = None
    mean_q_residual: float | None = None
    mean_angle_residual: float | None = None
    matched_template_fraction: float | None = None
    unexplained_experiment_fraction: float | None = None
    validation_status: str = "not_scored"
    # Hybrid validation (v3)
    matched_observable_template_fraction: float | None = None
    hybrid_score: float | None = None
    phase_call: str = "not_scored"
    candidate_group: str | None = None
    # Phase/orientation evidence (v4)
    top_phase_matches: list[dict[str, Any]] | None = None
    best_phase: str | None = None
    best_orientation: dict[str, Any] | None = None
    phase_margin: float | None = None
    orientation_margin: float | None = None
    radial_support_score: float | None = None
    radial_gate_status: str | None = None
    orientation_confidence: str = "not_scored"
    mapping_confidence: str = "not_scored"
    ambiguity_reason: str | None = None


def run_stage2_indexing(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Run Stage 2B indexing against accepted Stage 2A ROIs.

    CIFs with parseable lattice parameters are converted into a kinematic
    template stack. Each accepted ROI is scored by normalized correlation
    between its mean diffraction pattern and all generated templates. If no
    template can be generated for any candidate, the legacy mock peak-count
    score remains available for scaffold tests only.
    """
    cfg, base_dir = _load_indexing_config(config)
    stage2_dir = _resolve_path(cfg["stage2_dir"], base_dir)
    output_dir = _resolve_path(
        cfg.get("output_dir") or stage2_dir / "stage2b_indexing",
        base_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stage2a_summary_path = stage2_dir / "stage2_summary.json"
    stage2a_summary = json.loads(stage2a_summary_path.read_text(encoding="utf-8"))

    template_cfg = _template_config(cfg.get("template_generation") or {})
    matching_cfg = _matching_config(cfg.get("matching") or {})
    candidates = _load_candidates(cfg.get("candidate_cifs") or [], base_dir)
    accepted_rois = [
        roi for roi in stage2a_summary.get("roi_results", [])
        if is_roi_ready_for_indexing(roi)
    ]
    geometry = _stage2_geometry(stage2a_summary, required=any(c.cell is not None for c in candidates))
    candidates = _generate_candidate_templates(
        candidates,
        template_cfg,
        output_dir,
        geometry,
    )

    roi_results = [
        _match_roi_against_candidates(roi, candidates, matching_cfg)
        for roi in accepted_rois
    ]

    any_template = any(c.template_count > 0 for c in candidates)
    summary = {
        "schema_version": "stage2b-indexing-v4",
        "stage": "2B",
        "status": _stage2b_status(accepted_rois, candidates),
        "stage2a": {
            "stage2_dir": str(stage2_dir),
            "summary_path": str(stage2a_summary_path),
            "run_name": stage2a_summary.get("run_name"),
        },
        "output_dir": str(output_dir),
        "geometry": geometry,
        "template_generation": {
            **template_cfg,
            "backend": "analytic_cif",
            "templates_generated": int(sum(c.template_count for c in candidates)),
        },
        "matching": matching_cfg,
        "candidate_cifs": [
            {
                "name": c.name,
                "phase": c.phase,
                "path": str(c.path),
                "sha256": c.sha256,
                "cell": c.cell,
                "reference_peak_count": len(c.reference_peaks),
                "template_count": c.template_count,
                "template_stack_path": str(c.template_stack_path) if c.template_stack_path else None,
                "template_metadata_path": str(c.template_metadata_path) if c.template_metadata_path else None,
                "scoring_mode": _candidate_scoring_mode(c),
                "space_group": c.space_group,
            }
            for c in candidates
        ],
        "accepted_roi_count": len(accepted_rois),
        "roi_results": [
            {
                "name": r.name,
                "status": r.status,
                "stage2a_bragg_summary_path": r.stage2a_bragg_summary_path,
                "n_bragg_peaks": r.n_bragg_peaks,
                "candidate_phase": r.candidate_phase,
                "match_score": r.match_score,
                "match_quality": r.match_quality,
                "orientation_candidate_deg": r.orientation_candidate_deg,
                "scoring_mode": r.scoring_mode,
                "best_zone_axis": r.best_zone_axis,
                "score_margin": r.score_margin,
                "phase_confidence": r.phase_confidence,
                "second_best_candidate": r.second_best_candidate,
                "second_best_zone_axis": r.second_best_zone_axis,
                "second_best_score": r.second_best_score,
                "matched_peak_count": r.matched_peak_count,
                "mean_q_residual": r.mean_q_residual,
                "mean_angle_residual": r.mean_angle_residual,
                "matched_template_fraction": r.matched_template_fraction,
                "unexplained_experiment_fraction": r.unexplained_experiment_fraction,
                "validation_status": r.validation_status,
                "matched_observable_template_fraction": r.matched_observable_template_fraction,
                "hybrid_score": r.hybrid_score,
                "phase_call": r.phase_call,
                "candidate_group": r.candidate_group,
                "top_phase_matches": r.top_phase_matches,
                "best_phase": r.best_phase,
                "best_orientation": r.best_orientation,
                "phase_margin": r.phase_margin,
                "orientation_margin": r.orientation_margin,
                "radial_support_score": r.radial_support_score,
                "radial_gate_status": r.radial_gate_status,
                "orientation_confidence": r.orientation_confidence,
                "mapping_confidence": r.mapping_confidence,
                "ambiguity_reason": r.ambiguity_reason,
            }
            for r in roi_results
        ],
        "notes": [
            "Stage 2B uses analytic kinematic CIF templates when lattice parameters are available.",
            "Schema v4 (stage2b-indexing-v4): Added ROI-level top-k phase/orientation evidence, "
            "radial q-profile support scoring, separate phase/orientation/mapping confidence fields, "
            "and per-ROI phase_orientation_topk.json diagnostics. Phase maps remain cluster-level "
            "screening visualizations, not dense pointwise EBSD maps.",
            "Scores are normalized template correlations on ROI mean diffraction patterns.",
            "Full structure-factor intensities and py4DSTEM/pyxem backend adapters are future extensions.",
            "Schema v3 (stage2b-indexing-v3): Added peak-position residual metrics "
            "(matched_peak_count, mean_q_residual, mean_angle_residual, matched_template_fraction, "
            "unexplained_experiment_fraction). Phase confidence now uses tiers: "
            "HIGH_CONFIDENCE / MEDIUM_CONFIDENCE / LOW_CONFIDENCE / UNINDEXED. "
            "Validation criteria incorporate matched peak fraction and mean q residual "
            "in addition to correlation score and margin.",
            "Schema v2: field renames — best_candidate→candidate_phase, phase_score→match_score, "
            "best_orientation_deg→orientation_candidate_deg. Removed: orientation_score, template_score "
            "(consolidated into match_score). New: best_zone_axis, score_margin, phase_confidence, "
            "second_best_candidate, second_best_zone_axis, second_best_score.",
        ],
    }
    if not any_template:
        summary["notes"].append("No analytic templates were generated; mock scoring may appear only for test fixtures.")

    # --- Score-sign QC check --------------------------------------------------
    _check_score_signs(roi_results, summary)

    summary_path = output_dir / "stage2_indexing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # --- Phase match map (navigation-space overview) --------------------
    phase_map_path: Path | None = None
    phase_legend_path: Path | None = None
    try:
        labels = _load_stage1_label_map(stage2a_summary)
        contrast_image = _load_stage1_contrast_image(stage2a_summary, labels.shape if labels is not None else None)
        if labels is not None and accepted_rois:
            cluster_phase_entries = []
            result_by_name = {r.name: r for r in roi_results}
            for s2a_roi in accepted_rois:
                result = result_by_name.get(str(s2a_roi.get("name", "")))
                if result is None:
                    continue
                cluster_phase_entries.append({
                    "name": s2a_roi.get("name", "unknown"),
                    "cluster_id": s2a_roi.get("cluster_id"),
                    "candidate_phase": result.candidate_phase,
                    "match_score": result.match_score,
                    "phase_confidence": result.phase_confidence,
                })
            phase_map_path = save_cluster_phase_map(
                output_dir / "phase_match_map.png",
                labels,
                cluster_phase_entries,
                legend_path=output_dir / "phase_match_legend.png",
                contrast_image=contrast_image,
            )
            phase_legend_path = output_dir / "phase_match_legend.png"
            log.info("Dense cluster phase map saved: %s", phase_map_path)
        else:
            nav_shape = stage2a_summary.get("manifest", {}).get("nav_shape")
            if not (nav_shape and accepted_rois):
                nav_shape = None
        if phase_map_path is None and nav_shape and accepted_rois:
            # Merge stage2a bboxes with stage2b match results.
            phase_map_entries: list[dict[str, Any]] = []
            for s2a_roi in accepted_rois:
                entry: dict[str, Any] = {
                    "name": s2a_roi.get("name", "unknown"),
                    "stage1_bbox": s2a_roi.get("stage1_bbox"),
                }
                # Find matching Stage 2B result.
                for r in roi_results:
                    if r.name == entry["name"]:
                        entry["candidate_phase"] = r.candidate_phase
                        entry["match_score"] = r.match_score
                        entry["phase_confidence"] = r.phase_confidence
                        break
                phase_map_entries.append(entry)

            phase_map_path = save_phase_match_map(
                output_dir / "phase_match_map.png",
                (int(nav_shape[0]), int(nav_shape[1])),
                phase_map_entries,
            )
            log.info("Phase match map saved: %s", phase_map_path)
    except Exception as exc:
        log.warning("Failed to generate phase match map: %s", exc)

    # --- Update the Stage 2 PNG gallery (now includes Stage 2B match PNGs) ---
    try:
        from .export_stage2 import save_stage2_gallery

        gallery_summary = _merge_stage2b_results_into_stage2a_summary(stage2a_summary, roi_results)
        gallery_path = save_stage2_gallery(
            stage2_dir, gallery_summary,
            global_pngs=(
                [
                    {"path": str(phase_map_path), "caption": "Phase Map - Stage 1 cluster labels replaced by matched Stage 2B phases"},
                    {"path": str(phase_legend_path), "caption": "Phase Map Legend"},
                ]
                if phase_map_path and phase_map_path.is_file() else None
            ),
        )
        if gallery_path is not None:
            log.info("Stage 2 PNG gallery updated: %s", gallery_path)
    except Exception as exc:
        log.warning("Failed to update Stage 2 PNG gallery: %s", exc)

    try:
        report_md, report_html = _write_stage2b_phase_mapping_report(
            output_dir,
            summary,
            stage2a_summary,
            phase_map_path=phase_map_path,
            phase_legend_path=phase_legend_path,
        )
        log.info("Stage 2B phase mapping report saved: %s, %s", report_md, report_html)
    except Exception as exc:
        log.warning("Failed to write Stage 2B phase mapping report: %s", exc)

    return summary


def _merge_stage2b_results_into_stage2a_summary(
    stage2a_summary: dict[str, Any],
    roi_results: list[ROIIndexingResult],
) -> dict[str, Any]:
    """Return a Stage 2A-like summary enriched with Stage 2B match metadata."""
    by_name = {r.name: r for r in roi_results}
    merged = dict(stage2a_summary)
    merged_rois: list[dict[str, Any]] = []
    for roi in stage2a_summary.get("roi_results", []):
        entry = dict(roi)
        match = by_name.get(str(entry.get("name", "")))
        if match is not None:
            entry.update({
                "stage2b_status": match.status,
                "candidate_phase": match.candidate_phase,
                "match_score": match.match_score,
                "match_quality": match.match_quality,
                "orientation_candidate_deg": match.orientation_candidate_deg,
                "best_zone_axis": match.best_zone_axis,
                "score_margin": match.score_margin,
                "phase_confidence": match.phase_confidence,
                "second_best_candidate": match.second_best_candidate,
                "second_best_score": match.second_best_score,
                "matched_peak_count": match.matched_peak_count,
                "mean_q_residual": match.mean_q_residual,
                "mean_angle_residual": match.mean_angle_residual,
                "matched_template_fraction": match.matched_template_fraction,
                "unexplained_experiment_fraction": match.unexplained_experiment_fraction,
                "validation_status": match.validation_status,
                "matched_observable_template_fraction": match.matched_observable_template_fraction,
                "hybrid_score": match.hybrid_score,
                "phase_call": match.phase_call,
                "candidate_group": match.candidate_group,
            })
        merged_rois.append(entry)
    merged["roi_results"] = merged_rois
    return merged


def _load_stage1_label_map(stage2a_summary: dict[str, Any]) -> np.ndarray | None:
    """Load the Stage 1 fingerprint-class labels used for dense phase maps."""
    stage1_dir_raw = stage2a_summary.get("stage1_dir")
    if not stage1_dir_raw:
        return None
    stage1_dir = Path(stage1_dir_raw)
    stage1_summary_path = stage1_dir / "stage1_summary.json"
    try:
        stage1_summary = json.loads(stage1_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not load Stage 1 summary for phase map: %s", exc)
        return None

    labels_path_raw = stage1_summary.get("labels_path")
    if not labels_path_raw:
        return None
    labels_path = Path(labels_path_raw)
    if not labels_path.is_absolute():
        labels_path = stage1_dir / labels_path
    try:
        labels = np.load(labels_path)
    except OSError as exc:
        log.warning("Could not load Stage 1 labels for phase map: %s", exc)
        return None
    if labels.ndim != 2:
        log.warning("Stage 1 labels for phase map must be 2D, got shape %s", labels.shape)
        return None
    return labels


def _load_stage1_contrast_image(
    stage2a_summary: dict[str, Any],
    expected_shape: tuple[int, int] | None,
) -> np.ndarray | None:
    """Load a Stage 1 virtual image for EBSD-like phase-map contrast."""
    stage1_dir_raw = stage2a_summary.get("stage1_dir")
    if not stage1_dir_raw:
        return None
    stage1_dir = Path(stage1_dir_raw)
    stage1_summary_path = stage1_dir / "stage1_summary.json"
    try:
        stage1_summary = json.loads(stage1_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    virtual_path_raw = stage1_summary.get("virtual_images_path")
    if not virtual_path_raw:
        return None
    virtual_path = Path(virtual_path_raw)
    if not virtual_path.is_absolute():
        virtual_path = stage1_dir / virtual_path
    try:
        with np.load(virtual_path) as virtual:
            for key in ("adf", "haadf", "bf"):
                if key in virtual:
                    image = np.asarray(virtual[key], dtype=np.float32)
                    if expected_shape is None or image.shape == expected_shape:
                        return image
    except OSError as exc:
        log.warning("Could not load Stage 1 virtual images for phase map contrast: %s", exc)
    return None


def _write_stage2b_phase_mapping_report(
    output_dir: Path,
    summary: dict[str, Any],
    stage2a_summary: dict[str, Any],
    *,
    phase_map_path: Path | None,
    phase_legend_path: Path | None,
) -> tuple[Path, Path]:
    """Write a human-readable Stage 2B phase-map interpretation report."""
    md_path = output_dir / "stage2_phase_mapping_report.md"
    html_path = output_dir / "stage2_phase_mapping_report.html"
    roi_results = summary.get("roi_results", [])
    phase_fractions = _phase_map_fractions(summary, stage2a_summary)
    low_reasons = _phase_mapping_low_confidence_reasons(roi_results)

    lines: list[str] = [
        "# Stage 2B Phase Mapping Report",
        "",
        f"**Status:** `{summary.get('status', 'unknown')}`",
        f"**Schema:** `{summary.get('schema_version', 'unknown')}`",
        f"**Accepted ROIs:** `{summary.get('accepted_roi_count', 0)}`",
        f"**Templates generated:** `{(summary.get('template_generation') or {}).get('templates_generated', 0)}`",
        "",
        "## Outputs",
        "",
    ]
    if phase_map_path is not None:
        lines.append(f"- Phase map: `{phase_map_path.name}`")
    if phase_legend_path is not None:
        lines.append(f"- Legend: `{phase_legend_path.name}`")
    lines.extend([
        "- Raw machine-readable results: `stage2_indexing_summary.json`",
        "",
        "## What The Phase Map Means",
        "",
        "The PNG phase map is a Stage 1 cluster-label map recolored by the best current Stage 2B candidate call. "
        "It is not a point-by-point crystallographic indexing result like EBSD. Each pixel inherits the phase assigned "
        "to its Stage 1 fingerprint cluster representative.",
        "",
        "## Phase Fractions From The Current Map",
        "",
        "| Phase call | Fraction | Pixels | Source clusters |",
        "| --- | ---: | ---: | --- |",
    ])
    if phase_fractions:
        for row in phase_fractions:
            lines.append(
                f"| `{row['phase']}` | {row['fraction']:.2%} | {row['pixels']} | `{row['clusters']}` |"
            )
    else:
        lines.append("| _No mapped phase labels_ | - | - | - |")

    lines.extend([
        "",
        "## Why Confidence Is Low",
        "",
    ])
    for reason in low_reasons:
        lines.append(f"- {reason}")

    lines.extend([
        "",
        "## Per-ROI Matching Evidence",
        "",
        "| ROI | Cluster | Phase call | Phase conf. | Orient. conf. | Mapping conf. | Evidence margin | Orient. margin | Radial support | Radial gate | Corr. score | Hybrid | Observable template matched | q residual | Ambiguity reason |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    cluster_by_name = {
        str(r.get("name")): r.get("cluster_id")
        for r in stage2a_summary.get("roi_results", [])
    }
    for r in roi_results:
        lines.append(
            "| "
            f"`{r.get('name', '?')}` | "
            f"{_fmt_report_value(cluster_by_name.get(str(r.get('name'))))} | "
            f"`{_fmt_report_value(r.get('candidate_phase'))}` | "
            f"`{_fmt_report_value(r.get('phase_confidence'))}` | "
            f"`{_fmt_report_value(r.get('orientation_confidence'))}` | "
            f"`{_fmt_report_value(r.get('mapping_confidence'))}` | "
            f"{_fmt_report_float(r.get('phase_margin'))} | "
            f"{_fmt_report_float(r.get('orientation_margin'))} | "
            f"{_fmt_report_float(r.get('radial_support_score'))} | "
            f"`{_fmt_report_value(r.get('radial_gate_status'))}` | "
            f"{_fmt_report_float(r.get('match_score'))} | "
            f"{_fmt_report_float(r.get('hybrid_score'))} | "
            f"{_fmt_report_float(r.get('matched_observable_template_fraction'))} | "
            f"{_fmt_report_float(r.get('mean_q_residual'))} | "
            f"`{_fmt_report_value(r.get('ambiguity_reason'))}` |"
        )

    lines.extend([
        "",
        "## Confidence Thresholds Used",
        "",
        "- `HIGH_CONFIDENCE`: correlation score > 0.55, score margin > 0.10, matched template fraction > 0.50, and mean q residual < 5 px.",
        "- `MEDIUM_CONFIDENCE`: correlation score > 0.40, score margin > 0.06, and matched template fraction > 0.30.",
        "- `LOW_CONFIDENCE`: scored, but one or more of the above criteria failed.",
        "- `AMBIGUOUS` phase calls are downgraded to `LOW_CONFIDENCE` even when raw scores look adequate.",
        "",
        "## Practical Interpretation",
        "",
        "Use this map as a screening visualization: it shows which Stage 1 fingerprint clusters currently prefer which candidate phase group. "
        "Do not treat it as a confirmed EBSD-equivalent phase assignment until the ambiguous candidate groups separate and the matched-template fractions improve.",
        "",
    ])

    markdown = "\n".join(lines)
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_stage2b_report_html(markdown, phase_map_path, phase_legend_path), encoding="utf-8")
    return md_path, html_path


def _phase_map_fractions(
    summary: dict[str, Any],
    stage2a_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = _load_stage1_label_map(stage2a_summary)
    if labels is None:
        return []
    roi_by_name = {str(r.get("name")): r for r in stage2a_summary.get("roi_results", [])}
    best_by_cluster: dict[int, dict[str, Any]] = {}
    for r in summary.get("roi_results", []):
        phase = r.get("candidate_phase")
        if not phase:
            continue
        stage2a_roi = roi_by_name.get(str(r.get("name")))
        if not stage2a_roi or stage2a_roi.get("cluster_id") is None:
            continue
        cluster_id = int(stage2a_roi["cluster_id"])
        score = r.get("match_score")
        score_value = float(score) if score is not None else -np.inf
        previous = best_by_cluster.get(cluster_id)
        previous_score = float(previous.get("match_score", -np.inf)) if previous else -np.inf
        if previous is None or score_value >= previous_score:
            best_by_cluster[cluster_id] = r

    total_mapped = 0
    by_phase: dict[str, dict[str, Any]] = {}
    for cluster_id, r in best_by_cluster.items():
        count = int(np.sum(labels == cluster_id))
        if count <= 0:
            continue
        total_mapped += count
        phase = str(r.get("candidate_phase"))
        row = by_phase.setdefault(phase, {"phase": phase, "pixels": 0, "clusters": []})
        row["pixels"] += count
        row["clusters"].append(str(cluster_id))

    if total_mapped <= 0:
        return []
    result = []
    for row in by_phase.values():
        result.append({
            "phase": row["phase"],
            "pixels": row["pixels"],
            "fraction": row["pixels"] / total_mapped,
            "clusters": ", ".join(row["clusters"]),
        })
    return sorted(result, key=lambda x: x["pixels"], reverse=True)


def _phase_mapping_low_confidence_reasons(roi_results: list[dict[str, Any]]) -> list[str]:
    if not roi_results:
        return ["No Stage 2B ROI match results were available."]
    reasons: list[str] = []
    ambiguous = sum(1 for r in roi_results if r.get("phase_call") == "AMBIGUOUS")
    if ambiguous:
        reasons.append(f"{ambiguous}/{len(roi_results)} ROI matches are `AMBIGUOUS`, meaning multiple candidate phases or candidate groups remain effectively tied.")
    phase_margins = [float(r["phase_margin"]) for r in roi_results if r.get("phase_margin") is not None]
    if phase_margins:
        reasons.append(
            f"Phase evidence margins are small: max {max(phase_margins):.4f}, median {float(np.median(phase_margins)):.4f}. "
            "Small margins mean the best phase is not clearly separated from the runner-up after radial and peak evidence."
        )
    orientation_margins = [float(r["orientation_margin"]) for r in roi_results if r.get("orientation_margin") is not None]
    if orientation_margins:
        reasons.append(
            f"Orientation margins are small: max {max(orientation_margins):.4f}, median {float(np.median(orientation_margins)):.4f}. "
            "This indicates 2D orientation matching is not uniquely resolved within the winning phase."
        )
    radial_scores = [float(r["radial_support_score"]) for r in roi_results if r.get("radial_support_score") is not None]
    if radial_scores:
        low_radial = sum(1 for v in radial_scores if v < 0.25)
        reasons.append(
            f"Radial q-profile support: min {min(radial_scores):.3f}, median {float(np.median(radial_scores)):.3f}; "
            f"{low_radial}/{len(radial_scores)} ROI(s) are below the default 0.25 support gate."
        )
    margins = [float(r["score_margin"]) for r in roi_results if r.get("score_margin") is not None]
    if margins:
        reasons.append(
            f"Correlation score margins are small: max {max(margins):.4f}, median {float(np.median(margins)):.4f}. "
            "Medium confidence requires margin > 0.06 and high confidence requires > 0.10."
        )
    matched = [
        float(r["matched_observable_template_fraction"])
        for r in roi_results
        if r.get("matched_observable_template_fraction") is not None
    ]
    if matched:
        reasons.append(
            f"Matched observable template fractions are low: max {max(matched):.3f}, median {float(np.median(matched)):.3f}. "
            "Medium confidence expects > 0.30 and high confidence expects > 0.50."
        )
    unexplained = [
        float(r["unexplained_experiment_fraction"])
        for r in roi_results
        if r.get("unexplained_experiment_fraction") is not None
    ]
    if unexplained:
        reasons.append(
            f"A large fraction of experimental peaks remains unexplained: median {float(np.median(unexplained)):.3f}."
        )
    return reasons or ["The current results did not expose enough confidence diagnostics to explain the tier."]


def _stage2b_report_html(
    markdown: str,
    phase_map_path: Path | None,
    phase_legend_path: Path | None,
) -> str:
    def esc(text: Any) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    body = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\">",
        "<title>Stage 2B Phase Mapping Report</title>",
        "<style>body{font-family:system-ui,-apple-system,sans-serif;margin:28px;line-height:1.45;color:#222;max-width:1200px}"
        "table{border-collapse:collapse;margin:14px 0;width:100%;font-size:13px}th,td{border:1px solid #ddd;padding:6px 8px;text-align:left}"
        "th{background:#f4f4f4}code{background:#f5f5f5;padding:1px 4px;border-radius:3px}pre{white-space:pre-wrap;background:#fafafa;padding:16px;border:1px solid #ddd}"
        ".figs{display:flex;gap:24px;align-items:flex-start;margin:16px 0}.figs img{border:1px solid #ddd;max-width:520px;height:auto}</style>",
        "</head><body>",
    ]
    if phase_map_path is not None or phase_legend_path is not None:
        body.append('<div class="figs">')
        if phase_map_path is not None:
            body.append(f'<figure><img src="{esc(phase_map_path.name)}" alt="phase map"><figcaption>Phase map</figcaption></figure>')
        if phase_legend_path is not None:
            body.append(f'<figure><img src="{esc(phase_legend_path.name)}" alt="phase legend"><figcaption>Legend</figcaption></figure>')
        body.append("</div>")
    body.append("<pre>")
    body.append(esc(markdown))
    body.append("</pre></body></html>")
    return "\n".join(body)


def _fmt_report_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_report_value(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def _load_indexing_config(config: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path]:
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        base_dir = Path.cwd()
    else:
        cfg = config
        base_dir = Path.cwd()

    if not isinstance(cfg, dict):
        raise ValueError("Stage 2B config must be a mapping.")
    if "stage2_dir" not in cfg:
        raise ValueError("Stage 2B config must contain 'stage2_dir'.")
    return cfg, base_dir


def _load_candidates(items: list[dict[str, Any]] | None, base_dir: Path) -> list[IndexingCandidate]:
    if items is None:
        items = []
    candidates: list[IndexingCandidate] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Each candidate_cifs entry must be a mapping.")
        raw_path = item.get("path")
        if not raw_path:
            raise ValueError("Each candidate_cifs entry must contain 'path'.")
        path = _resolve_path(raw_path, base_dir)
        cell = _parse_cif_cell(path)
        # Space group override: config value takes precedence over CIF content.
        # Use to apply correct extinction rules when CIF uses P1 for convenience.
        space_group = item.get("space_group")
        if space_group is not None:
            space_group = int(space_group)
        candidates.append(
            IndexingCandidate(
                name=str(item.get("name") or path.stem or f"candidate_{i:03d}"),
                phase=item.get("phase"),
                path=path,
                reference_peaks=tuple(item.get("reference_peaks") or ()),
                sha256=_sha256_file(path),
                cell=cell,
                space_group=space_group,
            )
        )
    return candidates


def _generate_candidate_templates(
    candidates: list[IndexingCandidate],
    template_cfg: dict[str, Any],
    output_dir: Path,
    geometry: dict[str, Any] | None,
) -> list[IndexingCandidate]:
    result: list[IndexingCandidate] = []
    template_dir = output_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    if geometry is None:
        return candidates
    sig_shape = tuple(int(v) for v in geometry["sig_shape"])
    beam_center = tuple(float(v) for v in geometry["beam_center_yx"])

    for candidate in candidates:
        if candidate.cell is None:
            result.append(candidate)
            continue

        try:
            all_stacks: list[np.ndarray] = []
            all_orientations: list[float] = []
            zone_axis_index: list[int] = []
            per_zone_metadata: list[dict[str, Any]] = []

            for zi, zone_axis in enumerate(template_cfg["zone_axes"]):
                stack, meta = _generate_kinematic_template_stack(
                    candidate.cell,
                    sig_shape=sig_shape,
                    beam_center_yx=beam_center,
                    max_index=int(template_cfg["max_index"]),
                    orientations_deg=[float(v) for v in template_cfg["orientations_deg"]],
                    zone_axis=tuple(float(v) for v in zone_axis),
                    peak_sigma_px=float(template_cfg["peak_sigma_px"]),
                    reciprocal_pixels_per_inv_angstrom=template_cfg["reciprocal_pixels_per_inv_angstrom"],
                    intensity_power=float(template_cfg["intensity_power"]),
                    space_group=candidate.space_group,
                )
                all_stacks.append(stack)
                all_orientations.extend(meta["orientations_deg"])
                zone_axis_index.extend([zi] * len(meta["orientations_deg"]))
                per_zone_metadata.append(meta)

            combined_stack = np.concatenate(all_stacks, axis=0)
            metadata: dict[str, Any] = {
                "cell": per_zone_metadata[0]["cell"],
                "max_index": per_zone_metadata[0]["max_index"],
                "hkl_count_total": int(sum(m["hkl_count"] for m in per_zone_metadata)),
                "orientations_deg": all_orientations,
                "zone_axes": template_cfg["zone_axes"],
                "zone_axis_index": zone_axis_index,
                "projections": [m["projection"] for m in per_zone_metadata],
                "per_zone_hkls": [m["hkls"] for m in per_zone_metadata],
                "per_zone_qxy": [m["qxy"] for m in per_zone_metadata],
                "per_zone_qnorm": [m["qnorm"] for m in per_zone_metadata],
                "sig_shape": per_zone_metadata[0]["sig_shape"],
                "beam_center_yx": per_zone_metadata[0]["beam_center_yx"],
                "peak_sigma_px": per_zone_metadata[0]["peak_sigma_px"],
                "reciprocal_pixels_per_inv_angstrom": per_zone_metadata[0]["reciprocal_pixels_per_inv_angstrom"],
                "reciprocal_scale_source": per_zone_metadata[0]["reciprocal_scale_source"],
                "intensity_power": per_zone_metadata[0]["intensity_power"],
                "space_group": per_zone_metadata[0].get("space_group"),
                "n_extinct_removed": int(sum(m.get("n_extinct_removed", 0) for m in per_zone_metadata)),
            }
        except ValueError as exc:
            log.warning("Could not generate templates for %s: %s", candidate.name, exc)
            result.append(candidate)
            continue

        stack_path = template_dir / f"{candidate.name}_template_stack.npy"
        metadata_path = template_dir / f"{candidate.name}_template_metadata.json"
        np.save(stack_path, combined_stack.astype(np.float32))
        metadata.update({
            "candidate": candidate.name,
            "phase": candidate.phase,
            "cif_path": str(candidate.path),
            "cif_sha256": candidate.sha256,
            "backend": "analytic_cif",
        })
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        result.append(
            IndexingCandidate(
                name=candidate.name,
                phase=candidate.phase,
                path=candidate.path,
                reference_peaks=candidate.reference_peaks,
                sha256=candidate.sha256,
                cell=candidate.cell,
                template_stack_path=stack_path,
                template_metadata_path=metadata_path,
                template_count=int(combined_stack.shape[0]),
                scoring_mode="template_match",
                space_group=candidate.space_group,
            )
        )
    return result


def _match_roi_against_candidates(
    roi: dict[str, Any],
    candidates: list[IndexingCandidate],
    matching_cfg: dict[str, Any] | None = None,
) -> ROIIndexingResult:
    n_bragg_peaks = int(roi.get("n_bragg_peaks", 0) or 0)

    template_result = _template_match_roi(roi, candidates, n_bragg_peaks, matching_cfg or _matching_config({}))
    if template_result is not None:
        return template_result

    return _mock_score_roi_against_candidates(roi, candidates, n_bragg_peaks)


def _template_match_roi(
    roi: dict[str, Any],
    candidates: list[IndexingCandidate],
    n_bragg_peaks: int,
    matching_cfg: dict[str, Any],
) -> ROIIndexingResult | None:
    mean_dp = _load_roi_match_pattern(roi)
    if mean_dp is None:
        return None
    pattern_vec = _normalize_pattern(mean_dp)
    if pattern_vec is None:
        return None

    all_hits: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.template_stack_path is None or candidate.template_metadata_path is None:
            continue
        try:
            stack = np.load(candidate.template_stack_path)
            metadata = json.loads(candidate.template_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not load templates for %s: %s", candidate.name, exc)
            continue

        # Bin templates to match the data resolution when necessary
        # (templates are generated at the pre-Q-bin sig_shape; the actual
        # ROI data / Bragg vector map may be at a binned resolution).
        tmpl_h, tmpl_w = stack.shape[1], stack.shape[2]
        data_h, data_w = mean_dp.shape
        if tmpl_h != data_h or tmpl_w != data_w:
            bin_y = tmpl_h // data_h
            bin_x = tmpl_w // data_w
            if bin_y > 1 and bin_x > 1 and tmpl_h % bin_y == 0 and tmpl_w % bin_x == 0:
                stack = stack.reshape(
                    stack.shape[0], data_h, bin_y, data_w, bin_x,
                ).mean(axis=(2, 4))
            else:
                log.warning(
                    "Template shape %s cannot be binned to match data shape %s for %s",
                    (tmpl_h, tmpl_w), (data_h, data_w), candidate.name,
                )
                continue

        flat_templates = stack.reshape((stack.shape[0], -1))
        normalized = [
            vec for vec in (_normalize_pattern(t.reshape(mean_dp.shape)) for t in flat_templates)
            if vec is not None
        ]
        if not normalized:
            continue
        template_vecs = np.vstack(normalized)
        scores = template_vecs @ pattern_vec  # shape (n_templates,)

        orientations = metadata.get("orientations_deg", [])
        zone_axis_index = metadata.get("zone_axis_index", [0] * len(orientations))
        zone_axes = metadata.get("zone_axes", [[0.0, 0.0, 1.0]])

        for i, score in enumerate(scores):
            zi = int(zone_axis_index[i]) if i < len(zone_axis_index) else 0
            za = zone_axes[zi] if zi < len(zone_axes) else [0.0, 0.0, 1.0]
            all_hits.append({
                "candidate": candidate,
                "score": float(score),
                "zone_axis": [float(v) for v in za],
                "zone_axis_idx": zi,
                "orientation_deg": float(orientations[i]) if i < len(orientations) else None,
                "template_idx": i,
                "stack": stack,
                "all_scores": [float(s) for s in scores],
                "orientations_deg": orientations,
                "metadata": metadata,
            })

    if not all_hits:
        return None

    all_hits.sort(key=lambda x: x["score"], reverse=True)
    best = all_hits[0]

    # --- Save visualisations (best by correlation) ---------------------------
    _save_match_visuals(
        roi.get("name", "unknown"),
        roi.get("bragg_summary_path", ""),
        mean_dp,
        best,
    )

    # --- Hybrid validation: score top candidates by correlation + peaks ------
    # Take top 3 distinct candidates by correlation for hybrid scoring.
    seen_phases: set[str] = set()
    top_candidates: list[dict[str, Any]] = []
    for hit in all_hits:
        phase_name = _candidate_display_name(hit["candidate"])
        if phase_name not in seen_phases:
            seen_phases.add(phase_name)
            top_candidates.append(hit)
        if len(top_candidates) >= 3:
            break

    hybrid_candidates: list[dict[str, Any]] = []
    for hit in top_candidates:
        candidate = hit["candidate"]
        res: dict[str, Any] = {}
        try:
            if candidate.template_metadata_path is not None:
                tmpl_meta = json.loads(candidate.template_metadata_path.read_text(encoding="utf-8"))
                res = _compute_peak_residual_metrics(
                    roi, tmpl_meta, hit["template_idx"],
                )
        except Exception as exc:
            log.warning("Peak residual skipped for %s: %s", candidate.name, exc)

        # Compute observable template fraction
        observable_frac: float | None = None
        try:
            if candidate.template_metadata_path is not None:
                tmpl_meta2 = json.loads(candidate.template_metadata_path.read_text(encoding="utf-8"))
                tmpl_peaks = _reconstruct_template_peak_positions(tmpl_meta2, hit["template_idx"])
                if tmpl_peaks is not None and len(tmpl_peaks) > 0:
                    observable_frac = _compute_observable_template_fraction(
                        tmpl_peaks, tuple(tmpl_meta2["sig_shape"]),
                    )
                    # Matched-observable = matched / (template * observable)
                    raw_matched_frac = res.get("matched_template_fraction") or 0.0
                    if observable_frac > 0:
                        matched_obs = round(raw_matched_frac / observable_frac, 4)
                    else:
                        matched_obs = 0.0
                else:
                    matched_obs = None
            else:
                matched_obs = None
        except Exception:
            observable_frac = None
            matched_obs = None

        hybrid = _compute_hybrid_validation_score(
            correlation_score=float(hit["score"]),
            matched_observable_fraction=matched_obs,
            mean_q_residual=res.get("mean_q_residual"),
            unexplained_experiment_fraction=res.get("unexplained_experiment_fraction"),
        )

        hybrid_candidates.append({
            "phase": _candidate_display_name(candidate),
            "correlation_score": round(float(hit["score"]), 4),
            "hybrid_score": hybrid,
            "matched_peak_count": res.get("matched_peak_count"),
            "mean_q_residual": res.get("mean_q_residual"),
            "mean_angle_residual": res.get("mean_angle_residual"),
            "matched_template_fraction": res.get("matched_template_fraction"),
            "unexplained_experiment_fraction": res.get("unexplained_experiment_fraction"),
            "matched_observable_template_fraction": matched_obs,
            "observable_template_fraction": observable_frac,
            "zone_axis": hit["zone_axis"],
            "orientation_deg": hit["orientation_deg"],
            "template_idx": hit["template_idx"],
            "stack": hit["stack"],
            "all_scores": hit["all_scores"],
            "orientations_deg": hit["orientations_deg"],
            "candidate_obj": candidate,
        })

    # Sort by hybrid score descending
    hybrid_candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)

    # --- v4 phase/orientation evidence: top-k per phase + radial support -----
    phase_evidence = _build_phase_orientation_evidence(roi, mean_dp, all_hits, matching_cfg)
    if phase_evidence:
        hybrid_candidates = phase_evidence
        _write_phase_orientation_topk(roi, phase_evidence, matching_cfg)

    # --- Resolve phase call --------------------------------------------------
    resolution = _resolve_phase_call(best, hybrid_candidates)
    phase_call = resolution["phase_call"]
    candidate_group = resolution.get("candidate_group")
    resolution_reason = resolution.get("reason", "")

    # Select the reported candidate: hybrid-winner when unambiguous, else corr-winner
    if phase_call == "AMBIGUOUS":
        reported = hybrid_candidates[0] if hybrid_candidates else best
    elif phase_call == "UNINDEXED":
        reported = {"phase": None, "correlation_score": 0.0, "hybrid_score": 0.0}
    else:
        reported = hybrid_candidates[0]

    # Best and second-best by hybrid score
    hybrid_best = hybrid_candidates[0] if hybrid_candidates else None
    hybrid_second = hybrid_candidates[1] if len(hybrid_candidates) > 1 else None

    match_score = round(float(best["score"]), 4)
    second_by_corr = _second_best_candidate_hit(all_hits, best)
    second_best_score = round(float(second_by_corr["score"]), 4) if second_by_corr else None
    score_margin = round(match_score - second_best_score, 4) if second_best_score is not None else None

    # Use hybrid-winner's residuals for reporting
    best_residual = hybrid_best or {}
    obs_frac = best_residual.get("matched_observable_template_fraction")
    phase_margin = _phase_evidence_margin(phase_evidence)
    radial_support_score = best_residual.get("radial_support_score")
    orientation_margin = best_residual.get("orientation_margin")

    phase_conf = _phase_evidence_confidence(
        best_residual.get("evidence_score"),
        phase_margin,
        radial_support_score,
        matching_cfg,
    )
    if phase_conf == "UNINDEXED":
        phase_conf = _phase_confidence(
            match_score, score_margin,
            matched_template_fraction=obs_frac if obs_frac is not None else best_residual.get("matched_template_fraction"),
            mean_q_residual=best_residual.get("mean_q_residual"),
        )
    orient_conf = _orientation_confidence(orientation_margin, matching_cfg)
    mapping_conf = _combined_mapping_confidence(phase_conf, orient_conf)

    # Downgrade to LOW_CONFIDENCE when phase is AMBIGUOUS
    if phase_call == "AMBIGUOUS" and phase_conf in ("HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE"):
        phase_conf = "LOW_CONFIDENCE"
        mapping_conf = _combined_mapping_confidence(phase_conf, orient_conf)

    best_orientation = None
    if best_residual:
        best_orientation = {
            "zone_axis": best_residual.get("zone_axis"),
            "orientation_deg": best_residual.get("orientation_deg"),
            "template_idx": best_residual.get("template_idx"),
            "orientation_margin": orientation_margin,
            "orientation_confidence": orient_conf,
        }

    return ROIIndexingResult(
        name=str(roi.get("name", "unknown")),
        status="TEMPLATE_MATCHED",
        stage2a_bragg_summary_path=roi.get("bragg_summary_path"),
        n_bragg_peaks=n_bragg_peaks,
        candidate_phase=(reported.get("phase") if phase_call != "AMBIGUOUS" else candidate_group),
        match_score=match_score,
        match_quality=_template_quality(match_score),
        orientation_candidate_deg=best_residual.get("orientation_deg") or best.get("orientation_deg"),
        best_zone_axis=best_residual.get("zone_axis") or best.get("zone_axis"),
        score_margin=score_margin,
        phase_confidence=phase_conf,
        second_best_candidate=_candidate_display_name(second_by_corr["candidate"]) if second_by_corr else None,
        second_best_zone_axis=second_by_corr["zone_axis"] if second_by_corr else None,
        second_best_score=second_best_score,
        scoring_mode="template_match",
        matched_peak_count=best_residual.get("matched_peak_count"),
        mean_q_residual=best_residual.get("mean_q_residual"),
        mean_angle_residual=best_residual.get("mean_angle_residual"),
        matched_template_fraction=best_residual.get("matched_template_fraction"),
        unexplained_experiment_fraction=best_residual.get("unexplained_experiment_fraction"),
        validation_status=phase_conf,
        matched_observable_template_fraction=obs_frac,
        hybrid_score=best_residual.get("hybrid_score"),
        phase_call=phase_call,
        candidate_group=candidate_group,
        top_phase_matches=[_jsonable_phase_evidence(row) for row in phase_evidence],
        best_phase=best_residual.get("phase"),
        best_orientation=best_orientation,
        phase_margin=phase_margin,
        orientation_margin=orientation_margin,
        radial_support_score=radial_support_score,
        radial_gate_status=best_residual.get("radial_gate_status"),
        orientation_confidence=orient_conf,
        mapping_confidence=mapping_conf,
        ambiguity_reason=resolution_reason,
    )


def _load_roi_match_pattern(roi: dict[str, Any]) -> np.ndarray | None:
    """Load the numeric pattern used for template matching.

    Prefer the ROI mean diffraction pattern when ``roi_data.npy`` was kept.
    If Stage 2A ran with ``save_roi_data: false``, fall back to the much
    smaller Bragg vector map so Stage 2B does not silently degrade to mock
    peak-count scoring.
    """
    roi_name = roi.get("name", "unknown")
    roi_data_path = roi.get("roi_data_path")
    if roi_data_path:
        try:
            return _mean_diffraction_pattern(np.load(roi_data_path))
        except (OSError, ValueError) as exc:
            log.warning("Could not load ROI data for %s: %s", roi_name, exc)

    bragg_vector_map_path = roi.get("bragg_vector_map_path")
    if bragg_vector_map_path:
        try:
            arr = np.asarray(np.load(bragg_vector_map_path), dtype=np.float32)
        except OSError as exc:
            log.warning("Could not load Bragg vector map for %s: %s", roi_name, exc)
            return None
        if arr.ndim == 2:
            return arr
        log.warning("Bragg vector map for %s must be 2D, got shape %s", roi_name, arr.shape)
    return None


def _candidate_display_name(candidate: IndexingCandidate) -> str:
    """Human-facing phase label, falling back to the candidate key."""
    return str(candidate.phase or candidate.name)


def _second_best_candidate_hit(
    hits: list[dict[str, Any]],
    best: dict[str, Any],
) -> dict[str, Any] | None:
    """Best hit from a competing candidate, not another orientation of the winner."""
    best_name = best["candidate"].name
    best_phase = _candidate_display_name(best["candidate"])
    for hit in hits:
        candidate = hit["candidate"]
        if candidate.name != best_name and _candidate_display_name(candidate) != best_phase:
            return hit
    return None


def _save_match_visuals(
    roi_name: str,
    bragg_summary_path: str,
    mean_dp: np.ndarray,
    best: dict[str, Any],
) -> None:
    """Save Stage 2B matching PNGs for the best candidate.

    Writes into the same directory as the ROI's ``bragg_summary.json``:
    - ``template_best_match.png`` — the best-matching template
    - ``template_match_overlay.png`` — mean DP (gray) + template peaks (green)
    - ``correlation_vs_angle.png`` — bar chart of correlation vs. orientation
    """
    try:
        roi_dir = Path(bragg_summary_path).parent
        roi_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    try:
        # Best template image — use viridis for consistency with pyxem style.
        best_template = best["stack"][best["template_idx"]]
        save_png(roi_dir / "template_best_match.png", best_template, cmap="viridis")

        # Overlay: mean DP (viridis) + explicit green template peaks.
        # Apply pyxem-style direct-beam masking for cleaner visual focus.
        base = np.asarray(mean_dp, dtype=np.float32)
        base_masked = mask_center_for_display(base, radius_px=35.0)
        overlay = _gray_rgb(base_masked, cmap="viridis")
        tmpl_norm = _scale_unit_interval(best_template)
        peak_mask = tmpl_norm >= 0.20
        green = (tmpl_norm * 255).astype(np.uint8)
        overlay[:, :, 0] = np.where(peak_mask, (overlay[:, :, 0] * 0.25).astype(np.uint8), overlay[:, :, 0])
        overlay[:, :, 1] = np.where(peak_mask, np.maximum(overlay[:, :, 1], green), overlay[:, :, 1])
        overlay[:, :, 2] = np.where(peak_mask, (overlay[:, :, 2] * 0.25).astype(np.uint8), overlay[:, :, 2])
        save_png(roi_dir / "template_match_overlay.png", overlay)

        metadata = best.get("metadata")
        if isinstance(metadata, dict):
            _save_experimental_template_peak_overlay(
                roi_dir / "experimental_template_peak_overlay.png",
                roi={"name": roi_name, "bragg_summary_path": bragg_summary_path},
                mean_dp=mean_dp,
                template_metadata=metadata,
                template_idx=int(best["template_idx"]),
            )
            _save_radial_q_profile_validation(
                roi_dir / "radial_q_profile_validation.png",
                mean_dp=mean_dp,
                template_metadata=metadata,
                template_idx=int(best["template_idx"]),
            )

        # --- IPF-coloured orientation overlay + legend (pyxem-style) ---------
        _save_ipf_orientation_overlay(
            roi_dir / "ipf_orientation_overlay.png",
            mean_dp=mean_dp,
            best=best,
        )

        # Correlation vs angle bar chart + heatmap
        scores_arr = np.asarray(best["all_scores"], dtype=np.float32)
        if scores_arr.size > 0:
            save_bar_png(
                roi_dir / "correlation_vs_angle.png",
                scores_arr.reshape(1, -1),
                xlabel="orientation index",
                ylabel="correlation",
            )
            # Build a 2D heatmap: orientations × 1, or multi-zone-axis rows.
            _save_correlation_heatmap(
                roi_dir / "correlation_heatmap.png",
                best,
            )

        # --- Multi-panel overlay figure (pyxem-style plot_over_signal) --------
        _save_multi_panel_match_figure(
            roi_dir / "template_match_figure.png",
            mean_dp=mean_dp,
            best=best,
            roi={"name": roi_name, "bragg_summary_path": bragg_summary_path},
        )
    except Exception:
        pass


def _save_ipf_orientation_overlay(
    path: Path,
    *,
    mean_dp: np.ndarray,
    best: dict[str, Any],
) -> None:
    """Save an IPF-coloured orientation overlay on the mean diffraction pattern.

    Mimics pyxem's ``OrientationMap.plot_over_signal()``: the best-matching
    template peaks are coloured by their zone-axis direction using the cubic
    IPF convention, and the zone axis is marked with an IPF-coloured cross.
    An IPF legend is saved alongside.
    """
    try:
        metadata = best.get("metadata")
        if not isinstance(metadata, dict):
            return
        zone_axis = best.get("zone_axis")
        if zone_axis is None:
            return

        # Render the mean DP as a viridis base.
        base = np.asarray(mean_dp, dtype=np.float32)
        canvas = _gray_rgb(mask_center_for_display(base, radius_px=35.0), cmap="viridis")

        # Colour template peak markers by their hkl direction via IPF.
        template_idx = int(best["template_idx"])
        tmpl_peaks = _reconstruct_template_peak_positions(metadata, template_idx)
        if tmpl_peaks is not None and len(tmpl_peaks) > 0:
            # Get the hkls for this template.
            zi = int(metadata.get("zone_axis_index", [0] * len(metadata.get("orientations_deg", [])))[template_idx]) if template_idx < len(metadata.get("zone_axis_index", [0])) else 0
            per_zone_hkls = metadata.get("per_zone_hkls")
            if per_zone_hkls is not None and zi < len(per_zone_hkls):
                hkls = np.asarray(per_zone_hkls[zi], dtype=np.float64)
                # Scale to display and apply IPF colours.
                dst_shape = tuple(int(v) for v in mean_dp.shape)
                src_shape = tuple(int(v) for v in metadata.get("sig_shape", dst_shape))
                display_peaks = _scale_points(tmpl_peaks, src_shape, dst_shape)
                ipf_rgb = apply_ipf_colors(hkls / np.maximum(np.linalg.norm(hkls, axis=1, keepdims=True), 1e-12))
                for i in range(min(len(display_peaks), len(ipf_rgb))):
                    py, px = float(display_peaks[i, 0]), float(display_peaks[i, 1])
                    color = tuple(int(c) for c in ipf_rgb[i])
                    _draw_cross(canvas, py, px, color, radius=3)

        # Mark the zone axis direction with a larger IPF-coloured cross at centre.
        za = np.asarray(zone_axis, dtype=np.float64).reshape(1, 3)
        za = za / np.maximum(np.linalg.norm(za), 1e-12)
        za_color = tuple(int(c) for c in apply_ipf_colors(za)[0])
        cy, cx = float(canvas.shape[0] - 1) / 2.0, float(canvas.shape[1] - 1) / 2.0
        _draw_cross(canvas, cy, cx, za_color, radius=6)

        save_png(path, canvas)

        # Save companion IPF legend.
        legend_path = path.with_stem(path.stem + "_legend")
        save_ipf_legend(legend_path, label="CUBIC IPF")
    except Exception:
        pass


def _save_correlation_heatmap(
    path: Path,
    best: dict[str, Any],
) -> None:
    """Save a 2D correlation heatmap (orientations × zone-axes) for the match.

    When multiple zone axes were searched, rows represent zone axes and columns
    represent in-plane orientations, giving an at-a-glance view of the
    correlation landscape — analogous to pyxem's
    ``add_ipf_correlation_heatmap=True`` panel.
    """
    try:
        from .export import save_heatmap_png
        metadata = best.get("metadata")
        if not isinstance(metadata, dict):
            return
        all_scores = np.asarray(best.get("all_scores", []), dtype=np.float64)
        if all_scores.size == 0:
            return
        zone_axis_index = metadata.get("zone_axis_index", [])
        orientations = metadata.get("orientations_deg", [])
        zone_axes = metadata.get("zone_axes", [])
        if len(zone_axis_index) != len(all_scores) or len(orientations) != len(all_scores):
            # Fall back to 1×N heatmap.
            heatmap_data = all_scores.reshape(1, -1)
            yticklabels = None
        else:
            # Pivot into (n_zone_axes, n_orientations_per_zone) matrix.
            indices = np.asarray(zone_axis_index, dtype=np.int32)
            n_za = int(indices.max()) + 1 if indices.size > 0 else 1
            # Assume each zone axis has the same number of orientations.
            per_za = int(np.sum(indices == 0)) if n_za > 0 and np.any(indices == 0) else len(all_scores) // max(n_za, 1)
            if per_za == 0:
                per_za = len(all_scores) // max(n_za, 1)
            heatmap_data = np.full((n_za, per_za), np.nan, dtype=np.float64)
            for za_idx in range(n_za):
                mask = indices == za_idx
                row_scores = all_scores[mask]
                heatmap_data[za_idx, :len(row_scores)] = row_scores
            yticklabels = [f"ZA {tuple(zone_axes[i])}" if i < len(zone_axes) else f"ZA {i}" for i in range(n_za)]

        save_heatmap_png(
            path,
            heatmap_data,
            cmap="viridis",
            xlabel="orientation",
            ylabel="zone axis",
            title="correlation landscape",
            add_colorbar=True,
        )
    except Exception:
        pass


def _save_multi_panel_match_figure(
    path: Path,
    *,
    mean_dp: np.ndarray,
    best: dict[str, Any],
    roi: dict[str, Any],
) -> None:
    """Composite a pyxem-style multi-panel figure for the template match.

    Layout: (left) mean DP + colour-coded peak markers,
    (right) side panels for IPF legend and correlation heatmap.
    """
    try:
        from .export import save_overlay_figure

        metadata = best.get("metadata")
        if not isinstance(metadata, dict):
            return

        tmpl_idx = int(best["template_idx"])
        tmpl_peaks = _reconstruct_template_peak_positions(metadata, tmpl_idx)
        dst_shape = tuple(int(v) for v in mean_dp.shape)
        src_shape = tuple(int(v) for v in metadata.get("sig_shape", dst_shape))

        # --- Build overlays for the main panel ---------------------------------
        overlays: list[dict[str, Any]] = []
        if tmpl_peaks is not None and len(tmpl_peaks) > 0:
            display_peaks = _scale_points(tmpl_peaks, src_shape, dst_shape)
            zi = int(metadata.get("zone_axis_index", [0])[tmpl_idx] if tmpl_idx < len(metadata.get("zone_axis_index", [0])) else 0)
            hkls = None
            per_zone_hkls = metadata.get("per_zone_hkls")
            if per_zone_hkls is not None and zi < len(per_zone_hkls):
                hkls = np.asarray(per_zone_hkls[zi], dtype=np.float64)
            if hkls is not None and len(hkls) > 0:
                norms = np.maximum(np.linalg.norm(hkls, axis=1, keepdims=True), 1e-12)
                ipf_rgb = apply_ipf_colors(hkls / norms)
                overlays.append({
                    "positions_yx": display_peaks,
                    "colors": ipf_rgb,
                    "marker": "cross",
                    "radius": 4,
                })

        # Also overlay measured peaks.
        measured = _load_measured_peak_positions_for_display(roi, mean_dp)
        if measured.size > 0:
            n_meas = len(measured)
            overlays.append({
                "positions_yx": measured,
                "colors": np.tile(np.asarray([20, 230, 80], dtype=np.uint8), (n_meas, 1)),
                "marker": "circle",
                "radius": 5,
            })

        # --- Build side panels -------------------------------------------------
        side_panels: list[dict[str, Any]] = []
        # IPF legend.
        side_panels.append({
            "image": _ipf_legend_image(128),
            "title": "CUBIC IPF",
            "width": 128,
        })

        # Correlation bar strip.
        scores = np.asarray(best.get("all_scores", []), dtype=np.float64)
        if scores.size > 0:
            heatmap_2d = scores.reshape(-1, 1) if scores.ndim == 1 else scores
            side_panels.append({
                "image": heatmap_2d,
                "title": "correlation",
                "width": 64,
                "cmap": "viridis",
            })

        save_overlay_figure(
            path,
            mean_dp,
            overlays,
            side_panels=side_panels,
            cmap="viridis",
            center_mask_radius=35.0,
            title=f"template match: {roi.get('name', 'unknown')}",
        )
    except Exception:
        pass


def _ipf_legend_image(size: int = 128) -> np.ndarray:
    """Return a square RGB image of the cubic IPF stereographic triangle."""
    from .export import _CUBIC_IPF_LUT as ipf_lut
    h, w = ipf_lut.shape[:2]
    sy = np.linspace(0, h - 1, size).astype(np.int32)
    sx = np.linspace(0, w - 1, size).astype(np.int32)
    return ipf_lut[sy[:, None], sx[None, :]]


def _mock_score_roi_against_candidates(
    roi: dict[str, Any],
    candidates: list[IndexingCandidate],
    n_bragg_peaks: int,
) -> ROIIndexingResult:
    best_candidate: IndexingCandidate | None = None
    best_score: float | None = None

    for candidate in candidates:
        if not candidate.reference_peaks:
            continue
        reference_count = len(candidate.reference_peaks)
        denom = max(n_bragg_peaks, reference_count, 1)
        score = 1.0 - abs(n_bragg_peaks - reference_count) / denom
        if best_score is None or score > best_score:
            best_candidate = candidate
            best_score = round(float(score), 4)

    if best_candidate is None:
        return ROIIndexingResult(
            name=str(roi.get("name", "unknown")),
            status="PENDING_TEMPLATE_MATCHING",
            stage2a_bragg_summary_path=roi.get("bragg_summary_path"),
            n_bragg_peaks=n_bragg_peaks,
            candidate_phase=None,
            match_score=None,
            match_quality="not_scored",
            scoring_mode="not_scored",
        )

    return ROIIndexingResult(
        name=str(roi.get("name", "unknown")),
        status="MOCK_SCORED",
        stage2a_bragg_summary_path=roi.get("bragg_summary_path"),
        n_bragg_peaks=n_bragg_peaks,
        candidate_phase=best_candidate.name,
        match_score=best_score,
        match_quality="mock_scored",
        scoring_mode="mock_peak_count",
    )


def _template_config(raw: dict[str, Any]) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("template_generation must be a mapping.")
    step = float(raw.get("orientation_step_deg", 5.0))
    if "orientations_deg" in raw and raw["orientations_deg"] is not None:
        orientations = [float(v) for v in raw["orientations_deg"]]
    else:
        orientations = [float(v) for v in np.arange(0.0, 360.0, step)]
    return {
        "max_index": int(raw.get("max_index", 4)),
        "orientations_deg": orientations,
        "zone_axes": _parse_zone_axes(raw),
        "peak_sigma_px": float(raw.get("peak_sigma_px", 5.0)),
        "reciprocal_pixels_per_inv_angstrom": (
            None if raw.get("reciprocal_pixels_per_inv_angstrom") is None
            else float(raw["reciprocal_pixels_per_inv_angstrom"])
        ),
        "intensity_power": float(raw.get("intensity_power", 2.0)),
    }


def _matching_config(raw: dict[str, Any]) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("matching must be a mapping.")
    return {
        "top_k_per_phase": max(1, int(raw.get("top_k_per_phase", 5))),
        "radial_gate_enabled": bool(raw.get("radial_gate_enabled", True)),
        "radial_min_support": float(raw.get("radial_min_support", 0.25)),
        "phase_margin_threshold": float(raw.get("phase_margin_threshold", 0.08)),
        "orientation_margin_threshold": float(raw.get("orientation_margin_threshold", 0.05)),
    }


def _parse_zone_axis(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("template_generation.zone_axis must be [u, v, w].")
    axis = [float(v) for v in value]
    if float(np.linalg.norm(axis)) <= 1e-12:
        raise ValueError("template_generation.zone_axis must be non-zero.")
    return axis


def _parse_zone_axes(raw: dict[str, Any]) -> list[list[float]]:
    """Parse ``zone_axes`` (plural) with backward compat for ``zone_axis`` (singular).

    Precedence: ``zone_axes`` > ``zone_axis`` > default ``[[0, 0, 1]]``.
    Always returns a list of zone-axis triplets.
    """
    if "zone_axes" in raw and raw["zone_axes"] is not None:
        axes = [_parse_zone_axis(z) for z in raw["zone_axes"]]
    elif "zone_axis" in raw and raw["zone_axis"] is not None:
        axes = [_parse_zone_axis(raw["zone_axis"])]
    else:
        axes = [_parse_zone_axis([0, 0, 1])]
    if not axes:
        raise ValueError("zone_axes must contain at least one zone axis.")
    return axes


def _stage2_geometry(stage2a_summary: dict[str, Any], *, required: bool = True) -> dict[str, Any] | None:
    roi_results = stage2a_summary.get("roi_results", [])
    sig_shape = None
    beam_center = None
    for roi in roi_results:
        if sig_shape is None and roi.get("sig_shape"):
            sig_shape = [int(v) for v in roi["sig_shape"]]
        if beam_center is None and roi.get("beam_center_yx"):
            beam_center = [float(v) for v in roi["beam_center_yx"]]
    if sig_shape is None:
        manifest_sig = stage2a_summary.get("manifest", {}).get("sig_shape")
        if manifest_sig:
            sig_shape = [int(v) for v in manifest_sig]
    if sig_shape is None:
        if not required:
            return None
        raise ValueError("Stage 2A summary does not provide sig_shape for template generation.")
    if beam_center is None:
        beam_center = [float(sig_shape[0] - 1) / 2.0, float(sig_shape[1] - 1) / 2.0]

    # Record the data (post-binning) sig_shape when it differs, so the
    # template matcher can downsample templates to match the actual data.
    bin_q = int(stage2a_summary.get("parameters", {}).get("bin_q", 1))
    data_sig_shape = None
    if bin_q > 1 and not any(r.get("sig_shape_after_bin") for r in roi_results):
        data_sig_shape = [max(1, s // bin_q) for s in sig_shape]

    return {
        "sig_shape": sig_shape,
        "beam_center_yx": beam_center,
        "beam_center_source": "stage2a_roi" if any(r.get("beam_center_yx") for r in roi_results) else "detector_center_fallback",
        "data_sig_shape": data_sig_shape,
    }


def _parse_cif_cell(path: Path) -> dict[str, float] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        log.warning("Could not read candidate CIF %s: %s", path, exc)
        return None
    fields = {
        "a": "_cell_length_a",
        "b": "_cell_length_b",
        "c": "_cell_length_c",
        "alpha": "_cell_angle_alpha",
        "beta": "_cell_angle_beta",
        "gamma": "_cell_angle_gamma",
    }
    parsed: dict[str, float] = {}
    for key, token in fields.items():
        value = _extract_cif_number(text, token)
        if value is not None:
            parsed[key] = value
    if "a" in parsed and "b" not in parsed:
        parsed["b"] = parsed["a"]
    if "a" in parsed and "c" not in parsed:
        parsed["c"] = parsed["a"]
    parsed.setdefault("alpha", 90.0)
    parsed.setdefault("beta", 90.0)
    parsed.setdefault("gamma", 90.0)
    required = {"a", "b", "c", "alpha", "beta", "gamma"}
    if not required.issubset(parsed):
        return None
    return parsed


def _extract_cif_number(text: str, token: str) -> float | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not parts or parts[0] != token or len(parts) < 2:
            continue
        raw = parts[1].strip("'\"")
        raw = raw.split("(")[0]
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _gray_rgb(image: np.ndarray, *, cmap: str = "gray") -> np.ndarray:
    """Return a log-scaled RGB image for diffraction diagnostics.

    Parameters
    ----------
    cmap:
        Colormap name: ``"gray"`` (default) or ``"viridis"``.
    """
    from .export import _get_colormap

    base = np.asarray(image, dtype=np.float32)
    finite = base[np.isfinite(base)]
    if finite.size == 0:
        scaled = np.zeros(base.shape, dtype=np.uint8)
    else:
        lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip(
            (np.log1p(np.maximum(base, 0.0)) - np.log1p(max(lo, 0.0)))
            / max(np.log1p(max(hi, 0.0)) - np.log1p(max(lo, 0.0)), 1e-12)
            * 255.0,
            0,
            255,
        ).astype(np.uint8)
    lut = _get_colormap(cmap)
    return lut[scaled]


def _draw_cross(
    canvas: np.ndarray,
    y: float,
    x: float,
    color: tuple[int, int, int],
    *,
    radius: int = 4,
) -> None:
    h, w = canvas.shape[:2]
    cy, cx = int(round(y)), int(round(x))
    if cy < 0 or cy >= h or cx < 0 or cx >= w:
        return
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    canvas[y0:y1, cx] = color
    canvas[cy, x0:x1] = color
    if radius >= 3:
        for d in range(-radius + 1, radius):
            yy, xx = cy + d, cx + d
            if 0 <= yy < h and 0 <= xx < w:
                canvas[yy, xx] = color
            yy, xx = cy + d, cx - d
            if 0 <= yy < h and 0 <= xx < w:
                canvas[yy, xx] = color


def _draw_polyline_local(
    canvas: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    pts = np.asarray(points, dtype=np.int32)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps + 1).round().astype(int)
        ys = np.linspace(y0, y1, steps + 1).round().astype(int)
        valid = (ys >= 0) & (ys < canvas.shape[0]) & (xs >= 0) & (xs < canvas.shape[1])
        canvas[ys[valid], xs[valid]] = color


def _scale_points(
    points_yx: np.ndarray,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> np.ndarray:
    pts = np.asarray(points_yx, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 2)
    sy = float(dst_shape[0]) / max(float(src_shape[0]), 1.0)
    sx = float(dst_shape[1]) / max(float(src_shape[1]), 1.0)
    out = pts.copy()
    out[:, 0] *= sy
    out[:, 1] *= sx
    return out


def _load_measured_peak_positions_for_display(
    roi: dict[str, Any],
    mean_dp: np.ndarray,
) -> np.ndarray:
    bvm_path = roi.get("bragg_vector_map_path")
    if not bvm_path and roi.get("bragg_summary_path"):
        candidate = Path(str(roi["bragg_summary_path"])).parent / "bragg_vector_map.npy"
        if candidate.is_file():
            bvm_path = str(candidate)
    if bvm_path:
        try:
            vmap = np.load(bvm_path)
            peaks = _extract_measured_peak_positions(vmap, min_spacing=2)
            return _scale_points(peaks, tuple(vmap.shape), tuple(mean_dp.shape))
        except (OSError, ValueError):
            pass

    arr = np.asarray(mean_dp, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    threshold = max(float(np.percentile(finite, 99.5)), 0.20 * float(np.max(finite)))
    return _extract_measured_peak_positions(arr, min_intensity=threshold, min_spacing=2).astype(np.float64)


def _peak_match_assignments(
    template_peaks: np.ndarray,
    measured_peaks: np.ndarray,
    tolerance_px: float,
) -> dict[str, Any]:
    n_template = len(template_peaks)
    n_measured = len(measured_peaks)
    empty = {
        "matched_peak_count": 0,
        "mean_q_residual": None,
        "mean_angle_residual": None,
        "matched_template_fraction": 0.0,
        "unexplained_experiment_fraction": 1.0 if n_measured > 0 else 0.0,
        "matched_template_indices": [],
        "matched_measured_indices": [],
        "unmatched_template_indices": list(range(n_template)),
        "unmatched_measured_indices": list(range(n_measured)),
        "matched_distances": [],
    }
    if n_template == 0 or n_measured == 0:
        return empty

    from scipy.spatial import cKDTree
    tree = cKDTree(measured_peaks.astype(np.float64))
    distances, indices = tree.query(template_peaks.astype(np.float64), distance_upper_bound=tolerance_px)
    candidate_mask = np.isfinite(distances)

    matched_template: list[int] = []
    matched_measured: list[int] = []
    matched_distances: list[float] = []
    matched_angles: list[float] = []
    used_measured: set[int] = set()

    order = np.argsort(np.where(candidate_mask, distances, np.inf))
    template_center = np.asarray(template_peaks, dtype=np.float64).mean(axis=0)
    measured_center = np.asarray(measured_peaks, dtype=np.float64).mean(axis=0)
    for template_idx in order:
        if not candidate_mask[template_idx]:
            continue
        measured_idx = int(indices[template_idx])
        if measured_idx in used_measured:
            continue
        used_measured.add(measured_idx)
        matched_template.append(int(template_idx))
        matched_measured.append(measured_idx)
        matched_distances.append(float(distances[template_idx]))
        t_angle = math.atan2(
            float(template_peaks[template_idx, 0] - template_center[0]),
            float(template_peaks[template_idx, 1] - template_center[1]),
        )
        m_angle = math.atan2(
            float(measured_peaks[measured_idx, 0] - measured_center[0]),
            float(measured_peaks[measured_idx, 1] - measured_center[1]),
        )
        angle_diff = abs(t_angle - m_angle)
        matched_angles.append(float(min(angle_diff, 2 * math.pi - angle_diff)))

    matched_template_set = set(matched_template)
    matched_measured_set = set(matched_measured)
    n_matched = len(matched_template)
    return {
        "matched_peak_count": n_matched,
        "mean_q_residual": round(float(np.mean(matched_distances)), 3) if n_matched else None,
        "mean_angle_residual": round(float(np.mean(matched_angles)), 4) if n_matched else None,
        "matched_template_fraction": round(n_matched / max(n_template, 1), 4),
        "unexplained_experiment_fraction": round((n_measured - n_matched) / max(n_measured, 1), 4),
        "matched_template_indices": matched_template,
        "matched_measured_indices": matched_measured,
        "unmatched_template_indices": [i for i in range(n_template) if i not in matched_template_set],
        "unmatched_measured_indices": [i for i in range(n_measured) if i not in matched_measured_set],
        "matched_distances": matched_distances,
    }


def _template_peak_tolerance_px(
    template_metadata: dict[str, Any],
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> float:
    sigma_px = float(template_metadata.get("peak_sigma_px", 5.0))
    scale = min(float(dst_shape[0]) / max(float(src_shape[0]), 1.0), float(dst_shape[1]) / max(float(src_shape[1]), 1.0))
    return max(1.5, round(2.5 * sigma_px * scale, 1))


def _save_experimental_template_peak_overlay(
    path: Path,
    *,
    roi: dict[str, Any],
    mean_dp: np.ndarray,
    template_metadata: dict[str, Any],
    template_idx: int,
) -> None:
    template_peaks = _reconstruct_template_peak_positions(template_metadata, template_idx)
    if template_peaks is None:
        return
    dst_shape = tuple(int(v) for v in mean_dp.shape)
    src_shape = tuple(int(v) for v in template_metadata.get("sig_shape", dst_shape))
    template_display = _scale_points(template_peaks, src_shape, dst_shape)
    measured = _load_measured_peak_positions_for_display(roi, mean_dp)
    tol = _template_peak_tolerance_px(template_metadata, src_shape, dst_shape)
    assignments = _peak_match_assignments(template_display, measured, tol)

    canvas = _gray_rgb(mask_center_for_display(mean_dp, radius_px=35.0), cmap="viridis")
    for idx in assignments["unmatched_template_indices"]:
        _draw_cross(canvas, template_display[idx, 0], template_display[idx, 1], (30, 120, 255), radius=4)
    for idx in assignments["unmatched_measured_indices"]:
        _draw_cross(canvas, measured[idx, 0], measured[idx, 1], (255, 40, 40), radius=4)
    for idx in assignments["matched_measured_indices"]:
        _draw_cross(canvas, measured[idx, 0], measured[idx, 1], (20, 230, 80), radius=5)
    save_png(path, canvas)


def _radial_profile(image: np.ndarray, center_yx: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=np.float64)
    h, w = arr.shape
    cy, cx = center_yx if center_yx is not None else ((h - 1) / 2.0, (w - 1) / 2.0)
    yy, xx = np.indices(arr.shape)
    radii = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.floor(radii).astype(np.int32)
    max_bin = int(bins.max())
    sums = np.bincount(bins.ravel(), weights=np.nan_to_num(arr, nan=0.0).ravel(), minlength=max_bin + 1)
    counts = np.bincount(bins.ravel(), minlength=max_bin + 1)
    profile = sums / np.maximum(counts, 1)
    return np.arange(max_bin + 1, dtype=np.float64), profile.astype(np.float64)


def _radial_local_maxima(profile: np.ndarray) -> np.ndarray:
    y = np.asarray(profile, dtype=np.float64)
    if y.size < 3:
        return np.zeros((0,), dtype=np.int64)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return np.zeros((0,), dtype=np.int64)
    threshold = max(float(np.percentile(finite, 80)), 0.10 * float(np.max(finite)))
    mask = (y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > threshold)
    return np.nonzero(mask)[0].astype(np.int64) + 1


def _expected_template_radii_px(
    template_metadata: dict[str, Any],
    template_idx: int,
    max_radius_px: float,
) -> np.ndarray:
    orientations = template_metadata.get("orientations_deg", [])
    zone_axis_index = template_metadata.get("zone_axis_index", [0] * len(orientations))
    zi = int(zone_axis_index[template_idx]) if template_idx < len(zone_axis_index) else 0
    qnorms = template_metadata.get("per_zone_qnorm")
    if qnorms is not None and zi < len(qnorms):
        q = np.asarray(qnorms[zi], dtype=np.float64)
    elif template_metadata.get("per_zone_qxy") is not None and zi < len(template_metadata["per_zone_qxy"]):
        qxy = np.asarray(template_metadata["per_zone_qxy"][zi], dtype=np.float64)
        q = np.linalg.norm(qxy, axis=1)
    elif template_metadata.get("qxy") is not None:
        qxy = np.asarray(template_metadata["qxy"], dtype=np.float64)
        q = np.linalg.norm(qxy, axis=1)
    else:
        return np.zeros((0,), dtype=np.float64)
    scale = float(template_metadata.get("reciprocal_pixels_per_inv_angstrom", 1.0))
    radii = q * scale
    radii = radii[(radii > 1.0) & (radii < max_radius_px)]
    if radii.size == 0:
        return radii
    return _cluster_radii(radii, min_separation_px=3.0)


def _cluster_radii(radii: np.ndarray, *, min_separation_px: float) -> np.ndarray:
    values = np.sort(np.asarray(radii, dtype=np.float64))
    if values.size == 0:
        return values
    clusters: list[list[float]] = [[float(values[0])]]
    for value in values[1:]:
        if float(value) - clusters[-1][-1] <= min_separation_px:
            clusters[-1].append(float(value))
        else:
            clusters.append([float(value)])
    return np.asarray([float(np.mean(cluster)) for cluster in clusters], dtype=np.float64)


def _save_radial_q_profile_validation(
    path: Path,
    *,
    mean_dp: np.ndarray,
    template_metadata: dict[str, Any],
    template_idx: int,
) -> None:
    src_shape = tuple(int(v) for v in template_metadata.get("sig_shape", mean_dp.shape))
    dst_shape = tuple(int(v) for v in mean_dp.shape)
    beam = template_metadata.get("beam_center_yx")
    if beam is not None:
        center = (
            float(beam[0]) * dst_shape[0] / max(float(src_shape[0]), 1.0),
            float(beam[1]) * dst_shape[1] / max(float(src_shape[1]), 1.0),
        )
    else:
        center = None
    radii, profile = _radial_profile(mean_dp, center)
    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return
    norm = profile - float(np.min(finite))
    denom = float(np.max(norm)) if float(np.max(norm)) > 0 else 1.0
    norm = np.clip(norm / denom, 0, 1)

    canvas = np.full((420, 760, 3), 255, dtype=np.uint8)
    ml, mr, mt, mb = 46, 14, 16, 34  # tight margins — pyxem-style compact layout
    x0, x1 = ml, canvas.shape[1] - mr - 1
    y0, y1 = mt, canvas.shape[0] - mb - 1
    canvas[y0:y1 + 1, x0] = 35
    canvas[y1, x0:x1 + 1] = 35
    for frac in (0.25, 0.5, 0.75):
        gy = int(y1 - frac * (y1 - y0))
        canvas[gy, x0:x1 + 1] = 228

    px = (x0 + (radii / max(float(radii[-1]), 1.0)) * (x1 - x0)).astype(int)
    py = (y1 - norm * (y1 - y0)).astype(int)
    _draw_polyline_local(canvas, np.column_stack([px, py]), (25, 25, 25))

    radius_scale = min(
        float(dst_shape[0]) / max(float(src_shape[0]), 1.0),
        float(dst_shape[1]) / max(float(src_shape[1]), 1.0),
    )
    exp_peaks = radii[_radial_local_maxima(norm)]
    expected = _expected_template_radii_px(
        template_metadata,
        template_idx,
        float(radii[-1]) / max(radius_scale, 1e-12),
    )
    expected = expected * radius_scale
    expected = expected[(expected > 1.0) & (expected < float(radii[-1]))]
    expected = _cluster_radii(expected, min_separation_px=3.0)
    tol = max(2.0, float(template_metadata.get("peak_sigma_px", 5.0)))
    matched_expected: set[float] = set()
    for er in expected:
        if exp_peaks.size and float(np.min(np.abs(exp_peaks - er))) <= tol:
            matched_expected.add(float(er))

    for er in expected:
        x = int(x0 + er / max(float(radii[-1]), 1.0) * (x1 - x0))
        color = (20, 170, 80) if float(er) in matched_expected else (30, 120, 255)
        canvas[y0:y1 + 1, max(x - 1, x0):min(x + 2, x1 + 1)] = color
    for rr in exp_peaks:
        if expected.size == 0 or float(np.min(np.abs(expected - rr))) > tol:
            x = int(x0 + rr / max(float(radii[-1]), 1.0) * (x1 - x0))
            canvas[y0:y1 + 1, max(x - 1, x0):min(x + 2, x1 + 1)] = (255, 60, 60)

    save_png(path, canvas)


def _compute_radial_support_evidence(
    mean_dp: np.ndarray,
    template_metadata: dict[str, Any],
    template_idx: int,
) -> dict[str, Any]:
    src_shape = tuple(int(v) for v in template_metadata.get("sig_shape", mean_dp.shape))
    dst_shape = tuple(int(v) for v in mean_dp.shape)
    beam = template_metadata.get("beam_center_yx")
    if beam is not None:
        center = (
            float(beam[0]) * dst_shape[0] / max(float(src_shape[0]), 1.0),
            float(beam[1]) * dst_shape[1] / max(float(src_shape[1]), 1.0),
        )
    else:
        center = None

    radii, profile = _radial_profile(mean_dp, center)
    if radii.size == 0:
        return {"radial_support_score": None, "expected_q_bands": [], "matched_q_bands": [], "experimental_q_peaks": []}
    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return {"radial_support_score": None, "expected_q_bands": [], "matched_q_bands": [], "experimental_q_peaks": []}
    norm = profile - float(np.min(finite))
    denom = float(np.max(norm)) if float(np.max(norm)) > 0 else 1.0
    norm = np.clip(norm / denom, 0, 1)

    radius_scale = min(
        float(dst_shape[0]) / max(float(src_shape[0]), 1.0),
        float(dst_shape[1]) / max(float(src_shape[1]), 1.0),
    )
    experimental = radii[_radial_local_maxima(norm)]
    expected = _expected_template_radii_px(
        template_metadata,
        template_idx,
        float(radii[-1]) / max(radius_scale, 1e-12),
    )
    expected = expected * radius_scale
    expected = expected[(expected > 1.0) & (expected < float(radii[-1]))]
    expected = _cluster_radii(expected, min_separation_px=3.0)
    if expected.size == 0:
        return {
            "radial_support_score": None,
            "expected_q_bands": [],
            "matched_q_bands": [],
            "experimental_q_peaks": [round(float(v), 2) for v in experimental[:20]],
        }

    tol = max(2.0, float(template_metadata.get("peak_sigma_px", 5.0)) * radius_scale)
    matched: list[float] = []
    for er in expected:
        if experimental.size and float(np.min(np.abs(experimental - er))) <= tol:
            matched.append(float(er))
    score = len(matched) / max(len(expected), 1)
    return {
        "radial_support_score": round(float(score), 4),
        "expected_q_bands": [round(float(v), 2) for v in expected[:30]],
        "matched_q_bands": [round(float(v), 2) for v in matched[:30]],
        "experimental_q_peaks": [round(float(v), 2) for v in experimental[:30]],
        "radial_tolerance_px": round(float(tol), 2),
    }


def _build_phase_orientation_evidence(
    roi: dict[str, Any],
    mean_dp: np.ndarray,
    all_hits: list[dict[str, Any]],
    matching_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    top_k = int(matching_cfg["top_k_per_phase"])
    radial_gate_enabled = bool(matching_cfg["radial_gate_enabled"])
    radial_min_support = float(matching_cfg["radial_min_support"])

    grouped: dict[str, list[dict[str, Any]]] = {}
    for hit in all_hits:
        phase = _candidate_display_name(hit["candidate"])
        grouped.setdefault(phase, []).append(hit)

    evidence: list[dict[str, Any]] = []
    for phase, hits in grouped.items():
        hits.sort(key=lambda h: float(h["score"]), reverse=True)
        best_hit = hits[0]
        top_hits = hits[:top_k]
        metadata = best_hit.get("metadata") or {}
        candidate = best_hit["candidate"]

        res: dict[str, Any] = {}
        try:
            res = _compute_peak_residual_metrics(roi, metadata, int(best_hit["template_idx"]))
        except Exception as exc:
            log.warning("Peak residual skipped for %s: %s", candidate.name, exc)

        observable_frac: float | None = None
        matched_obs: float | None = None
        try:
            tmpl_peaks = _reconstruct_template_peak_positions(metadata, int(best_hit["template_idx"]))
            if tmpl_peaks is not None and len(tmpl_peaks) > 0:
                observable_frac = _compute_observable_template_fraction(tmpl_peaks, tuple(metadata["sig_shape"]))
                raw_matched_frac = res.get("matched_template_fraction") or 0.0
                matched_obs = round(raw_matched_frac / observable_frac, 4) if observable_frac > 0 else 0.0
        except Exception:
            observable_frac = None
            matched_obs = None

        radial = _compute_radial_support_evidence(mean_dp, metadata, int(best_hit["template_idx"]))
        radial_score = radial.get("radial_support_score")
        if radial_score is None:
            gate_status = "NOT_EVALUATED"
        elif float(radial_score) >= radial_min_support:
            gate_status = "PASS"
        else:
            gate_status = "LOW_SUPPORT"

        hybrid = _compute_hybrid_validation_score(
            correlation_score=float(best_hit["score"]),
            matched_observable_fraction=matched_obs,
            mean_q_residual=res.get("mean_q_residual"),
            unexplained_experiment_fraction=res.get("unexplained_experiment_fraction"),
        )
        evidence_score = hybrid
        if radial_gate_enabled and radial_score is not None:
            evidence_score = round(0.8 * hybrid + 0.2 * float(radial_score), 4)
            if gate_status == "LOW_SUPPORT":
                evidence_score = round(evidence_score * 0.75, 4)

        orientation_margin = None
        if len(hits) > 1:
            orientation_margin = round(float(hits[0]["score"]) - float(hits[1]["score"]), 4)

        evidence.append({
            "phase": phase,
            "candidate": candidate.name,
            "correlation_score": round(float(best_hit["score"]), 4),
            "hybrid_score": hybrid,
            "evidence_score": evidence_score,
            "radial_support_score": radial_score,
            "radial_gate_status": gate_status,
            "radial_evidence": radial,
            "matched_peak_count": res.get("matched_peak_count"),
            "mean_q_residual": res.get("mean_q_residual"),
            "mean_angle_residual": res.get("mean_angle_residual"),
            "matched_template_fraction": res.get("matched_template_fraction"),
            "unexplained_experiment_fraction": res.get("unexplained_experiment_fraction"),
            "matched_observable_template_fraction": matched_obs,
            "matched_observable_fraction": matched_obs,
            "observable_template_fraction": observable_frac,
            "zone_axis": best_hit["zone_axis"],
            "orientation_deg": best_hit["orientation_deg"],
            "template_idx": best_hit["template_idx"],
            "stack": best_hit["stack"],
            "all_scores": best_hit["all_scores"],
            "orientations_deg": best_hit["orientations_deg"],
            "candidate_obj": candidate,
            "orientation_margin": orientation_margin,
            "top_matches": [
                {
                    "rank": rank + 1,
                    "correlation_score": round(float(hit["score"]), 4),
                    "zone_axis": hit["zone_axis"],
                    "orientation_deg": hit["orientation_deg"],
                    "template_idx": int(hit["template_idx"]),
                }
                for rank, hit in enumerate(top_hits)
            ],
        })

    evidence.sort(key=lambda c: float(c.get("evidence_score", 0.0)), reverse=True)
    return evidence


def _write_phase_orientation_topk(
    roi: dict[str, Any],
    evidence: list[dict[str, Any]],
    matching_cfg: dict[str, Any],
) -> None:
    bragg_summary_path = roi.get("bragg_summary_path")
    if not bragg_summary_path:
        return
    try:
        roi_dir = Path(bragg_summary_path).parent
        payload = {
            "schema_version": "stage2b-phase-orientation-topk-v1",
            "roi": roi.get("name", "unknown"),
            "matching": matching_cfg,
            "phase_matches": [_jsonable_phase_evidence(row) for row in evidence],
        }
        (roi_dir / "phase_orientation_topk.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not write phase/orientation top-k evidence for %s: %s", roi.get("name", "unknown"), exc)


def _jsonable_phase_evidence(row: dict[str, Any]) -> dict[str, Any]:
    omit = {"stack", "candidate_obj", "all_scores", "orientations_deg"}
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in omit:
            continue
        if isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _phase_evidence_margin(evidence: list[dict[str, Any]]) -> float | None:
    if len(evidence) < 2:
        return None
    return round(float(evidence[0].get("evidence_score", 0.0)) - float(evidence[1].get("evidence_score", 0.0)), 4)


def _phase_evidence_confidence(
    evidence_score: float | None,
    phase_margin: float | None,
    radial_support_score: float | None,
    matching_cfg: dict[str, Any],
) -> str:
    if evidence_score is None or evidence_score <= 0.0:
        return "UNINDEXED"
    margin = phase_margin or 0.0
    radial_ok = radial_support_score is None or radial_support_score >= float(matching_cfg["radial_min_support"])
    if evidence_score > 0.55 and margin > max(float(matching_cfg["phase_margin_threshold"]), 0.10) and radial_ok:
        return "HIGH_CONFIDENCE"
    if evidence_score > 0.40 and margin >= float(matching_cfg["phase_margin_threshold"]) and radial_ok:
        return "MEDIUM_CONFIDENCE"
    return "LOW_CONFIDENCE"


def _orientation_confidence(
    orientation_margin: float | None,
    matching_cfg: dict[str, Any],
) -> str:
    if orientation_margin is None:
        return "LOW_CONFIDENCE"
    threshold = float(matching_cfg["orientation_margin_threshold"])
    if orientation_margin >= max(0.10, 2.0 * threshold):
        return "HIGH_CONFIDENCE"
    if orientation_margin >= threshold:
        return "MEDIUM_CONFIDENCE"
    return "LOW_CONFIDENCE"


def _combined_mapping_confidence(phase_confidence: str, orientation_confidence: str) -> str:
    order = {"UNINDEXED": 0, "LOW_CONFIDENCE": 1, "MEDIUM_CONFIDENCE": 2, "HIGH_CONFIDENCE": 3}
    reverse = {v: k for k, v in order.items()}
    return reverse[min(order.get(phase_confidence, 1), order.get(orientation_confidence, 1))]


def _apply_extinctions(
    hkls: np.ndarray,
    space_group: int | None,
) -> np.ndarray:
    """Filter *hkls* to remove systematically absent reflections.

    When a CIF uses P1 (space group 1) for convenience but the real
    structure has higher symmetry, the user can supply the true space
    group number in the config.  This function removes kinematically
    forbidden reflections that would add phantom spots to the templates.

    Parameters
    ----------
    hkls:
        (N, 3) int array of Miller indices.
    space_group:
        International Tables space group number, or *None* (P1 — no
        filtering).

    Returns
    -------
    Boolean mask ``(N,)`` where ``True`` = reflection is **allowed**.
    """
    if space_group is None or space_group == 1:
        return np.ones(len(hkls), dtype=bool)

    h = hkls[:, 0].astype(np.int64)
    k = hkls[:, 1].astype(np.int64)
    l = hkls[:, 2].astype(np.int64)

    # Start with all allowed, then knock out forbidden families.
    allowed = np.ones(len(hkls), dtype=bool)

    if space_group == 194:   # P6_3/mmc — α-Ti (hcp)
        # 6_3 screw along c: 00l with l odd → absent
        axial = (h == 0) & (k == 0)
        allowed[axial & (l % 2 != 0)] = False

    elif space_group == 229:  # Im-3m — β-Ti (bcc)
        # Body centering: h + k + l odd → absent
        allowed[(h + k + l) % 2 != 0] = False

    elif space_group == 225:  # Fm-3m — FCC
        # All-face centering: h,k,l must be all even or all odd
        parity_sum = (h % 2) + (k % 2) + (l % 2)
        allowed[(parity_sum != 0) & (parity_sum != 3)] = False

    elif space_group == 227:  # Fd-3m — diamond
        # h,k,l all even AND h+k+l = 4n, OR h,k,l all odd
        # (simplified: remove the most prominent forbidden families)
        all_even = (h % 2 == 0) & (k % 2 == 0) & (l % 2 == 0)
        all_odd  = (h % 2 != 0) & (k % 2 != 0) & (l % 2 != 0)
        diamond_forbidden = all_even & ((h + k + l) % 4 != 0)
        allowed[diamond_forbidden] = False
        # Also: 0kl with k+l not multiple of 4
        h0 = (h == 0)
        allowed[h0 & ((k + l) % 4 != 0)] = False

    elif space_group == 136:  # P4_2/mnm — rutile TiO2
        # 4_2 screw along c: 00l with l odd → absent
        axial = (h == 0) & (k == 0)
        allowed[axial & (l % 2 != 0)] = False
        # n-glide ⊥ [110]: 0kl with k+l odd → absent
        h0 = (h == 0)
        allowed[h0 & ((k + l) % 2 != 0)] = False

    elif space_group == 166:  # R-3m — rhombohedral
        # -h + k + l ≠ 3n → absent (obverse setting)
        allowed[(-h + k + l) % 3 != 0] = False

    elif space_group == 191:  # P6/mmm — simple hexagonal
        pass  # No systematic extinctions for primitive hexagonal

    # Unrecognised space groups pass through with a logged note.
    else:
        log.info(
            "No extinction rules implemented for space group %d; "
            "all %d reflections retained.",
            space_group, len(hkls),
        )

    return allowed


def _generate_kinematic_template_stack(
    cell: dict[str, float],
    *,
    sig_shape: tuple[int, int],
    beam_center_yx: tuple[float, float],
    max_index: int,
    orientations_deg: list[float],
    zone_axis: tuple[float, float, float],
    peak_sigma_px: float,
    reciprocal_pixels_per_inv_angstrom: float | None,
    intensity_power: float,
    space_group: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    hkls, qxy, qnorm, projection = _reciprocal_spots(cell, max_index, zone_axis)
    if len(hkls) == 0:
        raise ValueError("No reciprocal spots generated from CIF cell.")

    # --- Apply space-group extinction filtering -------------------------------
    n_before = len(hkls)
    extinction_mask = _apply_extinctions(hkls, space_group)
    hkls = hkls[extinction_mask]
    qxy = qxy[extinction_mask]
    qnorm = qnorm[extinction_mask]
    n_extinct = n_before - len(hkls)
    if n_extinct > 0:
        log.info(
            "Space group %s: %d/%d reflections removed by extinction rules (%d retained).",
            space_group, n_extinct, n_before, len(hkls),
        )

    detector_radius = 0.48 * min(sig_shape)
    scale = reciprocal_pixels_per_inv_angstrom
    scale_source = "config"
    if scale is None:
        scale = detector_radius / max(float(np.max(qnorm)), 1e-6)
        scale_source = "auto_fit_to_detector"
        log.info(
            "Auto-scaled Stage 2B reciprocal template scale to %.4f px/A^-1 "
            "(source=%s, detector_radius_fraction=0.48, zone_axis=%s). "
            "Set reciprocal_pixels_per_inv_angstrom for calibrated matching.",
            scale,
            scale_source,
            list(zone_axis),
        )

    stack = np.zeros((len(orientations_deg), sig_shape[0], sig_shape[1]), dtype=np.float32)
    for i, angle in enumerate(orientations_deg):
        rotated = _rotate_xy(qxy, angle)
        coords_yx = np.column_stack([
            beam_center_yx[0] + rotated[:, 1] * scale,
            beam_center_yx[1] + rotated[:, 0] * scale,
        ])
        intensities = 1.0 / np.maximum(qnorm, 1e-6) ** intensity_power
        stack[i] = _render_gaussian_spots(sig_shape, coords_yx, intensities, peak_sigma_px)
        stack[i] = _scale_unit_interval(stack[i])

    metadata = {
        "cell": cell,
        "max_index": max_index,
        "hkl_count": int(len(hkls)),
        "hkls": hkls.tolist(),
        "qxy": qxy.tolist(),
        "qnorm": qnorm.tolist(),
        "orientations_deg": orientations_deg,
        "zone_axis": list(zone_axis),
        "projection": projection,
        "sig_shape": list(sig_shape),
        "beam_center_yx": list(beam_center_yx),
        "peak_sigma_px": peak_sigma_px,
        "reciprocal_pixels_per_inv_angstrom": scale,
        "reciprocal_scale_source": scale_source,
        "intensity_power": intensity_power,
        "space_group": space_group,
        "n_extinct_removed": n_extinct,
    }
    return stack, metadata


def _reciprocal_spots(
    cell: dict[str, float],
    max_index: int,
    zone_axis: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    reciprocal = _reciprocal_basis(cell)
    zone_unit, plane_x, plane_y = _zone_projection_basis(zone_axis)
    hkls: list[tuple[int, int, int]] = []
    q_vectors: list[np.ndarray] = []
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                q_vec = h * reciprocal[0] + k * reciprocal[1] + l * reciprocal[2]
                q_norm = float(np.linalg.norm(q_vec))
                if q_norm <= 0:
                    continue
                hkls.append((h, k, l))
                q_vectors.append(q_vec)
    q_array = np.asarray(q_vectors, dtype=np.float64)
    hkls_array = np.asarray(hkls, dtype=np.int16)
    order = np.argsort(np.linalg.norm(q_array, axis=1))
    q_array = q_array[order]
    hkls_array = hkls_array[order]
    qxy = np.column_stack([q_array @ plane_x, q_array @ plane_y])
    projection = {
        "mode": "single_zone_axis_orthographic",
        "zone_axis": zone_unit.tolist(),
        "plane_x": plane_x.tolist(),
        "plane_y": plane_y.tolist(),
        "limitations": (
            "Single-zone kinematic approximation: in-plane rotation only; "
            "tilt, precession, excitation error, and multi-zone coverage are not modeled."
        ),
    }
    return hkls_array, qxy, np.linalg.norm(q_array, axis=1), projection


# ---------------------------------------------------------------------------
# Peak-position residual analysis (P0 validation)
# ---------------------------------------------------------------------------


def _reconstruct_template_peak_positions(
    metadata: dict[str, Any],
    template_idx: int,
) -> np.ndarray | None:
    """Return (N, 2) float64 array of template peak pixel positions [y, x].

    Uses per-zone hkls/qxy from metadata when available; falls back to
    regenerating from cell parameters for backward compatibility with
    template files that lack per-zone persistence.
    """
    try:
        orientations = metadata.get("orientations_deg", [])
        zone_axis_index = metadata.get("zone_axis_index", [0] * len(orientations))
        zone_axes = metadata.get("zone_axes", [[0.0, 0.0, 1.0]])
        sig_shape = tuple(metadata["sig_shape"])
        beam_center = tuple(metadata["beam_center_yx"])
        scale = float(metadata["reciprocal_pixels_per_inv_angstrom"])
        angle_deg = float(orientations[template_idx])
        zi = int(zone_axis_index[template_idx]) if template_idx < len(zone_axis_index) else 0

        per_zone_qxy = metadata.get("per_zone_qxy")
        cell = metadata.get("cell")
        max_index = metadata.get("max_index", 4)
        space_group = metadata.get("space_group")

        if per_zone_qxy is not None and zi < len(per_zone_qxy):
            qxy = np.asarray(per_zone_qxy[zi], dtype=np.float64)
        elif cell is not None and zone_axes and zi < len(zone_axes):
            # Backward-compat fallback: regenerate from cell
            hkls, qxy, _qnorm, _proj = _reciprocal_spots(
                cell, int(max_index), tuple(float(v) for v in zone_axes[zi]),
            )
            mask = _apply_extinctions(hkls, space_group)
            qxy = qxy[mask]
        else:
            return None

        if len(qxy) == 0:
            return None

        rotated = _rotate_xy(qxy, angle_deg)
        coords = np.column_stack([
            beam_center[0] + rotated[:, 1] * scale,
            beam_center[1] + rotated[:, 0] * scale,
        ])

        # Clip to detector bounds
        valid = (
            (coords[:, 0] >= 0) & (coords[:, 0] < sig_shape[0])
            & (coords[:, 1] >= 0) & (coords[:, 1] < sig_shape[1])
        )
        return coords[valid].astype(np.float64)

    except Exception as exc:
        log.warning("Failed to reconstruct template peak positions: %s", exc)
        return None


def _extract_measured_peak_positions(
    vmap: np.ndarray,
    *,
    min_intensity: float = 0.0,
    min_spacing: int = 2,
) -> np.ndarray:
    """Return (M, 2) int array of measured peak pixel positions [y, x].

    Finds local maxima in the Bragg vector map above *min_intensity*.
    """
    arr = np.asarray(vmap, dtype=np.float64)
    if arr.size == 0 or arr.max() <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    # Identify local maxima: pixel >= all neighbours within min_spacing
    from scipy.ndimage import maximum_filter
    footprint = np.ones((2 * min_spacing + 1, 2 * min_spacing + 1), dtype=bool)
    local_max = arr >= maximum_filter(arr, footprint=footprint)
    thresholded = arr > float(min_intensity)
    peaks = local_max & thresholded
    rows, cols = np.nonzero(peaks)
    return np.column_stack([rows, cols]).astype(np.int64)


def _match_peaks_residual(
    template_peaks: np.ndarray,
    measured_peaks: np.ndarray,
    tolerance_px: float,
) -> dict[str, Any]:
    """Match template peaks to measured peaks within a radial tolerance.

    Greedy closest-pair matching: each measured peak may match at most one
    template peak.

    Returns dict with matched_peak_count, mean_q_residual (px),
    mean_angle_residual (rad), matched_template_fraction,
    unexplained_experiment_fraction.
    """
    assignments = _peak_match_assignments(template_peaks, measured_peaks, tolerance_px)
    return {
        "matched_peak_count": assignments["matched_peak_count"],
        "mean_q_residual": assignments["mean_q_residual"],
        "mean_angle_residual": assignments["mean_angle_residual"],
        "matched_template_fraction": assignments["matched_template_fraction"],
        "unexplained_experiment_fraction": assignments["unexplained_experiment_fraction"],
    }


def _compute_peak_residual_metrics(
    roi: dict[str, Any],
    template_metadata: dict[str, Any],
    best_template_idx: int,
    *,
    tolerance_px: float | None = None,
) -> dict[str, Any]:
    """Orchestrate peak residual analysis for one ROI's best match.

    Loads the Bragg vector map, extracts measured peaks, reconstructs
    template peak positions, and runs the matching.
    """
    result: dict[str, Any] = {
        "matched_peak_count": None,
        "mean_q_residual": None,
        "mean_angle_residual": None,
        "matched_template_fraction": None,
        "unexplained_experiment_fraction": None,
        "tolerance_px": None,
        "n_template_peaks": None,
        "n_measured_peaks": None,
        "warning": None,
    }

    # Load Bragg vector map
    bvm_path = roi.get("bragg_vector_map_path")
    if not bvm_path:
        result["warning"] = "No bragg_vector_map_path for ROI; cannot extract measured peaks."
        return result
    try:
        vmap = np.load(bvm_path)
    except OSError as exc:
        result["warning"] = f"Could not load Bragg vector map: {exc}"
        return result

    # Handle binning mismatch (template sig_shape vs vmap shape)
    tmpl_h, tmpl_w = template_metadata["sig_shape"]
    vmap_h, vmap_w = vmap.shape
    bin_y, bin_x = tmpl_h // vmap_h, tmpl_w // vmap_w
    if bin_y > 1 and bin_x > 1 and tmpl_h % bin_y == 0 and tmpl_w % bin_x == 0:
        bin_factor = float(bin_y)
    else:
        bin_factor = 1.0

    sigma_px = float(template_metadata.get("peak_sigma_px", 5.0))
    tol = tolerance_px if tolerance_px is not None else round(2.5 * sigma_px / bin_factor, 1)
    result["tolerance_px"] = tol

    # Extract measured peaks
    measured = _extract_measured_peak_positions(vmap, min_spacing=2)
    # Scale measured coords to template pixel space
    if bin_factor != 1.0 and len(measured) > 0:
        measured = (measured.astype(np.float64) * bin_factor + bin_factor / 2.0).astype(np.int64)
    result["n_measured_peaks"] = len(measured)

    # Reconstruct template peaks
    template_peaks = _reconstruct_template_peak_positions(template_metadata, best_template_idx)
    if template_peaks is None:
        result["warning"] = "Could not reconstruct template peak positions."
        return result
    result["n_template_peaks"] = len(template_peaks)

    if len(template_peaks) < 3 or len(measured) < 3:
        result["warning"] = (
            f"Too few peaks for reliable matching "
            f"(template={len(template_peaks)}, measured={len(measured)})."
        )
        return result

    match = _match_peaks_residual(template_peaks, measured, tol)
    result.update(match)
    return result


# ---------------------------------------------------------------------------
# Hybrid validation scoring and ambiguity-aware phase resolution (v3)
# ---------------------------------------------------------------------------


def _compute_observable_template_fraction(
    template_peaks: np.ndarray,
    sig_shape: tuple[int, int],
) -> float:
    """Return fraction of template peaks within the detector bounds.

    High-q peaks that fall outside the detector are not observable at the
    current camera length / binning — they should not count against the
    matched fraction.
    """
    if len(template_peaks) == 0:
        return 0.0
    in_bounds = (
        (template_peaks[:, 0] >= 0) & (template_peaks[:, 0] < sig_shape[0])
        & (template_peaks[:, 1] >= 0) & (template_peaks[:, 1] < sig_shape[1])
    )
    return round(float(np.mean(in_bounds)), 4)


def _compute_hybrid_validation_score(
    correlation_score: float,
    matched_observable_fraction: float | None,
    mean_q_residual: float | None,
    unexplained_experiment_fraction: float | None,
) -> float:
    """Combine correlation and peak-matching evidence into a single score [0, 1].

    Weights reflect the relative reliability of each signal:
    - Correlation score (35%): overall pattern match quality
    - Matched observable fraction (40%): strongest discriminator for correct phase
    - q residual penalty (15%): penalises poor positional accuracy
    - Unexplained fraction penalty (10%): penalises many unmatched measured peaks
    """
    score = 0.0

    # Correlation: already in [0, 1] for positive matches
    score += 0.35 * max(0.0, float(correlation_score))

    # Observable matched fraction: best single discriminator
    mof = matched_observable_fraction if matched_observable_fraction is not None else 0.0
    score += 0.40 * max(0.0, min(1.0, float(mof)))

    # q residual: normalise to [0, 1] where 0 px → 1.0, 20+ px → 0.0
    qr = mean_q_residual if mean_q_residual is not None else 20.0
    qr_norm = max(0.0, 1.0 - float(qr) / 20.0)
    score += 0.15 * qr_norm

    # Unexplained fraction: lower is better
    uf = unexplained_experiment_fraction if unexplained_experiment_fraction is not None else 1.0
    uf_norm = max(0.0, 1.0 - float(uf))
    score += 0.10 * uf_norm

    return round(score, 4)


def _resolve_phase_call(
    best_by_corr: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    hybrid_margin_threshold: float = 0.08,
    matched_frac_threshold: float = 0.20,
) -> dict[str, Any]:
    """Determine the final phase call from hybrid-ranked candidates.

    Returns dict with:
    - ``phase_call``: candidate phase name, ``"AMBIGUOUS"``, or ``"UNINDEXED"``
    - ``candidate_group``: pipe-joined phase names when AMBIGUOUS
    - ``reason``: human-readable explanation
    """
    if not candidates:
        return {"phase_call": "UNINDEXED", "candidate_group": None, "reason": "No candidates scored."}

    best = candidates[0]
    best_corr_score = float(best.get("correlation_score", 0))
    best_hybrid = float(best.get("evidence_score", best.get("hybrid_score", 0)))
    best_mof = best.get("matched_observable_fraction", best.get("matched_observable_template_fraction"))

    # UNINDEXED: correlation too low
    if best_corr_score <= 0.0:
        return {
            "phase_call": "UNINDEXED",
            "candidate_group": None,
            "reason": f"Best correlation score ({best_corr_score:.4f}) <= 0.",
        }

    if len(candidates) == 1:
        return {
            "phase_call": str(best.get("phase", "unknown")),
            "candidate_group": None,
            "reason": "Single candidate.",
        }

    second = candidates[1]
    second_hybrid = float(second.get("evidence_score", second.get("hybrid_score", 0)))
    second_mof = second.get("matched_observable_fraction", second.get("matched_observable_template_fraction"))
    hybrid_margin = best_hybrid - second_hybrid

    # AMBIGUOUS: hybrid scores too close
    if hybrid_margin < hybrid_margin_threshold and (best_mof or 0) < matched_frac_threshold:
        names = sorted({str(best.get("phase", "?")), str(second.get("phase", "?"))})
        reason_parts = []
        if best_mof and second_mof:
            reason_parts.append(
                f"Hybrid margin {hybrid_margin:.3f} < {hybrid_margin_threshold}, "
                f"matched fractions {best_mof:.2%}/{second_mof:.2%}"
            )
        else:
            reason_parts.append(f"Hybrid margin {hybrid_margin:.3f} < {hybrid_margin_threshold}")
        if best.get("correlation_score", 0) < second.get("correlation_score", 0):
            reason_parts.append(
                f"Correlation winner ({second['phase']}) ≠ peak-matching winner ({best['phase']})"
            )
        return {
            "phase_call": "AMBIGUOUS",
            "candidate_group": " / ".join(names),
            "reason": ". ".join(reason_parts),
        }

    # AMBIGUOUS: correlation winner ≠ hybrid (peak-matching) winner
    corr_winner = max(candidates, key=lambda c: float(c.get("correlation_score", 0)))
    if corr_winner.get("phase") != best.get("phase"):
        corr_best_mof = corr_winner.get("matched_observable_fraction", corr_winner.get("matched_observable_template_fraction"))
        if corr_best_mof and best_mof and best_mof > corr_best_mof:
            names = sorted({str(corr_winner.get("phase", "?")), str(best.get("phase", "?"))})
            return {
                "phase_call": "AMBIGUOUS",
                "candidate_group": " / ".join(names),
                "reason": (
                    f"Correlation favours {corr_winner['phase']} "
                    f"({corr_winner.get('correlation_score', 0):.4f}) "
                    f"but peak matching favours {best['phase']} "
                    f"(obs-matched {best_mof:.2%} vs {corr_best_mof:.2%})"
                ),
            }

    # UNAMBIGUOUS
    return {
        "phase_call": str(best.get("phase", "unknown")),
        "candidate_group": None,
        "reason": (
            f"Hybrid margin {hybrid_margin:.3f} >= {hybrid_margin_threshold}, "
            f"obs-matched fraction {best_mof:.2%}" if best_mof else f"Hybrid margin {hybrid_margin:.3f}"
        ),
    }


def _zone_projection_basis(
    zone_axis: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zone = np.asarray(zone_axis, dtype=np.float64)
    norm = float(np.linalg.norm(zone))
    if norm <= 1e-12:
        raise ValueError("zone_axis must be non-zero.")
    zone_unit = zone / norm
    reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(zone_unit, reference))) > 0.95:
        reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    plane_x = np.cross(reference, zone_unit)
    plane_x = plane_x / max(float(np.linalg.norm(plane_x)), 1e-12)
    plane_y = np.cross(zone_unit, plane_x)
    plane_y = plane_y / max(float(np.linalg.norm(plane_y)), 1e-12)
    return zone_unit, plane_x, plane_y


def _reciprocal_basis(cell: dict[str, float]) -> np.ndarray:
    a = float(cell["a"])
    b = float(cell["b"])
    c = float(cell["c"])
    alpha = math.radians(float(cell["alpha"]))
    beta = math.radians(float(cell["beta"]))
    gamma = math.radians(float(cell["gamma"]))
    direct = np.asarray([
        [a, 0.0, 0.0],
        [b * math.cos(gamma), b * math.sin(gamma), 0.0],
        [
            c * math.cos(beta),
            c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / max(math.sin(gamma), 1e-12),
            c * math.sqrt(max(
                1.0 - math.cos(beta) ** 2
                - ((math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / max(math.sin(gamma), 1e-12)) ** 2,
                0.0,
            )),
        ],
    ], dtype=np.float64)
    volume = float(np.dot(direct[0], np.cross(direct[1], direct[2])))
    if abs(volume) < 1e-12:
        raise ValueError(f"Invalid CIF cell volume for cell {cell}.")
    return np.asarray([
        np.cross(direct[1], direct[2]) / volume,
        np.cross(direct[2], direct[0]) / volume,
        np.cross(direct[0], direct[1]) / volume,
    ], dtype=np.float64)


def _rotate_xy(xy: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    rot = np.asarray([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ], dtype=np.float64)
    return xy @ rot.T


def _render_gaussian_spots(
    sig_shape: tuple[int, int],
    coords_yx: np.ndarray,
    intensities: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    image = np.zeros(sig_shape, dtype=np.float32)
    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    for (cy, cx), intensity in zip(coords_yx, intensities, strict=False):
        if cy < -radius or cy >= sig_shape[0] + radius or cx < -radius or cx >= sig_shape[1] + radius:
            continue
        y0 = max(0, int(math.floor(cy)) - radius)
        y1 = min(sig_shape[0], int(math.floor(cy)) + radius + 1)
        x0 = max(0, int(math.floor(cx)) - radius)
        x1 = min(sig_shape[1], int(math.floor(cx)) + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        image[y0:y1, x0:x1] += float(intensity) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma_px ** 2)
        )
    return image


def _mean_diffraction_pattern(roi_data: np.ndarray) -> np.ndarray:
    arr = np.asarray(roi_data, dtype=np.float32)
    if arr.ndim == 4:
        return arr.mean(axis=(0, 1))
    if arr.ndim == 2:
        return arr
    raise ValueError(f"ROI data must be 2D or 4D, got shape {arr.shape}.")


def _normalize_pattern(pattern: np.ndarray) -> np.ndarray | None:
    vec = np.asarray(pattern, dtype=np.float32).ravel()
    vec = vec - float(vec.mean())
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return None
    return vec / norm


def _scale_unit_interval(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    max_v = float(np.max(arr))
    if max_v <= 0:
        return arr
    return arr / max_v


_VALIDATION_TIERS = ("UNINDEXED", "LOW_CONFIDENCE", "MEDIUM_CONFIDENCE", "HIGH_CONFIDENCE")


def _template_quality(score: float) -> str:
    if score >= 0.55:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def _phase_confidence(
    best_score: float,
    score_margin: float | None,
    matched_template_fraction: float | None = None,
    mean_q_residual: float | None = None,
) -> str:
    """Per-ROI phase confidence tier (v3).

    Incorporates peak-position residuals when available.  Falls back to
    score-only criteria when residual metrics are None (backward compat).

    Returns one of ``"HIGH_CONFIDENCE"``, ``"MEDIUM_CONFIDENCE"``,
    ``"LOW_CONFIDENCE"``, or ``"UNINDEXED"``.
    """
    if best_score is None or best_score <= 0.0:
        return "UNINDEXED"
    if score_margin is None:
        score_margin = 0.0

    # HIGH_CONFIDENCE: strong score, big margin, good peak matching
    if (
        best_score > 0.55
        and score_margin > 0.10
        and (matched_template_fraction is None or matched_template_fraction > 0.50)
        and (mean_q_residual is None or mean_q_residual < 5.0)
    ):
        return "HIGH_CONFIDENCE"

    # MEDIUM_CONFIDENCE: adequate score and margin, acceptable peak matching
    if (
        best_score > 0.40
        and score_margin > 0.06
        and (matched_template_fraction is None or matched_template_fraction > 0.30)
    ):
        return "MEDIUM_CONFIDENCE"

    # LOW_CONFIDENCE: scores present but below thresholds
    return "LOW_CONFIDENCE"


def _check_score_signs(
    roi_results: list[ROIIndexingResult],
    summary: dict[str, Any],
) -> None:
    """QC check: flag negative or near-zero template correlation scores.

    Negative scores indicate the templates are anti-correlated with the
    data — a strong signal that the matching is unphysical (e.g. wrong
    zone axes, uncalibrated reciprocal scale, or matching against a
    Bragg vector map instead of a mean DP).

    Adds ``"score_sign_qc"`` to *summary*.
    """
    scored = [r for r in roi_results if r.match_score is not None]
    if not scored:
        summary["score_sign_qc"] = {
            "status": "NO_SCORES",
            "message": "No template match scores available for sign check.",
        }
        return

    scores = [r.match_score for r in scored]
    n_negative = sum(1 for s in scores if s < 0)
    n_near_zero = sum(1 for s in scores if abs(s) < 0.01)
    n_total = len(scores)

    if n_negative == n_total:
        summary["score_sign_qc"] = {
            "status": "FAIL",
            "severity": "critical",
            "message": (
                f"ALL {n_total} ROI(s) have negative template match scores "
                f"(range [{min(scores):.4f}, {max(scores):.4f}]). "
                "Templates are anti-correlated with the data. Likely causes: "
                "matching against bragg_vector_map instead of mean DP "
                "(set save_roi_data: true), uncalibrated reciprocal scale "
                "(set reciprocal_pixels_per_inv_angstrom), or single-zone-axis "
                "limitation (enable zone_axes)."
            ),
            "evidence": {
                "n_total": n_total,
                "n_negative": n_negative,
                "score_min": round(min(scores), 4),
                "score_max": round(max(scores), 4),
            },
        }
    elif n_negative > 0 or n_near_zero == n_total:
        summary["score_sign_qc"] = {
            "status": "PASS_WITH_WARNINGS",
            "severity": "warning",
            "message": (
                f"{n_negative}/{n_total} ROI(s) have negative match scores; "
                f"{n_near_zero}/{n_total} are near zero (|score| < 0.01). "
                "Some templates may be anti-correlated or uninformative. "
                "Check per-ROI template_match_overlay.png for alignment."
            ),
            "evidence": {
                "n_total": n_total,
                "n_negative": n_negative,
                "n_near_zero": n_near_zero,
                "score_min": round(min(scores), 4),
                "score_max": round(max(scores), 4),
            },
        }
    else:
        summary["score_sign_qc"] = {
            "status": "PASS",
            "severity": "info",
            "message": (
                f"All {n_total} ROI(s) have positive template match scores "
                f"(range [{min(scores):.4f}, {max(scores):.4f}])."
            ),
            "evidence": {
                "n_total": n_total,
                "score_min": round(min(scores), 4),
                "score_max": round(max(scores), 4),
            },
        }


def _stage2b_status(
    accepted_rois: list[dict[str, Any]],
    candidates: list[IndexingCandidate],
) -> str:
    if not accepted_rois:
        return "NO_ACCEPTED_ROIS"
    if any(c.template_count > 0 for c in candidates):
        return "TEMPLATE_MATCHED"
    if any(c.reference_peaks for c in candidates):
        return "MOCK_SCORED"
    return "NO_TEMPLATES"


def _candidate_scoring_mode(candidate: IndexingCandidate) -> str:
    if candidate.scoring_mode != "not_scored":
        return candidate.scoring_mode
    if candidate.reference_peaks:
        return "mock_peak_count"
    return "not_scored"


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        log.warning("Could not hash candidate CIF %s: %s", path, exc)
        return None
