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


@dataclass(frozen=True)
class ROIIndexingResult:
    """Indexing contract result for one Stage 2A ROI."""

    name: str
    status: str
    stage2a_bragg_summary_path: str | None
    n_bragg_peaks: int
    best_candidate: str | None
    phase_score: float | None
    orientation_score: float | None
    match_quality: str
    best_orientation_deg: float | None = None
    template_score: float | None = None
    scoring_mode: str = "not_scored"


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

    template_cfg = _template_config(cfg.get("template_generation", {}))
    candidates = _load_candidates(cfg.get("candidate_cifs", []), base_dir)
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
        _match_roi_against_candidates(roi, candidates)
        for roi in accepted_rois
    ]

    any_template = any(c.template_count > 0 for c in candidates)
    summary = {
        "schema_version": "stage2b-indexing-v1",
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
                "best_candidate": r.best_candidate,
                "phase_score": r.phase_score,
                "orientation_score": r.orientation_score,
                "match_quality": r.match_quality,
                "best_orientation_deg": r.best_orientation_deg,
                "template_score": r.template_score,
                "scoring_mode": r.scoring_mode,
            }
            for r in roi_results
        ],
        "notes": [
            "Stage 2B uses analytic kinematic CIF templates when lattice parameters are available.",
            "Scores are normalized template correlations on ROI mean diffraction patterns.",
            "Full structure-factor intensities and py4DSTEM/pyxem backend adapters are future extensions.",
        ],
    }
    if not any_template:
        summary["notes"].append("No analytic templates were generated; mock scoring may appear only for test fixtures.")

    summary_path = output_dir / "stage2_indexing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


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


def _load_candidates(items: list[dict[str, Any]], base_dir: Path) -> list[IndexingCandidate]:
    candidates: list[IndexingCandidate] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Each candidate_cifs entry must be a mapping.")
        raw_path = item.get("path")
        if not raw_path:
            raise ValueError("Each candidate_cifs entry must contain 'path'.")
        path = _resolve_path(raw_path, base_dir)
        cell = _parse_cif_cell(path)
        candidates.append(
            IndexingCandidate(
                name=str(item.get("name") or path.stem or f"candidate_{i:03d}"),
                phase=item.get("phase"),
                path=path,
                reference_peaks=tuple(item.get("reference_peaks") or ()),
                sha256=_sha256_file(path),
                cell=cell,
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
            stack, metadata = _generate_kinematic_template_stack(
                candidate.cell,
                sig_shape=sig_shape,
                beam_center_yx=beam_center,
                max_index=int(template_cfg["max_index"]),
                orientations_deg=[float(v) for v in template_cfg["orientations_deg"]],
                zone_axis=tuple(float(v) for v in template_cfg["zone_axis"]),
                peak_sigma_px=float(template_cfg["peak_sigma_px"]),
                reciprocal_pixels_per_inv_angstrom=template_cfg["reciprocal_pixels_per_inv_angstrom"],
                intensity_power=float(template_cfg["intensity_power"]),
            )
        except ValueError as exc:
            log.warning("Could not generate templates for %s: %s", candidate.name, exc)
            result.append(candidate)
            continue

        stack_path = template_dir / f"{candidate.name}_template_stack.npy"
        metadata_path = template_dir / f"{candidate.name}_template_metadata.json"
        np.save(stack_path, stack.astype(np.float32))
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
                template_count=int(stack.shape[0]),
                scoring_mode="template_match",
            )
        )
    return result


def _match_roi_against_candidates(
    roi: dict[str, Any],
    candidates: list[IndexingCandidate],
) -> ROIIndexingResult:
    n_bragg_peaks = int(roi.get("n_bragg_peaks", 0) or 0)

    template_result = _template_match_roi(roi, candidates, n_bragg_peaks)
    if template_result is not None:
        return template_result

    return _mock_score_roi_against_candidates(roi, candidates, n_bragg_peaks)


def _template_match_roi(
    roi: dict[str, Any],
    candidates: list[IndexingCandidate],
    n_bragg_peaks: int,
) -> ROIIndexingResult | None:
    roi_data_path = roi.get("roi_data_path")
    if not roi_data_path:
        return None
    try:
        roi_data = np.load(roi_data_path)
    except OSError as exc:
        log.warning("Could not load ROI data for %s: %s", roi.get("name", "unknown"), exc)
        return None

    mean_dp = _mean_diffraction_pattern(roi_data)
    pattern_vec = _normalize_pattern(mean_dp)
    if pattern_vec is None:
        return None

    best: dict[str, Any] | None = None
    for candidate in candidates:
        if candidate.template_stack_path is None or candidate.template_metadata_path is None:
            continue
        try:
            stack = np.load(candidate.template_stack_path)
            metadata = json.loads(candidate.template_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not load templates for %s: %s", candidate.name, exc)
            continue
        flat_templates = stack.reshape((stack.shape[0], -1))
        normalized = [
            vec for vec in (_normalize_pattern(t.reshape(mean_dp.shape)) for t in flat_templates)
            if vec is not None
        ]
        if not normalized:
            continue
        template_vecs = np.vstack(normalized)
        scores = template_vecs @ pattern_vec
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if best is None or score > best["score"]:
            orientations = metadata.get("orientations_deg", [])
            best = {
                "candidate": candidate,
                "score": score,
                "orientation_deg": float(orientations[idx]) if idx < len(orientations) else None,
            }

    if best is None:
        return None

    score = round(float(best["score"]), 4)
    return ROIIndexingResult(
        name=str(roi.get("name", "unknown")),
        status="TEMPLATE_MATCHED",
        stage2a_bragg_summary_path=roi.get("bragg_summary_path"),
        n_bragg_peaks=n_bragg_peaks,
        best_candidate=best["candidate"].name,
        phase_score=score,
        orientation_score=score,
        match_quality=_template_quality(score),
        best_orientation_deg=best["orientation_deg"],
        template_score=score,
        scoring_mode="template_match",
    )


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
            best_candidate=None,
            phase_score=None,
            orientation_score=None,
            match_quality="not_scored",
            scoring_mode="not_scored",
        )

    return ROIIndexingResult(
        name=str(roi.get("name", "unknown")),
        status="MOCK_SCORED",
        stage2a_bragg_summary_path=roi.get("bragg_summary_path"),
        n_bragg_peaks=n_bragg_peaks,
        best_candidate=best_candidate.name,
        phase_score=best_score,
        orientation_score=None,
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
        "zone_axis": _parse_zone_axis(raw.get("zone_axis", [0, 0, 1])),
        "peak_sigma_px": float(raw.get("peak_sigma_px", 1.2)),
        "reciprocal_pixels_per_inv_angstrom": (
            None if raw.get("reciprocal_pixels_per_inv_angstrom") is None
            else float(raw["reciprocal_pixels_per_inv_angstrom"])
        ),
        "intensity_power": float(raw.get("intensity_power", 2.0)),
    }


def _parse_zone_axis(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("template_generation.zone_axis must be [u, v, w].")
    axis = [float(v) for v in value]
    if float(np.linalg.norm(axis)) <= 1e-12:
        raise ValueError("template_generation.zone_axis must be non-zero.")
    return axis


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
    return {
        "sig_shape": sig_shape,
        "beam_center_yx": beam_center,
        "beam_center_source": "stage2a_roi" if any(r.get("beam_center_yx") for r in roi_results) else "detector_center_fallback",
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
) -> tuple[np.ndarray, dict[str, Any]]:
    hkls, qxy, qnorm, projection = _reciprocal_spots(cell, max_index, zone_axis)
    if len(hkls) == 0:
        raise ValueError("No reciprocal spots generated from CIF cell.")
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
        "orientations_deg": orientations_deg,
        "zone_axis": list(zone_axis),
        "projection": projection,
        "sig_shape": list(sig_shape),
        "beam_center_yx": list(beam_center_yx),
        "peak_sigma_px": peak_sigma_px,
        "reciprocal_pixels_per_inv_angstrom": scale,
        "reciprocal_scale_source": scale_source,
        "intensity_power": intensity_power,
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


def _template_quality(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


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
