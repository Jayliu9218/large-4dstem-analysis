"""Stage 2A ROI Bragg detection using py4DSTEM.

Loads ROI candidates from a Stage-1 :class:`Stage1Manifest`, extracts
per-ROI sub-cubes from the original dataset, and runs py4DSTEM Bragg-disk
finding on each ROI.  All operations are isolated per ROI so that a failure
in one does not affect others.

Coordinate conventions
----------------------
* Stage 1 ROI bboxes are in **preprocessed (binned)** navigation coordinates
  (after ``r_bin`` reduction).
* py4DSTEM cubes are loaded in **original scan** coordinates.
* This module converts bboxes to raw coordinates before slicing,
  and records **both** ``stage1_bbox`` (binned) and ``raw_bbox`` (original).
* Beam centre is recorded as ``[qy, qx]`` per the ``center_order: y_x``
  convention in :class:`DataContract`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .contracts import Stage1Manifest
from .export import save_bar_png, save_png
from .logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration to py4DSTEM parameter mapping
# ---------------------------------------------------------------------------

# Map snake_case config keys (used in YAML) to the camelCase argument names
# that py4DSTEM's ``DataCube.find_Bragg_disks()`` expects.  Keys not listed
# here are passed through unchanged.
_BRAGG_CONFIG_TO_PY4DSTEM: dict[str, str] = {
    "corr_power": "corrPower",
    "edge_boundary": "edgeBoundary",
    "min_relative_intensity": "minRelativeIntensity",
    "min_peak_spacing": "minPeakSpacing",
    "max_num_peaks": "maxNumPeaks",
    "cuda": "CUDA",
}

# Default values for every py4DSTEM Bragg-disk parameter.  These are used
# when the config does not specify a value.
_DEFAULT_BRAGG_PARAMS: dict[str, Any] = {
    "corrPower": 1.0,
    "sigma_cc": 1,
    "edgeBoundary": 10,
    "minRelativeIntensity": 0.05,
    "minPeakSpacing": 4,
    "subpixel": "poly",
    "maxNumPeaks": 70,
    "CUDA": False,
}


def _convert_bragg_params(config_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert snake_case Stage-2 config keys to py4DSTEM camelCase kwargs.

    Keys that are already in camelCase form are passed through unchanged.
    Unknown keys are passed through with a debug log.

    Returns a dict that can be unpacked directly as
    ``DataCube.find_Bragg_disks(**result)``.
    """
    converted: dict[str, Any] = {}
    for key, value in config_kwargs.items():
        mapped = _BRAGG_CONFIG_TO_PY4DSTEM.get(key, key)
        if mapped != key:
            log.debug("Bragg param: %s -> %s = %s", key, mapped, value)
        elif mapped not in _DEFAULT_BRAGG_PARAMS and mapped not in (
            "corrPower", "sigma_cc", "edgeBoundary", "minRelativeIntensity",
            "minPeakSpacing", "subpixel", "maxNumPeaks", "CUDA",
        ):
            log.debug("Bragg param: passing through unknown key '%s'", mapped)
        converted[mapped] = value

    # Fill in defaults for any parameter not provided.
    for param, default in _DEFAULT_BRAGG_PARAMS.items():
        converted.setdefault(param, default)

    return converted


# ---------------------------------------------------------------------------
# Beam-centre helpers
# ---------------------------------------------------------------------------


def _parse_beam_center_txt(txt_path: Path) -> tuple[float, float] | None:
    """Parse ``beam_center_estimate.txt`` from Stage 1 preprocess.

    Returns ``(qy, qx)`` in detector pixels, or ``None`` if the file cannot
    be read or parsed.
    """
    if not txt_path.exists():
        return None
    try:
        text = txt_path.read_text(encoding="utf-8")
    except OSError:
        return None

    # The file has lines like:
    #   estimated_center_yx: [cy, cx]
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("estimated_center_yx:"):
            try:
                # Extract the list portion: "[cy, cx]"
                bracket = line[line.index("[") : line.rindex("]") + 1]
                parts = [float(v.strip()) for v in bracket.strip("[]").split(",")]
                if len(parts) == 2:
                    return (parts[0], parts[1])
            except (ValueError, IndexError):
                pass
    return None


def _extract_beam_center_from_calibration(cube: Any) -> tuple[float, float] | None:
    """Try to read the beam centre from py4DSTEM's calibration object.

    Returns ``(qy, qx)`` in detector pixels, or ``None`` on failure.
    """
    try:
        cal = cube.calibration
        qy = float(cal.get_qy0())
        qx = float(cal.get_qx0())
        return (qy, qx)
    except Exception:
        return None


def _to_raw_detector_center(
    center_yx: tuple[float, float],
    q_crop: list[int] | None,
    q_bin: int,
) -> tuple[float, float]:
    """Convert a beam centre from preprocessed to raw detector coordinates.

    Stage 1 computes the beam centre on the preprocessed data (after ``q_crop``
    and ``q_bin``).  The ROI data slice uses the raw detector, so the centre
    must be mapped back: ``raw = preprocessed * q_bin + crop_offset``.
    """
    q_bin = max(1, int(q_bin))
    qy0 = q_crop[0] if q_crop else 0
    qx0 = q_crop[2] if q_crop else 0
    return (center_yx[0] * q_bin + qy0, center_yx[1] * q_bin + qx0)


def _detector_center(sig_shape: tuple[int, ...]) -> tuple[float, float]:
    """Return the geometric centre of the detector ``(qy, qx)``."""
    return ((sig_shape[0] - 1) / 2.0, (sig_shape[1] - 1) / 2.0)


# ---------------------------------------------------------------------------
# Cluster / background validation helpers
# ---------------------------------------------------------------------------


def _validate_roi_cluster_binned(
    stage1_bbox: list[int],
    y0: int, y1: int, x0: int, x1: int,
    labels: np.ndarray | None,
    sample_mask: np.ndarray | None,
    r_bin: int,
) -> dict[str, Any]:
    """Compute cluster validation metrics for an ROI.

    Uses the clamped binned-coordinate bbox ``[y0, y1, x0, x1]`` to index
    into *labels* and *sample_mask* (both in binned navigation coordinates).

    Parameters
    ----------
    stage1_bbox:
        Original (unclamped) bbox from the ROI candidate, for recording.
    y0, y1, x0, x1:
        Clamped bbox in binned coordinates.
    labels:
        Fingerprint-class labels (``int16``, shape matches binned nav).
        Value ``-1`` = background.
    sample_mask:
        Boolean mask (``True`` = sample, shape matches binned nav).
    r_bin:
        Stage-1 navigation binning factor (for reporting).

    Returns
    -------
    dict with: ``background_fraction``, ``sample_mask_coverage``,
    ``labels_available``, ``sample_mask_available``, ``warning``.
    """
    result: dict[str, Any] = {
        "stage1_bbox": stage1_bbox,
        "background_fraction": None,
        "sample_mask_coverage": None,
        "labels_available": labels is not None,
        "sample_mask_available": sample_mask is not None,
        "warning": None,
        "r_bin": r_bin,
    }

    warnings: list[str] = []

    # --- Background fraction (from fingerprint class labels) -----------------
    if labels is not None:
        try:
            roi_labels = labels[y0:y1, x0:x1]
            n_total = roi_labels.size
            n_background = int(np.sum(roi_labels == -1))
            result["background_fraction"] = round(n_background / max(n_total, 1), 4)

            if n_background > 0:
                frac = result["background_fraction"]
                if frac > 0.5:
                    warnings.append(
                        f"ROI has {frac:.1%} background pixels (label -1). "
                        f"ROI may be positioned over vacuum/sample edge."
                    )
                elif frac > 0.1:
                    warnings.append(
                        f"ROI has {frac:.1%} background pixels (label -1)."
                    )
        except IndexError:
            result["warning"] = "Could not index labels array with ROI bbox."
            return result

    # --- Sample mask coverage -------------------------------------------------
    if sample_mask is not None:
        try:
            roi_mask = sample_mask[y0:y1, x0:x1]
            n_total = roi_mask.size
            n_sample = int(np.sum(roi_mask))
            result["sample_mask_coverage"] = round(n_sample / max(n_total, 1), 4)

            if n_sample == 0:
                warnings.append(
                    "ROI has 0% sample mask coverage. "
                    "ROI is entirely outside the sample region."
                )
            elif n_sample / max(n_total, 1) < 0.3:
                warnings.append(
                    f"ROI has only {result['sample_mask_coverage']:.1%} "
                    f"sample mask coverage."
                )
        except IndexError:
            result["warning"] = "Could not index sample mask with ROI bbox."
            return result

    if warnings:
        result["warning"] = " | ".join(warnings)

    return result


def _save_bragg_peaks_table(
    bragg: Any,
    roi_dir: Path,
    scan_shape: tuple[int, int],
) -> tuple[Path | None, dict[str, Any]]:
    """Extract per-peak tabular data from py4DSTEM and save as Parquet.

    Iterates the ``bragg.raw`` :class:`PointListArray` and collects every
    detected Bragg peak into a flat table with columns:

    - ``scan_y``, ``scan_x`` — navigation (real-space) position
    - ``qy``, ``qx`` — peak position in detector pixels
    - ``intensity`` — correlation intensity from the template match
    - ``snr`` — placeholder (NaN; requires subpixel noise model)

    Parameters
    ----------
    bragg:
        py4DSTEM :class:`BraggVectors` object returned by ``find_Bragg_disks()``.
    roi_dir:
        Per-ROI output directory.
    scan_shape:
        Navigation shape ``(ny, nx)`` of the extracted ROI (after thinning).

    Returns
    -------
    ``(parquet_path, summary)`` tuple.  *parquet_path* is *None* if no peaks
    were detected.  *summary* is a dict with keys ``parquet_rows``,
    ``parquet_columns``, ``peaks_per_pattern_mean``, ``peaks_per_pattern_std``.
    """
    scan_ys: list[np.ndarray] = []
    scan_xs: list[np.ndarray] = []
    qys: list[np.ndarray] = []
    qxs: list[np.ndarray] = []
    intensities: list[np.ndarray] = []

    ny, nx = scan_shape
    peaks_per_pattern: list[int] = []

    try:
        for ry in range(ny):
            for rx in range(nx):
                try:
                    v = bragg.raw[rx, ry]
                except (IndexError, AttributeError, TypeError):
                    continue
                n = len(v.data) if hasattr(v, "data") else 0
                if n == 0:
                    peaks_per_pattern.append(0)
                    continue
                peaks_per_pattern.append(n)
                scan_ys.append(np.full(n, ry, dtype=np.int32))
                scan_xs.append(np.full(n, rx, dtype=np.int32))
                try:
                    qys.append(np.asarray(v.qy, dtype=np.float32))
                    qxs.append(np.asarray(v.qx, dtype=np.float32))
                    intensities.append(np.asarray(v.I, dtype=np.float32))
                except AttributeError:
                    # Fallback: try accessing the structured data directly
                    data = np.asarray(v.data)
                    qys.append(data["qy"].astype(np.float32))
                    qxs.append(data["qx"].astype(np.float32))
                    intensities.append(data["intensity"].astype(np.float32))
    except Exception:
        # If the BraggVectors object doesn't support raw[rx, ry] indexing,
        # fall through with empty results.
        pass

    n_total = sum(arr.size for arr in scan_ys)
    if n_total == 0:
        return None, {
            "parquet_rows": 0,
            "parquet_columns": [],
            "peaks_per_pattern_mean": None,
            "peaks_per_pattern_std": None,
        }

    df = pd.DataFrame({
        "scan_y": np.concatenate(scan_ys),
        "scan_x": np.concatenate(scan_xs),
        "qy": np.concatenate(qys),
        "qx": np.concatenate(qxs),
        "intensity": np.concatenate(intensities),
        "snr": np.full(n_total, np.nan, dtype=np.float32),
    })

    parquet_path = roi_dir / "bragg_peaks.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    pp_arr = np.array(peaks_per_pattern, dtype=np.float64)
    return parquet_path, {
        "parquet_rows": n_total,
        "parquet_columns": list(df.columns),
        "peaks_per_pattern_mean": round(float(pp_arr.mean()), 2) if pp_arr.size > 0 else None,
        "peaks_per_pattern_std": round(float(pp_arr.std()), 2) if pp_arr.size > 0 else None,
    }


def _compute_bragg_qc_metrics(
    vmap: np.ndarray,
    beam_center_yx: tuple[float, float] | None,
    sig_shape: tuple[int, int],
    min_peak_spacing: float = 4.0,
    edge_boundary: int = 10,
    center_zone_radius: float = 5.0,
) -> dict[str, Any]:
    """Compute per-ROI Bragg-peak quality metrics from the vector map.

    Parameters
    ----------
    vmap:
        2D Bragg vector map histogram (shape ``(qy, qx)``).  Non-zero pixels
        mark detector bins where py4DSTEM found at least one Bragg disk.
    beam_center_yx:
        Beam centre ``(qy, qx)`` in the same (binned) detector coordinates as
        *vmap*, or *None* if no beam centre is available.
    sig_shape:
        Detector shape ``(qy, qx)`` in the same space as *vmap*.
    min_peak_spacing:
        Minimum allowed spacing between distinct Bragg peaks (pixels).  Used
        to compute the duplicate-peak fraction.
    edge_boundary:
        Distance from detector edges (pixels).  Peaks within this margin
        count toward the edge-peak fraction.
    center_zone_radius:
        Radius around the beam centre (pixels).  Peaks within this zone are
        likely central-beam / bright-field tail rather than genuine Bragg
        disks.

    Returns
    -------
    dict with keys:
        peak_pixel_count, total_peak_votes, mean_peak_intensity,
        radial_distances (list or null), radial_distance_mean,
        radial_distance_std, forbidden_center_zone_fraction,
        edge_peak_fraction, duplicate_peak_fraction,
        beam_center_error_estimate
    """
    peak_rows, peak_cols = np.nonzero(vmap)
    n_peaks = len(peak_rows)
    intensities = vmap[peak_rows, peak_cols].astype(np.float64)

    if n_peaks == 0:
        return {
            "peak_pixel_count": 0,
            "total_peak_votes": 0,
            "mean_peak_intensity": 0.0,
            "radial_distances": None,
            "radial_distance_mean": None,
            "radial_distance_std": None,
            "forbidden_center_zone_fraction": 0.0,
            "edge_peak_fraction": 0.0,
            "duplicate_peak_fraction": 0.0,
            "beam_center_error_estimate": None,
        }

    # --- Radial distances from beam centre ---------------------------------
    radial_distances: list[float] | None = None
    radial_mean: float | None = None
    radial_std: float | None = None
    center_zone_frac: float = 0.0
    beam_center_err: float | None = None

    if beam_center_yx is not None:
        dy = peak_rows.astype(np.float64) - float(beam_center_yx[0])
        dx = peak_cols.astype(np.float64) - float(beam_center_yx[1])
        radial = np.sqrt(dy ** 2 + dx ** 2)
        radial_distances = [float(r) for r in radial]
        radial_mean = float(np.mean(radial))
        radial_std = float(np.std(radial))
        center_zone_frac = float(np.mean(radial < center_zone_radius))

        # Beam-centre error: systematic offset between peak centroid and
        # nominal centre.  For a well-centred symmetric pattern this should
        # be small.
        centroid_y = float(np.mean(peak_rows.astype(np.float64)))
        centroid_x = float(np.mean(peak_cols.astype(np.float64)))
        beam_center_err = float(np.sqrt(
            (centroid_y - float(beam_center_yx[0])) ** 2
            + (centroid_x - float(beam_center_yx[1])) ** 2
        ))

    # --- Edge-peak fraction -------------------------------------------------
    qy, qx = sig_shape
    edge_mask = (
        (peak_rows < edge_boundary)
        | (peak_rows >= qy - edge_boundary)
        | (peak_cols < edge_boundary)
        | (peak_cols >= qx - edge_boundary)
    )
    edge_frac = float(np.mean(edge_mask))

    # --- Duplicate-peak fraction ---------------------------------------------
    # A peak is "duplicate" if it has at least one neighbour peak within
    # min_peak_spacing pixels.  Use a simple N^2 scan for small peak sets;
    # fall back to an approximate grid-based check for very large sets.
    dup_frac: float = 0.0
    if n_peaks > 1:
        positions = np.column_stack([peak_rows.astype(np.float64), peak_cols.astype(np.float64)])
        dup_mask = np.zeros(n_peaks, dtype=bool)
        # For up to ~2000 peaks the N^2 scan is fast enough.
        if n_peaks <= 2000:
            for i in range(n_peaks):
                if dup_mask[i]:
                    continue
                dists = np.sqrt(np.sum((positions - positions[i]) ** 2, axis=1))
                dists[i] = np.inf  # exclude self
                if np.any(dists < min_peak_spacing):
                    dup_mask[i] = True
        else:
            # Grid-based: bin peaks into cells of size min_peak_spacing;
            # check the 3×3 neighbourhood for each cell.
            cell = min_peak_spacing
            for i in range(n_peaks):
                if dup_mask[i]:
                    continue
                dy = np.abs(positions[:, 0] - positions[i, 0])
                dx = np.abs(positions[:, 1] - positions[i, 1])
                nearby = (dy < cell) & (dx < cell)
                nearby[i] = False
                if not np.any(nearby):
                    continue
                dists = np.sqrt(
                    (positions[nearby, 0] - positions[i, 0]) ** 2
                    + (positions[nearby, 1] - positions[i, 1]) ** 2
                )
                if np.any(dists < min_peak_spacing):
                    dup_mask[i] = True
        dup_frac = float(np.mean(dup_mask))

    return {
        "peak_pixel_count": n_peaks,
        "total_peak_votes": int(np.sum(intensities)),
        "mean_peak_intensity": round(float(np.mean(intensities)), 2),
        "radial_distances": radial_distances,
        "radial_distance_mean": round(radial_mean, 2) if radial_mean is not None else None,
        "radial_distance_std": round(radial_std, 2) if radial_std is not None else None,
        "forbidden_center_zone_fraction": round(center_zone_frac, 4),
        "edge_peak_fraction": round(edge_frac, 4),
        "duplicate_peak_fraction": round(dup_frac, 4),
        "beam_center_error_estimate": round(beam_center_err, 2) if beam_center_err is not None else None,
    }


def _save_bragg_visuals(
    roi_dir: Path,
    vmap: np.ndarray,
    mean_dp_binned: np.ndarray | None = None,
) -> None:
    """Save PNG visualisations of the Bragg vector map.

    Produces ``bragg_vector_map.png`` and, when *mean_dp_binned* is
    available, ``bragg_overlay.png`` with Bragg peak positions overlaid
    in green on the log-scale mean diffraction pattern.
    """
    try:
        save_png(roi_dir / "bragg_vector_map.png", vmap)
    except Exception:
        pass

    if mean_dp_binned is not None and np.count_nonzero(vmap) > 0:
        try:
            # Find Bragg peak positions (non-zero pixels in vmap)
            peaks_y, peaks_x = np.nonzero(vmap)
            base = np.asarray(mean_dp_binned, dtype=np.float32)
            # Scale base to uint8 gray
            finite = base[np.isfinite(base)]
            if finite.size > 0:
                lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
                if hi <= lo:
                    hi = lo + 1.0
                gray = np.clip((np.log1p(base) - np.log1p(lo)) / max(np.log1p(hi) - np.log1p(lo), 1e-12) * 255, 0, 255).astype(np.uint8)
            else:
                gray = np.zeros(base.shape, dtype=np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1).copy()
            # Mark Bragg peak positions in green, with a small cross
            for py, px in zip(peaks_y, peaks_x):
                y0 = max(0, int(py) - 2)
                y1 = min(rgb.shape[0], int(py) + 3)
                x0 = max(0, int(px) - 2)
                x1 = min(rgb.shape[1], int(px) + 3)
                rgb[y0:y1, int(px), :] = [0, 255, 0]
                rgb[int(py), x0:x1, :] = [0, 255, 0]
            save_png(roi_dir / "bragg_overlay.png", rgb)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ROIBraggResult:
    """Output from Bragg-disk detection on a single ROI.

    Attributes
    ----------
    name:
        ROI name from the candidate list.
    stage1_bbox:
        Bounding box ``[y0, y1, x0, x1]`` in Stage-1 **binned** navigation
        coordinates (as recorded in ``roi_candidates.yaml``).
    raw_bbox:
        Bounding box ``[y0, y1, x0, x1]`` in **original** (raw) scan
        coordinates - what was actually sliced from the py4DSTEM cube.
    nav_shape:
        Navigation shape ``(ny, nx)`` of the extracted ROI after thinning.
    sig_shape:
        Detector shape ``(qy, qx)`` of the extracted ROI.
    output_dir:
        Per-ROI output directory.
    n_peaks:
        Number of non-zero pixels in the Bragg vector map histogram.
    beam_center_yx:
        Beam centre ``[qy, qx]`` used for this ROI (in detector pixels).
    beam_center_source:
        Where the beam centre came from (``"stage1_com"``,
        ``"py4dstem_calibration"``, ``"detector_center_fallback"``).
    cluster_id:
        Fingerprint-class cluster id (from ``roi_candidates.yaml``).
    reason:
        Why this ROI was selected (from ``roi_candidates.yaml``).
    background_fraction:
        Fraction of pixels in this ROI with label ``-1`` (background).
        ``None`` if labels were not available.
    sample_mask_coverage:
        Fraction of pixels inside the sample mask.  ``None`` if sample mask
        was not available.
    cluster_validation_warning:
        Non-fatal warning from cluster / background validation, if any.
    error:
        Error message if processing failed, ``None`` on success.
    """

    name: str
    stage1_bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    raw_bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    nav_shape: tuple[int, int] = (0, 0)
    sig_shape: tuple[int, int] = (0, 0)
    output_dir: Path = field(default_factory=Path)
    roi_data_path: Path | None = None
    bragg_vector_map_path: Path | None = None
    bragg_summary_path: Path | None = None
    bragg_peaks_parquet_path: Path | None = None
    n_peaks: int = 0
    beam_center_yx: list[float] | None = None
    beam_center_source: str | None = None
    cluster_id: int | None = None
    reason: str | None = None
    background_fraction: float | None = None
    sample_mask_coverage: float | None = None
    cluster_validation_warning: str | None = None
    error: str | None = None
    bragg_qc: dict[str, Any] | None = None
    # Benchmark fields
    extraction_time_s: float = 0.0
    bragg_time_s: float = 0.0
    total_time_s: float = 0.0
    roi_data_size_bytes: int = 0


@dataclass
class Stage2Result:
    """Aggregated result from a Stage 2A run."""

    stage1_dir: Path
    output_dir: Path
    manifest: Stage1Manifest
    roi_results: list[ROIBraggResult] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.roi_results if r.error is None)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.roi_results if r.error is not None)

    @property
    def n_warnings(self) -> int:
        return sum(
            1 for r in self.roi_results
            if r.error is None and r.cluster_validation_warning is not None
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_roi_candidates(yaml_path: str | Path) -> list[dict[str, Any]]:
    """Load ROI candidates from a Stage-1 ``roi_candidates.yaml`` file.

    Returns a list of ROI dicts, each with keys ``name``, ``bbox``,
    ``center``, ``size``, ``reason``, and optionally ``cluster``.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"ROI candidates file not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "rois" not in data:
        raise ValueError(f"{yaml_path} does not contain a 'rois' key; unexpected format.")
    return data["rois"]


def run_roi_bragg_for_rois(
    manifest: Stage1Manifest,
    rois: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    data_path: str | Path,
    data_loader: str = "auto",
    max_rois: int | None = None,
    thin_r: int = 2,
    bin_q: int = 2,
    mem: str = "MEMMAP",
    bragg_kwargs: dict[str, Any] | None = None,
    scan_shape: tuple[int, int] | list[int] | None = None,
    beam_center_yx: tuple[float, float] | None = None,
    labels: np.ndarray | None = None,
    sample_mask: np.ndarray | None = None,
    save_roi_data: bool = False,
    central_exclusion_radius: float = 0.0,
) -> list[ROIBraggResult]:
    """Run py4DSTEM Bragg-disk detection on a set of ROIs.

    Parameters
    ----------
    manifest:
        Validated Stage-1 manifest providing nav/sig shapes and ``r_bin``.
    rois:
        List of ROI candidate dicts (from ``roi_candidates.yaml``).
    output_dir:
        Directory where per-ROI outputs are written.
    data_path:
        Path to the original 4D-STEM data file (MIB, HDF5, etc.).
    data_loader:
        How to load the data. ``"auto"`` uses py4DSTEM's ``import_file``.
    max_rois:
        Cap the number of ROIs processed (useful for quick tests).
    thin_r:
        Navigation thinning factor (take every N-th scan position).
    bin_q:
        Detector binning factor for the per-ROI DataCube.
    mem:
        py4DSTEM memory mode (``"MEMMAP"`` or ``"RAM"``).
    bragg_kwargs:
        Keyword arguments in **snake_case** config form (e.g. ``corr_power``).
        These are converted to py4DSTEM camelCase internally.
    scan_shape:
        Optional original/raw navigation shape ``(ny, nx)`` to pass to
        ``py4DSTEM.import_file``. If omitted, this is inferred as
        ``manifest.nav_shape * manifest.r_bin``.
    beam_center_yx:
        Beam centre ``(qy, qx)`` from Stage 1 COM estimate.  If ``None``,
        falls back to py4DSTEM calibration, then detector centre.
    labels:
        Fingerprint-class label array for cluster validation
        (``int16``, ``-1`` = background).  May be ``None``.
    sample_mask:
        Boolean sample mask for coverage validation.  May be ``None``.

    Returns
    -------
    list[ROIBraggResult]
        One result per ROI, including any per-ROI errors.
    """
    try:
        import py4DSTEM
    except ImportError as exc:
        raise ImportError(
            "Stage 2A ROI Bragg detection requires py4DSTEM. "
            "Install with: pip install py4DSTEM>=0.14"
        ) from exc

    # --- Convert Bragg params from config snake_case to py4DSTEM camelCase ---
    if bragg_kwargs is None:
        bragg_kwargs = {}
    bragg_kwargs_converted = _convert_bragg_params(bragg_kwargs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)

    if max_rois is not None:
        rois = rois[:max_rois]

    r_bin = max(1, int(manifest.r_bin))

    # Compute the original (pre-binned) scan shape. manifest.nav_shape is
    # after Stage-1 r_bin reduction, so multiplying back up is the default.
    # Explicit scan_shape handles detector/scan mismatches or non-even r_bin.
    if scan_shape is None:
        scan_shape_tuple: tuple[int, int] = (
            manifest.nav_shape[0] * r_bin,
            manifest.nav_shape[1] * r_bin,
        )
    else:
        if len(scan_shape) != 2:
            raise ValueError("scan_shape must contain exactly two values: [ny, nx].")
        scan_shape_tuple = (int(scan_shape[0]), int(scan_shape[1]))
        if scan_shape_tuple[0] <= 0 or scan_shape_tuple[1] <= 0:
            raise ValueError(f"scan_shape values must be positive, got {scan_shape_tuple}.")

    # --- Record py4DSTEM metadata -------------------------------------------
    py4dstem_version: str = getattr(py4DSTEM, "__version__", "unknown")
    log.info("py4DSTEM version: %s", py4dstem_version)
    log.info(
        "Loading 4D-STEM data from %s (mem=%s, scan=%s, r_bin=%d)",
        data_path, mem, scan_shape_tuple, r_bin,
    )
    suffix = Path(data_path).suffix.lower()
    if suffix in (".h5", ".hdf5", ".emd"):
        log.info("Using py4DSTEM.read() for HDF5/EMD file.")
        cube = py4DSTEM.read(
            str(data_path),
            mem=mem,
        )
    else:
        cube = py4DSTEM.import_file(
            str(data_path),
            mem=mem,
            scan=scan_shape_tuple,
        )

    # --- Verify import shape -------------------------------------------------
    actual_shape = tuple(int(v) for v in cube.data.shape)
    expected_nav = scan_shape_tuple
    if actual_shape[:2] != expected_nav:
        log.warning(
            "Imported cube shape %s nav does not match expected %s. "
            "Check scan_shape parameter.",
            actual_shape, expected_nav,
        )
    log.info("Imported cube shape: %s (detector: %s)", actual_shape, actual_shape[2:])

    # --- Resolve beam centre -------------------------------------------------
    # Stage 1 beam centre is computed on the *preprocessed* detector (after
    # q_crop and q_bin).  The ROI data extracted below uses the *raw* detector,
    # so we must convert back to raw coordinates when q_crop / q_bin are known.
    beam_center_source: str
    if beam_center_yx is not None:
        beam_center_source = "stage1_com"
        beam_center_yx = _to_raw_detector_center(
            beam_center_yx, manifest.q_crop, manifest.q_bin,
        )
        log.info(
            "Using Stage-1 COM beam centre (converted to raw detector): (%.3f, %.3f)",
            *beam_center_yx,
        )
    else:
        # Try py4DSTEM calibration
        cal_center = _extract_beam_center_from_calibration(cube)
        if cal_center is not None:
            beam_center_yx = cal_center
            beam_center_source = "py4dstem_calibration"
            log.info(
                "Using py4DSTEM calibration beam centre: (%.3f, %.3f)",
                *beam_center_yx,
            )
        else:
            # Fall back to detector centre
            det_shape = actual_shape[2:]
            beam_center_yx = _detector_center(det_shape)
            beam_center_source = "detector_center_fallback"
            log.warning(
                "No beam centre available from Stage 1 or py4DSTEM; "
                "falling back to detector centre: (%.3f, %.3f)",
                *beam_center_yx,
            )

    # --- Process each ROI ----------------------------------------------------
    results: list[ROIBraggResult] = []
    for idx, roi in enumerate(rois):
        name = roi.get("name", f"roi_{idx:03d}")
        roi_dir = output_dir / f"roi_{name}"
        roi_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = _process_one_roi(
                cube=cube,
                roi=roi,
                roi_dir=roi_dir,
                nav_shape_binned=tuple(manifest.nav_shape),
                r_bin=r_bin,
                thin_r=thin_r,
                bin_q=bin_q,
                bragg_kwargs=bragg_kwargs_converted,
                beam_center_yx=beam_center_yx,
                beam_center_source=beam_center_source,
                labels=labels,
                sample_mask=sample_mask,
                py4dstem_version=py4dstem_version,
                scan_shape=scan_shape_tuple,
                data_path_str=str(data_path),
                save_roi_data=save_roi_data,
                central_exclusion_radius=central_exclusion_radius,
            )
            results.append(result)
            log.info(
                "ROI '%s' complete: %d Bragg peaks found -> %s",
                name, result.n_peaks, roi_dir,
            )
        except Exception as exc:
            log.error("ROI '%s' failed: %s", name, exc, exc_info=True)
            # Build a best-effort error result
            bbox = [int(v) for v in roi.get("bbox", [0, 0, 0, 0])]
            raw_bbox = [v * r_bin for v in bbox]
            results.append(
                ROIBraggResult(
                    name=name,
                    stage1_bbox=bbox,
                    raw_bbox=raw_bbox,
                    nav_shape=(0, 0),
                    sig_shape=(0, 0),
                    output_dir=roi_dir,
                    beam_center_yx=list(beam_center_yx) if beam_center_yx else None,
                    beam_center_source=beam_center_source,
                    cluster_id=roi.get("cluster"),
                    reason=roi.get("reason"),
                    error=str(exc),
                )
            )

    return results


def _process_one_roi(
    cube: Any,
    roi: dict[str, Any],
    roi_dir: Path,
    nav_shape_binned: tuple[int, int],
    r_bin: int,
    thin_r: int,
    bin_q: int,
    bragg_kwargs: dict[str, Any],
    beam_center_yx: tuple[float, float] | None,
    beam_center_source: str,
    labels: np.ndarray | None,
    sample_mask: np.ndarray | None,
    py4dstem_version: str,
    scan_shape: tuple[int, int],
    data_path_str: str,
    save_roi_data: bool = False,
    central_exclusion_radius: float = 0.0,
) -> ROIBraggResult:
    """Extract and process a single ROI sub-cube.

    Parameters
    ----------
    cube:
        py4DSTEM DataCube of the full original dataset.
    roi:
        ROI candidate dict from ``roi_candidates.yaml``, with bbox in
        Stage-1 **binned** navigation coordinates.
    roi_dir:
        Per-ROI output directory.
    nav_shape_binned:
        Navigation shape after Stage-1 ``r_bin`` reduction.
    r_bin:
        Stage-1 navigation binning factor.  Used to convert binned bbox
        coordinates back to raw scan coordinates for slicing.
    thin_r:
        Navigation thinning factor for this ROI.
    bin_q:
        Detector binning factor for the per-ROI DataCube.
    bragg_kwargs:
        **Already-converted** py4DSTEM camelCase kwargs for
        ``find_Bragg_disks()``.
    beam_center_yx:
        Beam centre ``(qy, qx)`` in detector pixels.
    beam_center_source:
        Provenance tag for the beam centre.
    labels:
        Fingerprint-class label array (binned nav shape).
    sample_mask:
        Boolean sample mask (binned nav shape).
    py4dstem_version:
        py4DSTEM version string.
    scan_shape:
        Original (raw) navigation shape passed to ``import_file``.
    data_path_str:
        Original data file path (for provenance).

    Returns
    -------
    ROIBraggResult
    """
    import py4DSTEM

    name = roi.get("name", "unnamed")
    cluster_id = roi.get("cluster")
    reason = roi.get("reason")

    # --- Coordinate conversion: binned to raw --------------------------------
    # Stage 1 bbox is in binned navigation coordinates (after r_bin).
    # The py4DSTEM cube is in original scan coordinates.
    # We must convert the bbox to raw coordinates before slicing.
    bbox_binned = [int(v) for v in roi["bbox"]]
    by0, by1, bx0, bx1 = bbox_binned

    # Clamp to valid binned navigation range
    bny, bnx = nav_shape_binned
    by0 = max(0, min(by0, bny))
    by1 = max(by0 + 1, min(by1, bny))
    bx0 = max(0, min(bx0, bnx))
    bx1 = max(bx0 + 1, min(bx1, bnx))

    # Convert to raw (original scan) coordinates
    ry0 = by0 * r_bin
    ry1 = by1 * r_bin
    rx0 = bx0 * r_bin
    rx1 = bx1 * r_bin

    # Clamp to raw scan range
    sny, snx = scan_shape
    ry0 = max(0, min(ry0, sny))
    ry1 = max(ry0 + r_bin, min(ry1, sny))
    rx0 = max(0, min(rx0, snx))
    rx1 = max(rx0 + r_bin, min(rx1, snx))

    stage1_bbox_clamped = [by0, by1, bx0, bx1]
    raw_bbox = [ry0, ry1, rx0, rx1]

    log.info(
        "ROI '%s': stage1_bbox=%s -> raw_bbox=%s (r_bin=%d)",
        name, stage1_bbox_clamped, raw_bbox, r_bin,
    )

    # Validate that ROI is non-empty
    if ry1 <= ry0 or rx1 <= rx0:
        raise ValueError(
            f"ROI raw bbox {raw_bbox} has zero or negative area. "
            f"stage1_bbox={stage1_bbox_clamped}, r_bin={r_bin}, scan_shape={scan_shape}"
        )

    # --- Extract sub-cube from raw coordinates ------------------------------
    thin_r = max(1, int(thin_r))
    bin_q = max(1, int(bin_q))

    t_extract_start = time.perf_counter()
    roi_data = np.asarray(
        cube.data[ry0:ry1:thin_r, rx0:rx1:thin_r, :, :],
        dtype=np.float32,
    )

    if roi_data.size == 0:
        raise ValueError(
            f"ROI data is empty after slicing. "
            f"raw_bbox={raw_bbox}, thin_r={thin_r}, scan_shape={scan_shape}"
        )

    sig_shape = roi_data.shape[-2:]
    nav_shape_roi = roi_data.shape[:2]
    roi_data_size_bytes = int(roi_data.nbytes)
    t_extract_end = time.perf_counter()
    extraction_time_s = t_extract_end - t_extract_start

    # --- Cluster / background validation (on binned coords) -----------------
    cluster_validation = _validate_roi_cluster_binned(
        stage1_bbox=bbox_binned,
        y0=by0, y1=by1, x0=bx0, x1=bx1,
        labels=labels,
        sample_mask=sample_mask,
        r_bin=r_bin,
    )

    # --- Save raw ROI data (optional — large; off by default) ----------------
    roi_data_path: Path | None = None
    if save_roi_data:
        roi_data_path = roi_dir / "roi_data.npy"
        np.save(roi_data_path, roi_data)

    # --- Visualise mean diffraction pattern ----------------------------------
    try:
        mean_dp = np.asarray(roi_data.mean(axis=(0, 1)), dtype=np.float32)
        save_png(roi_dir / "mean_dp.png", np.log1p(mean_dp))
    except Exception:
        pass

    # --- Build py4DSTEM DataCube and bin ------------------------------------
    dc_roi = py4DSTEM.DataCube(roi_data, name=name, calibration=cube.calibration)
    binned_sig_shape = sig_shape
    if bin_q > 1:
        dc_roi = dc_roi.bin_Q(bin_q, dtype=np.float32)
        binned_sig_shape = dc_roi.data.shape[-2:]

    # --- Template: mean diffraction pattern over the ROI --------------------
    template = np.asarray(dc_roi.data.mean(axis=(0, 1)), dtype=np.float32)

    # --- Run Bragg disk detection -------------------------------------------
    # bragg_kwargs are already in camelCase form (converted upstream).
    t_bragg_start = time.perf_counter()
    bragg = dc_roi.find_Bragg_disks(
        template=template,
        corrPower=float(bragg_kwargs.get("corrPower", 1.0)),
        sigma_cc=float(bragg_kwargs.get("sigma_cc", 1)),
        edgeBoundary=int(bragg_kwargs.get("edgeBoundary", 10)),
        minRelativeIntensity=float(bragg_kwargs.get("minRelativeIntensity", 0.05)),
        minPeakSpacing=int(bragg_kwargs.get("minPeakSpacing", 4)),
        subpixel=bragg_kwargs.get("subpixel", "poly"),
        maxNumPeaks=int(bragg_kwargs.get("maxNumPeaks", 70)),
        CUDA=bool(bragg_kwargs.get("CUDA", False)),
    )

    # --- Save tabular Bragg peaks (parquet) ----------------------------------
    parquet_path, parquet_summary = _save_bragg_peaks_table(
        bragg, roi_dir, scan_shape=nav_shape_roi,
    )

    # --- Save Bragg vector map ----------------------------------------------
    histogram = bragg.histogram(mode="cal")
    vmap = np.asarray(histogram.data, dtype=np.float32)

    # --- Central exclusion: zero pixels within radius of beam centre --------
    central_exclusion_applied: bool = False
    if central_exclusion_radius > 0 and beam_center_yx is not None:
        bc_y_binned = float(beam_center_yx[0]) / float(bin_q)
        bc_x_binned = float(beam_center_yx[1]) / float(bin_q)
        yy, xx = np.indices(vmap.shape, dtype=np.float64)
        rr = np.sqrt((yy - bc_y_binned) ** 2 + (xx - bc_x_binned) ** 2)
        central_mask = rr < float(central_exclusion_radius)
        n_excluded = int(np.sum(central_mask & (vmap > 0)))
        vmap[central_mask] = 0.0
        central_exclusion_applied = True
        log.info(
            "ROI '%s': central exclusion radius=%.1f px → %d peak(s) removed (%d remaining).",
            name, central_exclusion_radius, n_excluded,
            int(np.count_nonzero(vmap)),
        )

    bragg_vector_map_path = roi_dir / "bragg_vector_map.npy"
    np.save(bragg_vector_map_path, vmap)
    n_peaks = int(np.count_nonzero(vmap))

    # --- Bragg peak QC metrics ------------------------------------------------
    beam_center_binned: tuple[float, float] | None = None
    if beam_center_yx is not None:
        beam_center_binned = (
            float(beam_center_yx[0]) / float(bin_q),
            float(beam_center_yx[1]) / float(bin_q),
        )
    bragg_qc = _compute_bragg_qc_metrics(
        vmap,
        beam_center_yx=beam_center_binned,
        sig_shape=tuple(binned_sig_shape),
        min_peak_spacing=float(bragg_kwargs.get("minPeakSpacing", 4)),
        edge_boundary=int(bragg_kwargs.get("edgeBoundary", 10)),
    )
    bragg_qc["central_exclusion_radius"] = float(central_exclusion_radius)
    bragg_qc["central_exclusion_applied"] = central_exclusion_applied

    # --- Per-pattern peak counts (from parquet summary) ----------------------
    bragg_qc["peaks_per_pattern_mean"] = parquet_summary.get("peaks_per_pattern_mean")
    bragg_qc["peaks_per_pattern_std"] = parquet_summary.get("peaks_per_pattern_std")

    # --- Radius histogram PNG -------------------------------------------------
    if bragg_qc.get("radial_distances") is not None:
        try:
            radial_arr = np.asarray(bragg_qc["radial_distances"], dtype=np.float32)
            if radial_arr.size > 0:
                save_bar_png(
                    roi_dir / "bragg_peak_radius_histogram.png",
                    radial_arr.reshape(1, -1),
                )
        except Exception:
            pass

    # --- Visualise Bragg vector map + overlay on mean DP ---------------------
    _save_bragg_visuals(roi_dir, vmap, mean_dp_binned=template)

    t_bragg_end = time.perf_counter()
    bragg_time_s = t_bragg_end - t_bragg_start

    # --- Record the effective beam centre for this ROI ----------------------
    # After bin_Q, the beam centre coordinates change.  Record the original
    # (pre-bin) centre so downstream indexing can apply the same binning.
    beam_cyx_list: list[float] | None = (
        [float(beam_center_yx[0]), float(beam_center_yx[1])]
        if beam_center_yx is not None
        else None
    )

    # --- Per-ROI bragg_summary.json -----------------------------------------
    # Exclude CUDA from human-readable params (it's in cuda_used).
    bragg_params_display = {
        k: v for k, v in bragg_kwargs.items()
        if k not in ("CUDA",)
    }
    bragg_summary = {
        "name": name,
        "stage1_bbox": stage1_bbox_clamped,
        "raw_bbox": raw_bbox,
        "nav_shape": list(nav_shape_roi),
        "nav_shape_before_thin": [ry1 - ry0, rx1 - rx0],
        "sig_shape": list(sig_shape),
        "sig_shape_after_bin": list(binned_sig_shape),
        "r_bin": r_bin,
        "thin_r": thin_r,
        "bin_q": bin_q,
        "n_bragg_peaks": n_peaks,
        "bragg_params": bragg_params_display,
        "cuda_used": bool(bragg_kwargs.get("CUDA", False)),
        "beam_center_yx": beam_cyx_list,
        "beam_center_source": beam_center_source,
        "cluster_id": cluster_id,
        "reason": reason,
        "cluster_validation": {
            "background_fraction": cluster_validation.get("background_fraction"),
            "sample_mask_coverage": cluster_validation.get("sample_mask_coverage"),
            "labels_available": cluster_validation.get("labels_available"),
            "sample_mask_available": cluster_validation.get("sample_mask_available"),
            "warning": cluster_validation.get("warning"),
        },
        "dependencies": {
            "py4dstem_version": py4dstem_version,
            "data_path": data_path_str,
            "scan_shape": list(scan_shape),
        },
        "bragg_peaks_parquet": {
            "path": str(parquet_path) if parquet_path else None,
            "rows": parquet_summary["parquet_rows"],
            "columns": parquet_summary["parquet_columns"],
        },
        "bragg_qc": bragg_qc,
        "benchmark": {
            "extraction_time_s": round(extraction_time_s, 4),
            "bragg_time_s": round(bragg_time_s, 4),
            "total_time_s": round(extraction_time_s + bragg_time_s, 4),
            "roi_data_size_bytes": roi_data_size_bytes,
        },
    }
    bragg_summary_path = roi_dir / "bragg_summary.json"
    bragg_summary_path.write_text(
        json.dumps(bragg_summary, indent=2, default=str),
        encoding="utf-8",
    )

    return ROIBraggResult(
        name=name,
        stage1_bbox=stage1_bbox_clamped,
        raw_bbox=raw_bbox,
        nav_shape=nav_shape_roi,
        sig_shape=sig_shape,
        output_dir=roi_dir,
        roi_data_path=roi_data_path,
        bragg_vector_map_path=bragg_vector_map_path,
        bragg_summary_path=bragg_summary_path,
        bragg_peaks_parquet_path=parquet_path,
        n_peaks=n_peaks,
        beam_center_yx=beam_cyx_list,
        beam_center_source=beam_center_source,
        cluster_id=cluster_id,
        reason=reason,
        background_fraction=cluster_validation.get("background_fraction"),
        sample_mask_coverage=cluster_validation.get("sample_mask_coverage"),
        cluster_validation_warning=cluster_validation.get("warning"),
        bragg_qc=bragg_qc,
        extraction_time_s=round(extraction_time_s, 4),
        bragg_time_s=round(bragg_time_s, 4),
        total_time_s=round(extraction_time_s + bragg_time_s, 4),
        roi_data_size_bytes=roi_data_size_bytes,
    )
