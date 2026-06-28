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
import yaml

from .contracts import Stage1Manifest
from .logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration → py4DSTEM parameter mapping
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
            log.debug("Bragg param: %s → %s = %s", key, mapped, value)
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
        coordinates — what was actually sliced from the py4DSTEM cube.
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
    n_peaks: int = 0
    beam_center_yx: list[float] | None = None
    beam_center_source: str | None = None
    cluster_id: int | None = None
    reason: str | None = None
    background_fraction: float | None = None
    sample_mask_coverage: float | None = None
    cluster_validation_warning: str | None = None
    error: str | None = None
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
    beam_center_yx: tuple[float, float] | None = None,
    labels: np.ndarray | None = None,
    sample_mask: np.ndarray | None = None,
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

    # --- Convert Bragg params from config snake_case → py4DSTEM camelCase ---
    if bragg_kwargs is None:
        bragg_kwargs = {}
    bragg_kwargs_converted = _convert_bragg_params(bragg_kwargs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)

    if max_rois is not None:
        rois = rois[:max_rois]

    r_bin = max(1, int(manifest.r_bin))

    # Compute the original (pre-binned) scan shape.  manifest.nav_shape is
    # *after* Stage-1 r_bin reduction, so we multiply back up.  This is
    # correct when r_bin divides evenly; when it doesn't the original
    # scan_shape should be provided explicitly via the parameter.
    scan_shape: tuple[int, int] = (
        manifest.nav_shape[0] * r_bin,
        manifest.nav_shape[1] * r_bin,
    )

    # --- Record py4DSTEM metadata -------------------------------------------
    py4dstem_version: str = getattr(py4DSTEM, "__version__", "unknown")
    log.info("py4DSTEM version: %s", py4dstem_version)
    log.info(
        "Loading 4D-STEM data from %s (mem=%s, scan=%s, r_bin=%d)",
        data_path, mem, scan_shape, r_bin,
    )
    cube = py4DSTEM.import_file(
        str(data_path),
        mem=mem,
        scan=scan_shape,
    )

    # --- Verify import shape -------------------------------------------------
    actual_shape = tuple(int(v) for v in cube.data.shape)
    expected_nav = scan_shape
    if actual_shape[:2] != expected_nav:
        log.warning(
            "Imported cube shape %s nav does not match expected %s. "
            "Check scan_shape parameter.",
            actual_shape, expected_nav,
        )
    log.info("Imported cube shape: %s (detector: %s)", actual_shape, actual_shape[2:])

    # --- Resolve beam centre -------------------------------------------------
    beam_center_source: str
    if beam_center_yx is not None:
        beam_center_source = "stage1_com"
        log.info("Using Stage-1 COM beam centre: (%.3f, %.3f)", *beam_center_yx)
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
                scan_shape=scan_shape,
                data_path_str=str(data_path),
            )
            results.append(result)
            log.info(
                "ROI '%s' complete: %d Bragg peaks found → %s",
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

    # --- Coordinate conversion: binned → raw --------------------------------
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
        "ROI '%s': stage1_bbox=%s → raw_bbox=%s (r_bin=%d)",
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

    # --- Save raw ROI data --------------------------------------------------
    roi_data_path = roi_dir / "roi_data.npy"
    np.save(roi_data_path, roi_data)

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

    # --- Save Bragg vector map ----------------------------------------------
    histogram = bragg.histogram(mode="cal")
    vmap = np.asarray(histogram.data, dtype=np.float32)
    bragg_vector_map_path = roi_dir / "bragg_vector_map.npy"
    np.save(bragg_vector_map_path, vmap)
    n_peaks = int(np.count_nonzero(vmap))
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
        n_peaks=n_peaks,
        beam_center_yx=beam_cyx_list,
        beam_center_source=beam_center_source,
        cluster_id=cluster_id,
        reason=reason,
        background_fraction=cluster_validation.get("background_fraction"),
        sample_mask_coverage=cluster_validation.get("sample_mask_coverage"),
        cluster_validation_warning=cluster_validation.get("warning"),
        extraction_time_s=round(extraction_time_s, 4),
        bragg_time_s=round(bragg_time_s, 4),
        total_time_s=round(extraction_time_s + bragg_time_s, 4),
        roi_data_size_bytes=roi_data_size_bytes,
    )
