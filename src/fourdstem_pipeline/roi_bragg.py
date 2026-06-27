"""Stage 2A ROI Bragg detection using py4DSTEM.

Loads ROI candidates from a Stage-1 :class:`Stage1Manifest`, extracts
per-ROI sub-cubes from the original dataset, and runs py4DSTEM Bragg-disk
finding on each ROI.  All operations are isolated per ROI so that a failure
in one does not affect others.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .contracts import Stage1Manifest
from .logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ROIBraggResult:
    """Output from Bragg-disk detection on a single ROI."""

    name: str
    roi_bbox: list[int]
    nav_shape: tuple[int, int]
    sig_shape: tuple[int, int]
    output_dir: Path
    roi_data_path: Path | None = None
    bragg_vector_map_path: Path | None = None
    bragg_summary_path: Path | None = None
    n_peaks: int = 0
    error: str | None = None


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
) -> list[ROIBraggResult]:
    """Run py4DSTEM Bragg-disk detection on a set of ROIs.

    Parameters
    ----------
    manifest:
        Validated Stage-1 manifest providing nav/sig shapes.
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
        Keyword arguments forwarded to ``DataCube.find_Bragg_disks()``.

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

    if bragg_kwargs is None:
        bragg_kwargs = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)

    if max_rois is not None:
        rois = rois[:max_rois]

    log.info("Loading 4D-STEM data from %s (mem=%s)", data_path, mem)
    cube = py4DSTEM.import_file(
        str(data_path),
        mem=mem,
        scan=tuple(manifest.nav_shape),
    )

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
                nav_shape=tuple(manifest.nav_shape),
                thin_r=thin_r,
                bin_q=bin_q,
                bragg_kwargs=bragg_kwargs,
            )
            results.append(result)
            log.info(
                "ROI '%s' complete: %d Bragg peaks found → %s",
                name, result.n_peaks, roi_dir,
            )
        except Exception as exc:
            log.error("ROI '%s' failed: %s", name, exc)
            results.append(
                ROIBraggResult(
                    name=name,
                    roi_bbox=[int(v) for v in roi.get("bbox", [0, 0, 0, 0])],
                    nav_shape=(0, 0),
                    sig_shape=(0, 0),
                    output_dir=roi_dir,
                    error=str(exc),
                )
            )

    return results


def _process_one_roi(
    cube: Any,
    roi: dict[str, Any],
    roi_dir: Path,
    nav_shape: tuple[int, int],
    thin_r: int,
    bin_q: int,
    bragg_kwargs: dict[str, Any],
) -> ROIBraggResult:
    """Extract and process a single ROI sub-cube."""
    import py4DSTEM

    name = roi.get("name", "unnamed")
    bbox = [int(v) for v in roi["bbox"]]
    y0, y1, x0, x1 = bbox

    # Clamp to valid navigation range
    ny, nx = nav_shape
    y0 = max(0, min(y0, ny))
    y1 = max(y0 + 1, min(y1, ny))
    x0 = max(0, min(x0, nx))
    x1 = max(x0 + 1, min(x1, nx))

    thin_r = max(1, int(thin_r))
    bin_q = max(1, int(bin_q))

    # Extract sub-cube: thin navigation, keep full detector
    roi_data = np.asarray(
        cube.data[y0:y1:thin_r, x0:x1:thin_r, :, :],
        dtype=np.float32,
    )

    sig_shape = roi_data.shape[-2:]
    nav_shape_roi = roi_data.shape[:2]

    # Save raw ROI data
    roi_data_path = roi_dir / "roi_data.npy"
    np.save(roi_data_path, roi_data)

    # Build py4DSTEM DataCube and bin
    dc_roi = py4DSTEM.DataCube(roi_data, name=name, calibration=cube.calibration)
    if bin_q > 1:
        dc_roi = dc_roi.bin_Q(bin_q, dtype=np.float32)

    # Template: mean diffraction pattern over the ROI
    template = np.asarray(dc_roi.data.mean(axis=(0, 1)), dtype=np.float32)

    # Run Bragg disk detection
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

    # Save Bragg vector map
    histogram = bragg.histogram(mode="cal")
    vmap = np.asarray(histogram.data, dtype=np.float32)
    bragg_vector_map_path = roi_dir / "bragg_vector_map.npy"
    np.save(bragg_vector_map_path, vmap)
    n_peaks = int(np.count_nonzero(vmap))

    # Save per-ROI summary
    bragg_summary = {
        "name": name,
        "bbox": [y0, y1, x0, x1],
        "nav_shape": list(nav_shape_roi),
        "sig_shape": list(sig_shape),
        "thin_r": thin_r,
        "bin_q": bin_q,
        "n_bragg_peaks": n_peaks,
        "bragg_params": {k: v for k, v in bragg_kwargs.items() if k != "CUDA"},
        "cuda_used": bool(bragg_kwargs.get("CUDA", False)),
    }
    bragg_summary_path = roi_dir / "bragg_summary.json"
    bragg_summary_path.write_text(
        json.dumps(bragg_summary, indent=2, default=str),
        encoding="utf-8",
    )

    return ROIBraggResult(
        name=name,
        roi_bbox=[y0, y1, x0, x1],
        nav_shape=nav_shape_roi,
        sig_shape=sig_shape,
        output_dir=roi_dir,
        roi_data_path=roi_data_path,
        bragg_vector_map_path=bragg_vector_map_path,
        bragg_summary_path=bragg_summary_path,
        n_peaks=n_peaks,
    )
