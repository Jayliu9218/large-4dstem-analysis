"""Raw-data preprocessing utilities — bin, crop, and export to EMD/H5.

These operate directly on the original 4D-STEM data file (MIB, H5, etc.)
using py4DSTEM, without going through the full Stage-1 pipeline.  They are
intended for preparing compressed / region-of-interest datasets that load
faster in downstream analysis.

All functions require the ``large-4dstem`` conda environment (py4DSTEM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bin_and_export(
    input_path: str | Path,
    output_path: str | Path,
    *,
    r_bin: int = 1,
    q_bin: int = 1,
    mem: str = "MEMMAP",
    scan_shape: tuple[int, int] | list[int] | None = None,
) -> Path:
    """Load a 4D-STEM dataset, bin navigation and/or detector, and export as EMD/H5.

    Uses py4DSTEM's ``import_file`` (MIB) or ``read`` (H5/EMD) to load,
    ``bin_R`` / ``bin_Q`` to downsample, and ``py4DSTEM.save`` to write a
    standards-compliant EMD 1.0 file.

    Parameters
    ----------
    input_path:
        Path to the raw data file (``.mib``, ``.h5``, ``.hdf5``, ``.emd``).
    output_path:
        Desired output path.  ``.h5`` is appended if no recognised suffix.
    r_bin:
        Navigation (real-space) binning factor.  1 = no binning.
    q_bin:
        Detector (reciprocal-space) binning factor.  1 = no binning.
    mem:
        py4DSTEM memory mode — ``"MEMMAP"`` (recommended) or ``"RAM"``.
    scan_shape:
        Raw navigation shape ``(ny, nx)`` passed to ``import_file``.
        Required for MIB files; ignored for H5/EMD.

    Returns
    -------
    Path
        The resolved output path (may have ``.h5`` appended).
    """
    import py4DSTEM

    in_path = Path(input_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = Path(output_path).resolve()
    suffix = out_path.suffix.lower()
    if suffix not in {".h5", ".hdf5", ".emd", ".hspy"}:
        out_path = out_path.with_suffix(".h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix_in = in_path.suffix.lower()
    py4dstem_version: str = getattr(py4DSTEM, "__version__", "unknown")
    log.info("py4DSTEM version: %s", py4dstem_version)

    # --- Load ----------------------------------------------------------------
    if suffix_in in (".h5", ".hdf5", ".emd"):
        log.info("Loading H5/EMD file: %s", in_path)
        cube = py4DSTEM.read(str(in_path), mem=mem)
    else:
        scan_tuple: tuple[int, int] | None = (
            (int(scan_shape[0]), int(scan_shape[1])) if scan_shape else None
        )
        log.info("Loading MIB/raw file: %s (scan=%s, mem=%s)", in_path, scan_tuple, mem)
        cube = py4DSTEM.import_file(str(in_path), mem=mem, scan=scan_tuple)

    original_shape = tuple(int(v) for v in cube.data.shape)
    log.info("Loaded: shape=%s", original_shape)

    # --- Bin -----------------------------------------------------------------
    if r_bin > 1:
        log.info("Applying r_bin=%d …", r_bin)
        cube.bin_R(r_bin)
        new_shape = tuple(int(v) for v in cube.data.shape)
        log.info("  after bin_R(%d): %s → %s", r_bin, original_shape, new_shape)

    if q_bin > 1:
        log.info("Applying q_bin=%d …", q_bin)
        cube.bin_Q(q_bin)
        new_shape = tuple(int(v) for v in cube.data.shape)
        log.info("  after bin_Q(%d): %s → %s", q_bin, original_shape, new_shape)

    # --- Save ----------------------------------------------------------------
    try:
        py4DSTEM.save(str(out_path), cube, mode="o")
    except (TypeError, ValueError) as exc:
        # Fallback: some py4DSTEM versions use different signatures.
        log.warning("py4DSTEM.save(…, mode='o') failed, trying without mode: %s", exc)
        py4DSTEM.save(str(out_path), cube)

    final_shape = tuple(int(v) for v in cube.data.shape)
    log.info("Exported binned DataCube → %s  (shape=%s)", out_path, final_shape)
    return out_path


def crop_navigation_and_export(
    input_path: str | Path,
    output_path: str | Path,
    *,
    nav_crop: tuple[int, int, int, int] | list[int],
    mem: str = "MEMMAP",
    scan_shape: tuple[int, int] | list[int] | None = None,
) -> Path:
    """Load a 4D-STEM dataset, crop navigation dimensions, and export as EMD/H5.

    Only the first two (navigation) axes are cropped; detector (signal)
    dimensions are left unchanged.  Typical use: extract a 64×64 sub-region
    from a 512×512 scan for faster downstream screening.

    Parameters
    ----------
    input_path:
        Path to the raw data file (``.mib``, ``.h5``, ``.hdf5``, ``.emd``).
    output_path:
        Desired output path.  ``.h5`` is appended if no recognised suffix.
    nav_crop:
        Crop region in navigation pixels: ``[y0, y1, x0, x1]`` (half-open,
        following the pipeline bbox convention).  ``y1`` and ``x1`` are
        *exclusive*.
    mem:
        py4DSTEM memory mode.
    scan_shape:
        Raw navigation shape ``(ny, nx)`` for MIB import; ignored for H5/EMD.

    Returns
    -------
    Path
        The resolved output path.
    """
    import py4DSTEM

    in_path = Path(input_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = Path(output_path).resolve()
    suffix = out_path.suffix.lower()
    if suffix not in {".h5", ".hdf5", ".emd", ".hspy"}:
        out_path = out_path.with_suffix(".h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y0, y1, x0, x1 = [int(v) for v in nav_crop]
    if y0 < 0 or x0 < 0:
        raise ValueError(f"nav_crop start indices must be ≥ 0, got [{y0}, {y1}, {x0}, {x1}].")
    if y1 <= y0 or x1 <= x0:
        raise ValueError(
            f"nav_crop must have y1 > y0 and x1 > x0, got [{y0}, {y1}, {x0}, {x1}]."
        )

    suffix_in = in_path.suffix.lower()

    # --- Load ----------------------------------------------------------------
    if suffix_in in (".h5", ".hdf5", ".emd"):
        log.info("Loading H5/EMD file: %s", in_path)
        cube = py4DSTEM.read(str(in_path), mem=mem)
    else:
        scan_tuple: tuple[int, int] | None = (
            (int(scan_shape[0]), int(scan_shape[1])) if scan_shape else None
        )
        log.info("Loading MIB/raw file: %s (scan=%s, mem=%s)", in_path, scan_tuple, mem)
        cube = py4DSTEM.import_file(str(in_path), mem=mem, scan=scan_tuple)

    original_shape = tuple(int(v) for v in cube.data.shape)
    log.info("Loaded: shape=%s", original_shape)

    # Validate crop bounds against actual nav shape.
    nav_y, nav_x = original_shape[:2]
    if y1 > nav_y or x1 > nav_x:
        raise ValueError(
            f"nav_crop [{y0}, {y1}, {x0}, {x1}] exceeds data nav shape "
            f"({nav_y}, {nav_x})."
        )

    # --- Crop navigation -----------------------------------------------------
    cube.data = cube.data[y0:y1, x0:x1, :, :]
    cropped_shape = tuple(int(v) for v in cube.data.shape)
    log.info(
        "Cropped navigation: %s → %s  (nav_crop=[%d,%d,%d,%d])",
        original_shape, cropped_shape, y0, y1, x0, x1,
    )

    # --- Save ----------------------------------------------------------------
    try:
        py4DSTEM.save(str(out_path), cube, mode="o")
    except (TypeError, ValueError):
        py4DSTEM.save(str(out_path), cube)

    log.info("Exported cropped DataCube → %s  (shape=%s)", out_path, cropped_shape)
    return out_path
