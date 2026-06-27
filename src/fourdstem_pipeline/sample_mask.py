"""Sample mask generation from virtual images.

Creates a binary mask that separates sample from background / vacuum /
edge regions.  The mask is used to exclude non-sample areas from
clustering, diagnostics, and ROI candidate generation.

Conventions
-----------
* ``True``  = sample (keep)
* ``False`` = background / vacuum / excluded
* Labels outside the mask are set to ``-1`` (background / vacuum / excluded).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from .export import save_png


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_sample_mask(
    image: np.ndarray,
    percentile: float = 15,
) -> np.ndarray:
    """Create a binary sample mask by percentile thresholding.

    Parameters
    ----------
    image:
        2-D virtual image (typically ADF or HAADF).
    percentile:
        Intensity percentile below which pixels are considered
        background.

    Returns
    -------
    Boolean mask where ``True`` indicates sample.
    """
    image = np.asarray(image, dtype=np.float64)
    threshold = float(np.percentile(image, percentile))
    return image > threshold


def clean_mask(
    mask: np.ndarray,
    min_size: int = 100,
    fill_holes: bool = True,
) -> np.ndarray:
    """Remove small objects and optionally fill small holes.

    Parameters
    ----------
    mask:
        Boolean mask from :func:`make_sample_mask`.
    min_size:
        Minimum connected-component size in pixels.  Objects /
        holes smaller than this are removed.
    fill_holes:
        If ``True``, fill small background regions that are fully
        enclosed by sample (i.e. do not touch the image border).

    Returns
    -------
    Cleaned boolean mask.
    """
    mask = np.asarray(mask, dtype=bool).copy()

    # --- Remove small foreground objects ---------------------------------
    labels, n_labels = _label_components(mask)
    for i in range(1, n_labels + 1):
        if np.sum(labels == i) < min_size:
            mask[labels == i] = False

    # --- Fill small holes -------------------------------------------------
    if fill_holes:
        inverted = ~mask
        inv_labels, n_inv = _label_components(inverted)
        for i in range(1, n_inv + 1):
            component = inv_labels == i
            if component.sum() < min_size and not _touches_border(component):
                mask[component] = True  # fill the hole

    return mask


def save_sample_mask_outputs(
    output_dir: Path,
    png_dir: Path,
    mask: np.ndarray,
    source_image: np.ndarray,
) -> dict[str, Path]:
    """Save sample mask arrays and visualisations.

    Parameters
    ----------
    output_dir:
        Directory for ``.npy`` outputs (typically ``00_preprocess``).
    png_dir:
        Directory for PNG previews.
    mask:
        The boolean sample mask.
    source_image:
        The virtual image used to generate the mask (for overlay).

    Returns
    -------
    Dict mapping output names to paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / "sample_mask.npy"
    np.save(npy_path, mask.astype(np.bool_))

    mask_png = save_png(png_dir / "sample_mask.png", mask.astype(np.float32))

    overlay = _mask_overlay(source_image, mask)
    overlay_png = save_png(png_dir / "sample_mask_overlay_adf.png", overlay)

    return {
        "sample_mask_npy": npy_path,
        "sample_mask_png": mask_png,
        "sample_mask_overlay_png": overlay_png,
    }


def apply_mask_to_labels(
    labels: np.ndarray,
    mask: np.ndarray,
    background_label: int = -1,
) -> None:
    """Set *labels* outside the sample mask to *background_label*.

    The *labels* array is modified **in-place**.
    """
    labels[~mask] = background_label


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected components in a boolean mask.

    Returns
    -------
    (labels_array, n_components)
        *labels_array* is an ``int32`` array of the same shape as *mask*
        with 0 for background and 1..*n_components* for each object.
    """
    labels = np.zeros(mask.shape, dtype=np.int32)
    n_found = 0
    for y, x in np.argwhere(mask):
        if labels[y, x] != 0:
            continue
        n_found += 1
        queue = deque([(int(y), int(x))])
        labels[y, x] = n_found
        while queue:
            cy, cx = queue.popleft()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (
                    0 <= ny < mask.shape[0]
                    and 0 <= nx < mask.shape[1]
                    and mask[ny, nx]
                    and labels[ny, nx] == 0
                ):
                    labels[ny, nx] = n_found
                    queue.append((ny, nx))
    return labels, n_found


def _touches_border(component: np.ndarray) -> bool:
    """Return ``True`` if *component* touches any edge of the image."""
    return bool(
        component[0, :].any()
        or component[-1, :].any()
        or component[:, 0].any()
        or component[:, -1].any()
    )


def _scale_gray(image: np.ndarray) -> np.ndarray:
    """Scale a 2-D float array to ``uint8`` for overlay rendering."""
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    scaled = (arr - lo) / max(hi - lo, 1e-12)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def _mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render the mask boundary as a green outline on *image*."""
    gray = _scale_gray(image)
    rgb = np.repeat(gray[..., None], 3, axis=-1)

    # Compute boundary of the mask (pixels where mask changes)
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]

    rgb[boundary] = [0, 255, 0]  # green outline
    return rgb
