from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .array_utils import as_numpy_block, iter_navigation_slices, parse_roi
from .dataset import DatasetHandle
from .logging import get_logger, log_block_progress

log = get_logger(__name__)


@dataclass(slots=True)
class FingerprintResult:
    profiles: np.ndarray
    radii: np.ndarray
    roi: tuple[int, int, int, int] | None
    output_dir: Path | None = None


def compute_radial_fingerprints(
    dataset: DatasetHandle,
    geometry: dict,
    bins: int,
    *,
    roi: tuple[int, int, int, int] | list[int] | None = None,
    output_dir: str | Path | None = None,
    block_shape: tuple[int, int] = (8, 8),
) -> FingerprintResult:
    """Reduce each diffraction pattern to a radial profile."""
    nav_shape = dataset.navigation_shape
    sig_shape = dataset.signal_shape
    y_slice, x_slice = parse_roi(roi, nav_shape)
    out_shape = (y_slice.stop - y_slice.start, x_slice.stop - x_slice.start, int(bins))
    profiles = np.zeros(out_shape, dtype=np.float32)

    bin_index, counts, radii = _radial_bin_index(sig_shape, geometry.get("center"), int(bins))
    blocks = list(iter_navigation_slices(out_shape[:2], block_shape))
    n_blocks = len(blocks)
    log.info("Computing radial fingerprints (%d bins) across %d navigation blocks", int(bins), n_blocks)

    for idx, (ys, xs) in enumerate(blocks, start=1):
        log_block_progress(log, block=idx, total_blocks=n_blocks, stage="fingerprints")
        src_y = slice(ys.start + y_slice.start, ys.stop + y_slice.start)
        src_x = slice(xs.start + x_slice.start, xs.stop + x_slice.start)
        block = as_numpy_block(dataset.data[src_y, src_x, :, :]).astype(np.float32, copy=False)
        flat = block.reshape((-1,) + sig_shape)
        prof = np.zeros((flat.shape[0], int(bins)), dtype=np.float32)
        for pidx, pattern in enumerate(flat):
            sums = np.bincount(bin_index.ravel(), weights=pattern.ravel(), minlength=int(bins))
            prof[pidx] = sums[: int(bins)] / np.maximum(counts, 1)
        profiles[ys, xs, :] = prof.reshape((ys.stop - ys.start, xs.stop - xs.start, int(bins)))

    result = FingerprintResult(profiles=profiles, radii=radii, roi=tuple(roi) if roi is not None else None, output_dir=Path(output_dir) if output_dir else None)
    if output_dir:
        result.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(result.output_dir / "radial_fingerprints.npy", profiles)
        np.save(result.output_dir / "radial_radii.npy", radii)
    return result


def _radial_bin_index(signal_shape: tuple[int, int], center: tuple[float, float] | list[float] | None, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sy, sx = signal_shape
    cy, cx = center if center is not None else ((sy - 1) / 2, (sx - 1) / 2)
    yy, xx = np.indices(signal_shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = rr.max()
    edges = np.linspace(0, max_radius + 1e-6, bins + 1)
    bin_index = np.clip(np.digitize(rr, edges) - 1, 0, bins - 1).astype(np.int16)
    counts = np.bincount(bin_index.ravel(), minlength=bins).astype(np.float32)[:bins]
    radii = 0.5 * (edges[:-1] + edges[1:])
    return bin_index, counts, radii.astype(np.float32)
