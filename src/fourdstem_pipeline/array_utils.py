from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


def as_numpy_block(block: Any) -> np.ndarray:
    """Convert a sliced block to NumPy, computing lazy arrays only for that block."""
    if hasattr(block, "compute"):
        block = block.compute()
    return np.asarray(block)


def iter_navigation_slices(shape: tuple[int, int], block_shape: tuple[int, int] = (8, 8)) -> Iterator[tuple[slice, slice]]:
    ny, nx = shape
    by, bx = block_shape
    for y0 in range(0, ny, by):
        for x0 in range(0, nx, bx):
            yield slice(y0, min(y0 + by, ny)), slice(x0, min(x0 + bx, nx))


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(denom, eps)


def parse_roi(roi: tuple[int, int, int, int] | list[int] | None, shape: tuple[int, int]) -> tuple[slice, slice]:
    """Parse a ``[y0, y1, x0, x1]`` ROI into ``(y_slice, x_slice)``.

    Follows the unified data contract ``bbox_order: y0_y1_x0_x1``.
    Pass ``None`` to cover the full *shape*.
    """
    if roi is None:
        return slice(0, shape[0]), slice(0, shape[1])
    y0, y1, x0, x1 = [int(v) for v in roi]
    y0 = max(0, min(y0, shape[0]))
    y1 = max(y0, min(y1, shape[0]))
    x0 = max(0, min(x0, shape[1]))
    x1 = max(x0, min(x1, shape[1]))
    if y1 <= y0 or x1 <= x0:
        raise ValueError(
            f"ROI {[int(v) for v in roi]} clamped to shape {shape} results in "
            f"zero-area slice: y=({y0},{y1}), x=({x0},{x1}). "
            f"Adjust the ROI so that y1 > y0 and x1 > x0 after clamping."
        )
    return slice(y0, y1), slice(x0, x1)
