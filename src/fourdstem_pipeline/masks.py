from __future__ import annotations

import numpy as np


def annular_mask(
    signal_shape: tuple[int, int],
    *,
    inner_radius: float,
    outer_radius: float,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    sy, sx = signal_shape
    cy, cx = center if center is not None else ((sy - 1) / 2, (sx - 1) / 2)
    yy, xx = np.indices(signal_shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (rr >= float(inner_radius)) & (rr < float(outer_radius))


def build_annular_masks(
    signal_shape: tuple[int, int],
    mask_config: dict[str, dict[str, float]],
    *,
    center: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, spec in mask_config.items():
        masks[name] = annular_mask(
            signal_shape,
            inner_radius=spec["inner_radius"],
            outer_radius=spec["outer_radius"],
            center=center,
        )
    return masks
