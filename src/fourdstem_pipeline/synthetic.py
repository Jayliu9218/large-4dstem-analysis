from __future__ import annotations

import numpy as np


def make_synthetic_4dstem(
    navigation_shape: tuple[int, int] = (16, 16),
    signal_shape: tuple[int, int] = (64, 64),
    *,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a small 4D-STEM dataset with two ring-like phase regions."""
    rng = np.random.default_rng(seed)
    ny, nx = navigation_shape
    sy, sx = signal_shape
    yy, xx = np.indices(signal_shape)
    cy = (sy - 1) / 2
    cx = (sx - 1) / 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    angle = np.arctan2(yy - cy, xx - cx)

    data = np.empty((ny, nx, sy, sx), dtype=np.float32)
    labels = np.zeros((ny, nx), dtype=np.int16)

    for iy in range(ny):
        for ix in range(nx):
            phase = int((iy > ny // 2) or (ix > nx // 2))
            labels[iy, ix] = phase
            ring_a = 10 + 5 * phase
            ring_b = 20 + 4 * phase
            orientation = (iy / max(ny - 1, 1) + ix / max(nx - 1, 1)) * np.pi
            spots = np.exp(-0.5 * ((rr - ring_a) / 1.6) ** 2)
            spots += 0.8 * np.exp(-0.5 * ((rr - ring_b) / 1.8) ** 2)
            texture = 1.0 + 0.25 * np.cos(4 * (angle - orientation))
            disk = 3.0 * np.exp(-0.5 * (rr / 2.2) ** 2)
            noise = rng.normal(0, 0.03, signal_shape)
            data[iy, ix] = np.clip(disk + spots * texture + noise, 0, None)

    return data, labels
