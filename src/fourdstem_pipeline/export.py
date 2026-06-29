from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any
import struct
import zlib

import numpy as np


# ---------------------------------------------------------------------------
# Colormaps — 256×3 uint8 LUTs for dependency-free pseudocolor rendering.
# ---------------------------------------------------------------------------


def _interpolate_colormap(control_points: list[tuple[float, float, float]], n: int = 256) -> np.ndarray:
    """Linearly interpolate *(position, r, g, b)* control points to *n* entries."""
    positions = np.linspace(0, 1, len(control_points))
    rgb = np.asarray(control_points, dtype=np.float64)
    out = np.zeros((n, 3), dtype=np.float64)
    for channel in range(3):
        out[:, channel] = np.interp(np.linspace(0, 1, n), positions, rgb[:, channel])
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# Canonical matplotlib viridis control points (position, r, g, b).
_VIRIDIS_CONTROL: list[tuple[float, float, float]] = [
    (0.267004, 0.004874, 0.329415),   # 0.00  — deep purple
    (0.229739, 0.322361, 0.545706),   # 0.25  — blue
    (0.127568, 0.566949, 0.550556),   # 0.50  — teal
    (0.369214, 0.788888, 0.382914),   # 0.75  — green
    (0.993248, 0.906157, 0.143936),   # 1.00  — yellow
]

# Grayscale LUT for consistency.
_GRAY_CONTROL: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
]

_COLORMAPS: dict[str, np.ndarray] = {
    "viridis": _interpolate_colormap(_VIRIDIS_CONTROL),
    "gray": _interpolate_colormap(_GRAY_CONTROL),
}


def _get_colormap(name: str = "gray") -> np.ndarray:
    """Return a 256×3 uint8 colormap LUT by name."""
    return _COLORMAPS.get(name, _COLORMAPS["gray"])


# ---------------------------------------------------------------------------
# Cubic IPF (Inverse Pole Figure) colormap — maps crystal directions to RGB
# using the standard convention: [001]=red, [101]=green, [111]=blue.
# ---------------------------------------------------------------------------


def _build_cubic_ipf_lut(size: int = 256) -> np.ndarray:
    """Precompute a *(size, size, 3)* uint8 LUT for the cubic fundamental sector.

    The LUT covers the stereographic projection of the standard stereographic
    triangle with corners at [001] (red), [101] (green), and [111] (blue).
    Pixels outside the sector are set to white (255, 255, 255).
    """
    # Stereographic grid: (X, Y) in [-1, 1] for the projection plane.
    grid = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    XX, YY = np.meshgrid(grid, grid)
    denom = 1.0 + XX * XX + YY * YY
    # Inverse stereographic projection: (X,Y) → (x,y,z) unit vector.
    x = 2.0 * XX / denom
    y = 2.0 * YY / denom
    z = (1.0 - XX * XX - YY * YY) / denom  # strictly positive in the hemisphere

    # Take absolute values (cubic symmetry — all octants are equivalent).
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)

    # Sort each pixel's (ax, ay, az) into ascending order: a <= b <= c.
    stacked = np.stack([ax, ay, az], axis=-1)  # (size, size, 3)
    a = np.min(stacked, axis=-1)
    c = np.max(stacked, axis=-1)
    b = np.sum(stacked, axis=-1) - a - c  # middle value

    # Barycentric coordinates within the fundamental triangle [001]-[101]-[111].
    denom_c = np.maximum(c, 1e-12)
    w_001 = (c - b) / denom_c  # red   — proximity to [001]
    w_101 = (b - a) / denom_c  # green — proximity to [101]
    w_111 = a / denom_c        # blue  — proximity to [111]

    # Check if this pixel's sorted abs values correspond to the right sector.
    # The fundamental sector for sorted abs values a<=b<=c with corners
    # [001]=(0,0,1) sorted→(0,0,1), [101]=(1,0,1)/√2 sorted→(0,1,1)/√2,
    # [111]=(1,1,1)/√3 sorted→(1,1,1)/√3.
    # After sorting abs, we need |x|<=|y|<=|z| (which is a<=b<=c by construction).
    # Additionally we need the point to be in the upper hemisphere (z>0) and
    # within the stereographic triangle boundaries.
    # Valid points in the fundamental sector satisfy: z >= y >= x >= 0
    # After taking abs and sorting, this becomes a<=b<=c with the original
    # point having x<=y<=z (all non-negative).
    in_sector = (z > 0.0) & (_is_in_fundamental_sector(x, y, z))

    rgb = np.full((size, size, 3), 255, dtype=np.uint8)
    mask = in_sector
    rgb[mask, 0] = np.clip(w_001[mask] * 255, 0, 255).astype(np.uint8)
    rgb[mask, 1] = np.clip(w_101[mask] * 255, 0, 255).astype(np.uint8)
    rgb[mask, 2] = np.clip(w_111[mask] * 255, 0, 255).astype(np.uint8)
    return rgb


def _is_in_fundamental_sector(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
) -> np.ndarray:
    """Boolean mask: pixels whose *(x, y, z)* lies in the cubic fundamental sector.

    For a vector with all components non-negative, the cubic fundamental
    sector (stereographic triangle [001]–[101]–[111]) is: ``x <= y <= z``.
    """
    return (x >= 0.0) & (y >= x) & (z >= y)


_CUBIC_IPF_LUT: np.ndarray = _build_cubic_ipf_lut(256)


def apply_ipf_colors(directions_xyz: np.ndarray) -> np.ndarray:
    """Map crystal direction unit vectors to cubic-IPF RGB colours.

    Uses the standard convention: [001]→red, [101]→green, [111]→blue.

    Parameters
    ----------
    directions_xyz:
        ``(N, 3)`` float array of unit vectors in crystal (reciprocal)
        coordinates.  Vectors need not be exactly normalised — the
        function divides by the largest absolute component internally.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` uint8 RGB colours.
    """
    vecs = np.asarray(directions_xyz, dtype=np.float64)
    if vecs.ndim != 2 or vecs.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got shape {vecs.shape!r}.")
    # Normalise by max absolute component (robust to near-zero vectors).
    norms = np.max(np.abs(vecs), axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    v = vecs / norms
    # Take absolute values for cubic symmetry.
    av = np.abs(v)
    # Sort each row ascending: a <= b <= c.
    a = np.min(av, axis=1)
    c = np.max(av, axis=1)
    b = np.sum(av, axis=1) - a - c
    # Barycentric weights.
    denom = np.maximum(c, 1e-12)
    w_001 = (c - b) / denom
    w_101 = (b - a) / denom
    w_111 = a / denom
    rgb = np.zeros((vecs.shape[0], 3), dtype=np.uint8)
    rgb[:, 0] = np.clip(w_001 * 255, 0, 255).astype(np.uint8)
    rgb[:, 1] = np.clip(w_101 * 255, 0, 255).astype(np.uint8)
    rgb[:, 2] = np.clip(w_111 * 255, 0, 255).astype(np.uint8)
    return rgb


def save_ipf_legend(
    path: str | Path,
    *,
    size: tuple[int, int] = (256, 256),
    label: str = "",
) -> Path:
    """Save the cubic IPF stereographic-triangle legend as a PNG.

    Parameters
    ----------
    path:
        Output PNG path.
    size:
        Legend image dimensions ``(width, height)`` in pixels.
    label:
        Optional title drawn above the triangle.
    """
    h, w = int(size[1]), int(size[0])
    # Resample the precomputed LUT to the requested size.
    lut_h, lut_w = _CUBIC_IPF_LUT.shape[:2]
    sy = np.linspace(0, lut_h - 1, h).astype(np.int32)
    sx = np.linspace(0, lut_w - 1, w).astype(np.int32)
    legend = _CUBIC_IPF_LUT[sy[:, None], sx[None, :]]
    # Add a dark border around non-white pixels.
    is_colored = np.any(legend < 250, axis=-1)
    border = np.zeros_like(is_colored)
    if is_colored.shape[0] > 1 and is_colored.shape[1] > 1:
        border[1:, :] |= is_colored[:-1, :] & ~is_colored[1:, :]
        border[:-1, :] |= is_colored[1:, :] & ~is_colored[:-1, :]
        border[:, 1:] |= is_colored[:, :-1] & ~is_colored[:, 1:]
        border[:, :-1] |= is_colored[:, 1:] & ~is_colored[:, :-1]
    border_color = np.asarray([40, 40, 40], dtype=np.uint8)
    if label:
        # Add header row.
        header_h = 24
        canvas = np.full((h + header_h, w, 3), 255, dtype=np.uint8)
        canvas[header_h:, :, :] = legend
        _draw_text(canvas, label.upper(), w // 2 - len(label) * 3, 6, color=(30, 30, 30), scale=1)
    else:
        canvas = legend.copy()
    if np.any(border):
        if label:
            canvas[header_h:][border] = border_color
        else:
            canvas[border] = border_color
    return save_png(path, canvas)


# ---------------------------------------------------------------------------
# Tight-layout margin constants (pyxem-style compact plots).
# ---------------------------------------------------------------------------

# Default tight margins — match pyxem's compact, publication-ready aesthetic.
_TIGHT_MARGIN_L = 46
_TIGHT_MARGIN_R = 14
_TIGHT_MARGIN_T = 16
_TIGHT_MARGIN_B = 34


def save_summary(output_dir: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(output_dir) / "workflow_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return path


def save_report(output_dir: str | Path, summary: dict[str, Any], phase_labels: np.ndarray) -> Path:
    path = Path(output_dir) / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _render_report(summary, phase_labels, base_dir=path.parent)
    path.write_text(markdown, encoding="utf-8")
    html_path = path.with_suffix(".html")
    html_path.write_text(_render_html_report(summary, phase_labels, base_dir=path.parent), encoding="utf-8")
    return path


def save_npz(output_dir: str | Path, name: str, **arrays: np.ndarray) -> Path:
    path = Path(output_dir) / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def save_png(
    path: str | Path,
    image: np.ndarray,
    *,
    percentiles: tuple[float, float] = (1, 99),
    cmap: str = "gray",
) -> Path:
    """Save a 2D scalar image or RGB uint8 image as PNG without extra dependencies.

    Parameters
    ----------
    cmap:
        Colormap name for 2D images.  ``"gray"`` (default), ``"viridis"``.
        Ignored for 3D RGB inputs.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 2:
        scaled = _scale_to_uint8(arr, percentiles)
        lut = _get_colormap(cmap)
        rgb = lut[scaled]
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        rgb = arr.astype(np.uint8, copy=False) if arr.dtype == np.uint8 else _scale_to_uint8(arr, percentiles)
    else:
        raise ValueError(f"Expected 2D or RGB image, got shape {arr.shape!r}.")
    _write_rgb_png(out, rgb)
    return out


def save_label_png(path: str | Path, labels: np.ndarray) -> Path:
    arr = np.asarray(labels, dtype=np.int64)
    palette = _label_palette()
    return save_png(path, palette[np.mod(arr, len(palette))])


def save_annotated_label_png(path: str | Path, labels: np.ndarray, *, title: str = "PHASE CLUSTERS") -> Path:
    arr = np.asarray(labels, dtype=np.int64)
    palette = _label_palette()
    image = palette[np.mod(arr, len(palette))]
    labels_present, counts = np.unique(arr, return_counts=True)

    scale = max(1, int(np.ceil(192 / max(arr.shape[0], arr.shape[1]))))
    map_rgb = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    legend_width = 220
    height = max(map_rgb.shape[0], 46 + 28 * len(labels_present))
    canvas = np.full((height, map_rgb.shape[1] + legend_width, 3), 255, dtype=np.uint8)
    canvas[: map_rgb.shape[0], : map_rgb.shape[1]] = map_rgb
    x0 = map_rgb.shape[1] + 16

    _draw_text(canvas, title.upper(), x0, 14, color=(30, 30, 30), scale=2)
    for row, (label, count) in enumerate(zip(labels_present, counts)):
        y = 48 + row * 28
        color = palette[int(label) % len(palette)]
        canvas[y : y + 14, x0 : x0 + 18] = color
        _draw_text(canvas, f"CLUSTER {int(label)}  N={int(count)}", x0 + 28, y + 2, color=(30, 30, 30), scale=1)
    return save_png(path, canvas)


def save_profile_png(
    path: str | Path,
    radii: np.ndarray,
    profiles: np.ndarray,
    *,
    size: tuple[int, int] = (720, 420),
    xlabel: str = "",
    ylabel: str = "",
) -> Path:
    """Save the mean radial profile as a simple line plot PNG."""
    width, height = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    ml, mr, mt, mb = _TIGHT_MARGIN_L, _TIGHT_MARGIN_R, _TIGHT_MARGIN_T, _TIGHT_MARGIN_B
    x0, x1 = ml, width - mr - 1
    y0, y1 = mt, height - mb - 1

    canvas[y0:y1 + 1, x0] = 30
    canvas[y1, x0:x1 + 1] = 30
    for frac in np.linspace(0.25, 0.75, 3):
        y = int(y1 - frac * (y1 - y0))
        canvas[y, x0:x1 + 1] = 225

    profile = np.asarray(profiles, dtype=np.float32).reshape(-1, profiles.shape[-1]).mean(axis=0)
    radii = np.asarray(radii, dtype=np.float32)
    if radii.size != profile.size:
        radii = np.arange(profile.size, dtype=np.float32)
    px = _normalize_to_span(radii, x0, x1)
    py = y1 - _normalize_to_span(profile, 0, y1 - y0)
    points = np.column_stack([px.astype(int), py.astype(int)])
    _draw_polyline(canvas, points, color=(25, 90, 155))

    if xlabel:
        _draw_text(canvas, xlabel, x0 + (x1 - x0) // 2 - len(xlabel) * 3, height - mb + 8, color=(70, 70, 70), scale=1)
    if ylabel:
        _draw_text_vertical(canvas, ylabel, 8, mt + (y1 - mt) // 2 + len(ylabel) * 3, color=(70, 70, 70), scale=1)

    return save_png(path, canvas)


def save_lines_png(
    path: str | Path,
    x: np.ndarray,
    ys: np.ndarray,
    *,
    colors: list[tuple[int, int, int]] | None = None,
    size: tuple[int, int] = (720, 420),
    xlabel: str = "",
    ylabel: str = "",
) -> Path:
    width, height = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    ml, mr, mt, mb = _TIGHT_MARGIN_L, _TIGHT_MARGIN_R, _TIGHT_MARGIN_T, _TIGHT_MARGIN_B
    x0, x1 = ml, width - mr - 1
    y0, y1 = mt, height - mb - 1
    canvas[y0:y1 + 1, x0] = 30
    canvas[y1, x0:x1 + 1] = 30
    for frac in np.linspace(0.25, 0.75, 3):
        y_grid = int(y1 - frac * (y1 - y0))
        canvas[y_grid, x0:x1 + 1] = 225

    arr = np.asarray(ys, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    x_values = np.asarray(x, dtype=np.float32)
    if x_values.size != arr.shape[1]:
        x_values = np.arange(arr.shape[1], dtype=np.float32)
    px = _normalize_to_span(x_values, x0, x1).astype(int)
    finite = arr[np.isfinite(arr)]
    ymin = float(np.min(finite)) if finite.size else 0.0
    ymax = float(np.max(finite)) if finite.size else 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    palette = colors or [tuple(map(int, color)) for color in _label_palette()]
    for idx, series in enumerate(arr):
        py = (y1 - (series - ymin) / (ymax - ymin) * (y1 - y0)).astype(int)
        _draw_polyline(canvas, np.column_stack([px, py]), color=palette[idx % len(palette)])

    if xlabel:
        _draw_text(canvas, xlabel, x0 + (x1 - x0) // 2 - len(xlabel) * 3, height - mb + 8, color=(70, 70, 70), scale=1)
    if ylabel:
        _draw_text_vertical(canvas, ylabel, 8, mt + (y1 - mt) // 2 + len(ylabel) * 3, color=(70, 70, 70), scale=1)

    return save_png(path, canvas)


def save_bar_png(
    path: str | Path,
    values: np.ndarray,
    *,
    size: tuple[int, int] = (720, 420),
    xlabel: str = "",
    ylabel: str = "",
) -> Path:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    width, height = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    ml, mr, mt, mb = _TIGHT_MARGIN_L, _TIGHT_MARGIN_R, _TIGHT_MARGIN_T, _TIGHT_MARGIN_B
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    vmax = max(float(np.nanmax(np.abs(arr))), 1e-12)
    groups, bars = arr.shape
    slot = max(1, plot_w // max(groups, 1))
    bar_w = max(1, slot // max(bars + 1, 2))
    palette = _label_palette()
    baseline = height - mb - 1
    canvas[mt:baseline + 1, ml] = 30
    canvas[baseline, ml:width - mr] = 30
    for group in range(groups):
        for bar in range(bars):
            value = max(float(arr[group, bar]), 0.0)
            h = int(value / vmax * plot_h)
            bx0 = ml + group * slot + bar * bar_w + 2
            bx1 = min(bx0 + bar_w, width - mr)
            canvas[baseline - h:baseline, bx0:bx1] = palette[bar % len(palette)]

    if xlabel:
        _draw_text(canvas, xlabel, ml + plot_w // 2 - len(xlabel) * 3, height - mb + 8, color=(70, 70, 70), scale=1)
    if ylabel:
        _draw_text_vertical(canvas, ylabel, 8, mt + plot_h // 2 + len(ylabel) * 3, color=(70, 70, 70), scale=1)

    return save_png(path, canvas)


def save_heatmap_png(
    path: str | Path,
    data: np.ndarray,
    *,
    cmap: str = "viridis",
    xlabel: str = "",
    ylabel: str = "",
    xticklabels: list[str] | None = None,
    yticklabels: list[str] | None = None,
    size: tuple[int, int] = (720, 420),
    add_colorbar: bool = True,
    title: str = "",
) -> Path:
    """Save a 2D array as a pseudocolour heatmap with labelled axes and colour bar.

    Mimics pyxem's correlation-heatmap panel from
    ``OrientationMap.plot_over_signal(..., add_ipf_correlation_heatmap=True)``.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape!r}.")
    n_rows, n_cols = arr.shape
    width, height = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    ml, mr, mt, mb = _TIGHT_MARGIN_L, _TIGHT_MARGIN_R, _TIGHT_MARGIN_T, _TIGHT_MARGIN_B
    # Reserve space for colorbar on the right.
    cbar_w = 28 if add_colorbar else 0
    cbar_gap = 10 if add_colorbar else 0
    plot_w = width - ml - mr - cbar_w - cbar_gap
    plot_h = height - mt - mb
    x0, x1 = ml, ml + plot_w
    y0, y1 = mt, mt + plot_h

    # --- Draw the heatmap image -----------------------------------------------
    finite = arr[np.isfinite(arr)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = (arr - vmin) / (vmax - vmin)
    scaled = np.clip(norm * 255, 0, 255).astype(np.uint8)
    lut = _get_colormap(cmap)
    heatmap = lut[scaled]
    # Resample heatmap to fit the plot area via nearest-neighbour.
    sy = np.linspace(0, n_rows - 1, plot_h).astype(np.int32)
    sx = np.linspace(0, n_cols - 1, plot_w).astype(np.int32)
    canvas[y0:y1, x0:x1] = heatmap[sy[:, None], sx[None, :]]

    # --- Draw axes and tick labels --------------------------------------------
    # Left axis.
    canvas[y0:y1 + 1, x0] = 30
    # Bottom axis.
    canvas[y1, x0:x1 + 1] = 30
    # Top axis (thin).
    canvas[y0, x0:x1 + 1] = 180
    # Right axis (thin).
    canvas[y0:y1 + 1, x1] = 180

    # X tick labels.
    if xticklabels:
        n_xticks = len(xticklabels)
        for i, label in enumerate(xticklabels):
            tx = int(x0 + (i + 0.5) / n_xticks * plot_w)
            canvas[y1 + 2:y1 + 4, tx] = 60
            _draw_text(canvas, str(label)[:6], tx - len(str(label)[:6]) * 3, y1 + 6, color=(60, 60, 60), scale=1)
    elif n_cols <= 20:
        for i in range(n_cols):
            tx = int(x0 + (i + 0.5) / n_cols * plot_w)
            canvas[y1 + 2:y1 + 4, tx] = 60

    # Y tick labels.
    if yticklabels:
        n_yticks = len(yticklabels)
        for i, label in enumerate(yticklabels):
            ty = int(y0 + (i + 0.5) / n_yticks * plot_h)
            canvas[ty, x0 - 4:x0 - 2] = 60
            _draw_text(canvas, str(label)[:8], x0 - 8 - len(str(label)[:8]) * 6, ty - 3, color=(60, 60, 60), scale=1)
    elif n_rows <= 20:
        for i in range(n_rows):
            ty = int(y0 + (i + 0.5) / n_rows * plot_h)
            canvas[ty, x0 - 4:x0 - 2] = 60

    # Axis labels.
    if xlabel:
        _draw_text(canvas, xlabel, x0 + plot_w // 2 - len(xlabel) * 3, height - mb + 8, color=(70, 70, 70), scale=1)
    if ylabel:
        _draw_text_vertical(canvas, ylabel, 8, mt + plot_h // 2 + len(ylabel) * 3, color=(70, 70, 70), scale=1)
    if title:
        _draw_text(canvas, title.upper(), x0 + plot_w // 2 - len(title) * 3, 6, color=(30, 30, 30), scale=1)

    # --- Colour bar strip -----------------------------------------------------
    if add_colorbar:
        cb_x0 = x1 + cbar_gap
        cb_x1 = cb_x0 + cbar_w
        # Gradient from vmin (bottom) to vmax (top).
        cb_h = plot_h
        gradient = np.linspace(255, 0, cb_h, dtype=np.uint8)  # bottom=0→vmin, top=255→vmax
        cb = lut[gradient]
        canvas[y0:y1, cb_x0:cb_x1] = cb[:, None, :]
        # Border.
        canvas[y0:y1 + 1, cb_x0] = 30
        canvas[y0:y1 + 1, cb_x1] = 30
        canvas[y0, cb_x0:cb_x1 + 1] = 30
        canvas[y1, cb_x0:cb_x1 + 1] = 30
        # Tick labels on colorbar.
        for frac, lbl in [(0.0, f"{vmin:.2g}"), (0.5, f"{(vmin+vmax)/2:.2g}"), (1.0, f"{vmax:.2g}")]:
            cy = int(y1 - frac * cb_h)
            canvas[cy, cb_x1 + 1:cb_x1 + 5] = 60
            _draw_text(canvas, lbl, cb_x1 + 7, cy - 3, color=(60, 60, 60), scale=1)

    return save_png(path, canvas)


def save_colorbar_png(
    path: str | Path,
    cmap: str = "viridis",
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    label: str = "",
    orientation: str = "vertical",
    size: tuple[int, int] = (28, 256),
) -> Path:
    """Save a standalone colour-bar strip as a PNG.

    Useful as a companion to unadorned ``save_png`` images that need a scale
    reference, or embedded alongside figures.
    """
    w, h = int(size[0]), int(size[1])
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    lut = _get_colormap(cmap)

    if orientation == "vertical":
        gradient = np.linspace(255, 0, h, dtype=np.uint8)
        bar = lut[gradient]
        canvas[:, 2:w - 2] = bar[:, None, :]
        canvas[:, 0:2] = 30
        canvas[:, w - 2:w] = 30
        canvas[0, :] = 30
        canvas[-1, :] = 30
        # Tick labels.
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = int(h - 1 - frac * (h - 1))
            canvas[y, w - 2:w + 6] = 60
            val = vmin + frac * (vmax - vmin)
            _draw_text(canvas, f"{val:.2g}", w + 8, y - 3, color=(60, 60, 60), scale=1)
        if label:
            _draw_text_vertical(canvas, label, 6, h // 2 + len(label) * 3, color=(30, 30, 30), scale=1)
    else:
        gradient = np.linspace(0, 255, w, dtype=np.uint8)
        bar = lut[gradient]
        canvas[2:h - 2, :] = bar[None, :, :]
        canvas[0:2, :] = 30
        canvas[h - 2:h, :] = 30
        canvas[:, 0] = 30
        canvas[:, -1] = 30
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = int(frac * (w - 1))
            canvas[h - 2:h + 6, x] = 60
            val = vmin + frac * (vmax - vmin)
            _draw_text(canvas, f"{val:.2g}", x - 8, h + 8, color=(60, 60, 60), scale=1)
        if label:
            _draw_text(canvas, label, w // 2 - len(label) * 3, 6, color=(30, 30, 30), scale=1)

    return save_png(path, canvas)


def save_overlay_figure(
    path: str | Path,
    signal_image: np.ndarray,
    overlays: list[dict[str, Any]],
    *,
    side_panels: list[dict[str, Any]] | None = None,
    cmap: str = "viridis",
    percentiles: tuple[float, float] = (1, 99),
    center_mask_radius: float = 0.0,
    size: tuple[int, int] = (960, 420),
    panel_gap: int = 8,
    title: str = "",
) -> Path:
    """Composite a multi-panel figure: signal + overlays + optional side panels.

    Mimics pyxem's ``OrientationMap.plot_over_signal()`` layout: the main
    diffraction-pattern panel on the left, with optional side panels (e.g.
    correlation heatmap, IPF legend, phase legend) arranged to the right.

    Parameters
    ----------
    path:
        Output PNG path.
    signal_image:
        2D scalar image for the main panel.
    overlays:
        List of overlay dicts.  Each dict has keys:
        - ``positions_yx`` — (M, 2) array of marker centres in *(y, x)* pixels.
        - ``colors`` — (M, 3) uint8 RGB array, one colour per marker.
        - ``marker`` — ``"cross"`` (default) or ``"circle"``.
        - ``radius`` — marker size in pixels (default 3).
    side_panels:
        Optional list of panel dicts placed to the right of the signal.  Each
        dict has:
        - ``image`` — 2D array or (H, W, 3) uint8 RGB for the panel.
        - ``title`` — optional caption below the panel.
        - ``width`` — panel width in pixels.
        - ``cmap`` — colormap for 2D images (default ``"viridis"``).
    cmap:
        Colormap for the signal image.
    percentiles:
        Contrast stretch percentiles for the signal image.
    center_mask_radius:
        If > 0, apply direct-beam masking to the signal image before rendering.
    size:
        Canvas ``(width, height)`` in pixels.
    panel_gap:
        Gap in pixels between the main panel and side panels.
    title:
        Optional title drawn at the top of the figure.
    """
    width, height = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    ml, mr, mt, mb = _TIGHT_MARGIN_L, _TIGHT_MARGIN_R, _TIGHT_MARGIN_T, _TIGHT_MARGIN_B

    # --- Compute panel layout -------------------------------------------------
    n_side = len(side_panels) if side_panels else 0
    side_total_w = 0
    if side_panels:
        for sp in side_panels:
            side_total_w += int(sp.get("width", 128)) + panel_gap
        side_total_w -= panel_gap  # remove trailing gap

    main_w = width - ml - mr - (side_total_w + panel_gap if n_side > 0 else 0)
    main_h = height - mt - mb

    # --- Main panel: signal + overlays ----------------------------------------
    sig = np.asarray(signal_image, dtype=np.float32)
    if center_mask_radius > 0:
        sig = mask_center_for_display(sig, radius_px=center_mask_radius)
    scaled = _scale_to_uint8(sig, percentiles)
    lut = _get_colormap(cmap)
    main_rgb = lut[scaled]
    # Resample to fit.
    sh, sw = main_rgb.shape[:2]
    sy = np.linspace(0, sh - 1, main_h).astype(np.int32)
    sx = np.linspace(0, sw - 1, main_w).astype(np.int32)
    canvas[mt:mt + main_h, ml:ml + main_w] = main_rgb[sy[:, None], sx[None, :]]

    # Draw overlays on the main panel.
    for ov in overlays:
        positions = np.asarray(ov.get("positions_yx", np.zeros((0, 2))), dtype=np.float64)
        colors = np.asarray(ov.get("colors", np.zeros((0, 3))), dtype=np.uint8)
        marker = ov.get("marker", "cross")
        radius = int(ov.get("radius", 3))
        if positions.size == 0 or colors.size == 0:
            continue
        # Scale positions from signal-image coords to main-panel coords.
        sy_scale = main_h / max(float(sh), 1.0)
        sx_scale = main_w / max(float(sw), 1.0)
        for i in range(min(len(positions), len(colors))):
            py = int(positions[i, 0] * sy_scale) + mt
            px = int(positions[i, 1] * sx_scale) + ml
            color = tuple(int(c) for c in colors[i])
            if marker == "circle":
                _draw_circle(canvas, py, px, color, radius=radius)
            else:
                _draw_cross(canvas, py, px, color, radius=radius)

    # --- Side panels ----------------------------------------------------------
    if side_panels:
        sx_cursor = ml + main_w + panel_gap
        for sp in side_panels:
            sp_w = int(sp.get("width", 128))
            sp_img = np.asarray(sp.get("image", np.zeros((1, 1))))
            sp_title = str(sp.get("title", ""))
            sp_cmap = sp.get("cmap", "viridis")

            if sp_img.ndim == 2:
                sp_lut = _get_colormap(sp_cmap)
                sp_scaled = _scale_to_uint8(sp_img, (0, 100))
                sp_rgb = sp_lut[sp_scaled]
            else:
                sp_rgb = sp_img.astype(np.uint8, copy=False) if sp_img.dtype == np.uint8 else _scale_to_uint8(sp_img, (0, 100))

            # Resample to fit.
            sph, spw = sp_rgb.shape[:2]
            target_h = main_h
            target_w = sp_w
            spy = np.linspace(0, sph - 1, target_h).astype(np.int32)
            spx = np.linspace(0, spw - 1, target_w).astype(np.int32)
            canvas[mt:mt + target_h, sx_cursor:sx_cursor + target_w] = sp_rgb[spy[:, None], spx[None, :]]
            # Border.
            canvas[mt:mt + target_h + 1, sx_cursor] = 180
            canvas[mt:mt + target_h + 1, sx_cursor + target_w - 1] = 180
            canvas[mt, sx_cursor:sx_cursor + target_w] = 180
            canvas[mt + target_h - 1, sx_cursor:sx_cursor + target_w] = 180

            if sp_title:
                _draw_text(canvas, sp_title, sx_cursor + target_w // 2 - len(sp_title) * 3, mt + target_h + 6, color=(60, 60, 60), scale=1)

            sx_cursor += target_w + panel_gap

    # --- Title ----------------------------------------------------------------
    if title:
        _draw_text(canvas, title.upper(), width // 2 - len(title) * 3, 6, color=(30, 30, 30), scale=1)

    return save_png(path, canvas)


def _draw_cross(
    canvas: np.ndarray,
    y: float,
    x: float,
    color: tuple[int, int, int],
    *,
    radius: int = 4,
) -> None:
    """Draw a cross marker at *(y, x)* on *canvas*."""
    h, w = canvas.shape[:2]
    cy, cx = int(round(y)), int(round(x))
    if cy < 0 or cy >= h or cx < 0 or cx >= w:
        return
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    canvas[y0:y1, cx] = color
    canvas[cy, x0:x1] = color
    if radius >= 3:
        for d in range(-radius + 1, radius):
            yy, xx = cy + d, cx + d
            if 0 <= yy < h and 0 <= xx < w:
                canvas[yy, xx] = color
            yy, xx = cy + d, cx - d
            if 0 <= yy < h and 0 <= xx < w:
                canvas[yy, xx] = color


def _draw_circle(
    canvas: np.ndarray,
    y: float,
    x: float,
    color: tuple[int, int, int],
    *,
    radius: int = 3,
) -> None:
    """Draw a hollow circle marker at *(y, x)* on *canvas*."""
    h, w = canvas.shape[:2]
    cy, cx = int(round(y)), int(round(x))
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d2 = dy * dy + dx * dx
            if r2 - radius < d2 <= r2:
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < h and 0 <= xx < w:
                    canvas[yy, xx] = color


def save_phase_match_map(
    path: str | Path,
    nav_shape: tuple[int, int],
    roi_entries: list[dict[str, Any]],
    *,
    scale: int = 4,
) -> Path:
    """Render an EBSD-style navigation-space phase map.

    Each navigation pixel inside a matched micro-zone/ROI is coloured by the
    assigned candidate phase. Labels are kept out of the map body and shown
    only in the legend so the image reads like a discrete EBSD phase map.

    Parameters
    ----------
    path:
        Output PNG path.
    nav_shape:
        Binned navigation shape ``(ny, nx)`` (the coordinate system that
        ``stage1_bbox`` values are expressed in).
    roi_entries:
        List of ROI dicts.  Each must contain ``stage1_bbox`` (``[y0, y1, x0, x1]``)
        and ``name``.  Optional Stage 2B fields: ``candidate_phase``,
        ``match_score``, ``phase_confidence``.
    scale:
        Integer scale factor applied to the nav-shape canvas so text is
        legible.  Default 4 → each binned pixel becomes a 4×4 px block.
    """
    return _render_ebsd_phase_map(path, nav_shape, roi_entries, scale=scale)

    palette = _label_palette()
    ny, nx = int(nav_shape[0]), int(nav_shape[1])
    map_h, map_w = ny * scale, nx * scale

    # ── Build phase → colour mapping ──────────────────────────────────
    # Gather unique candidate phases in order of first appearance.
    phase_order: list[str] = []
    seen_phases: set[str] = set()
    for r in roi_entries:
        cp = r.get("candidate_phase")
        if cp and cp not in seen_phases:
            seen_phases.add(cp)
            phase_order.append(cp)

    phase_color: dict[str, np.ndarray] = {}
    for i, ph in enumerate(phase_order):
        phase_color[ph] = palette[i % len(palette)]
    unmatched_color = np.asarray([210, 210, 210], dtype=np.uint8)  # light grey
    fallback_color = np.asarray([180, 180, 180], dtype=np.uint8)

    # ── Legend panel width ────────────────────────────────────────────
    legend_w = 240 if phase_order else 0
    canvas_w = map_w + legend_w
    canvas_h = max(map_h, 40 + 28 * len(phase_order) + 20)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # ── Draw ROI rectangles ───────────────────────────────────────────
    for r in roi_entries:
        bbox = r.get("stage1_bbox")
        if bbox is None:
            continue
        y0, y1, x0, x1 = [int(v) for v in bbox]
        # Clamp to nav shape
        y0 = max(0, min(y0, ny))
        y1 = max(0, min(y1, ny))
        x0 = max(0, min(x0, nx))
        x1 = max(0, min(x1, nx))

        py0, py1 = y0 * scale, y1 * scale
        px0, px1 = x0 * scale, x1 * scale

        # Fill colour by phase
        cp = r.get("candidate_phase")
        if cp and cp in phase_color:
            fill = phase_color[cp]
        elif cp:
            fill = fallback_color
        else:
            fill = unmatched_color

        canvas[py0:py1, px0:px1] = fill

        # Gold border (1 px at canvas scale)
        border_color = np.asarray([255, 215, 0], dtype=np.uint8)
        canvas[py0:py1, px0] = border_color
        canvas[py0:py1, max(px1 - 1, px0)] = border_color
        canvas[py0, px0:px1] = border_color
        canvas[max(py1 - 1, py0), px0:px1] = border_color

    # ── Draw ROI labels ───────────────────────────────────────────────
    for r in roi_entries:
        bbox = r.get("stage1_bbox")
        if bbox is None:
            continue
        y0, _, x0, _ = [int(v) for v in bbox]
        tx = x0 * scale + 4
        ty = y0 * scale + 4

        # Phase name + confidence
        cp = r.get("candidate_phase")
        pc = r.get("phase_confidence", "")
        if cp:
            label = cp
            if pc and pc != "not_scored":
                label = f"{cp} ({pc})"
        else:
            label = r.get("name", "?")
        _draw_text(canvas, label, tx, ty, color=(20, 20, 20), scale=1)

        # Match score on second line if available
        ms = r.get("match_score")
        if ms is not None:
            score_text = f"{ms:.3f}"
            _draw_text(canvas, score_text, tx, ty + 12, color=(80, 80, 80), scale=1)

    # ── Legend ────────────────────────────────────────────────────────
    if phase_order:
        lx = map_w + 16
        _draw_text(canvas, "PHASE MAP", lx, 14, color=(30, 30, 30), scale=2)
        for row, ph in enumerate(phase_order):
            ly = 48 + row * 28
            color = phase_color[ph]
            canvas[ly:ly + 14, lx:lx + 18] = color
            _draw_text(canvas, ph, lx + 28, ly + 2, color=(30, 30, 30), scale=1)

    return save_png(path, canvas)


def save_cluster_phase_map(
    path: str | Path,
    labels: np.ndarray,
    cluster_phase_entries: list[dict[str, Any]],
    *,
    scale: int = 1,
    legend_path: str | Path | None = None,
    contrast_image: np.ndarray | None = None,
    phase_alpha: float = 0.62,
) -> Path:
    """Render a dense EBSD-style phase map by replacing cluster labels with phases.

    ``labels`` is the Stage 1 navigation-space fingerprint-class map. Each
    entry in ``cluster_phase_entries`` must contain ``cluster_id`` and
    ``candidate_phase``; optional ``match_score`` is used when more than one
    ROI matched the same cluster.
    """
    label_arr = np.asarray(labels)
    if label_arr.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {label_arr.shape!r}.")
    scale = max(1, int(scale))

    best_by_cluster: dict[int, dict[str, Any]] = {}
    for entry in cluster_phase_entries:
        if entry.get("cluster_id") is None or not entry.get("candidate_phase"):
            continue
        cluster_id = int(entry["cluster_id"])
        score = entry.get("match_score")
        score_value = float(score) if score is not None else -np.inf
        previous = best_by_cluster.get(cluster_id)
        previous_score = (
            float(previous["match_score"])
            if previous is not None and previous.get("match_score") is not None
            else -np.inf
        )
        if previous is None or score_value >= previous_score:
            best_by_cluster[cluster_id] = entry

    phase_order: list[str] = []
    seen_phases: set[str] = set()
    for entry in best_by_cluster.values():
        phase = str(entry["candidate_phase"])
        if phase not in seen_phases:
            seen_phases.add(phase)
            phase_order.append(phase)

    palette = _phase_palette()
    phase_color = {
        phase: palette[i % len(palette)]
        for i, phase in enumerate(phase_order)
    }

    unmatched_color = np.asarray([245, 245, 245], dtype=np.uint8)
    phase_index = np.full(label_arr.shape, -1, dtype=np.int16)
    phase_to_index = {phase: i for i, phase in enumerate(phase_order)}

    for cluster_id, entry in best_by_cluster.items():
        phase = str(entry["candidate_phase"])
        phase_index[label_arr == cluster_id] = phase_to_index[phase]

    if contrast_image is not None:
        contrast = np.asarray(contrast_image, dtype=np.float32)
        if contrast.shape != label_arr.shape:
            raise ValueError(
                f"contrast_image shape {contrast.shape!r} does not match labels shape {label_arr.shape!r}."
            )
        gray = _scale_to_uint8(contrast, (1, 99))
        nav_rgb = np.repeat(gray[..., None], 3, axis=-1)
    else:
        nav_rgb = np.full(label_arr.shape + (3,), unmatched_color, dtype=np.uint8)

    alpha = float(np.clip(phase_alpha, 0.0, 1.0))
    for phase, idx in phase_to_index.items():
        mask = phase_index == idx
        if contrast_image is None:
            nav_rgb[mask] = phase_color[phase]
        else:
            color = phase_color[phase].astype(np.float32)
            base = nav_rgb[mask].astype(np.float32)
            nav_rgb[mask] = np.clip((1.0 - alpha) * base + alpha * color, 0, 255).astype(np.uint8)

    assigned = phase_index >= 0
    boundary = np.zeros(label_arr.shape, dtype=bool)
    if label_arr.shape[0] > 1:
        diff_y = assigned[1:, :] & assigned[:-1, :] & (phase_index[1:, :] != phase_index[:-1, :])
        boundary[1:, :] |= diff_y
        boundary[:-1, :] |= diff_y
    if label_arr.shape[1] > 1:
        diff_x = assigned[:, 1:] & assigned[:, :-1] & (phase_index[:, 1:] != phase_index[:, :-1])
        boundary[:, 1:] |= diff_x
        boundary[:, :-1] |= diff_x
    if np.any(boundary):
        nav_rgb[boundary] = np.clip(nav_rgb[boundary].astype(np.float32) * 0.35, 0, 255).astype(np.uint8)

    map_rgb = np.repeat(np.repeat(nav_rgb, scale, axis=0), scale, axis=1)

    if legend_path is not None:
        save_phase_legend_png(legend_path, phase_order, phase_color, best_by_cluster, phase_index)

    return save_png(path, map_rgb)


def save_phase_legend_png(
    path: str | Path,
    phase_order: list[str],
    phase_color: dict[str, np.ndarray],
    best_by_cluster: dict[int, dict[str, Any]],
    phase_index: np.ndarray,
) -> Path:
    """Save the legend for a dense phase map as a separate PNG."""
    width = 360
    height = max(96, 44 + 42 * max(len(phase_order), 1) + 20)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_text(canvas, "PHASE MAP", 16, 14, color=(30, 30, 30), scale=2)
    for row, phase in enumerate(phase_order):
        y = 52 + row * 42
        canvas[y:y + 16, 16:38] = phase_color[phase]
        _draw_text(canvas, phase, 48, y + 3, color=(30, 30, 30), scale=1)
        clusters = [
            str(cluster_id)
            for cluster_id, entry in sorted(best_by_cluster.items())
            if str(entry["candidate_phase"]) == phase
        ]
        idx = phase_order.index(phase)
        pct = 100.0 * float(np.sum(phase_index == idx)) / max(int(np.sum(phase_index >= 0)), 1)
        _draw_text(canvas, f"{pct:.1f}%  C {','.join(clusters)}", 48, y + 18, color=(90, 90, 90), scale=1)
    return save_png(path, canvas)


def _render_ebsd_phase_map(
    path: str | Path,
    nav_shape: tuple[int, int],
    roi_entries: list[dict[str, Any]],
    *,
    scale: int,
) -> Path:
    ny, nx = int(nav_shape[0]), int(nav_shape[1])
    if ny <= 0 or nx <= 0:
        raise ValueError(f"nav_shape must be positive, got {nav_shape!r}.")
    scale = max(1, int(scale))

    palette = _phase_palette()
    phase_order: list[str] = []
    seen_phases: set[str] = set()
    for r in roi_entries:
        phase = r.get("candidate_phase")
        if phase and phase not in seen_phases:
            seen_phases.add(str(phase))
            phase_order.append(str(phase))

    phase_color = {
        phase: palette[i % len(palette)]
        for i, phase in enumerate(phase_order)
    }
    phase_to_index = {phase: i for i, phase in enumerate(phase_order)}

    phase_index = np.full((ny, nx), -1, dtype=np.int16)
    score_map = np.full((ny, nx), -np.inf, dtype=np.float32)

    for r in roi_entries:
        phase = r.get("candidate_phase")
        if not phase:
            continue
        phase = str(phase)
        if phase not in phase_to_index:
            continue
        bbox = r.get("stage1_bbox")
        if bbox is None:
            continue
        y0, y1, x0, x1 = [int(v) for v in bbox]
        y0 = max(0, min(y0, ny))
        y1 = max(0, min(y1, ny))
        x0 = max(0, min(x0, nx))
        x1 = max(0, min(x1, nx))
        if y1 <= y0 or x1 <= x0:
            continue

        raw_score = r.get("match_score")
        score = float(raw_score) if raw_score is not None else 0.0
        current = score_map[y0:y1, x0:x1]
        update = score >= current
        phase_index[y0:y1, x0:x1] = np.where(
            update,
            phase_to_index[phase],
            phase_index[y0:y1, x0:x1],
        )
        score_map[y0:y1, x0:x1] = np.where(update, score, current)

    unmatched_color = np.asarray([245, 245, 245], dtype=np.uint8)
    boundary_color = np.asarray([35, 35, 35], dtype=np.uint8)
    nav_rgb = np.full((ny, nx, 3), unmatched_color, dtype=np.uint8)
    for phase, idx in phase_to_index.items():
        nav_rgb[phase_index == idx] = phase_color[phase]

    assigned = phase_index >= 0
    boundary = np.zeros((ny, nx), dtype=bool)
    if ny > 1:
        diff_y = assigned[1:, :] & assigned[:-1, :] & (phase_index[1:, :] != phase_index[:-1, :])
        boundary[1:, :] |= diff_y
        boundary[:-1, :] |= diff_y
    if nx > 1:
        diff_x = assigned[:, 1:] & assigned[:, :-1] & (phase_index[:, 1:] != phase_index[:, :-1])
        boundary[:, 1:] |= diff_x
        boundary[:, :-1] |= diff_x
    nav_rgb[boundary] = boundary_color

    map_rgb = np.repeat(np.repeat(nav_rgb, scale, axis=0), scale, axis=1)
    map_h, map_w = map_rgb.shape[:2]
    legend_w = 260 if phase_order else 0
    canvas_h = max(map_h, 44 + 30 * max(len(phase_order), 1) + 20)
    canvas = np.full((canvas_h, map_w + legend_w, 3), 255, dtype=np.uint8)
    canvas[:map_h, :map_w] = map_rgb

    if phase_order:
        lx = map_w + 16
        _draw_text(canvas, "PHASE MAP", lx, 14, color=(30, 30, 30), scale=2)
        for row, phase in enumerate(phase_order):
            ly = 52 + row * 30
            canvas[ly:ly + 16, lx:lx + 22] = phase_color[phase]
            _draw_text(canvas, phase, lx + 32, ly + 3, color=(30, 30, 30), scale=1)

    return save_png(path, canvas)


def _phase_palette() -> np.ndarray:
    """High-contrast discrete colours for EBSD-style phase maps."""
    return np.asarray(
        [
            [46, 117, 182],
            [214, 73, 51],
            [76, 153, 77],
            [128, 88, 177],
            [230, 159, 0],
            [0, 158, 115],
            [86, 180, 233],
            [204, 121, 167],
            [110, 110, 110],
        ],
        dtype=np.uint8,
    )


def _label_palette() -> np.ndarray:
    return np.asarray(
        [
            [45, 105, 170],
            [220, 95, 60],
            [80, 155, 90],
            [180, 120, 190],
            [230, 180, 65],
            [70, 170, 180],
            [150, 150, 150],
            [195, 70, 120],
        ],
        dtype=np.uint8,
    )


def _scale_to_uint8(image: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, percentiles)
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    scaled = (arr - lo) / max(hi - lo, 1e-12)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def _normalize_to_span(values: np.ndarray, low: int, high: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    span = max(high - low, 1)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if vmax <= vmin:
        return np.full(vals.shape, (low + high) / 2, dtype=np.float32)
    return low + (vals - vmin) / (vmax - vmin) * span


def _draw_polyline(canvas: np.ndarray, points: np.ndarray, *, color: tuple[int, int, int]) -> None:
    for start, end in zip(points[:-1], points[1:]):
        x0, y0 = start
        x1, y1 = end
        steps = int(max(abs(x1 - x0), abs(y1 - y0), 1))
        xs = np.linspace(x0, x1, steps + 1).astype(int)
        ys = np.linspace(y0, y1, steps + 1).astype(int)
        valid = (xs >= 0) & (xs < canvas.shape[1]) & (ys >= 0) & (ys < canvas.shape[0])
        canvas[ys[valid], xs[valid]] = color


def _draw_text(canvas: np.ndarray, text: str, x: int, y: int, *, color: tuple[int, int, int], scale: int = 1) -> None:
    cursor = int(x)
    for char in text:
        glyph = _FONT_5X7.get(char.upper(), _FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    y0 = y + gy * scale
                    x0 = cursor + gx * scale
                    y1 = y0 + scale
                    x1 = x0 + scale
                    if y1 <= 0 or x1 <= 0 or y0 >= canvas.shape[0] or x0 >= canvas.shape[1]:
                        continue
                    canvas[max(y0, 0) : min(y1, canvas.shape[0]), max(x0, 0) : min(x1, canvas.shape[1])] = color
        cursor += 6 * scale


def _draw_text_vertical(
    canvas: np.ndarray, text: str, x: int, y: int, *, color: tuple[int, int, int], scale: int = 1,
) -> None:
    """Draw text rotated 90° counter-clockwise (bottom-to-top)."""
    cursor = int(y)
    for char in text:
        glyph = _FONT_5X7.get(char.upper(), _FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    # Rotate: (gx, gy) → (-gy, gx) in glyph space
                    x0 = x - gy * scale
                    y0 = cursor + gx * scale
                    if x0 < 0 or y0 < 0 or x0 >= canvas.shape[1] or y0 >= canvas.shape[0]:
                        continue
                    x1 = min(x0 + scale, canvas.shape[1])
                    y1 = min(y0 + scale, canvas.shape[0])
                    canvas[max(y0, 0):y1, max(x0, 0):x1] = color
        cursor += 6 * scale


def mask_center_for_display(
    image: np.ndarray,
    center_yx: tuple[float, float] | None = None,
    radius_px: float = 35.0,
    *,
    outer_radius_px: float | None = None,
) -> np.ndarray:
    """Zero out the direct-beam disk (and optionally the outer region) for display.

    Mimics pyxem's ``get_direct_beam_mask(radius)`` applied before
    ``plot_images`` — creates a "donut" that focuses attention on the
    diffraction ring where Bragg peaks actually reside.

    Parameters
    ----------
    image:
        2D scalar diffraction pattern.
    center_yx:
        Beam centre in *(y, x)* pixel coordinates.  Defaults to the
        geometric centre of the image.
    radius_px:
        Radius in pixels of the central mask (default 35, matching pyxem's
        typical direct-beam exclusion).
    outer_radius_px:
        If set, also zero out pixels beyond this radius.

    Returns
    -------
    np.ndarray
        A copy of *image* with the specified region(s) set to zero.
    """
    arr = np.asarray(image, dtype=np.float32).copy()
    h, w = arr.shape
    cy, cx = center_yx if center_yx is not None else ((h - 1) / 2.0, (w - 1) / 2.0)
    yy, xx = np.indices(arr.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if radius_px > 0:
        arr[r < radius_px] = 0.0
    if outer_radius_px is not None and outer_radius_px > 0:
        arr[r > outer_radius_px] = 0.0
    return arr


def polar_reproject(
    image: np.ndarray,
    *,
    npt: int = 100,
    npt_azim: int = 360,
    center_yx: tuple[float, float] | None = None,
    mean: bool = True,
) -> np.ndarray:
    """Reproject a 2D Cartesian diffraction pattern to polar coordinates.

    Pure-numpy equivalent of pyxem's
    ``Diffraction2D.get_azimuthal_integral2d(npt, npt_azim, mean=True)``.
    Uses fast ``np.bincount`` binning rather than interpolation, so it
    handles masked/zeroed pixels correctly.

    Parameters
    ----------
    image:
        2D scalar diffraction pattern.
    npt:
        Number of radial bins (default 100).
    npt_azim:
        Number of azimuthal bins (default 360, one per degree).
    center_yx:
        Beam centre in *(y, x)* pixels.  Defaults to the geometric centre.
    mean:
        If True, return the *mean* pixel value in each bin (matching pyxem).
        If False, return the sum.

    Returns
    -------
    np.ndarray
        ``(npt_azim, npt)`` float32 array.  Each row is an azimuthal slice
        at constant angle; each column is a radial bin.
    """
    arr = np.asarray(image, dtype=np.float64)
    h, w = arr.shape
    cy, cx = center_yx if center_yx is not None else ((h - 1) / 2.0, (w - 1) / 2.0)

    yy, xx = np.indices(arr.shape)
    radii = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    angles = np.arctan2(yy - cy, xx - cx)  # [-π, π]
    angles = np.where(angles < 0, angles + 2 * np.pi, angles)  # [0, 2π]

    max_radius = float(np.max(radii))
    if max_radius <= 0:
        max_radius = 1.0

    # Bin indices: radial [0, npt-1], azimuthal [0, npt_azim-1].
    r_idx = np.clip((radii / max_radius * (npt - 1)).astype(np.int32), 0, npt - 1)
    a_idx = np.clip((angles / (2 * np.pi) * (npt_azim - 1)).astype(np.int32), 0, npt_azim - 1)
    flat_idx = a_idx * npt + r_idx  # (npt_azim, npt) in row-major

    n_bins = npt_azim * npt
    sums = np.bincount(flat_idx.ravel(), weights=np.nan_to_num(arr, nan=0.0).ravel(), minlength=n_bins)
    counts = np.bincount(flat_idx.ravel(), minlength=n_bins)

    polar = (sums / np.maximum(counts, 1)).astype(np.float32) if mean else sums.astype(np.float32)
    return polar.reshape(npt_azim, npt)


def apply_gamma(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Apply power-law gamma correction to a diffraction pattern.

    ``gamma < 1`` boosts weak high-angle features at the expense of the
    bright direct beam — matching pyxem's ``polar_multi**0.5`` pattern.
    ``gamma > 1`` compresses dynamic range.
    ``gamma == 1`` is a no-op (returns a copy).

    Handles negative or zero values safely by working on the absolute value
    and preserving sign.
    """
    arr = np.asarray(image, dtype=np.float32)
    if gamma == 1.0:
        return arr.copy()
    # Clip negative values to zero (physical diffraction intensities are >= 0).
    pos = np.maximum(arr, 0.0)
    return np.where(arr >= 0, pos ** gamma, arr).astype(np.float32)


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("PNG writer expects an RGB uint8 array.")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    chunks = [
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw, level=6)),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _render_report(summary: dict[str, Any], phase_labels: np.ndarray, *, base_dir: Path) -> str:
    project = summary.get("project", {})
    data = summary.get("data_config", {})
    dataset = summary.get("dataset", {})
    outputs = summary.get("outputs", {})
    png = outputs.get("png", {}) if isinstance(outputs.get("png"), dict) else {}
    shapes = summary.get("shapes", {})
    preprocess = summary.get("preprocess", "see workflow_summary.json")
    title = str(project.get("name", "4D-STEM Diffraction-Class Screening Report"))

    data_contract = summary.get("data_contract", {})
    provenance = summary.get("provenance", {})
    qc = summary.get("qc", {})

    qc_status = qc.get("stage1_status", "UNKNOWN")
    qc_label = {"PASS": "[PASS]", "PASS_WITH_WARNINGS": "[WARN]", "FAIL": "[FAIL]"}.get(qc_status, "[N/A]")

    lines = [
        f"# {title}",
        "",
        f"## Stage 1 QC Status: {qc_label}",
        "",
        f"- Warnings: {qc.get('n_warnings', '?')}",
        f"- Critical: {qc.get('n_critical', '?')}",
        "",
    ]

    # Inline QC flag table
    qc_flags = qc.get("flags", [])
    if qc_flags:
        lines.append("| Severity | Code | Message |")
        lines.append("| --- | --- | --- |")
        for flag in qc_flags:
            sev = flag.get("severity", "info").upper()
            lines.append(f"| `{sev}` | `{flag.get('code', '?')}` | {flag.get('message', '')} |")
        lines.append("")

    lines.extend([
        "> **Caution:** The fingerprint-class labels are generated by unsupervised "
        "clustering of radial-fingerprint profiles (PCA/NMF + KMeans). They "
        "represent **radial-profile similarity groups**, not confirmed "
        "crystallographic phases. Treat them as **phase candidates** until "
        "validated by Stage-2 py4DSTEM / pyxem Bragg indexing or template matching.",
        "",
        "## Data",
        "",
        f"- Source: `{data.get('path', dataset.get('path', 'unknown'))}`",
        f"- Backend: `{dataset.get('source_backend', 'unknown')}`",
        f"- Preprocessed shape: `{dataset.get('shape', 'unknown')}`",
        f"- Navigation shape: `{dataset.get('navigation_shape', 'unknown')}`",
        f"- Signal shape: `{dataset.get('signal_shape', 'unknown')}`",
        "",
        "## Coordinate Conventions (Data Contract)",
        "",
        f"- Axis order: `{data_contract.get('axis_order', 'nav_y_nav_x_q_y_q_x')}`",
        f"- BBox order: `{data_contract.get('bbox_order', 'y0_y1_x0_x1')}`",
        f"- Centre order: `{data_contract.get('center_order', 'y_x')}`",
        "",
        "## Provenance",
        "",
        f"- Pipeline version: `{provenance.get('pipeline_version', 'unknown')}`",
        f"- Git commit: `{provenance.get('git_commit', 'unknown')}`",
        f"- Config path: `{provenance.get('config_path', 'inline')}`",
        f"- Config hash: `{provenance.get('config_hash', 'n/a')}`",
        f"- Input path: `{provenance.get('input_path', 'synthetic')}`",
        f"- Input file size: {_human_size(provenance.get('input_file_size'))}",
        f"- Input file mtime: `{provenance.get('input_file_mtime', 'n/a')}`",
        f"- Python version: `{provenance.get('python_version', 'unknown')}`",
        f"- Platform: `{provenance.get('platform', 'unknown')}`",
        f"- Random seed: `{provenance.get('random_seed', 'none')}`",
        f"- Start time: `{provenance.get('start_time', 'unknown')}`",
        f"- End time: `{provenance.get('end_time', 'unknown')}`",
        "",
        _render_package_versions(
            provenance.get("packages", {}),
            summary.get("dependencies"),
        ),
        "",
        "## Screening Settings",
        "",
        f"- q-crop/q-bin/r-bin: `{preprocess}`",
        f"- Virtual image shapes: `{shapes.get('virtual_images', 'unknown')}`",
        f"- Radial fingerprints: `{shapes.get('radial_fingerprints', 'unknown')}`",
        f"- Fingerprint-class labels: `{shapes.get('fingerprint_class_labels', shapes.get('phase_labels', 'unknown'))}`",
        f"- Orientation preview: `{shapes.get('orientation_index', 'unknown')}`",
        "",
        "## Interpretation Level",
        "",
        "Level A: Fingerprint class. Unsupervised radial-fingerprint class only. Not a crystallographic phase assignment.",
        "",
        "Level B: Phase candidate. Cluster average DP / radial peaks are consistent with a candidate phase. Requires indexing validation.",
        "",
        "Level C: Confirmed phase. Validated by py4DSTEM / pyxem template matching or Bragg indexing using candidate CIFs.",
        "",
        "Current Stage 1 result should be treated as Level A unless later indexing validation is added.",
        "",
        "## Unsupervised Fingerprint Classes (Phase Candidates)",
        "",
        "| Cluster | Pixels | Fraction |",
        "| --- | ---: | ---: |",
    ])

    labels, counts = np.unique(np.asarray(phase_labels), return_counts=True)
    total = max(int(counts.sum()), 1)
    for label, count in zip(labels, counts):
        lines.append(f"| {int(label)} | {int(count)} | {int(count) / total:.3f} |")

    # Sample mask statistics
    sample_mask = summary.get("sample_mask", {})
    if sample_mask and sample_mask.get("generated"):
        lines.extend([
            "## Sample Mask",
            "",
            f"- Source: `{sample_mask.get('source', 'adf')}`",
            f"- Sample pixels: `{sample_mask.get('sample_pixels', '?')}`",
            f"- Background pixels: `{sample_mask.get('background_pixels', '?')}`",
            f"- Sample fraction: `{sample_mask.get('sample_fraction', '?')}`",
            "",
            "Pixels outside the sample mask are marked as background "
            "(label `-1`) and excluded from fingerprint-class clustering.",
            "",
        ])

    lines.extend(["", "## Key PNG Outputs", ""])
    for key in [
        "fingerprint_class_labels_annotated",
        "cluster_cleaned_labels",
        "cluster_mean_radial_profiles",
        "cluster_virtual_image_statistics",
        "cluster_vs_orientation_heatmap",
        "ring_2_over_ring_1",
        "ring_3_over_ring_1",
        "ring_3_over_ring_2",
        "roi_candidates_overlay",
        "sample_mask",
        "sample_mask_overlay_adf",
        "virtual_bf",
        "virtual_adf",
        "virtual_haadf",
        "com_x",
        "com_y",
        "mean_radial_profile",
        "orientation_index",
        "orientation_score",
    ]:
        if key in png:
            png_path = Path(png[key])
            try:
                link = png_path.relative_to(base_dir).as_posix()
            except ValueError:
                link = png_path.as_posix()
            lines.append(f"- {key}: [{png_path.name}]({link})")

    diagnostics = outputs.get("diagnostics", {}) if isinstance(outputs.get("diagnostics"), dict) else {}
    lines.extend(["", "## Cluster Diagnostics", ""])
    for title_text, path_text in [
        ("Cluster virtual-image statistics", diagnostics.get("cluster_summary_csv")),
        ("Connected-component cleanup", (diagnostics.get("connected_components") or {}).get("connected_components_csv") if isinstance(diagnostics.get("connected_components"), dict) else None),
        ("Cluster vs orientation", (diagnostics.get("cluster_vs_orientation") or {}).get("csv") if isinstance(diagnostics.get("cluster_vs_orientation"), dict) else None),
        ("K-sweep metrics", (diagnostics.get("k_sweep") or {}).get("metrics_csv") if isinstance(diagnostics.get("k_sweep"), dict) else None),
    ]:
        table = _csv_to_markdown(path_text, base_dir=base_dir, max_rows=12)
        if table:
            lines.extend([f"### {title_text}", "", table, ""])

    roi_csv = (diagnostics.get("roi_outputs") or {}).get("csv") if isinstance(diagnostics.get("roi_outputs"), dict) else None
    roi_table = _csv_to_markdown(roi_csv, base_dir=base_dir, max_rows=12)
    if roi_table:
        lines.extend(["## Stage 2 ROI Candidates", "", roi_table, ""])

    ring_maps = diagnostics.get("ring_ratio_maps") if isinstance(diagnostics.get("ring_ratio_maps"), dict) else {}
    if ring_maps:
        lines.extend(["## Ring Ratio Map Arrays", ""])
        for name, path_text in ring_maps.items():
            lines.append(f"- {name}: `{path_text}`")
        lines.append("")

    lines.extend(
        [
            "",
            "## Output Directories",
            "",
            f"- Virtual images: `{outputs.get('virtual', 'not generated')}`",
            f"- Fingerprints: `{outputs.get('fingerprints', 'not generated')}`",
            f"- Fingerprint classes: `{outputs.get('fingerprint_classes', outputs.get('phase_screening', 'not generated'))}`",
            f"- Cluster diagnostics: `{outputs.get('cluster_diagnostics', 'not generated')}`",
            f"- ROI candidates: `{outputs.get('roi_candidates', 'not generated')}`",
            f"- Orientation: `{outputs.get('orientation', 'not generated')}`",
            f"- PNG previews: `{Path(next(iter(png.values()))).parent.as_posix() if png else 'not generated'}`",
            "",
            "## Interpretation Notes",
            "",
            "- **Fingerprint-class labels are unsupervised radial-profile clusters, not final crystallographic phase maps.** Formal phase assignment requires Stage-2 validation.",
            "- If clusters strongly follow HAADF / ADF intensity, they may reflect thickness, Z-contrast, or mass-thickness effects — not phase.",
            "- If clusters strongly follow COM-x / COM-y gradients, they may reflect bend, diffraction shift, or beam-center drift — not phase.",
            "- If clusters mainly differ in ring ratios and radial peak positions, they are more plausible **phase candidates**, but still require indexing validation.",
            "- **Stage-2 workflow**: Select ROI candidates from the most promising fingerprint classes, then run py4DSTEM / pyxem template matching or Bragg indexing against candidate CIFs for confirmed phase mapping.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_html_report(summary: dict[str, Any], phase_labels: np.ndarray, *, base_dir: Path) -> str:
    project = summary.get("project", {})
    outputs = summary.get("outputs", {})
    png = outputs.get("png", {}) if isinstance(outputs.get("png"), dict) else {}
    title = str(project.get("name", "4D-STEM Diffraction-Class Screening Report"))
    body = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}",
        "h1,h2{margin-top:1.4em} table{border-collapse:collapse;margin:12px 0;max-width:100%}",
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:right} th:first-child,td:first-child{text-align:left}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}",
        "figure{margin:0} img{max-width:100%;border:1px solid #ddd} figcaption{font-size:13px;color:#555;margin-top:4px}",
        "code{background:#f5f5f5;padding:1px 4px}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    body.extend(_html_qc_banner(summary.get("qc", {})))
    body.extend(_html_data_summary(summary))
    body.extend(_html_label_summary(phase_labels))
    body.extend(_html_sample_mask(summary.get("sample_mask", {})))
    body.extend(_html_provenance(summary.get("provenance", {}), summary.get("dependencies")))
    body.extend(_html_png_grid(png, base_dir))
    body.extend(_html_diagnostic_tables(outputs.get("diagnostics", {}), base_dir))
    body.extend(
        [
            "<h2>Interpretation Notes</h2>",
            "<ul>",
            "<li><strong>Fingerprint-class labels are unsupervised radial-profile clusters, not crystallographic phase assignments.</strong> Treat them as phase candidates.</li>",
            "<li>Use cluster mean DPs, radial profiles, virtual-image statistics, ring ratios, and orientation consistency to prioritize Stage-2 ROI candidates.</li>",
            "<li><strong>Stage-2:</strong> Select promising ROI candidates, then run py4DSTEM/pyxem Bragg indexing or template matching against candidate CIFs for confirmed phase mapping.</li>",
            "</ul>",
            "</body></html>",
        ]
    )
    return "\n".join(body)


def _html_qc_banner(qc: dict[str, Any]) -> list[str]:
    """Render the QC status banner and caution at the top of the report."""
    if not qc:
        return []
    status = qc.get("stage1_status", "UNKNOWN")
    label = {"PASS": "[PASS]", "PASS_WITH_WARNINGS": "[WARN]", "FAIL": "[FAIL]"}.get(status, "[N/A]")
    bg = {"PASS": "#d4edda", "PASS_WITH_WARNINGS": "#fff3cd", "FAIL": "#f8d7da"}.get(status, "#e2e3e5")
    border = {"PASS": "#28a745", "PASS_WITH_WARNINGS": "#ffc107", "FAIL": "#dc3545"}.get(status, "#6c757d")

    lines = [
        f"<div style=\"background:{bg};border-left:6px solid {border};padding:14px 20px;margin:16px 0;border-radius:4px\">",
        f"<h2 style=\"margin:0 0 6px 0\">Stage 1 QC Status: {label}</h2>",
        f"<p style=\"margin:0\">Warnings: {qc.get('n_warnings', '?')} &nbsp;|&nbsp; Critical: {qc.get('n_critical', '?')}</p>",
    ]

    qc_flags = qc.get("flags", [])
    if qc_flags:
        lines.append("<details style=\"margin-top:10px\"><summary><strong>QC Flags</strong></summary>")
        lines.append("<table style=\"margin-top:8px\"><thead><tr><th>Severity</th><th>Code</th><th>Message</th></tr></thead><tbody>")
        for flag in qc_flags:
            sev = flag.get("severity", "info").upper()
            lines.append(
                f"<tr><td><code>{html.escape(sev)}</code></td>"
                f"<td><code>{html.escape(flag.get('code', '?'))}</code></td>"
                f"<td>{html.escape(flag.get('message', ''))}</td></tr>"
            )
        lines.append("</tbody></table></details>")

    lines.append("</div>")

    lines.extend([
        "<blockquote style=\"border-left:4px solid #ffc107;padding:10px 16px;margin:12px 0;background:#fff8e1\">",
        "<strong>[WARN] Caution:</strong> The fingerprint-class labels are generated by unsupervised "
        "clustering of radial-fingerprint profiles (PCA/NMF + KMeans). They represent "
        "<strong>radial-profile similarity groups</strong>, not confirmed crystallographic "
        "phases. Treat them as <strong>phase candidates</strong> until validated by "
        "Stage-2 py4DSTEM / pyxem Bragg indexing or template matching.",
        "</blockquote>",
    ])
    return lines


def _html_data_summary(summary: dict[str, Any]) -> list[str]:
    data = summary.get("data_config", {})
    dataset = summary.get("dataset", {})
    shapes = summary.get("shapes", {})
    data_contract = summary.get("data_contract", {})
    provenance = summary.get("provenance", {})
    rows = [
        ("Source", data.get("path", dataset.get("path", "unknown"))),
        ("Backend", dataset.get("source_backend", "unknown")),
        ("Preprocessed shape", dataset.get("shape", "unknown")),
        ("Navigation shape", dataset.get("navigation_shape", "unknown")),
        ("Signal shape", dataset.get("signal_shape", "unknown")),
        ("Radial fingerprints", shapes.get("radial_fingerprints", "unknown")),
        ("Fingerprint labels", shapes.get("fingerprint_class_labels", "unknown")),
        ("Orientation preview", shapes.get("orientation_index", "unknown")),
        ("Axis order", data_contract.get("axis_order", "nav_y_nav_x_q_y_q_x")),
        ("BBox order", data_contract.get("bbox_order", "y0_y1_x0_x1")),
        ("Centre order", data_contract.get("center_order", "y_x")),
        ("Pipeline version", provenance.get("pipeline_version", "unknown")),
        ("Git commit", provenance.get("git_commit", "unknown")),
        ("Config hash", provenance.get("config_hash", "n/a")),
    ]
    lines = ["<h2>Data And Settings</h2>", "<table><tbody>"]
    for key, value in rows:
        lines.append(f"<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(str(value))}</code></td></tr>")
    lines.append("</tbody></table>")
    return lines


def _html_label_summary(phase_labels: np.ndarray) -> list[str]:
    labels, counts = np.unique(np.asarray(phase_labels), return_counts=True)
    total = max(int(counts.sum()), 1)
    lines = ["<h2>Fingerprint Classes (Phase Candidates)</h2>", "<table><thead><tr><th>Cluster</th><th>Pixels</th><th>Fraction</th></tr></thead><tbody>"]
    for label, count in zip(labels, counts):
        lines.append(f"<tr><td>{int(label)}</td><td>{int(count)}</td><td>{int(count) / total:.3f}</td></tr>")
    lines.append("</tbody></table>")
    return lines


def _html_sample_mask(sample_mask: dict[str, Any]) -> list[str]:
    """Render sample-mask statistics as an HTML section."""
    if not sample_mask or not sample_mask.get("generated"):
        return []
    lines = [
        "<h2>Sample Mask</h2>",
        "<table><tbody>",
        f"<tr><th>Sample pixels</th><td><code>{sample_mask.get('sample_pixels', '?')}</code></td></tr>",
        f"<tr><th>Background pixels</th><td><code>{sample_mask.get('background_pixels', '?')}</code></td></tr>",
        f"<tr><th>Sample fraction</th><td><code>{sample_mask.get('sample_fraction', '?')}</code></td></tr>",
        "</tbody></table>",
        "<p>Pixels outside the sample mask are marked as background "
        "(label <code>-1</code>) and excluded from fingerprint-class "
        "clustering.</p>",
    ]
    return lines


def _html_provenance(provenance: dict[str, Any], dependencies: dict[str, Any] | None = None) -> list[str]:
    """Render provenance as an HTML section, including dependency availability."""
    if not provenance:
        return []
    rows = [
        ("Pipeline version", provenance.get("pipeline_version")),
        ("Git commit", provenance.get("git_commit")),
        ("Config path", provenance.get("config_path")),
        ("Config hash", provenance.get("config_hash")),
        ("Input path", provenance.get("input_path")),
        ("Input file size", _human_size(provenance.get("input_file_size"))),
        ("Input file mtime", provenance.get("input_file_mtime")),
        ("Python version", provenance.get("python_version")),
        ("Platform", provenance.get("platform")),
        ("Random seed", provenance.get("random_seed")),
        ("Start time", provenance.get("start_time")),
        ("End time", provenance.get("end_time")),
    ]
    packages = provenance.get("packages", {})
    lines = ["<h2>Provenance</h2>", "<table><tbody>"]
    for key, value in rows:
        lines.append(
            f"<tr><th>{html.escape(str(key))}</th>"
            f"<td><code>{html.escape(str(value) if value is not None else 'n/a')}</code></td></tr>"
        )
    lines.append("</tbody></table>")
    if packages:
        lines.append("<h3>Runtime Dependency Availability</h3>")
        lines.append("<table><tbody>")
        for name in sorted(packages):
            version = packages.get(name)
            label = str(version) if version is not None else "not installed"
            lines.append(
                f"<tr><th>{html.escape(name)}</th>"
                f"<td><code>{html.escape(label)}</code></td></tr>"
            )
        lines.append("</tbody></table>")
        # Runtime usage notes
        if dependencies:
            notes: list[str] = []
            if not dependencies.get("pyxem_available", False) and packages.get("pyxem") is not None:
                notes.append("pyxem is installed but was not used by this run.")
            if not dependencies.get("py4DSTEM_used", False) and packages.get("py4DSTEM") is not None:
                notes.append("py4DSTEM is installed but was not used by this run (roi_bragg disabled or failed).")
            if dependencies.get("pyxem_signal_type"):
                notes.append(f"HyperSpy signal set to <code>{html.escape(str(dependencies['pyxem_signal_type']))}</code> for pyxem compatibility.")
            if dependencies.get("source_backend"):
                notes.append(f"Data loaded via <code>{html.escape(str(dependencies['source_backend']))}</code> backend.")
            if notes:
                lines.append("<p>" + "<br>".join(notes) + "</p>")
    return lines


def _html_png_grid(png: dict[str, Any], base_dir: Path) -> list[str]:
    keys = [
        "fingerprint_class_labels_annotated",
        "cluster_cleaned_labels",
        "cluster_mean_radial_profiles",
        "cluster_virtual_image_statistics",
        "cluster_vs_orientation_heatmap",
        "ring_2_over_ring_1",
        "ring_3_over_ring_1",
        "roi_candidates_overlay",
        "sample_mask",
        "sample_mask_overlay_adf",
        "mean_dp_with_center_marker",
        "orientation_score_masked",
        "k_sweep_metrics",
    ]
    lines = ["<h2>Visual Outputs</h2>", "<div class=\"grid\">"]
    for key in keys:
        if key not in png:
            continue
        path = Path(png[key])
        try:
            link = path.relative_to(base_dir).as_posix()
        except ValueError:
            link = path.as_posix()
        lines.append(
            "<figure>"
            f"<img src=\"{html.escape(link)}\" alt=\"{html.escape(key)}\">"
            f"<figcaption>{html.escape(key)}</figcaption>"
            "</figure>"
        )
    lines.append("</div>")
    return lines


def _html_diagnostic_tables(diagnostics: Any, base_dir: Path) -> list[str]:
    if not isinstance(diagnostics, dict):
        return []
    items = [
        ("Cluster virtual-image statistics", diagnostics.get("cluster_summary_csv")),
        ("Connected-component cleanup", (diagnostics.get("connected_components") or {}).get("connected_components_csv") if isinstance(diagnostics.get("connected_components"), dict) else None),
        ("Cluster vs orientation", (diagnostics.get("cluster_vs_orientation") or {}).get("csv") if isinstance(diagnostics.get("cluster_vs_orientation"), dict) else None),
        ("Stage 2 ROI candidates", (diagnostics.get("roi_outputs") or {}).get("csv") if isinstance(diagnostics.get("roi_outputs"), dict) else None),
        ("K-sweep metrics", (diagnostics.get("k_sweep") or {}).get("metrics_csv") if isinstance(diagnostics.get("k_sweep"), dict) else None),
    ]
    lines = ["<h2>Diagnostic Tables</h2>"]
    for title, path_text in items:
        table = _csv_to_html(path_text, max_rows=12)
        if table:
            lines.extend([f"<h3>{html.escape(title)}</h3>", table])
    return lines


def _csv_to_markdown(path_text: Any, *, base_dir: Path, max_rows: int) -> str:
    if not path_text:
        return ""
    path = Path(str(path_text))
    if not path.exists():
        return ""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ""
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    if len(rows) > max_rows:
        try:
            link = path.relative_to(base_dir).as_posix()
        except ValueError:
            link = path.as_posix()
        tail = [""] * len(keys)
        tail[0] = "..."
        if len(tail) > 1:
            tail[1] = f"see [{path.name}]({link})"
        lines.append("| " + " | ".join(tail) + " |")
    return "\n".join(lines)


def _csv_to_html(path_text: Any, *, max_rows: int) -> str:
    if not path_text:
        return ""
    path = Path(str(path_text))
    if not path.exists():
        return ""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ""
    keys = list(rows[0])
    lines = ["<table><thead><tr>"]
    lines.extend(f"<th>{html.escape(key)}</th>" for key in keys)
    lines.extend(["</tr></thead><tbody>"])
    for row in rows[:max_rows]:
        lines.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in keys) + "</tr>")
    if len(rows) > max_rows:
        lines.append(f"<tr><td colspan=\"{len(keys)}\">Showing {max_rows} of {len(rows)} rows. Full CSV: {html.escape(path.name)}</td></tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


_FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "=": ["00000", "11111", "00000", "00000", "11111", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
}


def _human_size(size: Any) -> str:
    """Format a byte count as a human-readable string."""
    if size is None:
        return "n/a"
    size = int(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024:
            return f"{size} {unit}"
        size //= 1024
    return f"{size} TiB"


def _render_package_versions(packages: dict[str, Any], dependencies: dict[str, Any] | None = None) -> str:
    """Render optional package versions as a Markdown list.

    Distinguishes between packages that are *not installed* (version is
    ``None``) and packages that were *not used* by this particular run
    (version is known but the feature was never invoked).
    """
    if not packages:
        return "### Runtime Dependency Availability\n\n_Dependency versions were not recorded._"

    items: list[str] = []
    for name, version in sorted(packages.items()):
        status: str
        if version is not None:
            status = f"`{version}`"
        else:
            status = "`not installed`"
        items.append(f"- `{name}`: {status}")

    # Add runtime usage annotations when dependency info is available.
    if dependencies:
        deps_note: list[str] = []
        if not dependencies.get("pyxem_available", False) and packages.get("pyxem") is not None:
            deps_note.append("pyxem is installed but was not used by this run.")
        if not dependencies.get("py4DSTEM_used", False) and packages.get("py4DSTEM") is not None:
            deps_note.append("py4DSTEM is installed but was not used by this run (roi_bragg disabled or failed).")
        if dependencies.get("pyxem_signal_type"):
            deps_note.append(f"HyperSpy signal set to `{dependencies['pyxem_signal_type']}` for pyxem compatibility.")
        if dependencies.get("source_backend"):
            deps_note.append(f"Data loaded via `{dependencies['source_backend']}` backend.")
        if deps_note:
            items.append("")
            items.extend(deps_note)

    return "### Runtime Dependency Availability\n\n" + "\n".join(items)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
