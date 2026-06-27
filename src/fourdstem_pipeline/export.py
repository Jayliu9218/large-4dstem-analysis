from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import struct
import zlib

import numpy as np


def save_summary(output_dir: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(output_dir) / "workflow_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return path


def save_npz(output_dir: str | Path, name: str, **arrays: np.ndarray) -> Path:
    path = Path(output_dir) / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def save_png(path: str | Path, image: np.ndarray, *, percentiles: tuple[float, float] = (1, 99)) -> Path:
    """Save a 2D scalar image or RGB uint8 image as PNG without extra dependencies."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = np.repeat(_scale_to_uint8(arr, percentiles)[..., None], 3, axis=-1)
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


def save_profile_png(path: str | Path, radii: np.ndarray, profiles: np.ndarray, *, size: tuple[int, int] = (720, 420)) -> Path:
    """Save the mean radial profile as a simple line plot PNG."""
    width, height = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 58, 22, 24, 46
    x0, x1 = margin_l, width - margin_r - 1
    y0, y1 = margin_t, height - margin_b - 1

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
    return save_png(path, canvas)


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
                    canvas[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor += 6 * scale


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
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "=": ["00000", "11111", "00000", "00000", "11111", "00000", "00000"],
}


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
