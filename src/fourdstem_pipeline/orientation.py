from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .array_utils import as_numpy_block, iter_navigation_slices, normalize_rows, parse_roi
from .dataset import DatasetHandle


@dataclass(slots=True)
class OrientationResult:
    orientation_index: np.ndarray
    phase_label: np.ndarray
    score: np.ndarray
    low_confidence_mask: np.ndarray
    roi: tuple[int, int, int, int] | None
    output_dir: Path | None = None


def run_orientation_preview(
    dataset: DatasetHandle,
    phase_candidates: list[dict[str, Any]] | None = None,
    *,
    binning: tuple[int, int] | list[int] = (2, 2),
    roi: tuple[int, int, int, int] | list[int] | None = None,
    confidence_threshold: float = 0.05,
    output_dir: str | Path | None = None,
    block_shape: tuple[int, int] = (8, 8),
) -> OrientationResult:
    """Run a conservative orientation preview.

    If candidate templates are provided, normalized dot-product template
    matching is used. Otherwise, a COM-angle proxy provides a fast orientation
    preview suitable for selecting ROIs.
    """
    y_slice, x_slice = parse_roi(roi, dataset.navigation_shape)
    out_nav = (y_slice.stop - y_slice.start, x_slice.stop - x_slice.start)
    by, bx = [max(1, int(v)) for v in binning]
    out_binned = (int(np.ceil(out_nav[0] / by)), int(np.ceil(out_nav[1] / bx)))

    templates, template_phases = _collect_templates(phase_candidates, dataset.signal_shape)
    orientation_index = np.zeros(out_binned, dtype=np.int16)
    phase_label = np.zeros(out_binned, dtype=np.int16)
    score = np.zeros(out_binned, dtype=np.float32)

    for ys, xs in iter_navigation_slices(out_binned, block_shape):
        src_y = slice(y_slice.start + ys.start * by, min(y_slice.start + ys.stop * by, y_slice.stop))
        src_x = slice(x_slice.start + xs.start * bx, min(x_slice.start + xs.stop * bx, x_slice.stop))
        block = as_numpy_block(dataset.data[src_y, src_x, :, :]).astype(np.float32, copy=False)
        patterns = _bin_navigation(block, by, bx).reshape((-1,) + dataset.signal_shape)
        if templates is not None:
            idx, scr = _match_templates(patterns, templates)
            orientation_index[ys, xs] = idx.reshape((ys.stop - ys.start, xs.stop - xs.start))
            phase_label[ys, xs] = template_phases[idx].reshape((ys.stop - ys.start, xs.stop - xs.start))
            score[ys, xs] = scr.reshape((ys.stop - ys.start, xs.stop - xs.start))
        else:
            idx, scr = _com_angle_preview(patterns)
            orientation_index[ys, xs] = idx.reshape((ys.stop - ys.start, xs.stop - xs.start))
            score[ys, xs] = scr.reshape((ys.stop - ys.start, xs.stop - xs.start))

    low_confidence_mask = score < float(confidence_threshold)
    result = OrientationResult(
        orientation_index=orientation_index,
        phase_label=phase_label,
        score=score,
        low_confidence_mask=low_confidence_mask,
        roi=tuple(roi) if roi is not None else None,
        output_dir=Path(output_dir) if output_dir else None,
    )
    if output_dir:
        _save_orientation_result(result)
    return result


def _collect_templates(phase_candidates: list[dict[str, Any]] | None, signal_shape: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray]:
    if not phase_candidates:
        return None, np.zeros(0, dtype=np.int16)
    templates = []
    phases = []
    for phase_idx, phase in enumerate(phase_candidates):
        for template in phase.get("templates", []) or []:
            arr = np.asarray(template, dtype=np.float32)
            if arr.shape == signal_shape:
                templates.append(arr.ravel())
                phases.append(phase_idx)
    if not templates:
        return None, np.zeros(0, dtype=np.int16)
    return normalize_rows(np.vstack(templates)), np.asarray(phases, dtype=np.int16)


def _match_templates(patterns: np.ndarray, templates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = normalize_rows(patterns.reshape(patterns.shape[0], -1))
    scores = flat @ templates.T
    idx = np.argmax(scores, axis=1)
    return idx.astype(np.int16), scores[np.arange(scores.shape[0]), idx].astype(np.float32)


def _com_angle_preview(patterns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sy, sx = patterns.shape[-2:]
    yy, xx = np.indices((sy, sx))
    total = np.maximum(patterns.sum(axis=(-2, -1)), 1e-12)
    com_x = (patterns * xx).sum(axis=(-2, -1)) / total - (sx - 1) / 2
    com_y = (patterns * yy).sum(axis=(-2, -1)) / total - (sy - 1) / 2
    angle = np.mod(np.arctan2(com_y, com_x), 2 * np.pi)
    idx = np.floor(angle / (2 * np.pi) * 36).astype(np.int16)
    score = np.sqrt(com_x**2 + com_y**2).astype(np.float32)
    if np.max(score) > 0:
        score = score / np.max(score)
    return idx, score.astype(np.float32)


def _bin_navigation(block: np.ndarray, by: int, bx: int) -> np.ndarray:
    ny, nx = block.shape[:2]
    out = []
    for y0 in range(0, ny, by):
        row = []
        for x0 in range(0, nx, bx):
            row.append(block[y0 : y0 + by, x0 : x0 + bx].mean(axis=(0, 1)))
        out.append(row)
    return np.asarray(out, dtype=np.float32)


def _save_orientation_result(result: OrientationResult) -> None:
    assert result.output_dir is not None
    result.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(result.output_dir / "orientation_index.npy", result.orientation_index)
    np.save(result.output_dir / "orientation_phase_label.npy", result.phase_label)
    np.save(result.output_dir / "orientation_score.npy", result.score)
    np.save(result.output_dir / "orientation_low_confidence_mask.npy", result.low_confidence_mask)
