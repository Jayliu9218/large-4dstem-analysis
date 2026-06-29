from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .array_utils import as_numpy_block
from .dataset import DatasetHandle
from .logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PreprocessSpec:
    q_crop: tuple[int, int, int, int] | None = None
    q_bin: int = 1
    r_bin: int = 1


class PreprocessedArray:
    """Lazy 4D array view applying q-crop, q-binning, and scan binning per slice."""

    def __init__(self, source: Any, spec: PreprocessSpec):
        if len(tuple(source.shape)) != 4:
            raise ValueError(f"Expected a 4D array, got shape {tuple(source.shape)!r}.")
        self.source = source
        self.spec = spec
        self._shape = self._compute_shape(tuple(int(v) for v in source.shape), spec)
        self.dtype = np.dtype(np.float32)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self._shape

    @property
    def chunks(self) -> Any | None:
        return getattr(self.source, "chunks", None)

    def __getitem__(self, key: Any) -> np.ndarray:
        y_key, x_key, qy_key, qx_key = _normalize_4d_key(key)
        out_y = _slice_from_key(y_key, self.shape[0])
        out_x = _slice_from_key(x_key, self.shape[1])
        out_qy = _slice_from_key(qy_key, self.shape[2])
        out_qx = _slice_from_key(qx_key, self.shape[3])

        ry = max(1, self.spec.r_bin)
        qbin = max(1, self.spec.q_bin)
        crop = self._crop

        src_y = slice(out_y.start * ry, out_y.stop * ry)
        src_x = slice(out_x.start * ry, out_x.stop * ry)
        src_qy = slice(crop[0] + out_qy.start * qbin, crop[0] + out_qy.stop * qbin)
        src_qx = slice(crop[2] + out_qx.start * qbin, crop[2] + out_qx.stop * qbin)

        block = as_numpy_block(self.source[src_y, src_x, src_qy, src_qx]).astype(np.float32, copy=False)
        if ry > 1:
            block = _bin_navigation(block, ry)
        if qbin > 1:
            block = _bin_signal(block, qbin)
        return block

    @property
    def _crop(self) -> tuple[int, int, int, int]:
        _, _, qy, qx = tuple(int(v) for v in self.source.shape)
        if self.spec.q_crop is None:
            return (0, qy, 0, qx)
        qy0, qy1, qx0, qx1 = [int(v) for v in self.spec.q_crop]
        qy0 = max(0, min(qy0, qy))
        qx0 = max(0, min(qx0, qx))
        qy1 = max(qy0, min(qy1, qy))
        qx1 = max(qx0, min(qx1, qx))
        return (qy0, qy1, qx0, qx1)

    @staticmethod
    def _compute_shape(shape: tuple[int, int, int, int], spec: PreprocessSpec) -> tuple[int, int, int, int]:
        ry, rx, qy, qx = shape
        if spec.q_crop is not None:
            qy0, qy1, qx0, qx1 = [int(v) for v in spec.q_crop]
            qy0 = max(0, min(qy0, qy))
            qx0 = max(0, min(qx0, qx))
            qy1 = max(qy0, min(qy1, qy))
            qx1 = max(qx0, min(qx1, qx))
            qy = qy1 - qy0
            qx = qx1 - qx0
        r_bin = max(1, int(spec.r_bin))
        q_bin = max(1, int(spec.q_bin))
        if ry % r_bin:
            log.warning(
                "Navigation y dimension %d is not divisible by r_bin=%d; %d row(s) will be dropped.",
                ry,
                r_bin,
                ry % r_bin,
            )
        if rx % r_bin:
            log.warning(
                "Navigation x dimension %d is not divisible by r_bin=%d; %d column(s) will be dropped.",
                rx,
                r_bin,
                rx % r_bin,
            )
        if qy % q_bin:
            log.warning(
                "Diffraction y dimension %d is not divisible by q_bin=%d; %d row(s) will be dropped.",
                qy,
                q_bin,
                qy % q_bin,
            )
        if qx % q_bin:
            log.warning(
                "Diffraction x dimension %d is not divisible by q_bin=%d; %d column(s) will be dropped.",
                qx,
                q_bin,
                qx % q_bin,
            )
        return (ry // r_bin, rx // r_bin, qy // q_bin, qx // q_bin)


def apply_preprocess(
    dataset: DatasetHandle,
    *,
    q_crop: tuple[int, int, int, int] | list[int] | None = None,
    q_bin: int = 1,
    r_bin: int = 1,
) -> DatasetHandle:
    """Return a dataset view with preprocessing applied lazily per requested block."""
    spec = PreprocessSpec(
        q_crop=tuple(int(v) for v in q_crop) if q_crop is not None else None,
        q_bin=max(1, int(q_bin)),
        r_bin=max(1, int(r_bin)),
    )
    if spec.q_crop is None and spec.q_bin == 1 and spec.r_bin == 1:
        return dataset
    data = PreprocessedArray(dataset.data, spec)
    metadata = dict(dataset.metadata)
    metadata["preprocess"] = {
        "q_crop": list(spec.q_crop) if spec.q_crop is not None else None,
        "q_bin": spec.q_bin,
        "r_bin": spec.r_bin,
    }
    return DatasetHandle(data=data, source=dataset.source, signal=dataset.signal, metadata=metadata)


def _normalize_4d_key(key: Any) -> tuple[Any, Any, Any, Any]:
    if not isinstance(key, tuple):
        key = (key,)
    key = key + (slice(None),) * (4 - len(key))
    if len(key) != 4:
        raise IndexError("PreprocessedArray expects 4D indexing.")
    return key


def _slice_from_key(key: Any, length: int) -> slice:
    if isinstance(key, slice):
        start, stop, step = key.indices(length)
        if step != 1:
            raise IndexError("Step slicing is not supported for preprocessed data.")
        return slice(start, stop)
    index = int(key)
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise IndexError(index)
    return slice(index, index + 1)


def _bin_navigation(block: np.ndarray, factor: int) -> np.ndarray:
    ry, rx, qy, qx = block.shape
    return block.reshape(ry // factor, factor, rx // factor, factor, qy, qx).mean(axis=(1, 3), dtype=np.float32)


def _bin_signal(block: np.ndarray, factor: int) -> np.ndarray:
    ry, rx, qy, qx = block.shape
    return block.reshape(ry, rx, qy // factor, factor, qx // factor, factor).mean(axis=(3, 5), dtype=np.float32)
