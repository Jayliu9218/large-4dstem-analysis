from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class DatasetHandle:
    """Small wrapper around a 4D-STEM data source.

    The wrapped data should expose a NumPy-like shape ordered as
    navigation_y, navigation_x, detector_y, detector_x.
    """

    data: Any
    source: str
    signal: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.data.shape)

    @property
    def navigation_shape(self) -> tuple[int, int]:
        if len(self.shape) != 4:
            raise ValueError(f"Expected 4D-STEM data, got shape {self.shape!r}.")
        return self.shape[:2]

    @property
    def signal_shape(self) -> tuple[int, int]:
        if len(self.shape) != 4:
            raise ValueError(f"Expected 4D-STEM data, got shape {self.shape!r}.")
        return self.shape[-2:]

    @property
    def dtype(self) -> str:
        return str(getattr(self.data, "dtype", "unknown"))

    @property
    def nbytes_estimate(self) -> int | None:
        dtype = getattr(self.data, "dtype", None)
        if dtype is None:
            return None
        return int(np.prod(self.shape) * np.dtype(dtype).itemsize)

    @property
    def chunks(self) -> Any | None:
        return getattr(self.data, "chunks", None)

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "shape": self.shape,
            "navigation_shape": self.navigation_shape,
            "signal_shape": self.signal_shape,
            "dtype": self.dtype,
            "chunks": self.chunks,
            "nbytes_estimate": self.nbytes_estimate,
            "path": self.metadata.get("path"),
        }


def save_jsonable_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    import json

    serializable = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v for k, v in metadata.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")
