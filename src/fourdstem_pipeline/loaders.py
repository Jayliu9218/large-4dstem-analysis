from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DatasetHandle
from .synthetic import make_synthetic_4dstem


def load_dataset(
    path: str | Path,
    *,
    lazy: bool = True,
    cache: str | Path | None = None,
    chunks: dict[str, Any] | tuple[int, ...] | None = None,
    backend: str | None = None,
) -> DatasetHandle:
    """Load a 4D-STEM dataset.

    `.mib` files are passed to HyperSpy with lazy loading when HyperSpy is
    installed. `.npy` and `.npz` are supported for tests and cached workflows.
    Use `synthetic://demo` for a self-contained demonstration dataset.
    """
    source = str(path)
    if source.startswith("synthetic://"):
        data, labels = make_synthetic_4dstem()
        return DatasetHandle(
            data=data,
            source="synthetic",
            metadata={
                "labels": labels,
                "path": source,
                "source_backend": backend or "synthetic",
                "scan_shape": data.shape[:2],
                "detector_shape": data.shape[-2:],
            },
        )

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(data_path, mmap_mode="r" if lazy else None)
        return DatasetHandle(
            data=data,
            source="npy",
            metadata={
                "path": str(data_path),
                "cache": str(cache) if cache else None,
                "source_backend": backend or "numpy",
                "scan_shape": tuple(data.shape[:2]),
                "detector_shape": tuple(data.shape[-2:]),
            },
        )

    if suffix == ".npz":
        archive = np.load(data_path, mmap_mode="r" if lazy else None)
        key = "data" if "data" in archive.files else archive.files[0]
        data = archive[key]
        return DatasetHandle(
            data=data,
            source="npz",
            metadata={
                "path": str(data_path),
                "npz_key": key,
                "source_backend": backend or "numpy",
                "scan_shape": tuple(data.shape[:2]),
                "detector_shape": tuple(data.shape[-2:]),
            },
        )

    if suffix == ".mib":
        return _load_mib_with_hyperspy(data_path, lazy=lazy, cache=cache, chunks=chunks, backend=backend)

    return _load_with_hyperspy(data_path, lazy=lazy, cache=cache, chunks=chunks, backend=backend)


def _load_mib_with_hyperspy(path: Path, *, lazy: bool, cache: str | Path | None, chunks: Any, backend: str | None) -> DatasetHandle:
    return _load_with_hyperspy(path, lazy=lazy, cache=cache, chunks=chunks, backend=backend or "hyperspy_pyxem", source="mib")


def _load_with_hyperspy(
    path: Path,
    *,
    lazy: bool,
    cache: str | Path | None,
    chunks: Any,
    backend: str | None,
    source: str | None = None,
) -> DatasetHandle:
    try:
        import hyperspy.api as hs
    except ImportError as exc:
        raise ImportError(
            "HyperSpy is required for this file type. Install the optional large-data dependencies "
            "or convert the dataset to .npy/.npz for the synthetic/test workflow."
        ) from exc

    signal = hs.load(str(path), lazy=lazy)
    pyxem_info = _try_make_pyxem_compatible(signal) if backend == "hyperspy_pyxem" else {"pyxem_available": False, "pyxem_signal_type": None, "pyxem_error": None}
    data = signal.data
    if chunks and hasattr(data, "rechunk"):
        data = data.rechunk(_normalize_chunks(chunks, data.shape))
        signal.data = data
    metadata = {
        "path": str(path),
        "cache": str(cache) if cache else None,
        "lazy": lazy,
        "source_backend": backend or "hyperspy",
        "axes": _extract_axes_metadata(signal),
        "scan_shape": tuple(int(v) for v in data.shape[:2]),
        "detector_shape": tuple(int(v) for v in data.shape[-2:]),
        **pyxem_info,
    }
    return DatasetHandle(data=data, signal=signal, source=source or path.suffix.lower().lstrip("."), metadata=metadata)


def _normalize_chunks(chunks: Any, shape: tuple[int, ...]) -> Any:
    if isinstance(chunks, dict):
        nav = tuple(chunks.get("navigation", (8, 8)))
        sig = tuple(chunks.get("signal", shape[-2:]))
        return nav + sig
    return chunks


def _try_make_pyxem_compatible(signal: Any) -> dict[str, Any]:
    try:
        import pyxem  # noqa: F401
    except ImportError as exc:
        return {"pyxem_available": False, "pyxem_signal_type": None, "pyxem_error": str(exc)}

    signal_type = None
    error = None
    setter = getattr(signal, "set_signal_type", None)
    if callable(setter):
        for candidate in ("electron_diffraction", "electron_diffraction2d"):
            try:
                setter(candidate)
                signal_type = candidate
                break
            except Exception as exc:  # pyxem/HyperSpy versions differ in registered names.
                error = str(exc)
    if signal_type is None:
        signal_type = _signal_type_name(signal)
    return {"pyxem_available": True, "pyxem_signal_type": signal_type, "pyxem_error": error}


def _signal_type_name(signal: Any) -> str | None:
    metadata = getattr(signal, "metadata", None)
    signal_node = getattr(metadata, "Signal", None)
    signal_type = getattr(signal_node, "signal_type", None)
    return str(signal_type) if signal_type else None


def _extract_axes_metadata(signal: Any) -> list[dict[str, Any]]:
    axes_manager = getattr(signal, "axes_manager", None)
    if axes_manager is None:
        return []
    axes = []
    for axis in axes_manager:
        axes.append(
            {
                "name": str(getattr(axis, "name", "")),
                "size": int(getattr(axis, "size", 0)),
                "scale": _safe_axis_value(getattr(axis, "scale", None)),
                "offset": _safe_axis_value(getattr(axis, "offset", None)),
                "units": str(getattr(axis, "units", "")),
                "navigate": bool(getattr(axis, "navigate", False)),
            }
        )
    return axes


def _safe_axis_value(value: Any) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
