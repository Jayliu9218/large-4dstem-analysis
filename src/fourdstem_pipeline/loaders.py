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
) -> DatasetHandle:
    """Load a 4D-STEM dataset.

    `.mib` files are passed to HyperSpy with lazy loading when HyperSpy is
    installed. `.npy` and `.npz` are supported for tests and cached workflows.
    Use `synthetic://demo` for a self-contained demonstration dataset.
    """
    source = str(path)
    if source.startswith("synthetic://"):
        data, labels = make_synthetic_4dstem()
        return DatasetHandle(data=data, source="synthetic", metadata={"labels": labels, "path": source})

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(data_path, mmap_mode="r" if lazy else None)
        return DatasetHandle(data=data, source="npy", metadata={"path": str(data_path), "cache": str(cache) if cache else None})

    if suffix == ".npz":
        archive = np.load(data_path, mmap_mode="r" if lazy else None)
        key = "data" if "data" in archive.files else archive.files[0]
        return DatasetHandle(data=archive[key], source="npz", metadata={"path": str(data_path), "npz_key": key})

    if suffix == ".mib":
        return _load_mib_with_hyperspy(data_path, lazy=lazy, cache=cache, chunks=chunks)

    return _load_with_hyperspy(data_path, lazy=lazy, cache=cache, chunks=chunks)


def _load_mib_with_hyperspy(path: Path, *, lazy: bool, cache: str | Path | None, chunks: Any) -> DatasetHandle:
    return _load_with_hyperspy(path, lazy=lazy, cache=cache, chunks=chunks, source="mib")


def _load_with_hyperspy(
    path: Path,
    *,
    lazy: bool,
    cache: str | Path | None,
    chunks: Any,
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
    data = signal.data
    if chunks and hasattr(data, "rechunk"):
        data = data.rechunk(_normalize_chunks(chunks, data.shape))
        signal.data = data
    metadata = {"path": str(path), "cache": str(cache) if cache else None, "lazy": lazy}
    return DatasetHandle(data=data, signal=signal, source=source or path.suffix.lower().lstrip("."), metadata=metadata)


def _normalize_chunks(chunks: Any, shape: tuple[int, ...]) -> Any:
    if isinstance(chunks, dict):
        nav = tuple(chunks.get("navigation", (8, 8)))
        sig = tuple(chunks.get("signal", shape[-2:]))
        return nav + sig
    return chunks
