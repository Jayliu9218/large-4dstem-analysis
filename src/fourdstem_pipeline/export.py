from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
