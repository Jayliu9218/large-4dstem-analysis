"""Runtime provenance collection for reproducible batch runs.

Every workflow invocation produces a ``provenance.json`` alongside the
summary so that results can be traced back to the exact code, config,
and input data that produced them.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_provenance(
    config: str | Path | dict[str, Any],
    input_path: str | Path | None,
    run_name: str,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Gather provenance metadata for a workflow run.

    Parameters
    ----------
    config:
        Path to the workflow YAML config, or an already-parsed dict
        (e.g. from an inline / programmatic invocation).
    input_path:
        Path to the primary input data file.  May be ``None`` for
        synthetic or memory-only sources.
    run_name:
        Human-readable label for this run (usually the project name).
    start_time:
        UTC timestamp captured at the start of the run.
    end_time:
        UTC timestamp captured when the run completes.
    random_seed:
        Explicit random seed used, if any.
    """
    provenance: dict[str, Any] = {
        "pipeline_version": _pipeline_version(),
        "git_commit": _git_commit(),
        "run_name": run_name,
        "config_path": str(config) if isinstance(config, (str, Path)) else None,
        "config_hash": _config_hash(config),
        "input_path": str(input_path) if input_path else None,
        "input_file_size": _file_size(input_path) if input_path else None,
        "input_file_mtime": _file_mtime(input_path) if input_path else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "random_seed": random_seed,
        "start_time": _isoformat(start_time) if start_time else None,
        "end_time": _isoformat(end_time) if end_time else None,
        "packages": _installed_packages(),
    }
    return provenance


def save_provenance(output_dir: str | Path, provenance: dict[str, Any]) -> Path:
    """Write *provenance* dict to ``<output_dir>/provenance.json``."""
    path = Path(output_dir) / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(provenance), indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pipeline_version() -> str | None:
    """Return the installed package version, if discoverable."""
    # Prefer importlib.metadata for PEP 621 installed packages.
    if sys.version_info >= (3, 10):
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("large-4dstem-analysis")
        except PackageNotFoundError:
            pass

    # Fall back to reading pyproject.toml from the repo root.
    pyproject = _repo_root() / "pyproject.toml"
    if pyproject.exists():
        try:
            text = pyproject.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("version"):
                    return stripped.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass

    return None


def _git_commit() -> str | None:
    """Return the short SHA of HEAD, or ``None`` if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_repo_root()),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _config_hash(config: str | Path | dict[str, Any]) -> str | None:
    """SHA-256 hex digest of the workflow configuration.

    - When *config* is a path, the file contents are hashed.
    - When *config* is a dict, it is serialised to canonical JSON first.
    """
    if isinstance(config, (str, Path)):
        return _sha256_file(Path(config))
    try:
        payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except (TypeError, ValueError):
        return None


def _sha256_file(path: Path) -> str | None:
    """SHA-256 hex digest of *path* (streamed, memory-safe)."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _file_size(path: str | Path) -> int | None:
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


def _file_mtime(path: str | Path) -> str | None:
    try:
        ts = Path(path).stat().st_mtime
        return _isoformat(datetime.fromtimestamp(ts, tz=timezone.utc))
    except OSError:
        return None


def _installed_packages() -> dict[str, str | None]:
    """Best-effort version lookup for optional dependencies."""
    candidates = [
        "numpy",
        "scipy",
        "sklearn",
        "hyperspy",
        "pyxem",
        "py4DSTEM",
    ]
    versions: dict[str, str | None] = {}
    for name in candidates:
        versions[name] = _try_package_version(name)
    return versions


def _try_package_version(name: str) -> str | None:
    if sys.version_info >= (3, 10):
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    # Python 3.9 fallback (though requires-python >= 3.10).
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def _repo_root() -> Path:
    """Return the repository root (directory containing .git)."""
    return Path(__file__).resolve().parents[2]


def _isoformat(dt: datetime) -> str:
    """Return ISO-8601 with ``Z`` suffix for UTC datetimes."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _jsonable(value: Any) -> Any:
    """Recursively convert *value* to JSON-serialisable primitives."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return _isoformat(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return str(value)
