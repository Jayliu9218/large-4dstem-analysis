"""Stage 2C pyxem pattern-matching validation contract.

The heavy pyxem/HyperSpy matching route is intentionally treated as an
independent validation branch.  This module standardises pyxem result arrays so
the pipeline and consensus layer can consume them without depending on pyxem at
import time.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


STANDARD_ARRAYS = {
    "phase_index": "phase_index.npy",
    "score": "score.npy",
    "score_margin": "score_margin.npy",
    "ambiguous_mask": "ambiguous_mask.npy",
    "high_confidence_mask": "high_confidence_mask.npy",
    "pyxem_phase_index": "pyxem_phase_index.npy",
    "pyxem_correlation": "pyxem_correlation.npy",
    "pyxem_margin": "pyxem_margin.npy",
    "pyxem_ambiguous_mask": "pyxem_ambiguous_mask.npy",
}


def run_stage2c_validation(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Run or standardise Stage 2C pyxem validation outputs.

    The first supported execution mode is ``results_npz`` / ``input.results_npz``:
    an existing output from ``scripts/pyxem_hyperspy_ti_phase_orientation.py`` is
    converted into the project-level Stage 2C schema.  ``dry_run: true`` writes a
    configuration/manifest summary without running pyxem matching.
    """

    cfg, config_path = _load_config(config)
    output_dir = Path(cfg.get("output_dir") or _default_output_dir(cfg))
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("enabled") is False:
        phase_names = _phase_names(cfg)
        return _write_summary(
            output_dir=output_dir,
            config_path=config_path,
            status="skipped",
            cfg=cfg,
            phase_names=phase_names,
            arrays={},
            errors=[],
            extra_metadata={"message": "Stage 2C disabled by config."},
        )

    results_npz = _resolve_optional_path(_nested_get(cfg, ("input", "results_npz")) or cfg.get("results_npz"))
    dry_run = bool(cfg.get("dry_run", False))
    phase_names = _phase_names(cfg)
    errors: list[dict[str, Any]] = []

    if results_npz is not None:
        if not results_npz.exists():
            errors.append({"stage": "stage2c", "error": f"pyxem results_npz does not exist: {results_npz}"})
            return _write_summary(
                output_dir=output_dir,
                config_path=config_path,
                status="failed",
                cfg=cfg,
                phase_names=phase_names,
                arrays={},
                errors=errors,
            )
        arrays, metadata = _standardise_pyxem_npz(results_npz, output_dir, phase_names=phase_names)
        return _write_summary(
            output_dir=output_dir,
            config_path=config_path,
            status="success",
            cfg=cfg,
            phase_names=metadata["phase_names"],
            arrays=arrays,
            source_results_npz=str(results_npz),
            extra_metadata=metadata,
            errors=[],
        )

    if dry_run:
        return _write_summary(
            output_dir=output_dir,
            config_path=config_path,
            status="dry_run",
            cfg=cfg,
            phase_names=phase_names,
            arrays={},
            errors=[],
            extra_metadata={"message": "Dry run only; pyxem pattern matching was not executed."},
        )

    missing = _missing_pyxem_dependencies()
    if missing:
        errors.append({
            "stage": "stage2c",
            "error": "Missing optional pyxem validation dependencies.",
            "missing_dependencies": missing,
        })
    else:
        errors.append({
            "stage": "stage2c",
            "error": (
                "Direct pyxem matching from unified YAML is not wired yet. "
                "Run scripts/pyxem_hyperspy_ti_phase_orientation.py and pass its NPZ as stage2c.input.results_npz."
            ),
        })
    return _write_summary(
        output_dir=output_dir,
        config_path=config_path,
        status="failed",
        cfg=cfg,
        phase_names=phase_names,
        arrays={},
        errors=errors,
    )


def _standardise_pyxem_npz(results_npz: Path, output_dir: Path, *, phase_names: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    with np.load(results_npz, allow_pickle=False) as data:
        if "phase_names" in data:
            phase_names = [str(v) for v in data["phase_names"].tolist()]
        phase_index = np.asarray(_require_array(data, "best_phase_index"), dtype=np.int16)
        score = np.asarray(_require_array(data, "best_correlation"), dtype=np.float32)
        score_margin = np.asarray(_require_array(data, "phase_margin"), dtype=np.float32)
        ambiguous = np.asarray(_require_array(data, "ambiguous_mask"), dtype=bool)
        high_conf = np.asarray(_require_array(data, "high_confidence_mask"), dtype=bool)

        arrays = {
            "phase_index": phase_index,
            "score": score,
            "score_margin": score_margin,
            "ambiguous_mask": ambiguous,
            "high_confidence_mask": high_conf,
            "pyxem_phase_index": phase_index,
            "pyxem_correlation": score,
            "pyxem_margin": score_margin,
            "pyxem_ambiguous_mask": ambiguous,
        }
        optional = {
            "orientation_deg": "best_rotation_deg",
            "template_index": "best_template_index",
            "second_phase_score": "second_phase_correlation",
        }
        for out_key, in_key in optional.items():
            if in_key in data:
                arrays[out_key] = np.asarray(data[in_key])

        paths: dict[str, str] = {}
        for key, arr in arrays.items():
            filename = STANDARD_ARRAYS.get(key, f"{key}.npy")
            path = output_dir / filename
            np.save(path, arr)
            paths[key] = filename

        total = int(phase_index.size)
        high_count = int(high_conf.sum())
        ambiguous_count = int(ambiguous.sum())
        metadata = {
            "backend": "pyxem_pattern",
            "method": "polar_template_matching",
            "phase_names": phase_names,
            "shape": list(phase_index.shape),
            "total_pixels": total,
            "high_confidence_fraction": float(high_count / max(total, 1)),
            "ambiguous_fraction": float(ambiguous_count / max(total, 1)),
            "score_mean": _nan_stat(score, "mean"),
            "score_margin_median": _nan_stat(score_margin, "median"),
            "phase_lookup_assumption": "template order = grouped by candidate phase unless source NPZ provided explicit phase_lookup",
            "priority_validation_targets": [
                "Ti-bcc candidate",
                "Ti-hcp candidate",
                "ambiguous",
                "conflict",
                "high score representative",
                "low score representative",
            ],
        }
        if "phase_lookup" in data:
            metadata["phase_lookup"] = data["phase_lookup"].astype(int).tolist()
        return paths, metadata


def _write_summary(
    *,
    output_dir: Path,
    config_path: Path | None,
    status: str,
    cfg: dict[str, Any],
    phase_names: list[str],
    arrays: dict[str, str],
    errors: list[dict[str, Any]],
    source_results_npz: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "stage2c-pyxem-validation-v1",
        "backend": "pyxem_pattern",
        "method": "polar_template_matching",
        "phase_names": phase_names,
        "arrays": arrays,
    }
    manifest_path = output_dir / "stage2c_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    summary = {
        "schema_version": "stage2c-pyxem-validation-v1",
        "status": status,
        "config_path": str(config_path) if config_path else None,
        "output_dir": str(output_dir),
        "source_results_npz": source_results_npz,
        "manifest_path": str(manifest_path),
        "backend": "pyxem_pattern",
        "method": "polar_template_matching",
        "phase_names": phase_names,
        "calibration": copy.deepcopy(cfg.get("calibration") or {}),
        "polar": copy.deepcopy(cfg.get("polar") or {}),
        "orientation": copy.deepcopy(cfg.get("orientation") or {}),
        "qc": copy.deepcopy(cfg.get("qc") or {}),
        "arrays": arrays,
        "metadata": extra_metadata or {},
        "errors": errors if errors else None,
    }
    summary_path = output_dir / "stage2c_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _load_config(config: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config, (str, Path)):
        path = Path(config)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Stage 2C config must be a YAML mapping, got {type(raw).__name__}.")
        if "stage2c" in raw:
            return copy.deepcopy(raw.get("stage2c") or {}), path
        return raw, path
    return copy.deepcopy(config), None


def _default_output_dir(cfg: dict[str, Any]) -> Path:
    stage2_dir = cfg.get("stage2_dir")
    if stage2_dir:
        return Path(stage2_dir) / "stage2c_pyxem_validation"
    stage2b_dir = cfg.get("stage2b_dir")
    if stage2b_dir:
        return Path(stage2b_dir).parent / "stage2c_pyxem_validation"
    return Path("outputs") / "stage2c_pyxem_validation"


def _phase_names(cfg: dict[str, Any]) -> list[str]:
    phases = _nested_get(cfg, ("candidates", "phases")) or cfg.get("candidate_phases") or []
    names = [str(item.get("name")) for item in phases if isinstance(item, dict) and item.get("name")]
    return names or ["Ti-bcc", "Ti-hcp"]


def _nested_get(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _resolve_optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _require_array(data: Any, key: str) -> np.ndarray:
    if key not in data:
        raise ValueError(f"pyxem result NPZ is missing required array {key!r}.")
    return data[key]


def _nan_stat(arr: np.ndarray, name: str) -> float | None:
    finite = np.asarray(arr, dtype=np.float64)
    if finite.size == 0 or not np.isfinite(finite).any():
        return None
    if name == "median":
        return float(np.nanmedian(finite))
    return float(np.nanmean(finite))


def _missing_pyxem_dependencies() -> list[str]:
    missing = []
    for module in ("hyperspy.api", "pyxem", "diffsims", "orix"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    return missing
