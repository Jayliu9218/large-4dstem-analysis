"""Unified three-stage pipeline orchestration.

This module keeps the existing Stage 1, Stage 2A, and Stage 2B entry points
intact, but lets users drive them from one nested YAML config.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .logging import configure_pipeline_logging, get_logger
from .stage2 import run_stage2
from .workflow import WorkflowResult, run_workflow
from .roi_bragg import Stage2Result
from .indexing import run_stage2_indexing

log = get_logger(__name__)


_STAGE1_KEYS = {
    "project",
    "data",
    "preprocess",
    "geometry",
    "virtual_images",
    "phase_screening",
    "orientation",
    "sample_mask",
    "block_shape",
    "random_seed",
}

_STAGE_ALIASES = {
    "stage1": "stage1",
    "stage_1": "stage1",
    "workflow": "stage1",
    "stage2": "stage2a",
    "stage2a": "stage2a",
    "stage_2a": "stage2a",
    "roi_bragg": "stage2a",
    "stage2b": "stage2b",
    "stage_2b": "stage2b",
    "indexing": "stage2b",
}


@dataclass(slots=True)
class PipelineStageRecord:
    """Serializable status record for one pipeline stage."""

    name: str
    status: str
    output_dir: str | None = None
    summary_path: str | None = None
    errors: list[Any] = field(default_factory=list)
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "output_dir": self.output_dir,
            "summary_path": self.summary_path,
            "errors": self.errors,
            "skipped_reason": self.skipped_reason,
        }


@dataclass(slots=True)
class PipelineResult:
    """Aggregated result from the unified Stage 1 -> 2A -> 2B pipeline."""

    output_dir: Path
    summary_path: Path
    stages: dict[str, PipelineStageRecord]
    stage1: WorkflowResult | None = None
    stage2a: Stage2Result | None = None
    stage2b: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)


def run_pipeline(
    config: str | Path | dict[str, Any] = "configs/pipeline.yaml",
    *,
    log_level: str = "INFO",
) -> PipelineResult:
    """Run selected stages from a unified pipeline config."""
    configure_pipeline_logging(level=log_level)
    cfg, config_path = _load_pipeline_config(config)
    stages = _resolve_stages(cfg)
    output_dir = _pipeline_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, PipelineStageRecord] = {}
    errors: list[dict[str, Any]] = []
    stage1_result: WorkflowResult | None = None
    stage2a_result: Stage2Result | None = None
    stage2b_result: dict[str, Any] | None = None

    log.info("Unified pipeline starting: stages=%s output=%s", stages, output_dir)

    if "stage1" in stages:
        stage1_cfg = _stage1_config(cfg)
        stage1_result = run_workflow(stage1_cfg, log_level=log_level)
        stage1_errors = list(stage1_result.errors or [])
        status = "failed" if stage1_errors else "success"
        records["stage1"] = PipelineStageRecord(
            name="stage1",
            status=status,
            output_dir=str(stage1_result.output_dir),
            summary_path=str(stage1_result.summary_path) if stage1_result.summary_path else None,
            errors=stage1_errors,
        )
        if stage1_errors:
            errors.extend({"stage": "stage1", **err} if isinstance(err, dict) else {"stage": "stage1", "error": str(err)} for err in stage1_errors)
    else:
        records["stage1"] = PipelineStageRecord(
            name="stage1",
            status="skipped",
            skipped_reason="Not listed in pipeline.stages.",
        )

    if "stage2a" in stages:
        if stage1_result is None or records["stage1"].status != "success":
            records["stage2a"] = PipelineStageRecord(
                name="stage2a",
                status="skipped",
                skipped_reason="Stage 1 did not complete successfully.",
            )
        else:
            try:
                stage2a_cfg = _stage2a_config(cfg, stage1_result.output_dir)
                stage2a_result = run_stage2(stage2a_cfg)
                stage2a_errors = [
                    {"roi": r.name, "error": r.error}
                    for r in stage2a_result.roi_results
                    if r.error
                ]
                status = "failed" if stage2a_errors else "success"
                records["stage2a"] = PipelineStageRecord(
                    name="stage2a",
                    status=status,
                    output_dir=str(stage2a_result.output_dir),
                    summary_path=str(stage2a_result.output_dir / "stage2_summary.json"),
                    errors=stage2a_errors,
                )
                if stage2a_errors:
                    errors.extend({"stage": "stage2a", **err} for err in stage2a_errors)
            except Exception as exc:
                records["stage2a"] = PipelineStageRecord(
                    name="stage2a",
                    status="failed",
                    errors=[{"error": str(exc)}],
                )
                errors.append({"stage": "stage2a", "error": str(exc)})
                log.exception("Stage 2A failed")
    else:
        records["stage2a"] = PipelineStageRecord(
            name="stage2a",
            status="skipped",
            skipped_reason="Not listed in pipeline.stages.",
        )

    if "stage2b" in stages:
        if stage2a_result is None or records["stage2a"].status != "success":
            records["stage2b"] = PipelineStageRecord(
                name="stage2b",
                status="skipped",
                skipped_reason="Stage 2A did not complete successfully.",
            )
        else:
            try:
                stage2b_cfg = _stage2b_config(cfg, stage2a_result.output_dir)
                stage2b_result = run_stage2_indexing(stage2b_cfg)
                records["stage2b"] = PipelineStageRecord(
                    name="stage2b",
                    status="success",
                    output_dir=str(stage2b_result.get("output_dir")) if stage2b_result else None,
                    summary_path=str(Path(str(stage2b_result.get("output_dir"))) / "stage2_indexing_summary.json")
                    if stage2b_result and stage2b_result.get("output_dir") else None,
                )
            except Exception as exc:
                records["stage2b"] = PipelineStageRecord(
                    name="stage2b",
                    status="failed",
                    errors=[{"error": str(exc)}],
                )
                errors.append({"stage": "stage2b", "error": str(exc)})
                log.exception("Stage 2B failed")
    else:
        records["stage2b"] = PipelineStageRecord(
            name="stage2b",
            status="skipped",
            skipped_reason="Not listed in pipeline.stages.",
        )

    summary_path = _write_pipeline_summary(
        output_dir=output_dir,
        config_path=config_path,
        stages=records,
        errors=errors,
    )
    log.info("Unified pipeline summary written to %s", summary_path)

    return PipelineResult(
        output_dir=output_dir,
        summary_path=summary_path,
        stages=records,
        stage1=stage1_result,
        stage2a=stage2a_result,
        stage2b=stage2b_result,
        errors=errors,
    )


def _load_pipeline_config(config: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config, (str, Path)):
        path = Path(config)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Pipeline config must be a YAML mapping, got {type(raw).__name__}.")
        return raw, path
    return copy.deepcopy(config), None


def _resolve_stages(cfg: dict[str, Any]) -> list[str]:
    raw = (cfg.get("pipeline") or {}).get("stages", ["stage1", "stage2a", "stage2b"])
    if not isinstance(raw, (list, tuple)):
        raise ValueError("pipeline.stages must be a list.")
    stages: list[str] = []
    for item in raw:
        key = _STAGE_ALIASES.get(str(item).strip().lower())
        if key is None:
            raise ValueError(f"Unknown pipeline stage: {item!r}.")
        if key not in stages:
            stages.append(key)
    return stages


def _pipeline_output_dir(cfg: dict[str, Any]) -> Path:
    pipeline_cfg = cfg.get("pipeline") or {}
    if pipeline_cfg.get("output_dir"):
        return Path(pipeline_cfg["output_dir"])
    project_cfg = cfg.get("project") or {}
    return Path(project_cfg.get("output_dir", "outputs")) / "pipeline"


def _stage1_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in cfg.items() if key in _STAGE1_KEYS}


def _stage2a_config(cfg: dict[str, Any], stage1_dir: Path) -> dict[str, Any]:
    stage2a = copy.deepcopy(cfg.get("stage2a") or {})
    stage2a["stage1_dir"] = str(stage1_dir)
    stage2a.setdefault("output_dir", None)
    return stage2a


def _stage2b_config(cfg: dict[str, Any], stage2a_dir: Path) -> dict[str, Any]:
    stage2b = copy.deepcopy(cfg.get("stage2b") or {})
    stage2b["stage2_dir"] = str(stage2a_dir)
    stage2b.setdefault("output_dir", None)
    return stage2b


def _write_pipeline_summary(
    *,
    output_dir: Path,
    config_path: Path | None,
    stages: dict[str, PipelineStageRecord],
    errors: list[dict[str, Any]],
) -> Path:
    summary = {
        "config_path": str(config_path) if config_path else None,
        "output_dir": str(output_dir),
        "status": "failed" if errors else "success",
        "stages": {name: record.to_dict() for name, record in stages.items()},
        "errors": errors if errors else None,
    }
    path = output_dir / "pipeline_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path
