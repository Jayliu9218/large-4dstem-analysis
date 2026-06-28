"""Stage 2B indexing contract scaffold.

This module consumes accepted Stage 2A ROI Bragg outputs and candidate CIF
metadata. It writes a stable ``stage2_indexing_summary.json`` contract before
the full template or Bragg matching implementation is added.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import is_roi_ready_for_indexing

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingCandidate:
    """Candidate phase/template metadata for Stage 2B."""

    name: str
    path: Path
    phase: str | None = None
    reference_peaks: tuple[Any, ...] = ()
    sha256: str | None = None


@dataclass(frozen=True)
class ROIIndexingResult:
    """Indexing contract result for one Stage 2A ROI."""

    name: str
    status: str
    stage2a_bragg_summary_path: str | None
    n_bragg_peaks: int
    best_candidate: str | None
    phase_score: float | None
    orientation_score: float | None
    match_quality: str


def run_stage2_indexing(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Create the Stage 2B indexing summary contract.

    The current implementation is intentionally lightweight. It validates the
    Stage 2A handoff, records candidate CIF provenance, and provides a
    deterministic mock-friendly score when a candidate supplies
    ``reference_peaks`` in the config. That score is labelled ``mock_scored``
    and must not be interpreted as crystallographic indexing.
    """
    cfg, base_dir = _load_indexing_config(config)
    stage2_dir = _resolve_path(cfg["stage2_dir"], base_dir)
    output_dir = _resolve_path(
        cfg.get("output_dir") or stage2_dir / "stage2b_indexing",
        base_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stage2a_summary_path = stage2_dir / "stage2_summary.json"
    stage2a_summary = json.loads(stage2a_summary_path.read_text(encoding="utf-8"))

    candidates = _load_candidates(cfg.get("candidate_cifs", []), base_dir)
    accepted_rois = [
        roi for roi in stage2a_summary.get("roi_results", [])
        if is_roi_ready_for_indexing(roi)
    ]

    roi_results = [
        _score_roi_against_candidates(roi, candidates)
        for roi in accepted_rois
    ]

    summary = {
        "schema_version": "stage2b-indexing-v0",
        "stage": "2B",
        "status": "READY_FOR_TEMPLATE_MATCHING" if accepted_rois else "NO_ACCEPTED_ROIS",
        "stage2a": {
            "stage2_dir": str(stage2_dir),
            "summary_path": str(stage2a_summary_path),
            "run_name": stage2a_summary.get("run_name"),
        },
        "output_dir": str(output_dir),
        "candidate_cifs": [
            {
                "name": c.name,
                "phase": c.phase,
                "path": str(c.path),
                "sha256": c.sha256,
                "reference_peak_count": len(c.reference_peaks),
                "scoring_mode": "mock_peak_count" if c.reference_peaks else "not_scored",
            }
            for c in candidates
        ],
        "accepted_roi_count": len(accepted_rois),
        "roi_results": [
            {
                "name": r.name,
                "status": r.status,
                "stage2a_bragg_summary_path": r.stage2a_bragg_summary_path,
                "n_bragg_peaks": r.n_bragg_peaks,
                "best_candidate": r.best_candidate,
                "phase_score": r.phase_score,
                "orientation_score": r.orientation_score,
                "match_quality": r.match_quality,
            }
            for r in roi_results
        ],
        "notes": [
            "This is the Stage 2B contract scaffold.",
            "Full CIF parsing, template generation, and orientation search are not implemented here.",
        ],
    }

    summary_path = output_dir / "stage2_indexing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _load_indexing_config(config: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path]:
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        base_dir = Path.cwd()
    else:
        cfg = config
        base_dir = Path.cwd()

    if not isinstance(cfg, dict):
        raise ValueError("Stage 2B config must be a mapping.")
    if "stage2_dir" not in cfg:
        raise ValueError("Stage 2B config must contain 'stage2_dir'.")
    return cfg, base_dir


def _load_candidates(items: list[dict[str, Any]], base_dir: Path) -> list[IndexingCandidate]:
    candidates: list[IndexingCandidate] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Each candidate_cifs entry must be a mapping.")
        raw_path = item.get("path")
        if not raw_path:
            raise ValueError("Each candidate_cifs entry must contain 'path'.")
        path = _resolve_path(raw_path, base_dir)
        candidates.append(
            IndexingCandidate(
                name=str(item.get("name") or path.stem or f"candidate_{i:03d}"),
                phase=item.get("phase"),
                path=path,
                reference_peaks=tuple(item.get("reference_peaks") or ()),
                sha256=_sha256_file(path),
            )
        )
    return candidates


def _score_roi_against_candidates(
    roi: dict[str, Any],
    candidates: list[IndexingCandidate],
) -> ROIIndexingResult:
    n_bragg_peaks = int(roi.get("n_bragg_peaks", 0) or 0)
    best_candidate: IndexingCandidate | None = None
    best_score: float | None = None

    for candidate in candidates:
        if not candidate.reference_peaks:
            continue
        reference_count = len(candidate.reference_peaks)
        denom = max(n_bragg_peaks, reference_count, 1)
        score = 1.0 - abs(n_bragg_peaks - reference_count) / denom
        if best_score is None or score > best_score:
            best_candidate = candidate
            best_score = round(float(score), 4)

    if best_candidate is None:
        return ROIIndexingResult(
            name=str(roi.get("name", "unknown")),
            status="PENDING_TEMPLATE_MATCHING",
            stage2a_bragg_summary_path=roi.get("bragg_summary_path"),
            n_bragg_peaks=n_bragg_peaks,
            best_candidate=None,
            phase_score=None,
            orientation_score=None,
            match_quality="not_scored",
        )

    return ROIIndexingResult(
        name=str(roi.get("name", "unknown")),
        status="MOCK_SCORED",
        stage2a_bragg_summary_path=roi.get("bragg_summary_path"),
        n_bragg_peaks=n_bragg_peaks,
        best_candidate=best_candidate.name,
        phase_score=best_score,
        orientation_score=None,
        match_quality="mock_scored",
    )


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        log.warning("Could not hash candidate CIF %s: %s", path, exc)
        return None
