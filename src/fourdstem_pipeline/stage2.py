"""Stage 2 orchestration: consume a Stage-1 manifest, run ROI Bragg detection.

Provides the public entry point :func:`run_stage2` and the convenience
function :func:`load_stage1_manifest` for programmatic use.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import load_workflow_config
from .contracts import Stage1Manifest, Stage1ManifestLoadError
from .logging import configure_pipeline_logging, get_logger
from .provenance import collect_provenance, save_provenance
from .roi_bragg import ROIBraggResult, Stage2Result, load_roi_candidates, run_roi_bragg_for_rois

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_stage1_manifest(stage1_dir: str | Path) -> Stage1Manifest:
    """Convenience wrapper around :meth:`Stage1Manifest.load`.

    Returns a validated manifest or raises :class:`Stage1ManifestLoadError`
    with a descriptive message.
    """
    return Stage1Manifest.load(stage1_dir)


def run_stage2(config: str | Path | dict[str, Any]) -> Stage2Result:
    """Execute Stage 2A ROI Bragg detection.

    Parameters
    ----------
    config:
        Path to a Stage-2 YAML config file, or an already-parsed dict.

    Returns
    -------
    Stage2Result
        Aggregated results including per-ROI outputs and any errors.

    Raises
    ------
    Stage1ManifestLoadError
        If the Stage-1 manifest is missing or invalid.
    ImportError
        If py4DSTEM is not installed.
    FileNotFoundError
        If the original data file cannot be found.
    """
    t0 = time.perf_counter()

    # --- Load config --------------------------------------------------------
    if isinstance(config, (str, Path)):
        cfg = _load_stage2_config(config)
    else:
        cfg = config

    stage1_dir = Path(cfg["stage1_dir"]).resolve()
    output_dir = Path(cfg.get("output_dir", stage1_dir / "stage2" / "roi_bragg")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Stage 2A starting: stage1_dir=%s, output_dir=%s", stage1_dir, output_dir)

    # --- Load and validate Stage-1 manifest ---------------------------------
    log.info("Loading Stage-1 manifest from %s", stage1_dir)
    manifest = Stage1Manifest.load(stage1_dir)
    if manifest.warnings:
        for w in manifest.warnings:
            log.warning("Stage-1 manifest warning: %s", w)
    log.info(
        "Manifest loaded: run=%s, nav=%s, sig=%s, qc=%s",
        manifest.run_name, manifest.nav_shape, manifest.sig_shape, manifest.qc_status,
    )

    # --- Load ROI candidates ------------------------------------------------
    roi_source = cfg.get("roi_source", "roi_candidates")
    if roi_source == "roi_candidates":
        roi_yaml = manifest.roi_candidates_path
    else:
        roi_yaml = Path(roi_source)
    log.info("Loading ROI candidates from %s", roi_yaml)
    rois = load_roi_candidates(roi_yaml)
    log.info("Found %d ROI candidate(s)", len(rois))

    max_rois = cfg.get("max_rois")
    if max_rois is not None:
        rois = rois[: int(max_rois)]
        log.info("Capped to %d ROI(s)", len(rois))

    # --- Resolve data path --------------------------------------------------
    data_path = _resolve_data_path(cfg, manifest)

    # --- Run ROI Bragg detection --------------------------------------------
    bragg_kwargs: dict[str, Any] = {}
    for key in (
        "corr_power", "sigma_cc", "edge_boundary", "min_relative_intensity",
        "min_peak_spacing", "subpixel", "max_num_peaks", "cuda",
    ):
        if key in cfg:
            bragg_kwargs[key] = cfg[key]

    roi_results = run_roi_bragg_for_rois(
        manifest=manifest,
        rois=rois,
        output_dir=output_dir,
        data_path=data_path,
        data_loader=cfg.get("data_loader", "auto"),
        max_rois=None,  # already capped above
        thin_r=int(cfg.get("thin_r", 2)),
        bin_q=int(cfg.get("bin_q", 2)),
        mem=cfg.get("mem", "MEMMAP"),
        bragg_kwargs=bragg_kwargs,
    )

    # --- Collect provenance -------------------------------------------------
    start_time = datetime.now(timezone.utc)
    provenance = collect_provenance(
        config=config if isinstance(config, (str, Path)) else cfg,
        input_path=str(data_path),
        run_name=f"{manifest.run_name}_stage2",
        start_time=start_time,
    )
    save_provenance(output_dir, provenance)

    # --- Compose result & summary -------------------------------------------
    result = Stage2Result(
        stage1_dir=stage1_dir,
        output_dir=output_dir,
        manifest=manifest,
        roi_results=roi_results,
    )

    summary = _build_stage2_summary(result, provenance, cfg)
    summary_path = output_dir / "stage2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("Stage 2 summary written to %s", summary_path)

    # --- QC summary ---------------------------------------------------------
    qc = _build_stage2_qc(result)
    qc_path = output_dir / "stage2_qc_summary.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    if result.n_failed > 0:
        log.warning(
            "Stage 2A finished with %d/%d ROI failures in %.1f s",
            result.n_failed, len(roi_results), elapsed,
        )
    else:
        log.info(
            "Stage 2A finished successfully: %d ROIs, %d total Bragg peaks in %.1f s",
            result.n_success, sum(r.n_peaks for r in roi_results), elapsed,
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_stage2_config(config_path: str | Path) -> dict[str, Any]:
    """Load a Stage-2 YAML config, validating the minimal required keys."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Stage 2 config must be a YAML mapping, got {type(cfg).__name__}.")
    if "stage1_dir" not in cfg:
        raise ValueError("Stage 2 config must contain 'stage1_dir'.")
    return cfg


def _resolve_data_path(cfg: dict[str, Any], manifest: Stage1Manifest) -> Path:
    """Find the original 4D-STEM data file from config or provenance."""
    # Explicit path in config takes priority.
    if cfg.get("data_path"):
        p = Path(cfg["data_path"])
        if p.exists():
            return p
        # Try relative to stage1_dir
        p = manifest.stage1_dir / cfg["data_path"]
        if p.exists():
            return p
        raise FileNotFoundError(f"Data file not found: {cfg['data_path']}")

    # Fall back to provenance.json input_path.
    prov_path = manifest.provenance_path
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
            input_path = prov.get("input_path")
            if input_path:
                p = Path(input_path)
                if p.exists():
                    return p
        except (json.JSONDecodeError, OSError):
            pass

    raise FileNotFoundError(
        "Cannot determine data file path.  Set 'data_path' in the Stage 2 "
        "config, or ensure provenance.json contains a valid 'input_path'."
    )


def _build_stage2_summary(
    result: Stage2Result,
    provenance: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build the stage2_summary.json dict."""
    return {
        "stage1_dir": str(result.stage1_dir),
        "output_dir": str(result.output_dir),
        "run_name": provenance.get("run_name"),
        "manifest": {
            "run_name": result.manifest.run_name,
            "nav_shape": result.manifest.nav_shape,
            "sig_shape": result.manifest.sig_shape,
            "qc_status": result.manifest.qc_status,
        },
        "parameters": {
            "thin_r": cfg.get("thin_r", 2),
            "bin_q": cfg.get("bin_q", 2),
            "max_rois": cfg.get("max_rois"),
            "roi_source": cfg.get("roi_source", "roi_candidates"),
        },
        "roi_results": [
            {
                "name": r.name,
                "roi_bbox": r.roi_bbox,
                "nav_shape": list(r.nav_shape),
                "sig_shape": list(r.sig_shape),
                "n_bragg_peaks": r.n_peaks,
                "roi_data_path": str(r.roi_data_path) if r.roi_data_path else None,
                "bragg_vector_map_path": str(r.bragg_vector_map_path) if r.bragg_vector_map_path else None,
                "bragg_summary_path": str(r.bragg_summary_path) if r.bragg_summary_path else None,
                "error": r.error,
            }
            for r in result.roi_results
        ],
        "provenance": provenance,
        "dependencies": {
            "py4DSTEM_used": True,
            "data_path": str(_resolve_data_path(cfg, result.manifest)),
        },
        "errors": result.errors if result.errors else None,
    }


def _build_stage2_qc(result: Stage2Result) -> dict[str, Any]:
    """Build a minimal QC summary for the Stage 2 run."""
    n_total = len(result.roi_results)
    n_success = result.n_success
    n_failed = result.n_failed

    flags: list[dict[str, Any]] = []
    if n_total == 0:
        flags.append({
            "severity": "warning",
            "code": "NO_ROIS_PROCESSED",
            "message": "No ROIs were processed. Check roi_candidates.yaml and max_rois setting.",
        })
    if n_failed > 0:
        flags.append({
            "severity": "warning",
            "code": "ROI_PROCESSING_FAILURES",
            "message": f"{n_failed}/{n_total} ROI(s) failed during Bragg detection.",
            "evidence": {
                "failed_rois": [r.name for r in result.roi_results if r.error is not None],
            },
        })
    if n_success > 0:
        total_peaks = sum(r.n_peaks for r in result.roi_results if r.error is None)
        if total_peaks == 0:
            flags.append({
                "severity": "warning",
                "code": "NO_BRAGG_PEAKS",
                "message": "No Bragg peaks found in any ROI. Check Bragg detection parameters.",
            })

    n_critical = sum(1 for f in flags if f.get("severity") == "critical")
    n_warnings = sum(1 for f in flags if f.get("severity") == "warning")

    if n_critical > 0:
        status = "FAIL"
    elif n_warnings > 0:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    return {
        "stage2_status": status,
        "n_rois_total": n_total,
        "n_rois_success": n_success,
        "n_rois_failed": n_failed,
        "n_warnings": n_warnings,
        "n_critical": n_critical,
        "flags": flags,
    }
