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

import numpy as np
import yaml

from .config import load_workflow_config
from .contracts import Stage1Manifest, Stage1ManifestLoadError
from .export_stage2 import build_benchmark, save_stage2_benchmark, save_stage2_gallery, save_stage2_report
from .logging import configure_pipeline_logging, get_logger
from .provenance import collect_provenance, save_provenance
from .roi_bragg import (
    ROIBraggResult,
    Stage2Result,
    _parse_beam_center_txt,
    load_roi_candidates,
    run_roi_bragg_for_rois,
)

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
    output_dir = Path(cfg.get("output_dir") or stage1_dir / "stage2" / "roi_bragg").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Stage 2A starting: stage1_dir=%s, output_dir=%s", stage1_dir, output_dir)

    # --- Load and validate Stage-1 manifest ---------------------------------
    log.info("Loading Stage-1 manifest from %s", stage1_dir)
    manifest = Stage1Manifest.load(stage1_dir)
    if manifest.warnings:
        for w in manifest.warnings:
            log.warning("Stage-1 manifest warning: %s", w)
    log.info(
        "Manifest loaded: run=%s, nav=%s, sig=%s, r_bin=%d, qc=%s",
        manifest.run_name, manifest.nav_shape, manifest.sig_shape,
        manifest.r_bin, manifest.qc_status,
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
    scan_shape = _parse_scan_shape(cfg.get("scan_shape"))

    # --- Load beam centre from Stage 1 --------------------------------------
    beam_center_yx: tuple[float, float] | None = None
    beam_center_txt = stage1_dir / "00_preprocess" / "beam_center_estimate.txt"
    if beam_center_txt.exists():
        parsed = _parse_beam_center_txt(beam_center_txt)
        if parsed is not None:
            beam_center_yx = parsed
            log.info(
                "Stage-1 beam centre loaded from %s: (%.3f, %.3f)",
                beam_center_txt, *beam_center_yx,
            )
        else:
            log.warning(
                "Found %s but could not parse beam centre coordinates.",
                beam_center_txt,
            )
    else:
        log.info(
            "No Stage-1 beam centre estimate at %s; will fall back to "
            "py4DSTEM calibration or detector centre.",
            beam_center_txt,
        )

    # --- Load labels for cluster validation ---------------------------------
    labels: np.ndarray | None = None
    if manifest.labels_path.exists():
        try:
            labels = np.load(manifest.labels_path)
            log.info("Labels loaded: shape=%s, background=%d pixels",
                     labels.shape, int(np.sum(labels == -1)))
        except Exception as exc:
            log.warning("Failed to load labels from %s: %s", manifest.labels_path, exc)

    # --- Load sample mask for coverage validation ---------------------------
    sample_mask: np.ndarray | None = None
    sample_mask_npy = stage1_dir / "00_preprocess" / "sample_mask.npy"
    if sample_mask_npy.exists():
        try:
            sample_mask = np.load(sample_mask_npy)
            log.info("Sample mask loaded: shape=%s, coverage=%.1f%%",
                     sample_mask.shape, 100.0 * float(sample_mask.mean()))
        except Exception as exc:
            log.warning("Failed to load sample mask from %s: %s", sample_mask_npy, exc)
    else:
        log.info("No sample mask at %s; skipping coverage validation.", sample_mask_npy)

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
        scan_shape=scan_shape,
        beam_center_yx=beam_center_yx,
        labels=labels,
        sample_mask=sample_mask,
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

    summary = _build_stage2_summary(result, provenance, cfg, beam_center_yx)
    summary_path = output_dir / "stage2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("Stage 2 summary written to %s", summary_path)

    # --- QC summary ---------------------------------------------------------
    qc = _build_stage2_qc(result)
    qc_path = output_dir / "stage2_qc_summary.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    # --- Report & benchmark --------------------------------------------------
    try:
        report_md, report_html = save_stage2_report(output_dir, summary)
        log.info("Stage 2 report written: %s, %s", report_md, report_html)
    except Exception as exc:
        log.warning("Failed to generate Stage 2 report: %s", exc)

    try:
        gallery_path = save_stage2_gallery(output_dir, summary)
        if gallery_path is not None:
            log.info("Stage 2 PNG gallery written: %s", gallery_path)
    except Exception as exc:
        log.warning("Failed to generate Stage 2 PNG gallery: %s", exc)

    try:
        benchmark_entries = _build_benchmark_entries(result, cfg)
        benchmark = build_benchmark(
            benchmark_entries,
            time.perf_counter() - t0,  # total elapsed
            mem=cfg.get("mem", "MEMMAP"),
            thin_r=int(cfg.get("thin_r", 2)),
            bin_q=int(cfg.get("bin_q", 2)),
        )
        bench_path = save_stage2_benchmark(output_dir, benchmark)
        log.info("Stage 2 benchmark written: %s", bench_path)
    except Exception as exc:
        log.warning("Failed to generate Stage 2 benchmark: %s", exc)

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

    # Emit cluster validation warnings to the log
    for r in roi_results:
        if r.error is None and r.cluster_validation_warning:
            log.warning("ROI '%s' cluster validation: %s", r.name, r.cluster_validation_warning)

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


def _parse_scan_shape(value: Any) -> tuple[int, int] | None:
    """Parse optional raw navigation scan shape from Stage-2 config."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Stage 2 config 'scan_shape' must be null or [ny, nx].")
    scan_shape = (int(value[0]), int(value[1]))
    if scan_shape[0] <= 0 or scan_shape[1] <= 0:
        raise ValueError(f"Stage 2 config 'scan_shape' values must be positive: {scan_shape}.")
    return scan_shape


def _build_stage2_summary(
    result: Stage2Result,
    provenance: dict[str, Any],
    cfg: dict[str, Any],
    beam_center_yx: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build the stage2_summary.json dict."""
    data_path_str = str(_resolve_data_path(cfg, result.manifest))
    configured_scan_shape = _parse_scan_shape(cfg.get("scan_shape"))

    roi_summaries: list[dict[str, Any]] = []
    for r in result.roi_results:
        roi_summaries.append({
            "name": r.name,
            "stage1_bbox": r.stage1_bbox,
            "raw_bbox": r.raw_bbox,
            "nav_shape": list(r.nav_shape),
            "sig_shape": list(r.sig_shape),
            "n_bragg_peaks": r.n_peaks,
            "beam_center_yx": r.beam_center_yx,
            "beam_center_source": r.beam_center_source,
            "cluster_id": r.cluster_id,
            "reason": r.reason,
            "background_fraction": r.background_fraction,
            "sample_mask_coverage": r.sample_mask_coverage,
            "cluster_validation_warning": r.cluster_validation_warning,
            "roi_data_path": str(r.roi_data_path) if r.roi_data_path else None,
            "bragg_vector_map_path": str(r.bragg_vector_map_path) if r.bragg_vector_map_path else None,
            "bragg_summary_path": str(r.bragg_summary_path) if r.bragg_summary_path else None,
            "bragg_peaks_parquet_path": str(r.bragg_peaks_parquet_path) if r.bragg_peaks_parquet_path else None,
            "bragg_qc": r.bragg_qc,
            "error": r.error,
        })

    return {
        "stage1_dir": str(result.stage1_dir),
        "output_dir": str(result.output_dir),
        "run_name": provenance.get("run_name"),
        "manifest": {
            "run_name": result.manifest.run_name,
            "nav_shape": result.manifest.nav_shape,
            "sig_shape": result.manifest.sig_shape,
            "r_bin": result.manifest.r_bin,
            "qc_status": result.manifest.qc_status,
        },
        "parameters": {
            "thin_r": cfg.get("thin_r", 2),
            "bin_q": cfg.get("bin_q", 2),
            "max_rois": cfg.get("max_rois"),
            "roi_source": cfg.get("roi_source", "roi_candidates"),
            "scan_shape": list(configured_scan_shape) if configured_scan_shape else None,
        },
        "beam_center": {
            "stage1_yx": list(beam_center_yx) if beam_center_yx else None,
            "source": (
                result.roi_results[0].beam_center_source
                if result.roi_results else None
            ),
        },
        "roi_results": roi_summaries,
        "provenance": provenance,
        "dependencies": {
            "py4DSTEM_used": True,
            "data_path": data_path_str,
            "scan_shape": list(configured_scan_shape) if configured_scan_shape else None,
        },
        "errors": result.errors if result.errors else None,
    }


def _build_benchmark_entries(
    result: Stage2Result,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build per-ROI benchmark entries for the benchmark JSON."""
    entries: list[dict[str, Any]] = []
    for r in result.roi_results:
        entries.append({
            "name": r.name,
            "error": r.error,
            "stage1_bbox": r.stage1_bbox,
            "raw_bbox": r.raw_bbox,
            "nav_shape": list(r.nav_shape),
            "sig_shape": list(r.sig_shape),
            "r_bin": result.manifest.r_bin,
            "n_bragg_peaks": r.n_peaks,
            "timing": {
                "extraction_s": r.extraction_time_s,
                "bragg_detection_s": r.bragg_time_s,
                "total_s": r.total_time_s,
            },
            "roi_data_size_bytes": r.roi_data_size_bytes,
        })
    return entries


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

    # --- Cluster validation warnings ----------------------------------------
    cluster_warnings = [
        r for r in result.roi_results
        if r.error is None and r.cluster_validation_warning is not None
    ]
    if cluster_warnings:
        flags.append({
            "severity": "warning",
            "code": "CLUSTER_VALIDATION_WARNINGS",
            "message": f"{len(cluster_warnings)} ROI(s) have cluster/background validation warnings.",
            "evidence": {
                "warned_rois": [
                    {
                        "name": r.name,
                        "background_fraction": r.background_fraction,
                        "sample_mask_coverage": r.sample_mask_coverage,
                        "warning": r.cluster_validation_warning,
                    }
                    for r in cluster_warnings
                ],
            },
        })

    # --- High-background ROIs (critical) ------------------------------------
    high_bg_rois = [
        r for r in result.roi_results
        if r.error is None
        and r.background_fraction is not None
        and r.background_fraction > 0.5
    ]
    if high_bg_rois:
        flags.append({
            "severity": "critical",
            "code": "HIGH_BACKGROUND_ROIS",
            "message": (
                f"{len(high_bg_rois)} ROI(s) have >50% background pixels "
                f"(label -1). These ROIs may be over vacuum/sample edge."
            ),
            "evidence": {
                "high_background_rois": [
                    {"name": r.name, "background_fraction": r.background_fraction}
                    for r in high_bg_rois
                ],
            },
        })

    # --- Zero-coverage ROIs (critical) --------------------------------------
    zero_cov_rois = [
        r for r in result.roi_results
        if r.error is None
        and r.sample_mask_coverage is not None
        and r.sample_mask_coverage == 0.0
    ]
    if zero_cov_rois:
        flags.append({
            "severity": "critical",
            "code": "ZERO_SAMPLE_COVERAGE_ROIS",
            "message": (
                f"{len(zero_cov_rois)} ROI(s) have 0% sample mask coverage. "
                f"These ROIs are entirely outside the sample region."
            ),
            "evidence": {
                "zero_coverage_rois": [r.name for r in zero_cov_rois],
            },
        })

    # --- Bragg peak QC flags --------------------------------------------------
    for r in result.roi_results:
        if r.error is not None or r.bragg_qc is None:
            continue
        bq = r.bragg_qc
        if bq.get("forbidden_center_zone_fraction", 0.0) > 0.3:
            flags.append({
                "severity": "warning",
                "code": "HIGH_CENTER_ZONE_PEAKS",
                "message": (
                    f"ROI '{r.name}' has {bq['forbidden_center_zone_fraction']:.1%} "
                    f"peaks within the central beam zone — likely BF tail, not Bragg disks."
                ),
                "evidence": {"roi": r.name, "center_zone_fraction": bq["forbidden_center_zone_fraction"]},
            })
        if bq.get("edge_peak_fraction", 0.0) > 0.3:
            flags.append({
                "severity": "warning",
                "code": "HIGH_EDGE_PEAKS",
                "message": (
                    f"ROI '{r.name}' has {bq['edge_peak_fraction']:.1%} "
                    f"peaks near detector edges — possible edge artifacts."
                ),
                "evidence": {"roi": r.name, "edge_peak_fraction": bq["edge_peak_fraction"]},
            })
        if bq.get("duplicate_peak_fraction", 0.0) > 0.3:
            flags.append({
                "severity": "warning",
                "code": "HIGH_DUPLICATE_PEAKS",
                "message": (
                    f"ROI '{r.name}' has {bq['duplicate_peak_fraction']:.1%} "
                    f"peaks too close together (< minPeakSpacing) — possible splitting."
                ),
                "evidence": {"roi": r.name, "duplicate_peak_fraction": bq["duplicate_peak_fraction"]},
            })
        if (bq.get("beam_center_error_estimate") or 0.0) > 5.0:
            flags.append({
                "severity": "warning",
                "code": "LARGE_BEAM_CENTER_ERROR",
                "message": (
                    f"ROI '{r.name}' peak centroid is {bq['beam_center_error_estimate']:.1f} px "
                    f"from the nominal beam centre — centre may be miscalibrated."
                ),
                "evidence": {"roi": r.name, "beam_center_error_px": bq["beam_center_error_estimate"]},
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
