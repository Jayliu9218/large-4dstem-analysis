"""Single non-visual 4D-STEM analysis workflow.

Orchestrates data loading, preprocessing, virtual imaging, radial
fingerprinting, unsupervised phase screening, orientation preview, and
optional ROI Bragg detection.  Each stage is logged with timing, and
individual stage failures are captured rather than aborting the entire run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import load_workflow_config, resolve_data_config
from .dataset import DatasetHandle
from .diagnostics import run_stage1_diagnostics
from .export import save_annotated_label_png, save_label_png, save_png, save_profile_png, save_report, save_summary
from .fingerprints import FingerprintResult, compute_radial_fingerprints
from .loaders import load_dataset
from .logging import configure_pipeline_logging, get_logger, log_stage_end, log_stage_start
from .masks import build_annular_masks
from .orientation import OrientationResult, run_orientation_preview
from .phase import PhaseScreeningResult, screen_phases
from .preprocess import apply_preprocess
from .virtual import VirtualImageResult, compute_virtual_images

log = get_logger(__name__)


@dataclass(slots=True)
class WorkflowResult:
    output_dir: Path
    dataset: DatasetHandle
    virtual_images: VirtualImageResult | None = None
    fingerprints: FingerprintResult | None = None
    phase_screening: PhaseScreeningResult | None = None
    orientation: OrientationResult | None = None
    roi_bragg: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    summary_path: Path | None = None
    report_path: Path | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)


def run_workflow(
    config: str | Path | dict[str, Any] = "configs/default_workflow.yaml",
    *,
    log_level: str = "INFO",
) -> WorkflowResult:
    """Run the single non-visual 4D-STEM analysis workflow.

    Parameters
    ----------
    config:
        Path to a YAML config file, or an already-parsed dict.
    log_level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.  Also
        controllable via the ``FOURDSTEM_LOG_LEVEL`` environment variable.
    """
    configure_pipeline_logging(level=log_level)
    t0 = time.perf_counter()

    cfg = load_workflow_config(config) if isinstance(config, (str, Path)) else dict(config)
    _validate_dict_config(cfg)

    project_cfg = cfg.get("project", {})
    output_dir = Path(project_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("4D-STEM analysis workflow starting")
    log.info("  project: %s", project_cfg.get("name", "unnamed"))
    log.info("  output:  %s", output_dir)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Stage 0: Load & preprocess
    # ------------------------------------------------------------------
    data_cfg = resolve_data_config(cfg.get("data", {}))
    block_shape = _navigation_block_shape(cfg, data_cfg)

    errors: list[dict[str, Any]] = []

    dataset = _run_stage("load", lambda: load_dataset(
        data_cfg.get("path", "synthetic://demo"),
        lazy=bool(data_cfg.get("lazy", True)),
        cache=data_cfg.get("cache"),
        chunks=data_cfg.get("chunks"),
        backend=data_cfg.get("backend"),
        scan_shape=data_cfg.get("scan_shape"),
        detector_shape=data_cfg.get("detector_shape"),
        dtype=data_cfg.get("dtype"),
        mib_header_bytes=data_cfg.get("mib_header_bytes"),
    ))

    if dataset is None:
        errors.append({"stage": "load", "error": "Data loading failed.", "elapsed_s": 0})
        return _make_error_result(output_dir, dataset, errors)

    log.info("  data backend:  %s", dataset.source_backend)
    log.info("  navigation:    %s", dataset.navigation_shape)
    log.info("  signal:        %s", dataset.signal_shape)
    log.info("  block shape:   %s", block_shape)

    dataset = _run_stage("preprocess", lambda: apply_preprocess(dataset, **cfg.get("preprocess", {})))

    if dataset is None:
        log.error("Preprocessing failed — cannot continue. Aborting workflow.")
        return _make_error_result(output_dir, dataset, errors)

    # ------------------------------------------------------------------
    # Set up output directories
    # ------------------------------------------------------------------
    geometry = cfg.get("geometry", {})
    masks = build_annular_masks(
        dataset.signal_shape,
        cfg.get("virtual_images", {}).get("masks", {}),
        center=geometry.get("center"),
    )

    preprocess_dir = output_dir / "00_preprocess"
    virtual_dir = output_dir / "01_virtual_images"
    fingerprints_dir = output_dir / "02_fingerprints"
    classes_dir = output_dir / "03_diffraction_classes"
    orientation_dir = output_dir / "04_orientation_preview"
    png_dir = output_dir / "png"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: Virtual images
    # ------------------------------------------------------------------
    virtual = _run_stage(
        "virtual images",
        lambda: compute_virtual_images(dataset, masks, output_dir=virtual_dir, block_shape=block_shape),
        errors=errors,
    )

    # ------------------------------------------------------------------
    # Stage 2: Radial fingerprints
    # ------------------------------------------------------------------
    fingerprints = _run_stage(
        "radial fingerprints",
        lambda: compute_radial_fingerprints(
            dataset,
            geometry,
            int(geometry.get("radial_bins", 48)),
            output_dir=fingerprints_dir,
            block_shape=block_shape,
        ),
        errors=errors,
    )

    # ------------------------------------------------------------------
    # Stage 3: Phase screening
    # ------------------------------------------------------------------
    phase_cfg = cfg.get("phase_screening", {})
    phase = _run_stage(
        "phase screening",
        lambda: screen_phases(
            fingerprints,
            method=phase_cfg.get("method", "pca_nmf_cluster"),
            candidate_phases=phase_cfg.get("candidate_phases"),
            n_components=int(phase_cfg.get("n_components", 3)),
            n_clusters=int(phase_cfg.get("n_clusters", phase_cfg.get("n_components", 3))),
            output_dir=classes_dir,
        ),
        errors=errors,
    )

    # ------------------------------------------------------------------
    # Stage 4: Orientation preview
    # ------------------------------------------------------------------
    orientation_cfg = cfg.get("orientation", {})
    orientation = _run_stage(
        "orientation preview",
        lambda: run_orientation_preview(
            dataset,
            phase_candidates=orientation_cfg.get("phase_candidates"),
            binning=orientation_cfg.get("preview_binning", (2, 2)),
            roi=orientation_cfg.get("roi"),
            confidence_threshold=float(orientation_cfg.get("confidence_threshold", 0.05)),
            output_dir=orientation_dir,
            block_shape=block_shape,
        ),
        errors=errors,
    )

    # ------------------------------------------------------------------
    # Stage 5: ROI Bragg (optional)
    # ------------------------------------------------------------------
    roi_bragg = None
    roi_cfg = cfg.get("roi_bragg", {})
    if bool(roi_cfg.get("enabled", False)):
        roi_bragg = _run_stage(
            "ROI Bragg detection",
            lambda: _run_roi_bragg(data_cfg, roi_cfg, output_dir / "roi_bragg"),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # PNG exports
    # ------------------------------------------------------------------
    png_outputs: dict[str, Path] = {}
    if virtual is not None and fingerprints is not None and phase is not None and orientation is not None:
        png_outputs = _save_png_outputs(png_dir, virtual, fingerprints, phase, orientation)
    else:
        log.warning("Skipping PNG exports because one or more upstream stages failed.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    diagnostics: dict[str, Any] = {}
    if virtual is not None and fingerprints is not None and phase is not None and orientation is not None:
        diagnostics = _run_stage(
            "stage-1 diagnostics",
            lambda: run_stage1_diagnostics(
                dataset,
                fingerprints,
                phase,
                virtual,
                orientation,
                output_dir=output_dir,
                png_dir=png_dir,
                block_shape=block_shape,
                confidence_threshold=float(orientation_cfg.get("confidence_threshold", 0.05)),
            ),
            errors=errors,
        ) or {}

        for key, filename in {
            "cluster_mean_radial_profiles": "cluster_mean_radial_profiles.png",
            "cluster_virtual_image_statistics": "cluster_virtual_image_statistics.png",
            "roi_candidates_overlay": "roi_candidates_overlay.png",
            "mean_dp_with_center_marker": "mean_dp_with_center_marker.png",
            "orientation_score_masked": "orientation_score_masked.png",
            "k_sweep_metrics": "k_sweep_metrics.png",
        }.items():
            path = png_dir / filename
            if path.exists():
                png_outputs[key] = path
    else:
        log.warning("Skipping diagnostics because one or more upstream stages failed.")

    # ------------------------------------------------------------------
    # Summary & report
    # ------------------------------------------------------------------
    summary = {
        "project": project_cfg,
        "data_config": data_cfg,
        "preprocess": cfg.get("preprocess", {}),
        "dataset": dataset.describe(),
        "outputs": {
            "virtual": str(virtual_dir),
            "fingerprints": str(fingerprints_dir),
            "diffraction_classes": str(classes_dir),
            "orientation": str(orientation_dir),
            "cluster_diagnostics": diagnostics.get("cluster_diagnostics"),
            "roi_candidates": diagnostics.get("roi_candidates"),
            "roi_bragg": roi_bragg,
            "png": {name: str(path) for name, path in png_outputs.items()},
            "diagnostics": diagnostics,
        },
        "shapes": {
            "virtual_images": {name: image.shape for name, image in virtual.images.items()} if virtual else {},
            "radial_fingerprints": fingerprints.profiles.shape if fingerprints else None,
            "diffraction_class_labels": phase.labels.shape if phase else None,
            "orientation_index": orientation.orientation_index.shape if orientation else None,
        },
        "errors": errors if errors else None,
    }

    import numpy as _np

    report_path = save_report(
        output_dir, summary,
        phase.labels if phase else _np.zeros(dataset.navigation_shape if dataset else (1, 1), dtype=_np.int16),
    )
    summary["outputs"]["report"] = str(report_path)
    summary_path = save_summary(output_dir, summary)

    elapsed = time.perf_counter() - t0
    if errors:
        log.warning("=" * 60)
        log.warning("Workflow finished with %d error(s) in %.1f s", len(errors), elapsed)
        for err in errors:
            log.warning("  ✗  %s: %s", err["stage"], err["error"])
    else:
        log.info("=" * 60)
        log.info("Workflow finished successfully in %.1f s", elapsed)
    log.info("  summary: %s", summary_path)
    log.info("  report:  %s", report_path)
    log.info("=" * 60)

    return WorkflowResult(
        output_dir=output_dir,
        dataset=dataset,
        virtual_images=virtual,
        fingerprints=fingerprints,
        phase_screening=phase,
        orientation=orientation,
        roi_bragg=roi_bragg,
        diagnostics=diagnostics,
        summary_path=summary_path,
        report_path=report_path,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_error_result(
    output_dir: Path,
    dataset: DatasetHandle | None,
    errors: list[dict[str, Any]],
) -> WorkflowResult:
    """Return a result with everything set to None when the workflow cannot proceed."""
    log.error("Workflow aborted due to unrecoverable error(s).")
    return WorkflowResult(
        output_dir=output_dir,
        dataset=dataset,
        summary_path=None,
        report_path=None,
        errors=errors,
    )


def _run_stage(
    label: str,
    fn: Any,
    *,
    errors: list[dict[str, Any]] | None = None,
) -> Any:
    """Run a pipeline stage with timing and optional error capture."""
    log_stage_start(log, label)
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log.error("✗  %s FAILED after %.1f s: %s", label, elapsed, exc)
        if errors is not None:
            errors.append({"stage": label, "error": str(exc), "elapsed_s": elapsed})
        return None
    elapsed = time.perf_counter() - t0
    log_stage_end(log, label, elapsed)
    return result


def _navigation_block_shape(cfg: dict[str, Any], data_cfg: dict[str, Any]) -> tuple[int, int]:
    block_shape = cfg.get("block_shape")
    if block_shape is None:
        chunks = data_cfg.get("chunks", {})
        if isinstance(chunks, dict):
            block_shape = chunks.get("navigation", (8, 8))
        else:
            block_shape = chunks[:2] if chunks else (8, 8)
    by, bx = [max(1, int(v)) for v in block_shape]
    return by, bx


def _validate_dict_config(cfg: dict[str, Any]) -> None:
    """Run schema validation on an inline config dict (not from file)."""
    from .config import validate_workflow_config

    validate_workflow_config(cfg)


def _save_png_outputs(
    output_dir: Path,
    virtual: VirtualImageResult,
    fingerprints: FingerprintResult,
    phase: PhaseScreeningResult,
    orientation: OrientationResult,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in virtual.images.items():
        paths[f"virtual_{name}"] = save_png(output_dir / f"virtual_{name}.png", image)
    paths["com_x"] = save_png(output_dir / "com_x.png", virtual.com_x)
    paths["com_y"] = save_png(output_dir / "com_y.png", virtual.com_y)
    paths["mean_diffraction"] = save_png(output_dir / "mean_diffraction.png", virtual.mean_diffraction)
    paths["max_diffraction"] = save_png(output_dir / "max_diffraction.png", virtual.max_diffraction)
    paths["mean_radial_profile"] = save_profile_png(
        output_dir / "mean_radial_profile.png",
        fingerprints.radii,
        fingerprints.profiles,
    )
    paths["diffraction_class_labels"] = save_label_png(output_dir / "diffraction_class_labels.png", phase.labels)
    paths["diffraction_class_labels_annotated"] = save_annotated_label_png(
        output_dir / "diffraction_class_labels_annotated.png",
        phase.labels,
        title="Diffraction-class map from radial fingerprints",
    )
    paths["diffraction_class_low_confidence"] = save_png(output_dir / "diffraction_class_low_confidence_mask.png", phase.low_confidence_mask)
    for name, score in phase.candidate_scores.items():
        paths[f"candidate_score_{name}"] = save_png(output_dir / f"candidate_score_{name}.png", score)
    paths["orientation_index"] = save_label_png(output_dir / "orientation_index.png", orientation.orientation_index)
    paths["orientation_score"] = save_png(output_dir / "orientation_score.png", orientation.score)
    paths["orientation_low_confidence"] = save_png(output_dir / "orientation_low_confidence_mask.png", orientation.low_confidence_mask)
    return paths


def _run_roi_bragg(data_cfg: dict[str, Any], roi_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    try:
        import py4DSTEM
    except ImportError as exc:
        raise ImportError("roi_bragg.enabled requires py4DSTEM. Install optional diffraction dependencies first.") from exc

    import numpy as np

    path = data_cfg.get("path")
    if not path:
        raise ValueError("roi_bragg requires data.path to point to a real MIB file.")
    scan_shape = tuple(data_cfg.get("scan_shape", (512, 512)))
    cube = py4DSTEM.import_file(str(path), mem=roi_cfg.get("mem", "MEMMAP"), scan=scan_shape)

    y0, y1, x0, x1 = [int(v) for v in roi_cfg.get("roi", (0, scan_shape[0], 0, scan_shape[1]))]
    thin_r = max(1, int(roi_cfg.get("thin_r", 2)))
    bin_q = max(1, int(roi_cfg.get("bin_q", 2)))
    roi_data = np.asarray(cube.data[y0:y1:thin_r, x0:x1:thin_r, :, :], dtype=np.float32)
    dc_roi = py4DSTEM.DataCube(roi_data, name="roi", calibration=cube.calibration)
    if bin_q > 1:
        dc_roi = dc_roi.bin_Q(bin_q, dtype=np.float32)

    template = np.asarray(dc_roi.data.mean(axis=(0, 1)), dtype=np.float32)
    bragg = dc_roi.find_Bragg_disks(
        template=template,
        corrPower=float(roi_cfg.get("corr_power", 1.0)),
        sigma_cc=float(roi_cfg.get("sigma_cc", 1)),
        edgeBoundary=int(roi_cfg.get("edge_boundary", 10)),
        minRelativeIntensity=float(roi_cfg.get("min_relative_intensity", 0.05)),
        minPeakSpacing=int(roi_cfg.get("min_peak_spacing", 4)),
        subpixel=roi_cfg.get("subpixel", "poly"),
        maxNumPeaks=int(roi_cfg.get("max_num_peaks", 70)),
        CUDA=bool(roi_cfg.get("cuda", False)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    histogram = bragg.histogram(mode="cal")
    np.save(output_dir / "roi_bragg_vector_map.npy", np.asarray(histogram.data, dtype=np.float32))
    return {
        "output_dir": str(output_dir),
        "roi": [y0, y1, x0, x1],
        "thin_r": thin_r,
        "bin_q": bin_q,
        "bragg_vector_map": str(output_dir / "roi_bragg_vector_map.npy"),
    }
