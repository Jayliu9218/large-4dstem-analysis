from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_workflow_config
from .dataset import DatasetHandle
from .export import save_annotated_label_png, save_label_png, save_png, save_profile_png, save_report, save_summary
from .fingerprints import FingerprintResult, compute_radial_fingerprints
from .loaders import load_dataset
from .masks import build_annular_masks
from .orientation import OrientationResult, run_orientation_preview
from .phase import PhaseScreeningResult, screen_phases
from .preprocess import apply_preprocess
from .virtual import VirtualImageResult, compute_virtual_images


@dataclass(slots=True)
class WorkflowResult:
    output_dir: Path
    dataset: DatasetHandle
    virtual_images: VirtualImageResult
    fingerprints: FingerprintResult
    phase_screening: PhaseScreeningResult
    orientation: OrientationResult
    roi_bragg: dict[str, Any] | None
    summary_path: Path
    report_path: Path


def run_workflow(config: str | Path | dict[str, Any] = "configs/default_workflow.yaml") -> WorkflowResult:
    """Run the single non-visual 4D-STEM analysis workflow."""
    cfg = load_workflow_config(config) if isinstance(config, (str, Path)) else dict(config)
    project_cfg = cfg.get("project", {})
    output_dir = Path(project_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _resolve_data_config(cfg.get("data", {}))
    dataset = load_dataset(
        data_cfg.get("path", "synthetic://demo"),
        lazy=bool(data_cfg.get("lazy", True)),
        cache=data_cfg.get("cache"),
        chunks=data_cfg.get("chunks"),
        backend=data_cfg.get("backend"),
        scan_shape=data_cfg.get("scan_shape"),
        detector_shape=data_cfg.get("detector_shape"),
        dtype=data_cfg.get("dtype"),
        mib_header_bytes=data_cfg.get("mib_header_bytes"),
    )
    dataset = apply_preprocess(dataset, **cfg.get("preprocess", {}))

    geometry = cfg.get("geometry", {})
    block_shape = _navigation_block_shape(cfg, data_cfg)
    masks = build_annular_masks(
        dataset.signal_shape,
        cfg.get("virtual_images", {}).get("masks", {}),
        center=geometry.get("center"),
    )
    virtual = compute_virtual_images(dataset, masks, output_dir=output_dir / "virtual", block_shape=block_shape)

    fingerprints = compute_radial_fingerprints(
        dataset,
        geometry,
        int(geometry.get("radial_bins", 48)),
        output_dir=output_dir / "fingerprints",
        block_shape=block_shape,
    )

    phase_cfg = cfg.get("phase_screening", {})
    phase = screen_phases(
        fingerprints,
        method=phase_cfg.get("method", "pca_nmf_cluster"),
        candidate_phases=phase_cfg.get("candidate_phases"),
        n_components=int(phase_cfg.get("n_components", 3)),
        n_clusters=int(phase_cfg.get("n_clusters", phase_cfg.get("n_components", 3))),
        output_dir=output_dir / "phase_screening",
    )

    orientation_cfg = cfg.get("orientation", {})
    orientation = run_orientation_preview(
        dataset,
        phase_candidates=orientation_cfg.get("phase_candidates"),
        binning=orientation_cfg.get("preview_binning", (2, 2)),
        roi=orientation_cfg.get("roi"),
        confidence_threshold=float(orientation_cfg.get("confidence_threshold", 0.05)),
        output_dir=output_dir / "orientation",
        block_shape=block_shape,
    )

    roi_bragg = None
    roi_cfg = cfg.get("roi_bragg", {})
    if bool(roi_cfg.get("enabled", False)):
        roi_bragg = _run_roi_bragg(data_cfg, roi_cfg, output_dir / "roi_bragg")

    png_outputs = _save_png_outputs(output_dir / "png", virtual, fingerprints, phase, orientation)

    summary = {
        "project": project_cfg,
        "data_config": data_cfg,
        "preprocess": cfg.get("preprocess", {}),
        "dataset": dataset.describe(),
        "outputs": {
            "virtual": str(virtual.output_dir),
            "fingerprints": str(fingerprints.output_dir),
            "phase_screening": str(phase.output_dir),
            "orientation": str(orientation.output_dir),
            "roi_bragg": roi_bragg,
            "png": {name: str(path) for name, path in png_outputs.items()},
        },
        "shapes": {
            "virtual_images": {name: image.shape for name, image in virtual.images.items()},
            "radial_fingerprints": fingerprints.profiles.shape,
            "phase_labels": phase.labels.shape,
            "orientation_index": orientation.orientation_index.shape,
        },
    }
    report_path = save_report(output_dir, summary, phase.labels)
    summary["outputs"]["report"] = str(report_path)
    summary_path = save_summary(output_dir, summary)
    return WorkflowResult(
        output_dir=output_dir,
        dataset=dataset,
        virtual_images=virtual,
        fingerprints=fingerprints,
        phase_screening=phase,
        orientation=orientation,
        roi_bragg=roi_bragg,
        summary_path=summary_path,
        report_path=report_path,
    )


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


def _resolve_data_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(data_cfg)
    if resolved.get("path"):
        return resolved

    directory = resolved.get("directory")
    if not directory:
        resolved["path"] = "synthetic://demo"
        return resolved

    pattern = resolved.get("pattern", "*.mib")
    candidates = sorted(Path(directory).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No data files matched {Path(directory) / pattern}.")

    index = int(resolved.get("index", 0))
    if index < 0 or index >= len(candidates):
        raise IndexError(f"data.index {index} is out of range for {len(candidates)} matched files.")
    resolved["path"] = str(candidates[index])
    resolved["matched_files"] = [str(path) for path in candidates]
    return resolved


def _save_png_outputs(
    output_dir: Path,
    virtual: VirtualImageResult,
    fingerprints: FingerprintResult,
    phase: PhaseScreeningResult,
    orientation: OrientationResult,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
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
    paths["phase_labels"] = save_label_png(output_dir / "phase_labels.png", phase.labels)
    paths["phase_labels_annotated"] = save_annotated_label_png(output_dir / "phase_labels_annotated.png", phase.labels)
    paths["phase_low_confidence"] = save_png(output_dir / "phase_low_confidence_mask.png", phase.low_confidence_mask)
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
