"""Notebook-first 4D-STEM analysis helpers."""

from .config import load_workflow_config, resolve_data_config, validate_workflow_config
from .contracts import (
    DataContract,
    DiffractionCalibration,
    Stage1Manifest,
    Stage1ManifestLoadError,
    is_roi_ready_for_indexing,
    roi_indexing_blockers,
    roi_indexing_readiness,
)
from .dataset import DatasetHandle
from .diffraction import DiffractionSignal, OrientationMap, PolarSignal
from .export import (
    apply_gamma,
    apply_ipf_colors,
    mask_center_for_display,
    polar_reproject,
    save_colorbar_png,
    save_heatmap_png,
    save_ipf_legend,
    save_overlay_figure,
)
from .fingerprints import FingerprintResult, compute_radial_fingerprints
from .indexing import IndexingCandidate, ROIIndexingResult, run_stage2_indexing
from .loaders import load_dataset
from .logging import configure_pipeline_logging, get_logger
from .masks import build_annular_masks
from .orientation import OrientationResult, run_orientation_preview
from .phase import PhaseScreeningResult, screen_phases
from .pipeline import PipelineResult, PipelineStageRecord, run_pipeline
from .preprocess import PreprocessedArray, PreprocessSpec, apply_preprocess
from .preprocess_raw import bin_and_export, crop_navigation_and_export
from .pyxem_validation import run_stage2c_validation
from .roi_bragg import ROIBraggResult, Stage2Result, load_roi_candidates, run_roi_bragg_for_rois
from .stage2 import load_stage1_manifest, run_stage2
from .consensus import run_consensus
from .synthetic import make_synthetic_4dstem
from .virtual import VirtualImageResult, compute_virtual_images
from .diagnostics import run_stage1_diagnostics
from .workflow import WorkflowResult, run_workflow

__all__ = [
    "DataContract",
    "DatasetHandle",
    "DiffractionCalibration",
    "DiffractionSignal",
    "FingerprintResult",
    "IndexingCandidate",
    "OrientationMap",
    "OrientationResult",
    "PhaseScreeningResult",
    "PipelineResult",
    "PipelineStageRecord",
    "PolarSignal",
    "PreprocessedArray",
    "PreprocessSpec",
    "ROIBraggResult",
    "ROIIndexingResult",
    "Stage1Manifest",
    "Stage1ManifestLoadError",
    "Stage2Result",
    "VirtualImageResult",
    "WorkflowResult",
    "apply_gamma",
    "apply_ipf_colors",
    "apply_preprocess",
    "bin_and_export",
    "build_annular_masks",
    "compute_radial_fingerprints",
    "crop_navigation_and_export",
    "compute_virtual_images",
    "configure_pipeline_logging",
    "get_logger",
    "is_roi_ready_for_indexing",
    "roi_indexing_blockers",
    "roi_indexing_readiness",
    "load_dataset",
    "load_roi_candidates",
    "load_stage1_manifest",
    "load_workflow_config",
    "make_synthetic_4dstem",
    "mask_center_for_display",
    "polar_reproject",
    "resolve_data_config",
    "run_orientation_preview",
    "run_pipeline",
    "run_consensus",
    "run_roi_bragg_for_rois",
    "run_stage1_diagnostics",
    "run_stage2",
    "run_stage2_indexing",
    "run_stage2c_validation",
    "run_workflow",
    "save_colorbar_png",
    "save_heatmap_png",
    "save_ipf_legend",
    "save_overlay_figure",
    "screen_phases",
    "validate_workflow_config",
]
