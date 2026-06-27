"""Stage-1 diagnostics orchestrator.

Delegates to sub-modules:
- ``diagnostics_cluster`` -- cluster mean DPs, radial profiles, virtual stats,
  normalisation comparison, K-sweep.
- ``diagnostics_spatial`` -- beam centre, connected components, orientation
  reliability, ROI candidates, overlays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DatasetHandle
from .diagnostics_cluster import (
    cluster_orientation_table,
    cluster_mean_diffraction,
    cluster_radial_profiles,
    cluster_virtual_statistics,
    interleave_mean_std,
    k_sweep,
    normalisation_comparison,
    ring_ratio_maps,
    stats_bar_values,
    write_cluster_summary,
)
from .diagnostics_spatial import (
    beam_diagnostics,
    connected_component_diagnostics,
    orientation_reliability,
    roi_candidates,
)
from .export import save_bar_png, save_lines_png, save_png
from .fingerprints import FingerprintResult
from .orientation import OrientationResult
from .phase import PhaseScreeningResult
from .virtual import VirtualImageResult


def run_stage1_diagnostics(
    dataset: DatasetHandle,
    fingerprints: FingerprintResult,
    phase: PhaseScreeningResult,
    virtual: VirtualImageResult,
    orientation: OrientationResult,
    *,
    output_dir: str | Path,
    png_dir: str | Path,
    block_shape: tuple[int, int],
    confidence_threshold: float,
) -> dict[str, Any]:
    """Run the full suite of stage-1 diagnostics and return output paths."""
    output_dir = Path(output_dir)
    png_dir = Path(png_dir)
    cluster_dir = output_dir / "05_cluster_diagnostics"
    roi_dir = output_dir / "roi_candidates"
    class_dir = output_dir / "fingerprint_classes"
    orientation_dir = output_dir / "orientation"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    labels = phase.labels
    cluster_ids = sorted(int(v) for v in np.unique(labels) if v >= 0)

    # --- Cluster mean diffraction patterns ---------------------------------
    cluster_mean_dps = cluster_mean_diffraction(dataset, labels, cluster_ids, block_shape)
    np.save(cluster_dir / "cluster_mean_dps.npy", cluster_mean_dps)
    for idx, cluster_id in enumerate(cluster_ids):
        save_png(png_dir / f"cluster_mean_dp_{cluster_id}.png", cluster_mean_dps[idx])
        save_png(cluster_dir / f"cluster_mean_dp_{cluster_id}.png", cluster_mean_dps[idx])
        save_png(png_dir / f"cluster_mean_dp_log_{cluster_id}.png", np.log1p(cluster_mean_dps[idx]))
        save_png(cluster_dir / f"cluster_mean_dp_log_{cluster_id}.png", np.log1p(cluster_mean_dps[idx]))

    # --- Cluster radial profiles -------------------------------------------
    radial_mean, radial_std = cluster_radial_profiles(fingerprints.profiles, labels, cluster_ids)
    np.save(cluster_dir / "cluster_mean_radial_profiles.npy", radial_mean)
    np.save(class_dir / "cluster_mean_radial_profiles.npy", radial_mean)
    np.save(cluster_dir / "cluster_radial_profile_std.npy", radial_std)
    save_lines_png(png_dir / "cluster_mean_radial_profiles.png", fingerprints.radii, radial_mean)
    save_lines_png(cluster_dir / "cluster_mean_radial_profiles.png", fingerprints.radii, radial_mean)
    save_lines_png(png_dir / "cluster_radial_profile_mean_std.png", fingerprints.radii, interleave_mean_std(radial_mean, radial_std))
    save_lines_png(cluster_dir / "cluster_radial_profile_mean_std.png", fingerprints.radii, interleave_mean_std(radial_mean, radial_std))

    # --- Cluster virtual-image statistics ----------------------------------
    stats_rows = cluster_virtual_statistics(labels, virtual, cluster_ids)
    write_cluster_summary(cluster_dir, stats_rows)
    write_cluster_summary(class_dir, stats_rows)
    save_bar_png(png_dir / "cluster_virtual_image_statistics.png", stats_bar_values(stats_rows))
    save_bar_png(cluster_dir / "cluster_virtual_image_statistics.png", stats_bar_values(stats_rows))
    ring_ratio_outputs = ring_ratio_maps(virtual, cluster_dir, png_dir)
    cluster_orientation_outputs = cluster_orientation_table(labels, orientation, cluster_ids, cluster_dir, png_dir)

    # --- Normalisation comparison & K-sweep --------------------------------
    norm_outputs = normalisation_comparison(fingerprints.profiles, len(cluster_ids), class_dir, png_dir)
    k_sweep_outputs = k_sweep(fingerprints.profiles, [2, 3, 4, 5, 6, 8], class_dir, png_dir)

    # --- Spatial diagnostics -----------------------------------------------
    beam_outputs = beam_diagnostics(virtual, png_dir, output_dir / "00_preprocess")
    component_outputs = connected_component_diagnostics(labels, virtual.images, cluster_ids, cluster_dir, png_dir, class_dir)
    orientation_outputs = orientation_reliability(orientation, confidence_threshold, png_dir, orientation_dir)
    roi_outputs = roi_candidates(labels, virtual.images, orientation, cluster_ids, roi_dir, png_dir)

    return {
        "cluster_diagnostics": str(cluster_dir),
        "roi_candidates": str(roi_dir),
        "cluster_summary_csv": str(cluster_dir / "cluster_summary.csv"),
        "cluster_summary_md": str(cluster_dir / "cluster_summary.md"),
        "ring_ratio_maps": ring_ratio_outputs,
        "cluster_vs_orientation": cluster_orientation_outputs,
        "normalization_comparison": norm_outputs,
        "k_sweep": k_sweep_outputs,
        "beam": beam_outputs,
        "connected_components": component_outputs,
        "orientation_reliability": orientation_outputs,
        "roi_outputs": roi_outputs,
    }
