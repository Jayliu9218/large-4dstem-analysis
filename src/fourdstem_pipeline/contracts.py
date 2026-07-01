"""Unified data contract and coordinate conventions for the 4D-STEM pipeline.

All modules MUST follow these conventions to avoid x/y or bbox-order
confusion downstream (Stage 2 ROI, py4DSTEM, PNG overlays, orientation
preview, etc.).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

AxisOrder = Literal["nav_y_nav_x_q_y_q_x"]
BBoxOrder = Literal["y0_y1_x0_x1"]
CenterOrder = Literal["y_x"]
ROIReadiness = Literal[
    "READY_FOR_INDEXING",
    "READY_FOR_SCREENING_ONLY",
    "NOT_INDEXABLE",
]


@dataclass(frozen=True, slots=True)
class DiffractionCalibration:
    """Beam centre and reciprocal-space calibration for a diffraction pattern.

    Bundles the metadata that pyxem carries in ``signal.calibration`` so it
    flows through the pipeline as a single, validated object rather than
    scattered ``beam_center_yx`` tuples and config strings.

    Attributes
    ----------
    beam_center_yx:
        Direct-beam position in detector pixels, ``(cy, cx)`` order.
    reciprocal_pixels_per_inv_angstrom:
        Reciprocal-space scale in px/Å.  Computed from camera length,
        wavelength, and pixel size when those are known; otherwise set
        directly from calibration.
    pixel_size_um:
        Detector pixel size in µm (optional).
    camera_length_mm:
        Camera length in mm (optional).
    accelerating_voltage_kv:
        TEM accelerating voltage in kV (default 200).
    """

    beam_center_yx: tuple[float, float]
    reciprocal_pixels_per_inv_angstrom: float | None = None
    pixel_size_um: float | None = None
    camera_length_mm: float | None = None
    accelerating_voltage_kv: float = 200.0

    @property
    def center_y(self) -> float:
        """Beam centre y-coordinate in detector pixels."""
        return self.beam_center_yx[0]

    @property
    def center_x(self) -> float:
        """Beam centre x-coordinate in detector pixels."""
        return self.beam_center_yx[1]

    @property
    def wavelength_angstrom(self) -> float:
        """Relativistically corrected electron wavelength in Å."""
        kv = max(self.accelerating_voltage_kv, 0.1)
        h = 6.62607015e-34
        m_e = 9.10938356e-31
        e = 1.602176634e-19
        c = 2.99792458e8
        V = kv * 1e3
        denom = math.sqrt(2 * m_e * e * V * (1 + e * V / (2 * m_e * c * c)))
        return float(h / denom * 1e10)  # metres → Å

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (suitable for JSON summaries)."""
        return {
            "beam_center_yx": list(self.beam_center_yx),
            "reciprocal_pixels_per_inv_angstrom": self.reciprocal_pixels_per_inv_angstrom,
            "pixel_size_um": self.pixel_size_um,
            "camera_length_mm": self.camera_length_mm,
            "accelerating_voltage_kv": self.accelerating_voltage_kv,
        }

    @classmethod
    def from_stage2_geometry(cls, geometry: dict[str, Any]) -> "DiffractionCalibration":
        """Build from the ``geometry`` dict in a Stage 2A/B summary."""
        return cls(
            beam_center_yx=(
                float(geometry["beam_center_yx"][0]),
                float(geometry["beam_center_yx"][1]),
            ),
            reciprocal_pixels_per_inv_angstrom=geometry.get("reciprocal_pixels_per_inv_angstrom"),
        )


def roi_indexing_blockers(roi: dict[str, Any]) -> list[str]:
    """Return evidence-quality blockers for using a Stage 2A ROI in indexing."""
    issues: list[str] = []
    bq = roi.get("bragg_qc") or {}
    n_peaks = int(roi.get("n_bragg_peaks", 0) or 0)
    bg_frac = roi.get("background_fraction")
    sample_cov = roi.get("sample_mask_coverage")
    beam_source = roi.get("beam_center_source", "")

    median_clean = bq.get("median_clean_peaks_per_DP")
    frac_ge6 = bq.get("fraction_DP_with_>=6_peaks")
    edge_frac = float(bq.get("edge_peak_fraction", 0.0) or 0.0)
    center_frac = float(
        bq.get("center_tail_peak_fraction", bq.get("forbidden_center_zone_fraction", 0.0)) or 0.0
    )
    splitting = bool(bq.get("peak_splitting_warning", False))

    if roi.get("error"):
        issues.append(f"ROI failed: {roi.get('error')}")
    if n_peaks <= 0:
        issues.append("zero Bragg peaks")
    if bg_frac is not None and bg_frac > 0.5:
        issues.append(f"high background ({bg_frac:.1%})")
    if sample_cov is not None and sample_cov == 0.0:
        issues.append("zero sample coverage")
    if beam_source == "detector_center_fallback":
        issues.append("no calibrated beam center")
    if median_clean is None:
        issues.append("missing per-DP clean peak metric")
    elif float(median_clean) < 6.0:
        issues.append(f"median clean peaks per DP < 6 ({float(median_clean):.2f})")
    if frac_ge6 is None:
        issues.append("missing fraction DP with >=6 peaks")
    elif float(frac_ge6) < 0.5:
        issues.append(f"fraction DP with >=6 peaks < 0.5 ({float(frac_ge6):.3f})")
    if splitting:
        issues.append("per-DP duplicate/peak-splitting warning")
    if edge_frac > 0.3:
        issues.append(f"high edge peak fraction ({edge_frac:.1%})")
    if center_frac > 0.3:
        issues.append(f"high center-tail peak fraction ({center_frac:.1%})")
    return issues


def roi_indexing_readiness(roi: dict[str, Any]) -> ROIReadiness:
    """Classify Stage 2A evidence quality for downstream indexing."""
    explicit = roi.get("indexing_readiness")
    if explicit in ("READY_FOR_INDEXING", "READY_FOR_SCREENING_ONLY", "NOT_INDEXABLE"):
        return explicit
    blockers = roi_indexing_blockers(roi)
    if not blockers:
        return "READY_FOR_INDEXING"
    if roi.get("error"):
        return "NOT_INDEXABLE"
    if int(roi.get("n_bragg_peaks", 0) or 0) <= 0:
        return "NOT_INDEXABLE"
    bq = roi.get("bragg_qc") or {}
    severe_artifact = (
        bool(bq.get("peak_splitting_warning", False))
        or float(bq.get("edge_peak_fraction", 0.0) or 0.0) > 0.5
        or float(bq.get("center_tail_peak_fraction", bq.get("forbidden_center_zone_fraction", 0.0)) or 0.0) > 0.5
    )
    bg_frac = roi.get("background_fraction")
    sample_cov = roi.get("sample_mask_coverage")
    if severe_artifact or (bg_frac is not None and bg_frac > 0.5) or sample_cov == 0.0:
        return "NOT_INDEXABLE"
    return "READY_FOR_SCREENING_ONLY"


def is_roi_ready_for_indexing(roi: dict[str, Any]) -> bool:
    """Return True only for Stage 2A ROIs with indexing-grade evidence."""
    return roi_indexing_readiness(roi) == "READY_FOR_INDEXING"


@dataclass
class DataContract:
    """Coordinate conventions used throughout the pipeline.

    Attributes
    ----------
    axis_order:
        4D data shape convention: ``nav_y_nav_x_q_y_q_x`` means
        ``[navigation_y, navigation_x, detector_q_y, detector_q_x]``.
    bbox_order:
        Bounding-box list convention: ``y0_y1_x0_x1`` means
        ``[y_start, y_end, x_start, x_end]``.
    center_order:
        Point / centre convention: ``y_x`` means ``[y, x]``
        (row-major, consistent with ``np.argwhere``).
    array_shape:
        Optional resolved 4D shape ``(nav_y, nav_x, q_y, q_x)``.
    nav_shape:
        Optional 2D navigation shape ``(nav_y, nav_x)``.
    sig_shape:
        Optional 2D signal shape ``(q_y, q_x)``.
    """

    axis_order: AxisOrder = "nav_y_nav_x_q_y_q_x"
    bbox_order: BBoxOrder = "y0_y1_x0_x1"
    center_order: CenterOrder = "y_x"
    array_shape: tuple[int, int, int, int] | None = None
    nav_shape: tuple[int, int] | None = None
    sig_shape: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, str | list[int] | None]:
        """Serialize to a plain dict suitable for JSON / YAML reports."""
        result: dict[str, str | list[int] | None] = {
            "axis_order": self.axis_order,
            "bbox_order": self.bbox_order,
            "center_order": self.center_order,
        }
        if self.array_shape is not None:
            result["array_shape"] = list(self.array_shape)
        if self.nav_shape is not None:
            result["nav_shape"] = list(self.nav_shape)
        if self.sig_shape is not None:
            result["sig_shape"] = list(self.sig_shape)
        return result


# ---------------------------------------------------------------------------
# Stage 1 manifest — validated bridge from Stage 1 to Stage 2
# ---------------------------------------------------------------------------


# Required keys in a valid stage1_summary.json.  Path-valued keys are
# resolved relative to the manifest directory and are checked for existence.
_REQUIRED_STAGE1_KEYS: set[str] = {
    "run_name",
    "nav_shape",
    "sig_shape",
    "labels_path",
    "roi_candidates_path",
    "qc_status",
    "virtual_images_path",
    "fingerprints_path",
    "radial_axis_path",
    "data_contract_path",
    "provenance_path",
    "qc_summary_path",
}

# Keys whose values are file paths that MUST exist on disk (relative to the
# manifest directory).
_REQUIRED_STAGE1_PATHS: set[str] = {
    "labels_path",
    "roi_candidates_path",
    "virtual_images_path",
    "fingerprints_path",
    "radial_axis_path",
    "data_contract_path",
    "provenance_path",
    "qc_summary_path",
}


class Stage1ManifestLoadError(ValueError):
    """Raised when a ``stage1_summary.json`` fails validation."""


@dataclass
class Stage1Manifest:
    """Validated bridge from Stage 1 to Stage 2.

    Loaded from a ``stage1_summary.json`` file.  All path-valued attributes
    are resolved to absolute paths so downstream code never needs to worry
    about the manifest base directory.

    Raises :class:`Stage1ManifestLoadError` on any validation failure.
    """

    # Raw manifest dictionary (paths are still relative strings here).
    raw: dict[str, Any] = field(repr=False)

    # Absolute directory containing the manifest file.
    stage1_dir: Path

    # Core metadata
    run_name: str
    nav_shape: list[int]
    sig_shape: list[int]
    qc_status: str

    # Resolved absolute paths to required outputs
    labels_path: Path
    roi_candidates_path: Path
    virtual_images_path: Path
    fingerprints_path: Path
    radial_axis_path: Path
    data_contract_path: Path
    provenance_path: Path
    qc_summary_path: Path

    # Optional paths (may be None)
    preprocessed_shape: list[int] | None = None
    orientation_index_path: Path | None = None
    orientation_score_path: Path | None = None
    cluster_summary_path: Path | None = None
    cluster_mean_radial_profiles_path: Path | None = None
    preprocess_info_path: Path | None = None

    # Preprocessing parameters
    q_crop: list[int] | None = None
    q_bin: int = 1
    r_bin: int = 1

    # Dependency info (optional but useful for Stage 2)
    dependencies: dict[str, Any] = field(default_factory=dict)

    # Errors encountered during validation that are non-fatal (warnings).
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, stage1_dir: str | Path) -> "Stage1Manifest":
        """Load and validate a ``stage1_summary.json`` from *stage1_dir*.

        Parameters
        ----------
        stage1_dir:
            Path to a Stage-1 output directory containing
            ``stage1_summary.json``.

        Returns
        -------
        Stage1Manifest
            A validated manifest with all paths resolved to absolute form.

        Raises
        ------
        Stage1ManifestLoadError
            If the manifest is missing, unreadable, has missing required
            keys, references missing files, reports a FAIL QC status, or
            violates coordinate conventions.
        """
        stage1_dir = Path(stage1_dir).resolve()
        manifest_path = stage1_dir / "stage1_summary.json"
        if not manifest_path.exists():
            raise Stage1ManifestLoadError(
                f"stage1_summary.json not found in {stage1_dir}"
            )

        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise Stage1ManifestLoadError(
                f"stage1_summary.json is not valid JSON: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise Stage1ManifestLoadError(
                "stage1_summary.json must contain a JSON object (dict)."
            )

        # --- Check required keys -------------------------------------------
        missing = _REQUIRED_STAGE1_KEYS - set(raw)
        if missing:
            raise Stage1ManifestLoadError(
                f"stage1_summary.json is missing required keys: {sorted(missing)}"
            )

        # --- QC status gate ------------------------------------------------
        qc_status = raw.get("qc_status", "UNKNOWN")
        if qc_status == "FAIL":
            # Gather critical flags from QC summary for better diagnostics.
            critical_details = ""
            qc_path = stage1_dir / raw.get("qc_summary_path", "qc_summary.json")
            if qc_path.exists():
                try:
                    qc_data = json.loads(qc_path.read_text(encoding="utf-8"))
                    critical_flags = [
                        f.get("code", "?") for f in qc_data.get("flags", [])
                        if f.get("severity") == "critical"
                    ]
                    if critical_flags:
                        critical_details = f" Critical flags: {critical_flags}"
                except (json.JSONDecodeError, OSError):
                    pass
            raise Stage1ManifestLoadError(
                f"Stage 1 QC status is FAIL.  The Stage 1 run did not pass "
                f"quality control and its outputs should not be used for "
                f"Stage 2 analysis.{critical_details}"
            )

        # --- Resolve & check required paths --------------------------------
        resolved: dict[str, Path | None] = {}
        warnings: list[str] = []
        for key in sorted(_REQUIRED_STAGE1_PATHS):
            rel = raw[key]
            if not rel:
                raise Stage1ManifestLoadError(
                    f"Required path '{key}' is empty in stage1_summary.json."
                )
            p = stage1_dir / rel
            if not p.exists():
                raise Stage1ManifestLoadError(
                    f"Required file '{rel}' (key '{key}') does not exist in "
                    f"{stage1_dir}."
                )
            resolved[key] = p

        # --- Validate data contract conventions ----------------------------
        dc_path = resolved["data_contract_path"]
        try:
            dc = json.loads(dc_path.read_text(encoding="utf-8"))
            if dc.get("bbox_order") != "y0_y1_x0_x1":
                raise Stage1ManifestLoadError(
                    f"Data contract bbox_order is '{dc.get('bbox_order')}', "
                    f"but Stage 2 requires 'y0_y1_x0_x1'."
                )
        except json.JSONDecodeError as exc:
            raise Stage1ManifestLoadError(
                f"Failed to read data_contract.json: {exc}"
            ) from exc

        # --- Resolve optional paths ----------------------------------------
        optional_path_keys = [
            "orientation_index_path",
            "orientation_score_path",
            "cluster_summary_path",
            "cluster_mean_radial_profiles_path",
            "preprocess_info_path",
        ]
        for key in optional_path_keys:
            rel = raw.get(key)
            if rel:
                p = stage1_dir / rel
                resolved[key] = p if p.exists() else None
                if p.exists() is False and rel:
                    warnings.append(f"Optional file '{rel}' (key '{key}') not found.")
            else:
                resolved[key] = None

        # --- Build manifest ------------------------------------------------
        return cls(
            raw=raw,
            stage1_dir=stage1_dir,
            run_name=raw["run_name"],
            nav_shape=raw["nav_shape"],
            sig_shape=raw["sig_shape"],
            qc_status=qc_status,
            labels_path=resolved["labels_path"],  # type: ignore[arg-type]
            roi_candidates_path=resolved["roi_candidates_path"],  # type: ignore[arg-type]
            virtual_images_path=resolved["virtual_images_path"],  # type: ignore[arg-type]
            fingerprints_path=resolved["fingerprints_path"],  # type: ignore[arg-type]
            radial_axis_path=resolved["radial_axis_path"],  # type: ignore[arg-type]
            data_contract_path=resolved["data_contract_path"],  # type: ignore[arg-type]
            provenance_path=resolved["provenance_path"],  # type: ignore[arg-type]
            qc_summary_path=resolved["qc_summary_path"],  # type: ignore[arg-type]
            preprocessed_shape=raw.get("preprocessed_shape"),
            orientation_index_path=resolved.get("orientation_index_path"),
            orientation_score_path=resolved.get("orientation_score_path"),
            cluster_summary_path=resolved.get("cluster_summary_path"),
            cluster_mean_radial_profiles_path=resolved.get("cluster_mean_radial_profiles_path"),
            preprocess_info_path=resolved.get("preprocess_info_path"),
            q_crop=raw.get("q_crop"),
            q_bin=int(raw.get("q_bin", 1)),
            r_bin=int(raw.get("r_bin", 1)),
            dependencies=raw.get("dependencies", {}),
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def has_orientation(self) -> bool:
        """True when orientation preview outputs are available."""
        return self.orientation_index_path is not None and self.orientation_index_path.exists()

    @property
    def qc_passed(self) -> bool:
        """True when QC status is PASS or PASS_WITH_WARNINGS."""
        return self.qc_status in ("PASS", "PASS_WITH_WARNINGS")

    @property
    def pyxem_available(self) -> bool:
        """True when pyxem was available at Stage 1 load time."""
        return bool(self.dependencies.get("pyxem_available", False))

    @property
    def py4DSTEM_used(self) -> bool:
        """True when py4DSTEM was used during Stage 1."""
        return bool(self.dependencies.get("py4DSTEM_used", False))

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path relative to the manifest directory."""
        return self.stage1_dir / relative
