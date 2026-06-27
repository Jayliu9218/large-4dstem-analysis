"""Unified data contract and coordinate conventions for the 4D-STEM pipeline.

All modules MUST follow these conventions to avoid x/y or bbox-order
confusion downstream (Stage 2 ROI, py4DSTEM, PNG overlays, orientation
preview, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

AxisOrder = Literal["nav_y_nav_x_q_y_q_x"]
BBoxOrder = Literal["y0_y1_x0_x1"]
CenterOrder = Literal["y_x"]


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
