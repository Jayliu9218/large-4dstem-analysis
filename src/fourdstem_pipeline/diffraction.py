"""Lightweight pyxem-style signal wrappers for 4D-STEM diffraction data.

Provides :class:`DiffractionSignal`, :class:`PolarSignal`, and
:class:`OrientationMap` — thin wrappers that bundle data arrays with
:class:`DiffractionCalibration` metadata and expose method-chaining
conveniences similar to pyxem's ``Diffraction2D`` / ``PolarDiffraction2D`` /
``OrientationMap`` classes.

These are *optional* notebook-friendly façades.  The existing imperative
pipeline (``run_workflow``, ``run_stage2``, ``run_stage2_indexing``) does
not depend on them and continues to work with plain arrays and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import DiffractionCalibration


# ---------------------------------------------------------------------------
# DiffractionSignal — wraps a 2D or 4D diffraction array + calibration
# ---------------------------------------------------------------------------


@dataclass
class DiffractionSignal:
    """A 2D diffraction pattern or 4D diffraction stack with calibration.

    Analogous to pyxem's ``Diffraction2D``.  For a 4D stack the first two
    axes are navigation (scan y, scan x) and the last two are signal
    (detector y, detector x).

    Parameters
    ----------
    data:
        NumPy array, shape ``(qy, qx)`` for a single pattern or
        ``(ny, nx, qy, qx)`` for a stack.
    calibration:
        Beam centre and reciprocal-scale metadata.
    """

    data: np.ndarray
    calibration: DiffractionCalibration

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def is_4d(self) -> bool:
        return self.data.ndim == 4

    @property
    def signal_shape(self) -> tuple[int, int]:
        if self.data.ndim == 4:
            return (int(self.data.shape[2]), int(self.data.shape[3]))
        if self.data.ndim == 2:
            return (int(self.data.shape[0]), int(self.data.shape[1]))
        raise ValueError(f"Cannot determine signal shape from {self.data.ndim}D data.")

    @property
    def navigation_shape(self) -> tuple[int, int] | None:
        if self.data.ndim == 4:
            return (int(self.data.shape[0]), int(self.data.shape[1]))
        return None

    # -- Transforms -----------------------------------------------------------

    def get_mean_pattern(self) -> "DiffractionSignal":
        """Return the mean diffraction pattern averaged over navigation."""
        if self.data.ndim == 4:
            data = np.asarray(self.data, dtype=np.float32)
            mean = data.mean(axis=(0, 1), dtype=np.float32)
            return DiffractionSignal(mean, self.calibration)
        return DiffractionSignal(np.asarray(self.data, dtype=np.float32).copy(), self.calibration)

    def get_polar(
        self, npt: int = 100, npt_azim: int = 360, *, mean: bool = True,
    ) -> "PolarSignal":
        """Reproject the mean pattern to polar coordinates.

        Analogous to pyxem's ``Diffraction2D.get_azimuthal_integral2d()``.
        """
        from .export import polar_reproject

        pattern = self.get_mean_pattern().data
        polar = polar_reproject(
            pattern,
            npt=npt,
            npt_azim=npt_azim,
            center_yx=self.calibration.beam_center_yx,
            mean=mean,
        )
        return PolarSignal(data=polar, calibration=self.calibration, npt_azim=npt_azim, npt_radial=npt)

    def get_radial_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(radii, intensity)`` for the mean 1D radial profile."""
        pattern = self.get_mean_pattern().data
        cy, cx = self.calibration.beam_center_yx
        h, w = pattern.shape
        yy, xx = np.indices(pattern.shape, dtype=np.float32)
        radii = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        bins = np.floor(radii).astype(np.int32)
        max_bin = int(bins.max())
        sums = np.bincount(bins.ravel(), weights=np.nan_to_num(pattern, nan=0.0).ravel(), minlength=max_bin + 1)
        counts = np.bincount(bins.ravel(), minlength=max_bin + 1)
        profile = sums / np.maximum(counts, 1)
        return np.arange(max_bin + 1, dtype=np.float32), profile.astype(np.float32)

    def apply_gamma(self, gamma: float = 0.5) -> "DiffractionSignal":
        """Apply power-law gamma correction (in-place on data)."""
        from .export import apply_gamma

        self.data = apply_gamma(np.asarray(self.data, dtype=np.float32), gamma)
        return self

    def mask_center(self, radius_px: float = 35.0, *, outer_radius_px: float | None = None) -> "DiffractionSignal":
        """Zero out the direct-beam disk for display (returns new signal)."""
        from .export import mask_center_for_display

        if self.data.ndim == 4:
            raise ValueError("mask_center operates on a single 2D pattern; call get_mean_pattern() first.")
        masked = mask_center_for_display(
            self.data, center_yx=self.calibration.beam_center_yx,
            radius_px=radius_px, outer_radius_px=outer_radius_px,
        )
        return DiffractionSignal(masked, self.calibration)

    # -- I/O ------------------------------------------------------------------

    def save_png(self, path: str | Path, **kwargs: Any) -> Path:
        """Save the mean pattern as a PNG via :func:`fourdstem_pipeline.export.save_png`."""
        from .export import save_png as _save

        return _save(path, self.get_mean_pattern().data, **kwargs)

    def plot(self, path: str | Path | None = None, **kwargs: Any) -> Path | None:
        """Quick-look PNG of the mean diffraction pattern.

        If *path* is ``None`` a temp-file path is returned (caller must clean up).
        """
        from .export import save_png as _save

        if path is None:
            import tempfile, os
            path = Path(tempfile.mkdtemp()) / "diffraction_signal.png"
        kwargs.setdefault("cmap", "viridis")
        return _save(path, self.get_mean_pattern().data, **kwargs)


# ---------------------------------------------------------------------------
# PolarSignal — result of polar reprojection
# ---------------------------------------------------------------------------


@dataclass
class PolarSignal:
    """A polar-reprojected diffraction pattern.

    Analogous to pyxem's ``PolarDiffraction2D``.  The data array has shape
    ``(npt_azim, npt_radial)`` where rows are azimuthal slices and columns
    are radial bins.

    Parameters
    ----------
    data:
        ``(npt_azim, npt_radial)`` float32 array.
    calibration:
        Original calibration metadata.
    npt_azim:
        Number of azimuthal bins.
    npt_radial:
        Number of radial bins.
    """

    data: np.ndarray
    calibration: DiffractionCalibration
    npt_azim: int = 360
    npt_radial: int = 100

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    def apply_gamma(self, gamma: float = 0.5) -> "PolarSignal":
        """Apply gamma correction in-place."""
        from .export import apply_gamma

        self.data = apply_gamma(self.data, gamma)
        return self

    def get_orientation(
        self,
        templates: np.ndarray,
        *,
        n_best: int = -1,
        frac_keep: float = 1.0,
    ) -> "OrientationMap":
        """Match polar data against a template stack.

        Analogous to pyxem's ``PolarDiffraction2D.get_orientation()``.

        Parameters
        ----------
        templates:
            ``(n_templates, npt_azim, npt_radial)`` float32 template stack.
        n_best:
            Number of best matches to keep per probe position.  ``-1`` means
            keep all.
        frac_keep:
            Fraction of correlation values to retain (1.0 = all).
        """
        from .export import apply_gamma as _gamma

        n_templates = templates.shape[0]
        if templates.shape[1:] != self.shape:
            raise ValueError(
                f"Template shape {templates.shape[1:]} does not match "
                f"polar shape {self.shape}."
            )

        # Normalise templates and data.
        tmpl_flat = templates.reshape(n_templates, -1).astype(np.float32)
        tmpl_mean = tmpl_flat.mean(axis=1, keepdims=True, dtype=np.float32)
        tmpl_std = tmpl_flat.std(axis=1, keepdims=True, dtype=np.float32)
        tmpl_std = np.maximum(tmpl_std, 1e-12)
        tmpl_norm = (tmpl_flat - tmpl_mean) / tmpl_std

        data_flat = np.asarray(self.data, dtype=np.float32).ravel()
        data_mean = float(data_flat.mean())
        data_std = max(float(data_flat.std()), 1e-12)
        data_norm = (data_flat - data_mean) / data_std

        scores = tmpl_norm @ data_norm
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        orientation_index = np.array([best_idx]) if n_best < 0 else np.argsort(scores)[::-1][:n_best]
        score = np.array([best_score]) if n_best < 0 else scores[orientation_index]

        return OrientationMap(
            orientation_index=np.asarray(orientation_index, dtype=np.int16),
            score=np.asarray(score, dtype=np.float32),
            phase_label=np.zeros(len(orientation_index), dtype=np.int16),
            calibration=self.calibration,
            template_metadata={"n_templates": n_templates, "best_score": best_score},
        )

    def plot(self, path: str | Path | None = None, **kwargs: Any) -> Path | None:
        """Quick-look PNG of the polar representation."""
        from .export import save_png as _save

        if path is None:
            import tempfile
            path = Path(tempfile.mkdtemp()) / "polar_signal.png"
        kwargs.setdefault("cmap", "viridis")
        return _save(path, self.data, **kwargs)

    def save_png(self, path: str | Path, **kwargs: Any) -> Path:
        """Save the polar data as a PNG."""
        from .export import save_png as _save

        return _save(path, self.data, **kwargs)


# ---------------------------------------------------------------------------
# OrientationMap — template-matching orientation result
# ---------------------------------------------------------------------------


@dataclass
class OrientationMap:
    """Template-matching orientation result for one or more probe positions.

    Analogous to pyxem's ``OrientationMap``.  Carries the best-match
    orientation index, correlation score, and optional phase label per
    probe position.

    Parameters
    ----------
    orientation_index:
        ``(N,)`` int16 array of best-matching template indices.
    score:
        ``(N,)`` float32 array of correlation scores.
    phase_label:
        ``(N,)`` int16 array of phase labels (0 if single-phase).
    calibration:
        Diffraction calibration metadata.
    template_metadata:
        Arbitrary dict with template-generation parameters.
    """

    orientation_index: np.ndarray
    score: np.ndarray
    phase_label: np.ndarray
    calibration: DiffractionCalibration
    template_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.orientation_index)
        if len(self.score) != n:
            raise ValueError("orientation_index and score must have the same length.")
        if len(self.phase_label) != n:
            raise ValueError("orientation_index and phase_label must have the same length.")

    @property
    def best_score(self) -> float:
        return float(np.max(self.score)) if self.score.size > 0 else 0.0

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.score)) if self.score.size > 0 else -1

    def to_ipf_colors(self, directions_xyz: np.ndarray | None = None) -> np.ndarray:
        """Map orientations to cubic-IPF RGB colours.

        If *directions_xyz* is not provided, returns a fixed grayscale.
        """
        from .export import apply_ipf_colors

        if directions_xyz is not None:
            return apply_ipf_colors(directions_xyz)
        # Fallback: grayscale by score.
        norm = self.score / max(float(np.max(self.score)), 1e-12)
        gray = np.clip(norm * 255, 0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    def plot_over_signal(
        self,
        signal: np.ndarray,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> Path | None:
        """Render the orientation map overlaid on the diffraction signal.

        Analogous to pyxem's ``OrientationMap.plot_over_signal()``.
        """
        from .export import save_overlay_figure

        if path is None:
            import tempfile
            path = Path(tempfile.mkdtemp()) / "orientation_overlay.png"
        kwargs.setdefault("cmap", "viridis")
        kwargs.setdefault("center_mask_radius", 35.0)
        return save_overlay_figure(path, signal, overlays=[], **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "best_index": self.best_index,
            "best_score": self.best_score,
            "n_orientations": len(self.orientation_index),
            "calibration": self.calibration.to_dict(),
            "template_metadata": self.template_metadata,
        }
