#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pyxem + HyperSpy Ti-bcc / Ti-hcp phase identification and orientation mapping.

Purpose
-------
This script is intended as an independent pattern-matching validation route for
4D-STEM data, complementary to py4DSTEM Bragg-vector matching.

Pipeline
--------
1. Read 4D-STEM data with HyperSpy, preferably lazily.
2. Optionally crop a navigation ROI, useful for tile-wise or representative ROI runs.
3. Set reciprocal-space calibration if known.
4. Convert diffraction patterns to polar coordinates with pyxem.
5. Optionally subtract radial background and apply gamma correction.
6. Build Ti-bcc and Ti-hcp simulated diffraction libraries from CIF files.
7. Run pyxem PolarDiffraction2D.get_orientation().
8. Export best phase, correlation, phase-margin, rotation, and QC masks.

Notes
-----
- This is a pattern-matching route: it compares experimental diffraction patterns
  with simulated diffraction templates. It is not the same as py4DSTEM Bragg-vector
  matching.
- The phase-index mapping assumes that diffsims returns simulations in the same
  order as [Ti-bcc rotations, Ti-hcp rotations]. This is the behavior expected from
  the standard pyxem/diffsims multi-phase example, but the script saves enough raw
  output to audit the result.
- For strict Ti-bcc vs Ti-hcp discrimination, use n_best >= 8 initially. If the
  second phase is not represented among the returned candidates, the phase margin
  is marked NaN and the pixel is treated as ambiguous unless --allow-missing-opponent
  is set.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Avoid OpenMP/BLAS oversubscription when using dask processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hyperspy.api as hs
import pyxem as pxm  # noqa: F401  # imported to register pyxem signal classes

from diffpy.structure import loadStructure
from diffsims.generators.simulation_generator import SimulationGenerator
from orix.crystal_map import Phase
from orix.sampling import get_sample_reduced_fundamental


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="pyxem + HyperSpy Ti-bcc / Ti-hcp pattern-matching phase/orientation mapping"
    )
    p.add_argument("--input", required=True, type=Path, help="Input MIB/HDF5/HSPY/ZSPY file readable by HyperSpy.")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    p.add_argument("--cif-bcc", required=True, type=Path, help="Ti-bcc CIF path.")
    p.add_argument("--cif-hcp", required=True, type=Path, help="Ti-hcp CIF path.")

    p.add_argument("--beam-energy-kv", type=float, default=200.0, help="Accelerating voltage in kV.")
    p.add_argument("--inv-ang-per-pixel", type=float, default=None,
                   help="Reciprocal calibration, A^-1 per detector pixel. Strongly recommended.")
    p.add_argument("--center", nargs=2, type=float, default=None, metavar=("QX0", "QY0"),
                   help="Direct beam center in detector pixels. If omitted, pyxem/HyperSpy calibration.center is set to None.")

    p.add_argument("--roi-yx", "--roi", dest="roi_yx", nargs=4, type=int, default=None, metavar=("Y0", "Y1", "X0", "X1"),
                   help="Navigation ROI in project order y0 y1 x0 x1. Example: --roi-yx 224 288 224 288. Uses s.inav[y0:y1, x0:x1].")
    p.add_argument("--chunk-nav", type=int, default=16,
                   help="Dask navigation chunk size, e.g. 8, 16, or 32. Used only for lazy dask arrays.")

    p.add_argument("--n-workers", type=int, default=0,
                   help="Start a local dask cluster with this many process workers. 0 disables explicit cluster creation.")
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--memory-limit", default="auto")
    p.add_argument("--dashboard-address", default=":8787")

    p.add_argument("--npt", type=int, default=160, help="Number of radial bins for polar transform.")
    p.add_argument("--npt-azim", type=int, default=360, help="Number of azimuthal bins for polar transform.")
    p.add_argument("--background", choices=["none", "radial_median", "radial_percentile"],
                   default="radial_percentile")
    p.add_argument("--background-percentile", type=float, default=20.0)
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Gamma correction exponent applied to polar pattern. 0.5 is sqrt correction.")

    p.add_argument("--orientation-resolution-deg", type=float, default=1.0,
                   help="Sampling resolution for reduced fundamental zone rotations.")
    p.add_argument("--minimum-intensity", type=float, default=0.03,
                   help="Minimum simulated reflection intensity retained in diffsims.")
    p.add_argument("--max-excitation-error", type=float, default=0.06)
    p.add_argument("--reciprocal-radius", type=float, default=1.8,
                   help="Maximum reciprocal radius for simulated templates in A^-1.")
    p.add_argument("--n-best", type=int, default=8,
                   help="Number of returned best template candidates per probe position. Use >=8 for robust bcc/hcp margin.")
    p.add_argument("--frac-keep", type=float, default=0.05)
    p.add_argument("--margin-threshold", type=float, default=0.15,
                   help="Minimum bcc-vs-hcp phase correlation margin for high-confidence assignment.")
    p.add_argument("--min-correlation", type=float, default=None,
                   help="Optional minimum best correlation threshold. If omitted, only margin is used.")
    p.add_argument("--allow-missing-opponent", action="store_true",
                   help="If set, pixels where the opposite phase is absent from top-n candidates are not automatically ambiguous.")

    p.add_argument("--save-polar", action="store_true", help="Save polar signal if supported by current HyperSpy version.")
    p.add_argument("--no-compute", action="store_true", help="Do not explicitly compute lazy results before saving arrays.")
    return p.parse_args()


def maybe_start_dask(args: argparse.Namespace):
    if args.n_workers <= 0:
        return None
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
        memory_limit=args.memory_limit,
        dashboard_address=args.dashboard_address,
    )
    client = Client(cluster)
    print(client)
    return client


def ensure_single_signal(obj):
    """HyperSpy hs.load may return a list for some formats."""
    if isinstance(obj, (list, tuple)):
        if len(obj) != 1:
            raise RuntimeError(f"hs.load returned {len(obj)} signals. Please select the diffraction signal manually.")
        return obj[0]
    return obj


def set_signal_type_safely(s, signal_type: str = "electron_diffraction"):
    try:
        s.set_signal_type(signal_type)
    except Exception as exc:
        print(f"[warning] Could not set signal type to {signal_type!r}: {exc}")
    return s


def crop_navigation_roi(s, roi: Optional[List[int]]):
    if roi is None:
        return s
    y0, y1, x0, x1 = roi
    print(f"Cropping navigation ROI: y={y0}:{y1}, x={x0}:{x1}")
    return s.inav[y0:y1, x0:x1]


def rechunk_navigation(s, chunk_nav: int):
    data = getattr(s, "data", None)
    if data is None or not hasattr(data, "rechunk"):
        return s
    if data.ndim < 4:
        return s
    # HyperSpy stores navigation dimensions before signal dimensions for typical 4D data.
    # Keep each diffraction pattern whole; chunk only navigation dimensions.
    chunks = tuple([chunk_nav] * (data.ndim - 2) + [data.shape[-2], data.shape[-1]])
    try:
        s.data = data.rechunk(chunks)
        print(f"Rechunked data to chunks={chunks}")
    except Exception as exc:
        print(f"[warning] Rechunk failed: {exc}")
    return s


def set_reciprocal_calibration(s, inv_ang_per_pixel: Optional[float]):
    if inv_ang_per_pixel is None:
        print("[warning] No reciprocal calibration supplied. Polar radial axis may be in pixels, not A^-1.")
        return s
    try:
        for ax in s.axes_manager.signal_axes:
            ax.scale = inv_ang_per_pixel
            ax.units = "A^-1"
        print(f"Set signal-axis reciprocal calibration = {inv_ang_per_pixel} A^-1 / pixel")
    except Exception as exc:
        print(f"[warning] Failed to set reciprocal calibration: {exc}")
    return s


def set_center(s, center: Optional[Tuple[float, float]]):
    try:
        if center is None:
            s.calibration.center = None
            print("Set s.calibration.center = None")
        else:
            s.calibration.center = tuple(center)
            print(f"Set s.calibration.center = {center}")
    except Exception as exc:
        print(f"[warning] Could not set s.calibration.center: {exc}")
    return s


def make_phase_from_cif(cif_path: Path, name: str, color: str) -> Phase:
    """Create an orix Phase from CIF with fallbacks across orix versions."""
    cif_path = Path(cif_path)
    if not cif_path.exists():
        raise FileNotFoundError(cif_path)

    # Newer orix versions may expose Phase.from_cif.
    if hasattr(Phase, "from_cif"):
        try:
            phase = Phase.from_cif(str(cif_path))
            phase.name = name
            phase.color = color
            if getattr(phase, "point_group", None) is None:
                raise RuntimeError("Phase.from_cif returned phase without point_group")
            return phase
        except Exception as exc:
            print(f"[warning] Phase.from_cif failed for {name}: {exc}")

    struct = loadStructure(str(cif_path))
    try:
        phase = Phase(name=name, structure=struct)
    except TypeError:
        phase = Phase(name=name)
        phase.structure = struct

    try:
        phase.color = color
    except Exception:
        pass

    if getattr(phase, "point_group", None) is None:
        raise RuntimeError(
            f"Could not infer point_group for {name} from {cif_path}. "
            "Check the CIF symmetry fields or construct an orix Phase manually for your version."
        )
    return phase


def build_simulation(args: argparse.Namespace):
    bcc = make_phase_from_cif(args.cif_bcc, "Ti-bcc", "blue")
    hcp = make_phase_from_cif(args.cif_hcp, "Ti-hcp", "red")

    generator = SimulationGenerator(
        args.beam_energy_kv,
        minimum_intensity=args.minimum_intensity,
    )

    rot_bcc = get_sample_reduced_fundamental(
        resolution=args.orientation_resolution_deg,
        point_group=bcc.point_group,
    )
    rot_hcp = get_sample_reduced_fundamental(
        resolution=args.orientation_resolution_deg,
        point_group=hcp.point_group,
    )

    print(f"Ti-bcc rotations: {len(rot_bcc)}")
    print(f"Ti-hcp rotations: {len(rot_hcp)}")

    sim = generator.calculate_diffraction2d(
        [bcc, hcp],
        rotation=[rot_bcc, rot_hcp],
        max_excitation_error=args.max_excitation_error,
        reciprocal_radius=args.reciprocal_radius,
        with_direct_beam=False,
    )

    phase_lookup = np.concatenate([
        np.zeros(len(rot_bcc), dtype=np.int16),
        np.ones(len(rot_hcp), dtype=np.int16),
    ])
    phase_names = np.array(["Ti-bcc", "Ti-hcp"])
    return sim, phase_lookup, phase_names, {"Ti-bcc": len(rot_bcc), "Ti-hcp": len(rot_hcp)}


def polar_transform(s, args: argparse.Namespace):
    print("Converting to polar diffraction signal...")
    polar = s.get_azimuthal_integral2d(
        npt=args.npt,
        npt_azim=args.npt_azim,
        inplace=False,
        mean=True,
    )

    if args.background == "radial_median":
        print("Subtracting radial median background...")
        polar = polar.subtract_diffraction_background(method="radial median", inplace=False)
    elif args.background == "radial_percentile":
        print(f"Subtracting radial percentile background, percentile={args.background_percentile}...")
        polar = polar.subtract_diffraction_background(
            method="radial percentile",
            percentile=args.background_percentile,
            inplace=False,
        )

    if args.gamma != 1.0:
        print(f"Applying gamma correction: polar ** {args.gamma}")
        polar = polar ** args.gamma
    return polar


def compute_if_lazy(signal, label: str):
    if hasattr(signal, "compute"):
        print(f"Computing lazy result: {label}")
        try:
            out = signal.compute()
            return signal if out is None else out
        except TypeError:
            try:
                signal.compute(progressbar=True)
                return signal
            except Exception:
                raise
    return signal


def normalize_orientation_array(arr: np.ndarray, n_best: int) -> np.ndarray:
    """Return orientation data as shape nav..., n_best, 4."""
    arr = np.asarray(arr)
    if arr.shape[-2:] == (n_best, 4):
        return arr
    if arr.shape[-2:] == (4, n_best):
        return np.swapaxes(arr, -1, -2)
    if arr.shape[-1] == 4:
        return arr
    raise ValueError(f"Unexpected orientation_map.data shape {arr.shape}; cannot locate (n_best, 4) columns.")


def topn_to_phase_maps(
    orientation_data: np.ndarray,
    phase_lookup: np.ndarray,
    phase_names: np.ndarray,
    margin_threshold: float,
    min_correlation: Optional[float],
    allow_missing_opponent: bool,
) -> Dict[str, np.ndarray]:
    """Convert top-n template results to phase/correlation/margin/QC maps."""
    nav_shape = orientation_data.shape[:-2]
    n_best = orientation_data.shape[-2]
    n_phases = len(phase_names)

    template_index = orientation_data[..., :, 0].astype(np.int64)
    correlation = orientation_data[..., :, 1].astype(np.float32)
    rotation = orientation_data[..., :, 2].astype(np.float32)
    factor = orientation_data[..., :, 3].astype(np.float32)

    valid_index = (template_index >= 0) & (template_index < len(phase_lookup))
    candidate_phase = np.full(template_index.shape, -1, dtype=np.int16)
    candidate_phase[valid_index] = phase_lookup[template_index[valid_index]]

    phase_best_corr = np.full(nav_shape + (n_phases,), -np.inf, dtype=np.float32)
    phase_best_rank = np.full(nav_shape + (n_phases,), -1, dtype=np.int16)

    # Top-n is small; a Python loop over n_best is fine and easier to audit.
    for rank in range(n_best):
        ph = candidate_phase[..., rank]
        corr = correlation[..., rank]
        for pidx in range(n_phases):
            m = ph == pidx
            better = m & (corr > phase_best_corr[..., pidx])
            phase_best_corr[..., pidx] = np.where(better, corr, phase_best_corr[..., pidx])
            phase_best_rank[..., pidx] = np.where(better, rank, phase_best_rank[..., pidx])

    best_phase = np.argmax(phase_best_corr, axis=-1).astype(np.int16)
    best_phase_corr = np.max(phase_best_corr, axis=-1)

    # For two phases, phase margin is best - other. Generalized via sorting finite phase correlations.
    sorted_corr = np.sort(phase_best_corr, axis=-1)
    second_phase_corr = sorted_corr[..., -2] if n_phases >= 2 else np.full(nav_shape, np.nan, dtype=np.float32)
    phase_margin = best_phase_corr - second_phase_corr

    missing_opponent = ~np.isfinite(second_phase_corr) | (second_phase_corr == -np.inf)
    phase_margin = phase_margin.astype(np.float32)
    phase_margin[missing_opponent] = np.nan

    best_rank = np.take_along_axis(phase_best_rank, best_phase[..., None], axis=-1)[..., 0]
    best_template_index = np.take_along_axis(template_index, best_rank[..., None], axis=-1)[..., 0]
    best_rotation = np.take_along_axis(rotation, best_rank[..., None], axis=-1)[..., 0]
    best_factor = np.take_along_axis(factor, best_rank[..., None], axis=-1)[..., 0]

    ambiguous = np.zeros(nav_shape, dtype=bool)
    ambiguous |= ~np.isfinite(best_phase_corr)
    ambiguous |= np.isnan(phase_margin)
    ambiguous |= phase_margin < margin_threshold
    if min_correlation is not None:
        ambiguous |= best_phase_corr < min_correlation
    if allow_missing_opponent:
        ambiguous &= ~missing_opponent

    high_confidence = ~ambiguous

    return {
        "template_index_topn": template_index,
        "correlation_topn": correlation,
        "candidate_phase_topn": candidate_phase,
        "best_phase_index": best_phase,
        "best_phase_name": phase_names[best_phase],
        "best_template_index": best_template_index,
        "best_correlation": best_phase_corr.astype(np.float32),
        "second_phase_correlation": second_phase_corr.astype(np.float32),
        "phase_margin": phase_margin.astype(np.float32),
        "missing_opponent_phase_in_topn": missing_opponent,
        "best_rotation_deg": best_rotation.astype(np.float32),
        "best_factor": best_factor.astype(np.float32),
        "ambiguous_mask": ambiguous,
        "high_confidence_mask": high_confidence,
        "phase_best_correlation_stack": phase_best_corr.astype(np.float32),
    }


def savefig(path: Path, dpi: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


def plot_scalar(arr: np.ndarray, title: str, path: Path, cmap: str = "viridis"):
    plt.figure(figsize=(5, 4))
    im = plt.imshow(np.asarray(arr).T, origin="lower", interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.xlabel("scan x")
    plt.ylabel("scan y")
    plt.colorbar(im, shrink=0.8)
    savefig(path)


def plot_phase_map(phase_idx: np.ndarray, high_conf: np.ndarray, phase_names: Iterable[str], path: Path):
    labels = list(phase_names) + ["AMBIGUOUS"]
    display = np.asarray(phase_idx, dtype=np.int16).copy()
    display[~high_conf] = len(labels) - 1
    plt.figure(figsize=(5, 4))
    im = plt.imshow(display.T, origin="lower", interpolation="nearest", vmin=0, vmax=len(labels) - 1)
    plt.title("pyxem QC-filtered Ti phase map")
    plt.xlabel("scan x")
    plt.ylabel("scan y")
    cbar = plt.colorbar(im, ticks=np.arange(len(labels)))
    cbar.ax.set_yticklabels(labels)
    savefig(path)


def plot_hist(arr: np.ndarray, title: str, path: Path, bins: int = 80):
    vals = np.asarray(arr).ravel()
    vals = vals[np.isfinite(vals)]
    plt.figure(figsize=(5, 4))
    if vals.size:
        plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("count")
    savefig(path)


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(type(obj).__name__)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = maybe_start_dask(args)

    print(f"HyperSpy version: {hs.__version__}")
    try:
        import pyxem
        print(f"pyxem version: {pyxem.__version__}")
    except Exception:
        pass

    print(f"Loading: {args.input}")
    s = ensure_single_signal(hs.load(args.input, lazy=True))
    s = set_signal_type_safely(s)
    s = crop_navigation_roi(s, args.roi_yx)
    s = rechunk_navigation(s, args.chunk_nav)
    s = set_reciprocal_calibration(s, args.inv_ang_per_pixel)
    s = set_center(s, tuple(args.center) if args.center is not None else None)

    print(s)
    print(s.axes_manager)

    sim, phase_lookup, phase_names, n_templates_by_phase = build_simulation(args)
    polar = polar_transform(s, args)

    if args.save_polar:
        try:
            polar.save(args.out_dir / "polar_signal.zspy", overwrite=True)
            print(f"[saved] {args.out_dir / 'polar_signal.zspy'}")
        except Exception as exc:
            print(f"[warning] Could not save polar signal: {exc}")

    print("Running pyxem pattern matching with get_orientation()...")
    orientation_map = polar.get_orientation(
        sim,
        n_best=args.n_best,
        frac_keep=args.frac_keep,
        normalize_templates=True,
    )

    if not args.no_compute:
        orientation_map = compute_if_lazy(orientation_map, "orientation_map")

    raw_orientation = np.asarray(orientation_map.data)
    orientation_data = normalize_orientation_array(raw_orientation, args.n_best)
    maps = topn_to_phase_maps(
        orientation_data=orientation_data,
        phase_lookup=phase_lookup,
        phase_names=phase_names,
        margin_threshold=args.margin_threshold,
        min_correlation=args.min_correlation,
        allow_missing_opponent=args.allow_missing_opponent,
    )

    npz_path = args.out_dir / "pyxem_ti_phase_orientation_results.npz"
    np.savez_compressed(
        npz_path,
        phase_names=phase_names,
        phase_lookup=phase_lookup,
        n_templates_by_phase=np.array([n_templates_by_phase["Ti-bcc"], n_templates_by_phase["Ti-hcp"]]),
        orientation_data=orientation_data.astype(np.float32),
        **{k: v for k, v in maps.items() if k != "best_phase_name"},
    )
    print(f"[saved] {npz_path}")

    # Plots
    plot_phase_map(
        maps["best_phase_index"],
        maps["high_confidence_mask"],
        phase_names,
        args.out_dir / "phase_map_pyxem_qc_filtered.png",
    )
    plot_scalar(maps["best_phase_index"], "pyxem raw best phase index", args.out_dir / "phase_map_pyxem_raw_best_phase.png")
    plot_scalar(maps["best_correlation"], "pyxem best phase correlation", args.out_dir / "pyxem_best_correlation.png")
    plot_scalar(maps["phase_margin"], "pyxem Ti-bcc/Ti-hcp phase margin", args.out_dir / "pyxem_phase_margin.png")
    plot_scalar(maps["ambiguous_mask"].astype(float), "pyxem ambiguous mask", args.out_dir / "pyxem_ambiguous_mask.png", cmap="gray")
    plot_scalar(maps["missing_opponent_phase_in_topn"].astype(float), "opponent phase missing in top-n", args.out_dir / "pyxem_missing_opponent_mask.png", cmap="gray")
    plot_hist(maps["best_correlation"], "pyxem best correlation", args.out_dir / "hist_pyxem_best_correlation.png")
    plot_hist(maps["phase_margin"], "pyxem phase margin", args.out_dir / "hist_pyxem_phase_margin.png")

    total = int(np.prod(maps["best_phase_index"].shape))
    high = maps["high_confidence_mask"]
    summary = {
        "settings": vars(args),
        "n_templates_by_phase": n_templates_by_phase,
        "phase_names": phase_names.tolist(),
        "total_pixels": total,
        "confidence_summary": {
            "high_confidence_fraction": float(np.sum(high) / total),
            "ambiguous_fraction": float(np.sum(maps["ambiguous_mask"]) / total),
            "missing_opponent_phase_in_topn_fraction": float(np.sum(maps["missing_opponent_phase_in_topn"]) / total),
            "best_correlation_mean": float(np.nanmean(maps["best_correlation"])),
            "best_correlation_median": float(np.nanmedian(maps["best_correlation"])),
            "phase_margin_mean": float(np.nanmean(maps["phase_margin"])),
            "phase_margin_median": float(np.nanmedian(maps["phase_margin"])),
        },
        "phase_results": [],
    }
    for pidx, pname in enumerate(phase_names):
        raw = maps["best_phase_index"] == pidx
        qc = raw & high
        summary["phase_results"].append({
            "phase": str(pname),
            "raw_winning_fraction": float(np.sum(raw) / total),
            "qc_high_confidence_fraction_of_all_pixels": float(np.sum(qc) / total),
            "qc_fraction_within_high_confidence_pixels": float(np.sum(qc) / max(np.sum(high), 1)),
            "best_phase_correlation_mean": float(np.nanmean(maps["phase_best_correlation_stack"][..., pidx])),
            "best_phase_correlation_median": float(np.nanmedian(maps["phase_best_correlation_stack"][..., pidx])),
        })

    summary_path = args.out_dir / "pyxem_ti_phase_orientation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=json_safe)
    print(f"[saved] {summary_path}")

    if client is not None:
        client.close()

    print("Done.")


if __name__ == "__main__":
    main()
