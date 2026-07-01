#!/usr/bin/env python
"""Parameter stability sweep for Stage 2B crystallographic indexing.

Runs ``run_stage2_indexing`` across a grid of template-generation parameters
and reports which candidate phase wins for each ROI under each combination.

Usage::

    python scripts/run_stage2b_sweep.py --config configs/pipeline.yaml
    python scripts/run_stage2b_sweep.py --config configs/pipeline.yaml \\
        --output-dir outputs/sweep --peak-sigma 3,4,5,6 --orient-step 5,2,1
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the source package is importable when run as a script.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from fourdstem_pipeline.indexing import run_stage2_indexing  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_SWEEP_GRID: dict[str, list[Any]] = {
    "peak_sigma_px": [3.0, 4.0, 5.0, 6.0],
    "orientation_step_deg": [5.0, 2.0, 1.0],
    "reciprocal_pixels_per_inv_angstrom": [53.1, 55.9, 58.7],
}

P1_EVIDENCE_SWEEP_GRID: dict[str, list[Any]] = {
    "k_max": [1.4, 1.7, 2.0],
    "reciprocal_scale_factor": [0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03],
    "beam_center_offset_yx": [(0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0)],
    "bragg_threshold_mode": ["conservative", "medium"],
}


def _parse_sweep_grid(raw: dict[str, str]) -> dict[str, list[Any]]:
    """Convert CLI string values to typed lists."""
    grid: dict[str, list[Any]] = {}
    type_map = {
        "peak_sigma_px": float,
        "orientation_step_deg": float,
        "reciprocal_pixels_per_inv_angstrom": float,
    }
    for key, val_str in raw.items():
        convert = type_map.get(key, float)
        grid[key] = [convert(v.strip()) for v in val_str.split(",") if v.strip()]
    return grid


def _grid_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of grid values → list of parameter dicts."""
    keys = list(grid.keys())
    if not keys:
        return [{}]
    values = [grid[k] for k in keys]
    combos: list[dict[str, Any]] = []
    _cartesian(keys, values, 0, {}, combos)
    return combos


def _cartesian(
    keys: list[str],
    values: list[list[Any]],
    depth: int,
    current: dict[str, Any],
    result: list[dict[str, Any]],
) -> None:
    if depth == len(keys):
        result.append(dict(current))
        return
    for v in values[depth]:
        current[keys[depth]] = v
        _cartesian(keys, values, depth + 1, current, result)
        current.pop(keys[depth])


def run_sweep(
    base_config_path: Path,
    sweep_grid: dict[str, list[Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Run the parameter sweep and return a stability report.

    Parameters
    ----------
    base_config_path:
        Path to the base Stage 2B YAML config.
    sweep_grid:
        Dict mapping parameter names to lists of values.
    output_dir:
        Directory for per-run subdirectories and the sweep report.

    Returns
    -------
    dict with keys ``grid_runs``, ``stability_matrix``, ``summary``.
    """
    import yaml

    base_cfg = _load_stage2b_sweep_config(base_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    combos = _grid_combinations(sweep_grid)
    log.info("Sweep grid: %d parameter combinations", len(combos))
    for key, vals in sweep_grid.items():
        log.info("  %s: %s", key, vals)

    grid_runs: list[dict[str, Any]] = []
    roi_stability: dict[str, dict[str, int]] = {}  # roi_name → {candidate: count}

    for idx, combo in enumerate(combos):
        label = "_".join(f"{k}={v}" for k, v in combo.items())
        run_dir = output_dir / f"run_{idx:03d}_{label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Deep-copy base config and override sweep params
        cfg = copy.deepcopy(base_cfg)
        cfg["output_dir"] = str(run_dir / "stage2b_indexing")
        tg = cfg.setdefault("template_generation", {})
        for key, val in combo.items():
            if key == "reciprocal_scale_factor":
                base_scale = float(tg.get("reciprocal_pixels_per_inv_angstrom", 55.9) or 55.9)
                tg["reciprocal_pixels_per_inv_angstrom"] = base_scale * float(val)
            elif key == "k_max":
                tg["k_max"] = float(val)
                tg["reciprocal_radius"] = float(val)
            elif key == "beam_center_offset_yx":
                cfg["stage2_dir"] = str(_stage2_dir_with_beam_offset(Path(cfg["stage2_dir"]), run_dir, val))
            elif key == "bragg_threshold_mode":
                cfg.setdefault("evidence_sweep", {})["bragg_threshold_mode"] = str(val)
            else:
                tg[key] = val
            # orientation_step_deg controls the step; regenerate orientations
            if key == "orientation_step_deg":
                tg["orientations_deg"] = None  # force auto-generation
            if key == "peak_sigma_px":
                tg["peak_sigma_px"] = float(val)

        try:
            summary = run_stage2_indexing(cfg)
        except Exception as exc:
            log.error("Sweep run %d (%s) failed: %s", idx, label, exc)
            grid_runs.append({
                "idx": idx,
                "label": label,
                "params": combo,
                "error": str(exc),
            })
            continue

        # Extract per-ROI winners
        roi_results = summary.get("roi_results", [])
        winners: dict[str, dict[str, Any]] = {}
        for r in roi_results:
            roi_name = str(r.get("name", "unknown"))
            candidate = r.get("candidate_phase") or "none"
            winners[roi_name] = {
                "candidate": candidate,
                "phase_call": r.get("phase_call"),
                "score": r.get("match_score"),
                "margin": r.get("score_margin"),
                "confidence": r.get("phase_confidence"),
                "matched_frac": r.get("matched_template_fraction"),
                "radial_support": r.get("radial_support_score"),
                "q_residual": r.get("mean_q_residual"),
            }

        grid_runs.append({
            "idx": idx,
            "label": label,
            "params": combo,
            "winners": winners,
        })

        # Accumulate stability counts
        for roi_name, info in winners.items():
            if roi_name not in roi_stability:
                roi_stability[roi_name] = {}
            cand = info["candidate"]
            roi_stability[roi_name][cand] = roi_stability[roi_name].get(cand, 0) + 1

    # --- Build stability report -----------------------------------------------
    stability_report: dict[str, Any] = {}
    for roi_name, counts in sorted(roi_stability.items()):
        total = sum(counts.values())
        winner = max(counts, key=counts.get) if counts else "none"
        stability_report[roi_name] = {
            "total_runs": total,
            "winner": winner,
            "winner_fraction": round(counts[winner] / total, 3) if total > 0 else 0.0,
            "counts": counts,
            "stable": len(counts) == 1,
        }

    all_stable = all(v["stable"] for v in stability_report.values())
    total_rois = len(stability_report)
    stable_rois = sum(1 for v in stability_report.values() if v["stable"])
    unanimous = all(
        v["winner"] == list(stability_report.values())[0]["winner"]
        for v in stability_report.values()
    ) if stability_report else False

    summary_stats = {
        "n_combos": len(combos),
        "n_succeeded": sum(1 for r in grid_runs if "error" not in r),
        "n_failed": sum(1 for r in grid_runs if "error" in r),
        "n_rois": total_rois,
        "stable_rois": stable_rois,
        "all_stable": all_stable,
        "unanimous_winner": unanimous,
        "dominant_candidate": (
            list(stability_report.values())[0]["winner"]
            if unanimous and stability_report else None
        ),
        "best_calibration_candidate": _best_calibration_candidate(grid_runs),
        "best_K_MAX": _best_param_value(grid_runs, "k_max"),
        "stability_of_phase_call": {
            name: info["winner_fraction"] for name, info in stability_report.items()
        },
        "sensitivity_flags": (
            ["PARAMETER_SENSITIVE_LOW_CONFIDENCE"]
            if any(not info["stable"] or info["winner_fraction"] < 0.8 for info in stability_report.values())
            else []
        ),
    }

    # --- Write sweep outputs --------------------------------------------------
    sweep_summary = {
        "sweep_grid": {k: [str(v) for v in vals] for k, vals in sweep_grid.items()},
        "summary": summary_stats,
        "stability": stability_report,
        "grid_runs": grid_runs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(sweep_summary, indent=2, default=str), encoding="utf-8")
    log.info("Sweep summary written: %s", summary_path)

    # --- Print report ---------------------------------------------------------
    _print_stability_report(summary_stats, stability_report, sweep_grid)

    return sweep_summary


def _print_stability_report(
    stats: dict[str, Any],
    stability: dict[str, Any],
    grid: dict[str, list[Any]],
) -> None:
    """Print a human-readable stability report to stdout."""
    total_params = 1
    for vals in grid.values():
        total_params *= len(vals)

    print()
    print("=" * 72)
    print("  Stage 2B Parameter Stability Sweep")
    print("=" * 72)
    print(f"  Combinations: {total_params} ({stats['n_succeeded']} OK, {stats['n_failed']} failed)")
    print(f"  ROIs analysed: {stats['n_rois']}")
    print(f"  Stable ROIs:   {stats['stable_rois']} / {stats['n_rois']}")
    if stats["unanimous_winner"]:
        print(f"  Winner:        {stats['dominant_candidate']} (unanimous across all ROIs)")
    print()

    if stability:
        print(f"  {'ROI':<30s} {'Winner':<18s} {'Frac':>6s}  {'Distribution'}")
        print(f"  {'─'*30} {'─'*18} {'─'*6}  {'─'*40}")
        for roi_name, info in stability.items():
            frac = f"{info['winner_fraction']:.1%}"
            dist = ", ".join(
                f"{cand}:{cnt}" for cand, cnt in sorted(info["counts"].items(), key=lambda x: -x[1])
            )
            print(f"  {roi_name:<30s} {info['winner']:<18s} {frac:>6s}  {dist}")
        print()

    flagged = [n for n, v in stability.items() if not v["stable"]]
    if flagged:
        print(f"  ⚠  {len(flagged)} ROI(s) changed winner across parameters:")
        for name in flagged:
            info = stability[name]
            print(f"     - {name}: {info['counts']}")
        print()
    else:
        print("  ✓  All ROIs stable — same winner across all parameter combinations.")
        print()


def _load_stage2b_sweep_config(config_path: Path) -> dict[str, Any]:
    """Load Stage 2B config, extracting it from unified pipeline YAML."""
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg).__name__}.")
    if "stage2b" not in cfg:
        return cfg

    stage2b = dict(cfg.get("stage2b") or {})
    project_cfg = cfg.get("project") or {}
    stage2a_cfg = cfg.get("stage2a") or {}
    stage1_dir = Path(project_cfg.get("output_dir", "outputs"))
    stage2a_dir = Path(stage2a_cfg.get("output_dir") or stage1_dir / "stage2" / "roi_bragg")
    stage2b.setdefault("stage2_dir", str(stage2a_dir))
    stage2b.setdefault("output_dir", None)
    return stage2b


def _stage2_dir_with_beam_offset(stage2_dir: Path, run_dir: Path, offset: Any) -> Path:
    """Create a temporary Stage 2A summary with beam center offset applied."""
    if not stage2_dir.is_absolute():
        stage2_dir = Path.cwd() / stage2_dir
    dy, dx = [float(v) for v in offset]
    src = stage2_dir / "stage2_summary.json"
    dst_dir = run_dir / "stage2a_beam_offset"
    dst_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(src.read_text(encoding="utf-8"))
    for roi in summary.get("roi_results", []):
        if roi.get("beam_center_yx"):
            roi["beam_center_yx"] = [
                float(roi["beam_center_yx"][0]) + dy,
                float(roi["beam_center_yx"][1]) + dx,
            ]
    summary["beam_center_sweep_offset_yx"] = [dy, dx]
    (dst_dir / "stage2_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return dst_dir


def _best_param_value(grid_runs: list[dict[str, Any]], key: str) -> Any:
    """Return the parameter value from the run with strongest median evidence."""
    scored: list[tuple[tuple[float, float, float], Any]] = []
    for run in grid_runs:
        if "error" in run or key not in run.get("params", {}):
            continue
        winners = list((run.get("winners") or {}).values())
        if not winners:
            continue
        radial = _median([w.get("radial_support") for w in winners])
        matched = _median([w.get("matched_frac") for w in winners])
        margin = _median([w.get("margin") for w in winners])
        scored.append(((radial, matched, margin), run["params"][key]))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _best_calibration_candidate(grid_runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for run in grid_runs:
        if "error" in run:
            continue
        params = run.get("params", {})
        winners = list((run.get("winners") or {}).values())
        if not winners:
            continue
        score = (
            _median([w.get("radial_support") for w in winners]),
            _median([w.get("matched_frac") for w in winners]),
            _median([w.get("margin") for w in winners]),
        )
        scored.append((score, {
            "reciprocal_scale_factor": params.get("reciprocal_scale_factor"),
            "beam_center_offset_yx": params.get("beam_center_offset_yx"),
            "bragg_threshold_mode": params.get("bragg_threshold_mode"),
        }))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _median(values: list[Any]) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(np.median(vals)) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2B parameter stability sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="configs/pipeline.yaml",
        help="Unified or Stage 2B config (default: configs/pipeline.yaml)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Sweep output directory (default: <stage2_dir>/stage2b_sweep/)",
    )
    parser.add_argument(
        "--peak-sigma", default=None,
        help="Comma-separated peak_sigma_px values (default: 3,4,5,6)",
    )
    parser.add_argument(
        "--orient-step", default=None,
        help="Comma-separated orientation_step_deg values (default: 5,2,1)",
    )
    parser.add_argument(
        "--recip-scale", default=None,
        help="Comma-separated reciprocal_pixels_per_inv_angstrom values (default: 53.1,55.9,58.7)",
    )
    parser.add_argument(
        "--p1-evidence-grid", action="store_true",
        help="Use P1 grid: K_MAX 1.4/1.7/2.0, q calibration +/-1-3%, beam center +/-1-2 px, Bragg threshold labels.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Build sweep grid from CLI or defaults
    sweep_grid: dict[str, list[Any]] = {}
    if args.p1_evidence_grid:
        sweep_grid = dict(P1_EVIDENCE_SWEEP_GRID)
    elif args.peak_sigma:
        sweep_grid["peak_sigma_px"] = [float(v.strip()) for v in args.peak_sigma.split(",")]
    if args.orient_step:
        sweep_grid["orientation_step_deg"] = [float(v.strip()) for v in args.orient_step.split(",")]
    if args.recip_scale:
        sweep_grid["reciprocal_pixels_per_inv_angstrom"] = [float(v.strip()) for v in args.recip_scale.split(",")]
    if not sweep_grid:
        sweep_grid = dict(DEFAULT_SWEEP_GRID)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        base_cfg = _load_stage2b_sweep_config(config_path)
        stage2_dir = Path(base_cfg["stage2_dir"])
        if not stage2_dir.is_absolute():
            stage2_dir = Path.cwd() / stage2_dir
        output_dir = stage2_dir / "stage2b_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_sweep(config_path, sweep_grid, output_dir)


if __name__ == "__main__":
    main()
