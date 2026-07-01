"""P0 evidence trustworthiness diagnostics for Stage 2A/2B outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .export import save_png
from .indexing import run_stage2_indexing


INV_ANG_SWEEP_FACTORS = (0.97, 0.99, 1.0, 1.01, 1.03)
BEAM_CENTER_OFFSETS = ((0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0))
MANUAL_LABELS = ("GOOD_PEAKS", "NOISY_OR_DUPLICATE_PEAKS", "INSUFFICIENT_PEAKS")


@dataclass(frozen=True)
class CalibrationSweepPoint:
    inv_ang_per_pixel: float
    beam_center_offset_yx: tuple[float, float]


def calibration_sweep_grid(
    base_inv_ang_per_pixel: float = 0.0192,
    offsets: tuple[tuple[float, float], ...] = BEAM_CENTER_OFFSETS,
) -> list[CalibrationSweepPoint]:
    """Return the fixed P0 calibration sweep grid."""
    return [
        CalibrationSweepPoint(
            inv_ang_per_pixel=round(float(base_inv_ang_per_pixel) * float(factor), 8),
            beam_center_offset_yx=(float(dy), float(dx)),
        )
        for factor in INV_ANG_SWEEP_FACTORS
        for dy, dx in offsets
    ]


def select_best_calibration_run(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the sweep row with best radial/peak evidence."""
    if not rows:
        return None

    def score(row: dict[str, Any]) -> tuple[float, float, float, float]:
        radial = float(row.get("radial_support_median") or 0.0)
        matched = float(row.get("matched_template_fraction_median") or 0.0)
        unexplained = float(row.get("unexplained_peak_fraction_median") or 1.0)
        residual = float(row.get("q_residual_median") or 1e9)
        return (radial, matched, -unexplained, -residual)

    return max(rows, key=score)


def run_evidence_qc(
    *,
    stage2_dir: Path,
    stage2b_dir: Path | None,
    config_path: Path | None,
    output_dir: Path,
    base_inv_ang_per_pixel: float = 0.0192,
    samples_per_roi: int = 20,
) -> dict[str, Any]:
    """Run P0 diagnostics and write a report directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2_summary_path = stage2_dir / "stage2_summary.json"
    stage2_summary = json.loads(stage2_summary_path.read_text(encoding="utf-8"))
    stage2b_summary = _load_stage2b_summary(stage2b_dir)

    overlay_rows = _write_overlay_samples(
        stage2_summary=stage2_summary,
        output_dir=output_dir / "bragg_overlay_samples",
        samples_per_roi=samples_per_roi,
        base_inv_ang_per_pixel=base_inv_ang_per_pixel,
    )
    sweep_rows = _run_calibration_sweep(
        stage2_summary=stage2_summary,
        stage2_dir=stage2_dir,
        config_path=config_path,
        output_dir=output_dir / "calibration_sweep",
        base_inv_ang_per_pixel=base_inv_ang_per_pixel,
    )
    best = select_best_calibration_run(sweep_rows)
    baseline = _baseline_sweep_like_row(stage2b_summary)
    recommendation = _calibration_recommendation(baseline, best)

    summary = {
        "schema_version": "p0-evidence-qc-v1",
        "stage2_dir": str(stage2_dir),
        "stage2b_dir": str(stage2b_dir) if stage2b_dir else None,
        "overlay_samples": overlay_rows,
        "calibration_sweep": {
            "grid_size": len(calibration_sweep_grid(base_inv_ang_per_pixel)),
            "rows": sweep_rows,
            "best": best,
            "baseline": baseline,
            "recommendation": recommendation,
        },
        "manual_label_categories": list(MANUAL_LABELS),
    }
    (output_dir / "p0_evidence_qc_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    _write_markdown_report(output_dir / "p0_evidence_qc_report.md", summary)
    return summary


def _load_stage2b_summary(stage2b_dir: Path | None) -> dict[str, Any] | None:
    if stage2b_dir is None:
        return None
    path = stage2b_dir / "stage2_indexing_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_overlay_samples(
    *,
    stage2_summary: dict[str, Any],
    output_dir: Path,
    samples_per_roi: int,
    base_inv_ang_per_pixel: float,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for roi in stage2_summary.get("roi_results", []):
        name = str(roi.get("name", "unknown"))
        roi_out = output_dir / name
        roi_out.mkdir(parents=True, exist_ok=True)
        roi_data_path = roi.get("roi_data_path")
        peaks_path = roi.get("bragg_peaks_parquet_path")
        if not roi_data_path or not Path(roi_data_path).exists() or not peaks_path or not Path(peaks_path).exists():
            rows.append({"roi": name, "status": "SKIPPED", "reason": "missing roi_data.npy or bragg_peaks.parquet"})
            continue
        roi_data = np.load(roi_data_path, mmap_mode="r")
        peaks = pd.read_parquet(peaks_path)
        selected = _select_sample_points(peaks, tuple(int(v) for v in roi_data.shape[:2]), samples_per_roi)
        beam = tuple(float(v) for v in (roi.get("beam_center_yx") or _detector_center(roi_data.shape[-2:])))
        circle_radii = _ti_low_order_radii_px(base_inv_ang_per_pixel)
        for i, (sy, sx, reason) in enumerate(selected):
            dp = np.asarray(roi_data[int(sy), int(sx)], dtype=np.float32)
            group = peaks[(peaks["scan_y"] == int(sy)) & (peaks["scan_x"] == int(sx))]
            peak_xy = [(float(row.qy), float(row.qx)) for row in group.itertuples()]
            raw_path = roi_out / f"sample_{i:03d}_{reason}_raw_dp.png"
            overlay_path = roi_out / f"sample_{i:03d}_{reason}_bragg_overlay.png"
            save_png(raw_path, np.log1p(dp))
            save_png(overlay_path, _render_overlay(dp, peak_xy, beam, circle_radii))
            rows.append({
                "roi": name,
                "sample_index": i,
                "scan_y": int(sy),
                "scan_x": int(sx),
                "selection_reason": reason,
                "raw_dp_path": str(raw_path),
                "overlay_path": str(overlay_path),
                "detected_peak_count": int(len(group)),
                "beam_center_yx": list(beam),
                "manual_label": None,
                "manual_label_options": list(MANUAL_LABELS),
            })
    return rows


def _select_sample_points(
    peaks: pd.DataFrame,
    nav_shape: tuple[int, int],
    samples_per_roi: int,
) -> list[tuple[int, int, str]]:
    counts = peaks.groupby(["scan_y", "scan_x"]).size().reset_index(name="count")
    selected: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    def add(sy: int, sx: int, reason: str) -> None:
        key = (int(sy), int(sx))
        if key in seen or len(selected) >= samples_per_roi:
            return
        seen.add(key)
        selected.append((key[0], key[1], reason))

    if not counts.empty:
        high = counts.sort_values("count", ascending=False).head(5)
        low = counts[counts["count"] > 0].sort_values("count", ascending=True).head(5)
        mid_value = float(counts["count"].median())
        amb = counts.assign(delta=np.abs(counts["count"] - mid_value)).sort_values("delta").head(5)
        for row in high.itertuples():
            add(row.scan_y, row.scan_x, "high_peak_count")
        for row in low.itertuples():
            add(row.scan_y, row.scan_x, "low_peak_count")
        for row in amb.itertuples():
            add(row.scan_y, row.scan_x, "ambiguous_peak_count")

    ny, nx = nav_shape
    boundary = [(0, 0), (0, nx - 1), (ny - 1, 0), (ny - 1, nx - 1), (ny // 2, nx // 2)]
    for sy, sx in boundary:
        add(sy, sx, "boundary_or_center")
    rng = np.random.default_rng(20260701)
    while len(selected) < samples_per_roi and ny > 0 and nx > 0:
        add(int(rng.integers(0, ny)), int(rng.integers(0, nx)), "random")
        if len(seen) >= ny * nx:
            break
    return selected


def _render_overlay(
    dp: np.ndarray,
    peaks_yx: list[tuple[float, float]],
    beam_center_yx: tuple[float, float],
    circle_radii_px: list[tuple[str, float]],
) -> np.ndarray:
    finite = dp[np.isfinite(dp)]
    if finite.size:
        lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    gray = np.clip((np.log1p(dp) - np.log1p(lo)) / max(np.log1p(hi) - np.log1p(lo), 1e-12) * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    _draw_cross(rgb, beam_center_yx, (255, 255, 0), radius=5)
    for py, px in peaks_yx:
        _draw_cross(rgb, (py, px), (0, 255, 0), radius=3)
    for phase, radius in circle_radii_px:
        color = (80, 160, 255) if "bcc" in phase else (255, 80, 80)
        _draw_circle(rgb, beam_center_yx, radius, color)
    return rgb


def _draw_cross(rgb: np.ndarray, yx: tuple[float, float], color: tuple[int, int, int], radius: int) -> None:
    y, x = int(round(yx[0])), int(round(yx[1]))
    if y < 0 or y >= rgb.shape[0] or x < 0 or x >= rgb.shape[1]:
        return
    y0, y1 = max(0, y - radius), min(rgb.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(rgb.shape[1], x + radius + 1)
    rgb[y0:y1, x, :] = color
    rgb[y, x0:x1, :] = color


def _draw_circle(rgb: np.ndarray, center_yx: tuple[float, float], radius: float, color: tuple[int, int, int]) -> None:
    cy, cx = center_yx
    if radius <= 0:
        return
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    ys = np.rint(cy + radius * np.sin(theta)).astype(int)
    xs = np.rint(cx + radius * np.cos(theta)).astype(int)
    valid = (ys >= 0) & (ys < rgb.shape[0]) & (xs >= 0) & (xs < rgb.shape[1])
    rgb[ys[valid], xs[valid], :] = color


def _detector_center(sig_shape: tuple[int, int]) -> tuple[float, float]:
    return ((float(sig_shape[0]) - 1.0) / 2.0, (float(sig_shape[1]) - 1.0) / 2.0)


def _ti_low_order_radii_px(inv_ang_per_pixel: float) -> list[tuple[str, float]]:
    # Low-order approximate q=1/d values for Ti bcc (a=3.306 A) and hcp
    # (a=2.951 A, c=4.684 A). These overlays are QC guides, not indexing proof.
    q_values = [
        ("Ti-bcc 110", np.sqrt(2.0) / 3.306),
        ("Ti-bcc 200", 2.0 / 3.306),
        ("Ti-hcp 100", 2.0 / (np.sqrt(3.0) * 2.951)),
        ("Ti-hcp 002", 2.0 / 4.684),
        ("Ti-hcp 101", np.sqrt(4.0 / (3.0 * 2.951 ** 2) + 1.0 / 4.684 ** 2)),
    ]
    scale = max(float(inv_ang_per_pixel), 1e-12)
    return [(name, float(q / scale)) for name, q in q_values]


def _run_calibration_sweep(
    *,
    stage2_summary: dict[str, Any],
    stage2_dir: Path,
    config_path: Path | None,
    output_dir: Path,
    base_inv_ang_per_pixel: float,
) -> list[dict[str, Any]]:
    if config_path is None or not config_path.exists():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, point in enumerate(calibration_sweep_grid(base_inv_ang_per_pixel)):
        run_dir = output_dir / f"run_{idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        modified = _stage2_summary_with_geometry_override(stage2_summary, point)
        temp_stage2 = run_dir / "stage2a_override"
        temp_stage2.mkdir(exist_ok=True)
        (temp_stage2 / "stage2_summary.json").write_text(json.dumps(modified, indent=2), encoding="utf-8")
        cfg = _stage2b_config_from_unified(config_path, temp_stage2, run_dir / "stage2b")
        try:
            result = run_stage2_indexing(cfg)
            row = _summarize_stage2b_sweep_result(result)
            row.update({
                "run_index": idx,
                "inv_ang_per_pixel": point.inv_ang_per_pixel,
                "beam_center_offset_yx": list(point.beam_center_offset_yx),
                "output_dir": str(result.get("output_dir")),
            })
        except Exception as exc:
            row = {
                "run_index": idx,
                "inv_ang_per_pixel": point.inv_ang_per_pixel,
                "beam_center_offset_yx": list(point.beam_center_offset_yx),
                "error": str(exc),
            }
        rows.append(row)
    (output_dir / "calibration_sweep_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _stage2_summary_with_geometry_override(
    summary: dict[str, Any],
    point: CalibrationSweepPoint,
) -> dict[str, Any]:
    data = json.loads(json.dumps(summary))
    for roi in data.get("roi_results", []):
        if roi.get("beam_center_yx"):
            roi["beam_center_yx"] = [
                float(roi["beam_center_yx"][0]) + point.beam_center_offset_yx[0],
                float(roi["beam_center_yx"][1]) + point.beam_center_offset_yx[1],
            ]
        _ensure_diagnostic_ready_qc(roi)
    data.setdefault("p0_calibration_override", {})
    data["p0_calibration_override"] = {
        "inv_ang_per_pixel": point.inv_ang_per_pixel,
        "beam_center_offset_yx": list(point.beam_center_offset_yx),
    }
    return data


def _ensure_diagnostic_ready_qc(roi: dict[str, Any]) -> None:
    """Backfill readiness metrics in temporary sweep summaries only."""
    bq = roi.setdefault("bragg_qc", {})
    if bq.get("median_clean_peaks_per_DP") is None:
        bq["median_clean_peaks_per_DP"] = float(bq.get("peaks_per_pattern_mean") or roi.get("n_bragg_peaks") or 0)
    if bq.get("fraction_DP_with_>=6_peaks") is None:
        bq["fraction_DP_with_>=6_peaks"] = 1.0 if float(bq.get("median_clean_peaks_per_DP") or 0.0) >= 6.0 else 0.0
    bq.setdefault("fraction_DP_with_>=4_peaks", bq.get("fraction_DP_with_>=6_peaks", 0.0))
    bq.setdefault("fraction_DP_with_>=8_peaks", 0.0)
    bq.setdefault("duplicate_fraction_per_pattern_median", 0.0)
    bq.setdefault("duplicate_fraction_per_pattern_p90", 0.0)
    bq.setdefault("peak_splitting_warning", False)
    bq.setdefault("edge_peak_fraction", 0.0)
    bq.setdefault("center_tail_peak_fraction", bq.get("forbidden_center_zone_fraction", 0.0))


def _stage2b_config_from_unified(config_path: Path, stage2_dir: Path, output_dir: Path) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage2b = dict((cfg or {}).get("stage2b") or cfg or {})
    stage2b["stage2_dir"] = str(stage2_dir)
    stage2b["output_dir"] = str(output_dir)
    tg = dict(stage2b.get("template_generation") or {})
    override = json.loads((stage2_dir / "stage2_summary.json").read_text(encoding="utf-8")).get("p0_calibration_override", {})
    inv_ang = override.get("inv_ang_per_pixel")
    if inv_ang:
        tg["reciprocal_pixels_per_inv_angstrom"] = 1.0 / float(inv_ang)
    stage2b["template_generation"] = tg
    return stage2b


def _summarize_stage2b_sweep_result(summary: dict[str, Any]) -> dict[str, Any]:
    rois = summary.get("roi_results", [])
    return {
        "accepted_roi_count": summary.get("accepted_roi_count", 0),
        "radial_support_median": _median_field(rois, "radial_support_score"),
        "q_residual_median": _median_field(rois, "mean_q_residual"),
        "matched_template_fraction_median": _median_field(rois, "matched_template_fraction"),
        "unexplained_peak_fraction_median": _median_field(rois, "unexplained_experiment_fraction"),
        "ti_bcc_hcp_margin_median": _median_field(rois, "phase_margin"),
    }


def _median_field(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(float(np.median(vals)), 4) if vals else None


def _baseline_sweep_like_row(stage2b_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if stage2b_summary is None:
        return None
    return _summarize_stage2b_sweep_result(stage2b_summary)


def _calibration_recommendation(baseline: dict[str, Any] | None, best: dict[str, Any] | None) -> str:
    if best is None:
        return "NO_SWEEP_RUNS_AVAILABLE"
    if baseline is None:
        return "REVIEW_BEST_SWEEP_RUN"
    base_radial = float(baseline.get("radial_support_median") or 0.0)
    best_radial = float(best.get("radial_support_median") or 0.0)
    base_matched = float(baseline.get("matched_template_fraction_median") or 0.0)
    best_matched = float(best.get("matched_template_fraction_median") or 0.0)
    if best_radial >= base_radial + 0.1 or best_matched >= base_matched + 0.1:
        return "FIX_CALIBRATION_BEFORE_PHASE_THRESHOLDS"
    return "NO_CLEAR_CALIBRATION_IMPROVEMENT"


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# P0 Evidence QC Report",
        "",
        f"Stage 2A: `{summary['stage2_dir']}`",
        f"Stage 2B: `{summary.get('stage2b_dir')}`",
        "",
        "## Manual Overlay Samples",
        "",
        f"Samples written: **{len(summary.get('overlay_samples', []))}**",
        f"Manual label options: `{', '.join(summary['manual_label_categories'])}`",
        "",
        "## Calibration Sweep",
        "",
        f"Recommendation: **{summary['calibration_sweep']['recommendation']}**",
        "",
    ]
    best = summary["calibration_sweep"].get("best")
    if best:
        lines.extend([
            "Best sweep row:",
            "",
            "```json",
            json.dumps(best, indent=2),
            "```",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run P0 Stage 2A evidence trustworthiness diagnostics.")
    parser.add_argument("--stage2-dir", required=True, help="Stage 2A roi_bragg directory.")
    parser.add_argument("--stage2b-dir", default=None, help="Existing Stage 2B output directory.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Unified pipeline config for calibration sweep.")
    parser.add_argument("--output-dir", default=None, help="Output directory for P0 diagnostics.")
    parser.add_argument("--inv-ang-per-pixel", type=float, default=0.0192)
    parser.add_argument("--samples-per-roi", type=int, default=20)
    args = parser.parse_args(argv)

    stage2_dir = Path(args.stage2_dir).resolve()
    stage2b_dir = Path(args.stage2b_dir).resolve() if args.stage2b_dir else stage2_dir / "stage2b_indexing"
    config_path = Path(args.config).resolve() if args.config else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else stage2_dir / "p0_evidence_qc"
    summary = run_evidence_qc(
        stage2_dir=stage2_dir,
        stage2b_dir=stage2b_dir,
        config_path=config_path,
        output_dir=output_dir,
        base_inv_ang_per_pixel=args.inv_ang_per_pixel,
        samples_per_roi=args.samples_per_roi,
    )
    print(f"P0 evidence QC written to {output_dir}")
    print(f"Recommendation: {summary['calibration_sweep']['recommendation']}")


if __name__ == "__main__":
    main()
