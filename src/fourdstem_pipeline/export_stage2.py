"""Stage 2A report and benchmark generation.

Produces ``stage2_report.md`` (and ``.html``) and
``stage2_benchmark.json`` from a :class:`Stage2Result`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import is_roi_ready_for_indexing


def save_stage2_report(output_dir: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``stage2_report.md`` and ``stage2_report.html``.

    Parameters
    ----------
    output_dir:
        Stage 2 output directory.
    summary:
        The ``stage2_summary.json`` dict (as returned by
        :func:`fourdstem_pipeline.stage2._build_stage2_summary`).

    Returns
    -------
    (md_path, html_path)
    """
    md_path = output_dir / "stage2_report.md"
    html_path = output_dir / "stage2_report.html"

    markdown = _render_stage2_markdown(summary)
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_render_stage2_html(markdown, summary), encoding="utf-8")
    return md_path, html_path


def save_stage2_benchmark(output_dir: Path, benchmark: dict[str, Any]) -> Path:
    """Write ``stage2_benchmark.json``.

    Parameters
    ----------
    output_dir:
        Stage 2 output directory.
    benchmark:
        Benchmark dict (see :func:`build_benchmark`).

    Returns
    -------
    Path to the written file.
    """
    path = output_dir / "stage2_benchmark.json"
    path.write_text(
        json.dumps(benchmark, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def build_benchmark(
    roi_results: list[dict[str, Any]],
    total_elapsed_s: float,
    *,
    mem: str = "MEMMAP",
    thin_r: int = 2,
    bin_q: int = 2,
) -> dict[str, Any]:
    """Assemble the benchmark dict from per-ROI timings.

    Parameters
    ----------
    roi_results:
        List of per-ROI benchmark entries (see :func:`roi_benchmark_entry`).
    total_elapsed_s:
        Wall-clock time for the entire Stage 2A run.
    mem:
        py4DSTEM memory mode.
    thin_r / bin_q:
        Extraction parameters.

    Returns
    -------
    Benchmark dict ready for JSON serialisation.
    """
    per_roi: list[dict[str, Any]] = []
    for entry in roi_results:
        per_roi.append({
            "name": entry.get("name"),
            "error": entry.get("error"),
            "stage1_bbox": entry.get("stage1_bbox"),
            "raw_bbox": entry.get("raw_bbox"),
            "nav_shape": entry.get("nav_shape"),
            "sig_shape": entry.get("sig_shape"),
            "r_bin": entry.get("r_bin"),
            "n_bragg_peaks": entry.get("n_bragg_peaks"),
            "timing_s": entry.get("timing"),
            "roi_data_size_bytes": entry.get("roi_data_size_bytes"),
        })

    return {
        "parameters": {
            "mem": mem,
            "thin_r": thin_r,
            "bin_q": bin_q,
        },
        "aggregate": {
            "total_elapsed_s": round(total_elapsed_s, 3),
            "n_rois_processed": len(roi_results),
            "n_rois_succeeded": sum(1 for r in roi_results if not r.get("error")),
            "n_rois_failed": sum(1 for r in roi_results if r.get("error")),
            "total_bragg_peaks": sum(r.get("n_bragg_peaks", 0) or 0 for r in roi_results if not r.get("error")),
        },
        "per_roi": per_roi,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _render_stage2_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []

    manifest = summary.get("manifest", {})
    params = summary.get("parameters", {})
    beam = summary.get("beam_center", {})
    provenance = summary.get("provenance", {})
    roi_results: list[dict[str, Any]] = summary.get("roi_results", [])
    errors = summary.get("errors")

    # --- Header ----------------------------------------------------------
    lines.extend([
        f"# Stage 2A ROI Bragg Detection Report - {summary.get('run_name', 'unknown')}",
        "",
        f"**Output directory:** `{summary.get('output_dir', '?')}`",
        "",
    ])

    # --- Run metadata ----------------------------------------------------
    lines.extend([
        "## Run Metadata",
        "",
        f"| Parameter | Value |",
        f"| --- | --- |",
        f"| Stage-1 run | `{manifest.get('run_name', '?')}` |",
        f"| Stage-1 nav shape | `{manifest.get('nav_shape', '?')}` |",
        f"| Stage-1 sig shape | `{manifest.get('sig_shape', '?')}` |",
        f"| r_bin (Stage 1) | `{manifest.get('r_bin', '?')}` |",
        f"| Stage-1 QC status | `{manifest.get('qc_status', '?')}` |",
        f"| thin_r | `{params.get('thin_r', '?')}` |",
        f"| bin_q | `{params.get('bin_q', '?')}` |",
        f"| max_rois | `{params.get('max_rois', 'all')}` |",
        f"| ROI source | `{params.get('roi_source', '?')}` |",
        f"| Beam centre (y, x) | `{beam.get('stage1_yx', '?')}` |",
        f"| Beam centre source | `{beam.get('source', '?')}` |",
        "",
    ])

    # --- Dependencies ----------------------------------------------------
    deps = summary.get("dependencies", {})
    pkg = provenance.get("packages", {})
    lines.extend([
        "## Dependencies",
        "",
        f"| Package | Version |",
        f"| --- | --- |",
        f"| py4DSTEM | `{pkg.get('py4DSTEM', '?')}` |",
        f"| numpy | `{pkg.get('numpy', '?')}` |",
        f"| scipy | `{pkg.get('scipy', '?')}` |",
        f"| Python | `{provenance.get('python_version', '?')}` |",
        f"| Platform | `{provenance.get('platform', '?')}` |",
        f"| Git commit | `{provenance.get('git_commit', '?')}` |",
        f"| Data file | `{deps.get('data_path', '?')}` |",
        "",
    ])

    # --- Per-ROI results -------------------------------------------------
    n_success = sum(1 for r in roi_results if not r.get("error"))
    n_failed = sum(1 for r in roi_results if r.get("error"))
    total_peaks = sum(r.get("n_bragg_peaks", 0) or 0 for r in roi_results if not r.get("error"))

    lines.extend([
        "## Overview",
        "",
        f"- ROIs processed: **{len(roi_results)}**",
        f"- Succeeded: **{n_success}**",
        f"- Failed: **{n_failed}**",
        f"- Total Bragg peaks: **{total_peaks}**",
        "",
    ])

    if roi_results:
        lines.extend([
            "## ROI Results",
            "",
            "| # | Name | Cluster | Reason | Stage1 BBox | Raw BBox | Nav | Sig | Peaks | Beam Source | BG Frac | Sample Cov | Verdict |",
            "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ])
        for i, r in enumerate(roi_results, 1):
            error = r.get("error")
            if error:
                lines.append(
                    f"| {i} | `{r.get('name', '?')}` | - | - | - | - "
                    f"| - | - | - | - | - | - "
                    f"| [FAIL] {_escape_md(str(error))} |"
                )
                continue

            verdict = _indexing_verdict(r)
            lines.append(
                f"| {i} "
                f"| `{r.get('name', '?')}` "
                f"| {r.get('cluster_id', '-')} "
                f"| {_escape_md(str(r.get('reason', '-')))} "
                f"| `{r.get('stage1_bbox', '?')}` "
                f"| `{r.get('raw_bbox', '?')}` "
                f"| `{r.get('nav_shape', '?')}` "
                f"| `{r.get('sig_shape', '?')}` "
                f"| **{r.get('n_bragg_peaks', 0)}** "
                f"| `{r.get('beam_center_source', '?')}` "
                f"| {_fmt_frac(r.get('background_fraction'))} "
                f"| {_fmt_frac(r.get('sample_mask_coverage'))} "
                f"| {verdict} |"
            )

        lines.append("")

    # --- Bragg Peak QC ----------------------------------------------------
    roi_with_qc = [
        r for r in roi_results
        if not r.get("error") and r.get("bragg_qc") is not None
    ]
    if roi_with_qc:
        lines.extend([
            "## Bragg Peak QC",
            "",
            "Per-ROI diagnostics for detected Bragg peak quality.  High fractions",
            "in the centre zone, edge, or duplicate columns suggest the Bragg",
            "detection may be picking up non-diffraction signals (central beam",
            "tail, edge artifacts, hot pixels, or peak splitting).",
            "",
            "| ROI | Peaks | Mean Int | Centre Zone | Edge | Duplicates | R Mean | R Std | BC Err |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for r in roi_with_qc:
            bq = r.get("bragg_qc", {})
            r_mean = _fmt_optional_float(bq.get("radial_distance_mean"))
            r_std = _fmt_optional_float(bq.get("radial_distance_std"))
            bc_err = _fmt_optional_float(bq.get("beam_center_error_estimate"))
            mean_int = bq.get("mean_peak_intensity", 0)
            lines.append(
                f"| `{r.get('name', '?')}` "
                f"| {bq.get('peak_pixel_count', 0)} "
                f"| {mean_int:.1f} "
                f"| {bq.get('forbidden_center_zone_fraction', 0):.1%} "
                f"| {bq.get('edge_peak_fraction', 0):.1%} "
                f"| {bq.get('duplicate_peak_fraction', 0):.1%} "
                f"| {r_mean} "
                f"| {r_std} "
                f"| {bc_err} |"
            )
        lines.append("")
        lines.append(
            "- **Centre Zone**: fraction of peak pixels within 5 px of beam centre "
            "(likely BF tail). **Edge**: fraction near detector boundary. "
            "**Duplicates**: fraction with a neighbour within `minPeakSpacing` px. "
            "**R Mean/Std**: radial distance mean ± std. "
            "**BC Err**: offset between peak centroid and nominal beam centre (px)."
        )
        lines.append("")

    # --- Indexing candidates ---------------------------------------------
    ready = [r for r in roi_results if is_roi_ready_for_indexing(r)]
    not_ready = [r for r in roi_results if not r.get("error") and not is_roi_ready_for_indexing(r)]
    failed = [r for r in roi_results if r.get("error")]

    if ready:
        lines.extend([
            "## [READY] Ready for Stage 2B Indexing",
            "",
            "These ROIs have non-zero Bragg peaks, acceptable background,",
            "and a recorded beam centre.  They can proceed to",
            "phase/orientation indexing.",
            "",
        ])
        for r in ready:
            lines.append(
                f"- **`{r.get('name')}`** - "
                f"cluster {r.get('cluster_id', '?')}, "
                f"{r.get('n_bragg_peaks', 0)} Bragg peaks, "
                f"bbox `{r.get('raw_bbox')}`"
            )
        lines.append("")

    if not_ready:
        lines.extend([
            "## [REVIEW] Not Ready for Stage 2B",
            "",
            "These ROIs succeeded but have issues that should be reviewed",
            "before indexing:",
            "",
        ])
        for r in not_ready:
            issues = _indexing_blockers(r)
            lines.append(
                f"- **`{r.get('name')}`**: {', '.join(issues)}"
            )
        lines.append("")

    if failed:
        lines.extend([
            "## [FAIL] Failed ROIs",
            "",
            "These ROIs encountered errors during Bragg detection and",
            "cannot proceed to Stage 2B.",
            "",
        ])
        for r in failed:
            lines.append(
                f"- **`{r.get('name')}`**: {_escape_md(str(r.get('error', 'unknown error')))}"
            )
        lines.append("")

    # --- Errors ----------------------------------------------------------
    if errors:
        lines.extend([
            "## Stage-Level Errors",
            "",
            "```",
            json.dumps(errors, indent=2),
            "```",
            "",
        ])

    # --- Interpretation guide --------------------------------------------
    lines.extend([
        "## Interpretation Guide",
        "",
        "### Verdicts",
        "",
        "- **[READY] Ready**: ROI has non-zero Bragg peaks, low background,",
        "  has a recorded beam centre, and sample coverage is acceptable.",
        "  Can proceed to Stage 2B phase/orientation indexing.",
        "- **[REVIEW] Review**: ROI has Bragg peaks but also warnings (high",
        "  background, missing sample coverage, zero peaks, etc.).",
        "  Review the diagnostic outputs before using for indexing.",
        "- **[SKIP] Skip**: ROI failed Bragg detection or has critical",
        "  validation failures (zero Bragg peaks, entirely outside",
        "  sample, etc.).  Do NOT use for indexing.",
        "",
        "### Column Notes",
        "",
        "- **BG Frac**: Fraction of pixels in this ROI with fingerprint",
        "  label `-1` (background/vacuum).  Values > 0.5 are flagged.",
        "- **Sample Cov**: Fraction of pixels covered by the Stage-1",
        "  sample mask.  ``-`` means no sample mask was generated.",
        "- **Beam Source**: Origin of the beam centre used for this ROI.",
        "  `stage1_com` = centre-of-mass from Stage 1 mean DP.",
        "  `py4dstem_calibration` = from file metadata.",
        "  `detector_center_fallback` = geometric centre (least reliable).",
        "",
        "---",
        "",
        f"*Report generated by fourdstem-pipeline v{provenance.get('pipeline_version', '0.1.0')}*",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report (lightweight, no external CSS dependency)
# ---------------------------------------------------------------------------


def _render_stage2_html(markdown: str, summary: dict[str, Any]) -> str:
    """Convert the markdown report to a self-contained HTML page."""
    # Very simple conversion: wrap in pre/code and do basic formatting
    lines: list[str] = []

    run_name = summary.get("run_name", "unknown")
    provenance = summary.get("provenance", {})

    lines.extend([
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Stage 2A Report - {run_name}</title>",
        "<style>",
        "  body { font-family: system-ui, -apple-system, sans-serif; ",
        "         max-width: 1100px; margin: 2em auto; padding: 0 1em; ",
        "         color: #1a1a1a; background: #fafafa; line-height: 1.5; }",
        "  table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "  th, td { padding: 6px 10px; text-align: left; ",
        "           border: 1px solid #ddd; font-size: 0.9em; }",
        "  th { background: #f0f0f0; font-weight: 600; }",
        "  tr:nth-child(even) { background: #f9f9f9; }",
        "  h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }",
        "  h2 { margin-top: 1.5em; border-bottom: 1px solid #ccc; }",
        "  code { background: #eee; padding: 1px 4px; border-radius: 3px; ",
        "          font-size: 0.95em; }",
        "  pre { background: #f5f5f5; padding: 1em; overflow-x: auto; ",
        "         border-radius: 4px; }",
        "  .pass { color: #2a7d2a; font-weight: bold; }",
        "  .warn { color: #b8860b; font-weight: bold; }",
        "  .fail { color: #c0392b; font-weight: bold; }",
        "</style>",
        "</head>",
        "<body>",
    ])

    # Simple markdown-to-HTML conversion
    in_table = False
    in_code = False
    for line in markdown.splitlines():
        stripped = line.strip()

        # Code blocks
        if stripped.startswith("```"):
            if in_code:
                lines.append("</pre>")
                in_code = False
            else:
                lines.append("<pre>")
                in_code = True
            continue
        if in_code:
            lines.append(line)
            continue

        # Headers
        if stripped.startswith("# "):
            lines.append(f"<h1>{stripped[2:]}</h1>")
            continue
        if stripped.startswith("## "):
            lines.append(f"<h2>{stripped[3:]}</h2>")
            continue
        if stripped.startswith("### "):
            lines.append(f"<h3>{stripped[4:]}</h3>")
            continue

        # Table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                lines.append("<table>")
                in_table = True
            # Skip separator rows
            if all(c in "|-: " for c in stripped):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            tag = "th" if in_table and len(lines) > 0 and lines[-1] == "<table>" else "td"
            row = "".join(f"<{tag}>{_html_escape(c)}</{tag}>" for c in cells)
            lines.append(f"<tr>{row}</tr>")
            continue
        elif in_table:
            lines.append("</table>")
            in_table = False

        # Lists
        if stripped.startswith("- "):
            lines.append(f"<li>{_html_escape(stripped[2:])}</li>")
            continue

        # Bold
        import re
        stripped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)
        # Inline code
        stripped = re.sub(r'`(.+?)`', r'<code>\1</code>', stripped)

        if stripped:
            lines.append(f"<p>{stripped}</p>")
        else:
            lines.append("<br>")

    if in_table:
        lines.append("</table>")
    if in_code:
        lines.append("</pre>")

    lines.extend([
        "</body>",
        "</html>",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _indexing_verdict(roi: dict[str, Any]) -> str:
    """Return a human-readable verdict for this ROI."""
    n_peaks = roi.get("n_bragg_peaks", 0) or 0
    bg_frac = roi.get("background_fraction")
    sample_cov = roi.get("sample_mask_coverage")
    warning = roi.get("cluster_validation_warning")
    beam_source = roi.get("beam_center_source", "")

    if n_peaks == 0:
        return "[SKIP] Skip (0 peaks)"
    if bg_frac is not None and bg_frac > 0.5:
        return "[SKIP] Skip (>50% bg)"
    if sample_cov is not None and sample_cov == 0.0:
        return "[SKIP] Skip (sample 0%)"
    if beam_source == "detector_center_fallback":
        return "[REVIEW] Review (no calib)"
    if warning:
        return "[REVIEW] Review"
    return "[READY] Ready"


def _indexing_blockers(roi: dict[str, Any]) -> list[str]:
    """List reasons why an ROI is not ready for Stage 2B."""
    issues: list[str] = []
    n_peaks = roi.get("n_bragg_peaks", 0) or 0
    bg_frac = roi.get("background_fraction")
    sample_cov = roi.get("sample_mask_coverage")
    beam_source = roi.get("beam_center_source", "")
    warning = roi.get("cluster_validation_warning")

    if n_peaks == 0:
        issues.append("zero Bragg peaks")
    if bg_frac is not None and bg_frac > 0.5:
        issues.append(f"high background ({bg_frac:.1%})")
    elif bg_frac is not None and bg_frac > 0.1:
        issues.append(f"moderate background ({bg_frac:.1%})")
    if sample_cov is not None and sample_cov == 0.0:
        issues.append("zero sample coverage")
    elif sample_cov is not None and sample_cov < 0.3:
        issues.append(f"low sample coverage ({sample_cov:.1%})")
    if beam_source == "detector_center_fallback":
        issues.append("no calibrated beam centre")
    if warning:
        issues.append(f"validation warning: {warning}")

    if not issues:
        issues.append("unknown - check outputs manually")
    return issues


def _escape_md(text: str) -> str:
    """Escape characters that could break markdown table cells."""
    return text.replace("|", "\\|").replace("\n", " ")


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _fmt_frac(value: float | None) -> str:
    """Format a fraction for display, returning '-' for None."""
    if value is None:
        return "-"
    return f"{value:.1%}"


def _fmt_optional_float(value: float | None) -> str:
    """Format an optional float, returning '-' for None."""
    if value is None:
        return "-"
    return f"{value:.1f}"


# ---------------------------------------------------------------------------
# PNG gallery — uses relative &lt;img&gt; references to the actual PNG files on
# disk (no base64 bloat).  Two views are generated:
#
# 1. Per-ROI detail — one section per ROI with all its diagnostic PNGs.
# 2. Cross-ROI comparison — for each image type, all ROIs side-by-side so
#    you can spot differences between fingerprint classes / candidates at a
#    glance.
# ---------------------------------------------------------------------------

# Canonical list of PNGs produced per ROI, in display order, with captions.
_GALLERY_PNG_SPEC: list[tuple[str, str]] = [
    ("mean_dp.png", "Mean diffraction pattern (log scale)"),
    ("bragg_vector_map.png", "Bragg vector map (peak vote histogram)"),
    ("bragg_overlay.png", "Bragg peaks overlaid on mean DP (green crosses)"),
    ("bragg_peak_radius_histogram.png", "Radial distance histogram of detected peaks"),
    ("template_best_match.png", "Best-matching kinematic template (Stage 2B)"),
    ("template_match_overlay.png", "Template peaks overlaid on mean DP (Stage 2B)"),
    ("correlation_vs_angle.png", "Correlation vs. in-plane orientation angle (Stage 2B)"),
]


def save_stage2_gallery(
    output_dir: Path,
    summary: dict[str, Any],
    *,
    global_pngs: list[dict[str, str]] | None = None,
) -> Path | None:
    """Generate ``stage2_gallery.html`` with per-ROI detail and cross-ROI comparisons.

    PNGs are referenced via relative ``<img src>`` paths — the HTML file must
    stay alongside the ROI directories for images to load.

    Parameters
    ----------
    output_dir:
        Stage 2 output directory (where ``stage2_summary.json`` lives).
    summary:
        The ``stage2_summary.json`` dict (must contain ``roi_results``).
    global_pngs:
        Optional list of ``{"path": ..., "caption": ...}`` dicts for overview
        PNGs that are not per-ROI (e.g. a phase match map).  Rendered in a
        "Global Overview" section at the top of the gallery.

    Returns
    -------
    Path to ``stage2_gallery.html``, or *None* if no PNGs were found.
    """
    roi_results: list[dict[str, Any]] = summary.get("roi_results", [])
    if not roi_results:
        return None

    # ── Collect per-ROI data ────────────────────────────────────────────
    roi_galleries: list[dict[str, Any]] = []
    # comparison_groups: key = filename, value = list of {roi_name, rel_path, caption, meta}
    comparison_groups: dict[str, list[dict[str, str]]] = {}
    total_pngs = 0

    for r in roi_results:
        name = r.get("name", "unknown")
        if r.get("error"):
            continue

        bsp = r.get("bragg_summary_path")
        roi_dir = Path(bsp).parent if bsp else output_dir / f"roi_{name}"
        roi_rel = _rel_path(output_dir, roi_dir)

        # Compact metadata for labels
        meta = _roi_meta_label(r)

        pngs: list[dict[str, str]] = []
        for filename, caption in _GALLERY_PNG_SPEC:
            png_path = roi_dir / filename
            if not png_path.is_file():
                continue
            rel_path = f"{roi_rel}/{filename}"
            pngs.append({
                "filename": filename,
                "caption": caption,
                "rel_path": rel_path,
            })
            total_pngs += 1

            # Cross-ROI comparison group
            comparison_groups.setdefault(filename, []).append({
                "roi_name": name,
                "rel_path": rel_path,
                "caption": caption,
                "meta": meta,
            })

        if pngs:
            roi_galleries.append({"name": name, "meta": meta, "pngs": pngs})

    if total_pngs == 0:
        return None

    # ── Render ──────────────────────────────────────────────────────────
    run_name = summary.get("run_name", "unknown")
    stage1_dir = summary.get("stage1_dir", "")
    # comparison groups sorted so Stage 2A PNGs come first
    comp_order = [fn for fn, _ in _GALLERY_PNG_SPEC if fn in comparison_groups]

    html = _render_gallery_html(
        run_name=run_name,
        stage1_dir=stage1_dir,
        n_rois=len(roi_galleries),
        total_pngs=total_pngs,
        roi_galleries=roi_galleries,
        comparison_groups=comparison_groups,
        comp_order=comp_order,
        global_pngs=_resolve_global_pngs(global_pngs, output_dir),
    )
    gallery_path = output_dir / "stage2_gallery.html"
    gallery_path.write_text(html, encoding="utf-8")
    return gallery_path


def _rel_path(base: Path, target: Path) -> str:
    """Return *target* as a relative path from *base*, using forwardslashes."""
    try:
        rel = target.relative_to(base)
    except ValueError:
        return target.as_posix()
    return rel.as_posix()


def _resolve_global_pngs(
    global_pngs: list[dict[str, str]] | None,
    base: Path,
) -> list[dict[str, str]]:
    """Resolve global PNG paths to relative URLs and filter to files that exist."""
    if not global_pngs:
        return []
    resolved: list[dict[str, str]] = []
    for entry in global_pngs:
        p = Path(entry["path"])
        if not p.is_file():
            continue
        resolved.append({
            "rel_path": _rel_path(base, p),
            "caption": entry.get("caption", p.stem),
            "filename": p.name,
        })
    return resolved


def _roi_meta_label(r: dict[str, Any]) -> str:
    """One-line metadata label for an ROI."""
    parts: list[str] = []
    n_peaks = r.get("n_bragg_peaks", 0)
    parts.append(f"{n_peaks} peaks")
    bc = r.get("beam_center_source", "")
    if bc:
        parts.append(f"beam:{bc}")
    bg = r.get("background_fraction")
    if bg is not None:
        parts.append(f"BG:{bg:.1%}")
    cp = r.get("candidate_phase")
    if cp:
        parts.append(f"{cp}")
        ms = r.get("match_score")
        if ms is not None:
            parts.append(f"score:{ms:.3f}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_GALLERY_CSS = """
*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 1500px; margin: 0 auto; padding: 20px;
    background: #f5f5f5; color: #222;
}
h1 { border-bottom: 2px solid #4472C4; padding-bottom: 8px; }
h2 { color: #4472C4; margin-top: 36px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
h3 { color: #333; margin-top: 24px; }
.summary-bar {
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 12px 20px; margin-bottom: 24px; font-size: 0.9em;
}
.toc { margin: 16px 0; line-height: 1.8; }
.toc a { margin-right: 14px; color: #4472C4; text-decoration: none; }
.toc a:hover { text-decoration: underline; }

/* ── Per-ROI detail ── */
.roi-section {
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 20px; margin-bottom: 24px;
}
.roi-meta {
    font-size: 0.9em; color: #555; margin-bottom: 16px;
    padding: 8px 12px; background: #f0f4ff; border-radius: 4px;
}
.png-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 16px;
}
.png-card {
    border: 1px solid #e0e0e0; border-radius: 4px;
    overflow: hidden; background: #fafafa;
}
.png-card img { width: 100%; height: auto; display: block; }
.png-caption {
    padding: 8px 12px; font-size: 0.85em; color: #555;
    background: #f8f8f8; border-top: 1px solid #eee;
}

/* ── Cross-ROI comparison ── */
.comp-section {
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 20px; margin-bottom: 24px;
}
.comp-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
}
.comp-card {
    border: 1px solid #e0e0e0; border-radius: 4px;
    overflow: hidden; background: #fafafa;
}
.comp-card img { width: 100%; height: auto; display: block; }
.comp-label {
    padding: 6px 10px; font-size: 0.8em; color: #4472C4; font-weight: 600;
    background: #f0f4ff; border-bottom: 1px solid #eee;
}
.comp-meta {
    padding: 4px 10px; font-size: 0.75em; color: #777;
    background: #f8f8f8; border-top: 1px solid #eee;
}

/* ── Global overview ── */
.global-section {
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 20px; margin-bottom: 24px; text-align: center;
}
.global-section img {
    max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 2px;
}
"""


def _render_gallery_html(
    *,
    run_name: str,
    stage1_dir: str,
    n_rois: int,
    total_pngs: int,
    roi_galleries: list[dict[str, Any]],
    comparison_groups: dict[str, list[dict[str, str]]],
    comp_order: list[str],
    global_pngs: list[dict[str, str]] | None = None,
) -> str:
    """Render the full gallery HTML."""
    global_list = global_pngs or []

    p: list[str] = []
    p.append("<!DOCTYPE html>")
    p.append('<html lang="en"><head>')
    p.append('<meta charset="utf-8">')
    p.append(f"<title>Stage 2 PNG Gallery — {run_name}</title>")
    p.append(f"<style>{_GALLERY_CSS}</style>")
    p.append("</head><body>")

    # Header
    p.append(f"<h1>Stage 2 PNG Gallery — {run_name}</h1>")
    p.append('<div class="summary-bar">')
    p.append(f"Stage 1: <code>{stage1_dir}</code><br>")
    p.append(f"ROIs with images: {n_rois} · Total PNGs: {total_pngs}")
    p.append("</div>")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 0 — Global overview (phase match map, etc.)
    # ══════════════════════════════════════════════════════════════════════
    if global_list:
        p.append("<h2>Global Overview</h2>")
        for gp in global_list:
            p.append('<div class="global-section">')
            p.append(f'<img src="{gp["rel_path"]}" alt="{gp["filename"]}" loading="lazy" style="max-width:100%;height:auto;">')
            p.append(f'<div class="png-caption">{gp["caption"]}</div>')
            p.append("</div>")

    # Table of contents
    p.append('<div class="toc"><strong>Jump to ROI:</strong> ')
    for i, rg in enumerate(roi_galleries):
        p.append(f'<a href="#roi-{i}">{rg["name"]}</a>')
    p.append("</div>")
    p.append('<div class="toc"><strong>Jump to comparison:</strong> ')
    for fn in comp_order:
        cap = next((c for f, c in _GALLERY_PNG_SPEC if f == fn), fn)
        p.append(f'<a href="#comp-{fn}">{cap}</a>')
    p.append("</div>")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — Per-ROI detail
    # ══════════════════════════════════════════════════════════════════════
    p.append("<h2>Per-ROI Detail</h2>")
    for i, rg in enumerate(roi_galleries):
        p.append(f'<div class="roi-section" id="roi-{i}">')
        p.append(f"<h3>{rg['name']}</h3>")
        p.append(f'<div class="roi-meta">{rg["meta"]}</div>')
        p.append('<div class="png-grid">')
        for png in rg["pngs"]:
            p.append('<div class="png-card">')
            p.append(f'<img src="{png["rel_path"]}" alt="{png["filename"]}" loading="lazy">')
            p.append(f'<div class="png-caption">{png["caption"]}</div>')
            p.append("</div>")
        p.append("</div></div>")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — Cross-ROI comparison (by image type)
    # ══════════════════════════════════════════════════════════════════════
    if len(roi_galleries) > 1:
        p.append("<h2>Cross-ROI Comparison</h2>")
        p.append("<p>Same diagnostic, all ROIs side-by-side. "
                  "Useful for spotting class differences at a glance.</p>")
        for fn in comp_order:
            entries = comparison_groups[fn]
            if len(entries) < 2:
                continue
            cap = next((c for f, c in _GALLERY_PNG_SPEC if f == fn), fn)
            p.append(f'<div class="comp-section" id="comp-{fn}">')
            p.append(f"<h3>{cap}</h3>")
            p.append('<div class="comp-grid">')
            for e in entries:
                p.append('<div class="comp-card">')
                p.append(f'<div class="comp-label">{e["roi_name"]}</div>')
                p.append(f'<img src="{e["rel_path"]}" alt="{fn} @ {e["roi_name"]}" loading="lazy">')
                p.append(f'<div class="comp-meta">{e["meta"]}</div>')
                p.append("</div>")
            p.append("</div></div>")

    p.append("</body></html>")
    return "\n".join(p)
