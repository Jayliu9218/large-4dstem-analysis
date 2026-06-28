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
