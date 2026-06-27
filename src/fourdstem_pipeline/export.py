from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any
import struct
import zlib

import numpy as np


def save_summary(output_dir: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(output_dir) / "workflow_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return path


def save_report(output_dir: str | Path, summary: dict[str, Any], phase_labels: np.ndarray) -> Path:
    path = Path(output_dir) / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _render_report(summary, phase_labels, base_dir=path.parent)
    path.write_text(markdown, encoding="utf-8")
    html_path = path.with_suffix(".html")
    html_path.write_text(_render_html_report(summary, phase_labels, base_dir=path.parent), encoding="utf-8")
    return path


def save_npz(output_dir: str | Path, name: str, **arrays: np.ndarray) -> Path:
    path = Path(output_dir) / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def save_png(path: str | Path, image: np.ndarray, *, percentiles: tuple[float, float] = (1, 99)) -> Path:
    """Save a 2D scalar image or RGB uint8 image as PNG without extra dependencies."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = np.repeat(_scale_to_uint8(arr, percentiles)[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        rgb = arr.astype(np.uint8, copy=False) if arr.dtype == np.uint8 else _scale_to_uint8(arr, percentiles)
    else:
        raise ValueError(f"Expected 2D or RGB image, got shape {arr.shape!r}.")
    _write_rgb_png(out, rgb)
    return out


def save_label_png(path: str | Path, labels: np.ndarray) -> Path:
    arr = np.asarray(labels, dtype=np.int64)
    palette = _label_palette()
    return save_png(path, palette[np.mod(arr, len(palette))])


def save_annotated_label_png(path: str | Path, labels: np.ndarray, *, title: str = "PHASE CLUSTERS") -> Path:
    arr = np.asarray(labels, dtype=np.int64)
    palette = _label_palette()
    image = palette[np.mod(arr, len(palette))]
    labels_present, counts = np.unique(arr, return_counts=True)

    scale = max(1, int(np.ceil(192 / max(arr.shape[0], arr.shape[1]))))
    map_rgb = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    legend_width = 220
    height = max(map_rgb.shape[0], 46 + 28 * len(labels_present))
    canvas = np.full((height, map_rgb.shape[1] + legend_width, 3), 255, dtype=np.uint8)
    canvas[: map_rgb.shape[0], : map_rgb.shape[1]] = map_rgb
    x0 = map_rgb.shape[1] + 16

    _draw_text(canvas, title.upper(), x0, 14, color=(30, 30, 30), scale=2)
    for row, (label, count) in enumerate(zip(labels_present, counts)):
        y = 48 + row * 28
        color = palette[int(label) % len(palette)]
        canvas[y : y + 14, x0 : x0 + 18] = color
        _draw_text(canvas, f"CLUSTER {int(label)}  N={int(count)}", x0 + 28, y + 2, color=(30, 30, 30), scale=1)
    return save_png(path, canvas)


def save_profile_png(path: str | Path, radii: np.ndarray, profiles: np.ndarray, *, size: tuple[int, int] = (720, 420)) -> Path:
    """Save the mean radial profile as a simple line plot PNG."""
    width, height = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 58, 22, 24, 46
    x0, x1 = margin_l, width - margin_r - 1
    y0, y1 = margin_t, height - margin_b - 1

    canvas[y0:y1 + 1, x0] = 30
    canvas[y1, x0:x1 + 1] = 30
    for frac in np.linspace(0.25, 0.75, 3):
        y = int(y1 - frac * (y1 - y0))
        canvas[y, x0:x1 + 1] = 225

    profile = np.asarray(profiles, dtype=np.float32).reshape(-1, profiles.shape[-1]).mean(axis=0)
    radii = np.asarray(radii, dtype=np.float32)
    if radii.size != profile.size:
        radii = np.arange(profile.size, dtype=np.float32)
    px = _normalize_to_span(radii, x0, x1)
    py = y1 - _normalize_to_span(profile, 0, y1 - y0)
    points = np.column_stack([px.astype(int), py.astype(int)])
    _draw_polyline(canvas, points, color=(25, 90, 155))
    return save_png(path, canvas)


def save_lines_png(
    path: str | Path,
    x: np.ndarray,
    ys: np.ndarray,
    *,
    colors: list[tuple[int, int, int]] | None = None,
    size: tuple[int, int] = (720, 420),
) -> Path:
    width, height = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 58, 22, 24, 46
    x0, x1 = margin_l, width - margin_r - 1
    y0, y1 = margin_t, height - margin_b - 1
    canvas[y0:y1 + 1, x0] = 30
    canvas[y1, x0:x1 + 1] = 30
    for frac in np.linspace(0.25, 0.75, 3):
        y_grid = int(y1 - frac * (y1 - y0))
        canvas[y_grid, x0:x1 + 1] = 225

    arr = np.asarray(ys, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    x_values = np.asarray(x, dtype=np.float32)
    if x_values.size != arr.shape[1]:
        x_values = np.arange(arr.shape[1], dtype=np.float32)
    px = _normalize_to_span(x_values, x0, x1).astype(int)
    finite = arr[np.isfinite(arr)]
    ymin = float(np.min(finite)) if finite.size else 0.0
    ymax = float(np.max(finite)) if finite.size else 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    palette = colors or [tuple(map(int, color)) for color in _label_palette()]
    for idx, series in enumerate(arr):
        py = (y1 - (series - ymin) / (ymax - ymin) * (y1 - y0)).astype(int)
        _draw_polyline(canvas, np.column_stack([px, py]), color=palette[idx % len(palette)])
    return save_png(path, canvas)


def save_bar_png(path: str | Path, values: np.ndarray, *, size: tuple[int, int] = (720, 420)) -> Path:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    width, height = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 58, 22, 24, 46
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    vmax = max(float(np.nanmax(np.abs(arr))), 1e-12)
    groups, bars = arr.shape
    slot = max(1, plot_w // max(groups, 1))
    bar_w = max(1, slot // max(bars + 1, 2))
    palette = _label_palette()
    baseline = height - margin_b - 1
    canvas[margin_t:baseline + 1, margin_l] = 30
    canvas[baseline, margin_l:width - margin_r] = 30
    for group in range(groups):
        for bar in range(bars):
            value = max(float(arr[group, bar]), 0.0)
            h = int(value / vmax * plot_h)
            x0 = margin_l + group * slot + bar * bar_w + 2
            x1 = min(x0 + bar_w, width - margin_r)
            canvas[baseline - h:baseline, x0:x1] = palette[bar % len(palette)]
    return save_png(path, canvas)


def _label_palette() -> np.ndarray:
    return np.asarray(
        [
            [45, 105, 170],
            [220, 95, 60],
            [80, 155, 90],
            [180, 120, 190],
            [230, 180, 65],
            [70, 170, 180],
            [150, 150, 150],
            [195, 70, 120],
        ],
        dtype=np.uint8,
    )


def _scale_to_uint8(image: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, percentiles)
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    scaled = (arr - lo) / max(hi - lo, 1e-12)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def _normalize_to_span(values: np.ndarray, low: int, high: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    span = max(high - low, 1)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if vmax <= vmin:
        return np.full(vals.shape, (low + high) / 2, dtype=np.float32)
    return low + (vals - vmin) / (vmax - vmin) * span


def _draw_polyline(canvas: np.ndarray, points: np.ndarray, *, color: tuple[int, int, int]) -> None:
    for start, end in zip(points[:-1], points[1:]):
        x0, y0 = start
        x1, y1 = end
        steps = int(max(abs(x1 - x0), abs(y1 - y0), 1))
        xs = np.linspace(x0, x1, steps + 1).astype(int)
        ys = np.linspace(y0, y1, steps + 1).astype(int)
        valid = (xs >= 0) & (xs < canvas.shape[1]) & (ys >= 0) & (ys < canvas.shape[0])
        canvas[ys[valid], xs[valid]] = color


def _draw_text(canvas: np.ndarray, text: str, x: int, y: int, *, color: tuple[int, int, int], scale: int = 1) -> None:
    cursor = int(x)
    for char in text:
        glyph = _FONT_5X7.get(char.upper(), _FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    y0 = y + gy * scale
                    x0 = cursor + gx * scale
                    canvas[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor += 6 * scale


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("PNG writer expects an RGB uint8 array.")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    chunks = [
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw, level=6)),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _render_report(summary: dict[str, Any], phase_labels: np.ndarray, *, base_dir: Path) -> str:
    project = summary.get("project", {})
    data = summary.get("data_config", {})
    dataset = summary.get("dataset", {})
    outputs = summary.get("outputs", {})
    png = outputs.get("png", {}) if isinstance(outputs.get("png"), dict) else {}
    shapes = summary.get("shapes", {})
    preprocess = summary.get("preprocess", "see workflow_summary.json")
    title = str(project.get("name", "4D-STEM Diffraction-Class Screening Report"))

    data_contract = summary.get("data_contract", {})
    provenance = summary.get("provenance", {})
    qc = summary.get("qc", {})

    qc_status = qc.get("stage1_status", "UNKNOWN")
    qc_emoji = {"PASS": "✅", "PASS_WITH_WARNINGS": "⚠️", "FAIL": "❌"}.get(qc_status, "❓")

    lines = [
        f"# {title}",
        "",
        f"## Stage 1 QC Status: {qc_emoji} {qc_status}",
        "",
        f"- Warnings: {qc.get('n_warnings', '?')}",
        f"- Critical: {qc.get('n_critical', '?')}",
        "",
    ]

    # Inline QC flag table
    qc_flags = qc.get("flags", [])
    if qc_flags:
        lines.append("| Severity | Code | Message |")
        lines.append("| --- | --- | --- |")
        for flag in qc_flags:
            sev = flag.get("severity", "info").upper()
            lines.append(f"| `{sev}` | `{flag.get('code', '?')}` | {flag.get('message', '')} |")
        lines.append("")

    lines.extend([
        "> **Caution:** The fingerprint-class labels are generated by unsupervised "
        "clustering of radial-fingerprint profiles (PCA/NMF + KMeans). They "
        "represent **radial-profile similarity groups**, not confirmed "
        "crystallographic phases. Treat them as **phase candidates** until "
        "validated by Stage-2 py4DSTEM / pyxem Bragg indexing or template matching.",
        "",
        "## Data",
        "",
        f"- Source: `{data.get('path', dataset.get('path', 'unknown'))}`",
        f"- Backend: `{dataset.get('source_backend', 'unknown')}`",
        f"- Preprocessed shape: `{dataset.get('shape', 'unknown')}`",
        f"- Navigation shape: `{dataset.get('navigation_shape', 'unknown')}`",
        f"- Signal shape: `{dataset.get('signal_shape', 'unknown')}`",
        "",
        "## Coordinate Conventions (Data Contract)",
        "",
        f"- Axis order: `{data_contract.get('axis_order', 'nav_y_nav_x_q_y_q_x')}`",
        f"- BBox order: `{data_contract.get('bbox_order', 'y0_y1_x0_x1')}`",
        f"- Centre order: `{data_contract.get('center_order', 'y_x')}`",
        "",
        "## Provenance",
        "",
        f"- Pipeline version: `{provenance.get('pipeline_version', 'unknown')}`",
        f"- Git commit: `{provenance.get('git_commit', 'unknown')}`",
        f"- Config path: `{provenance.get('config_path', 'inline')}`",
        f"- Config hash: `{provenance.get('config_hash', 'n/a')}`",
        f"- Input path: `{provenance.get('input_path', 'synthetic')}`",
        f"- Input file size: {_human_size(provenance.get('input_file_size'))}",
        f"- Input file mtime: `{provenance.get('input_file_mtime', 'n/a')}`",
        f"- Python version: `{provenance.get('python_version', 'unknown')}`",
        f"- Platform: `{provenance.get('platform', 'unknown')}`",
        f"- Random seed: `{provenance.get('random_seed', 'none')}`",
        f"- Start time: `{provenance.get('start_time', 'unknown')}`",
        f"- End time: `{provenance.get('end_time', 'unknown')}`",
        "",
        _render_package_versions(provenance.get("packages", {})),
        "",
        "## Screening Settings",
        "",
        f"- q-crop/q-bin/r-bin: `{preprocess}`",
        f"- Virtual image shapes: `{shapes.get('virtual_images', 'unknown')}`",
        f"- Radial fingerprints: `{shapes.get('radial_fingerprints', 'unknown')}`",
        f"- Fingerprint-class labels: `{shapes.get('fingerprint_class_labels', shapes.get('phase_labels', 'unknown'))}`",
        f"- Orientation preview: `{shapes.get('orientation_index', 'unknown')}`",
        "",
        "## Interpretation Level",
        "",
        "Level A: Fingerprint class. Unsupervised radial-fingerprint class only. Not a crystallographic phase assignment.",
        "",
        "Level B: Phase candidate. Cluster average DP / radial peaks are consistent with a candidate phase. Requires indexing validation.",
        "",
        "Level C: Confirmed phase. Validated by py4DSTEM / pyxem template matching or Bragg indexing using candidate CIFs.",
        "",
        "Current Stage 1 result should be treated as Level A unless later indexing validation is added.",
        "",
        "## Unsupervised Fingerprint Classes (Phase Candidates)",
        "",
        "| Cluster | Pixels | Fraction |",
        "| --- | ---: | ---: |",
    ])

    labels, counts = np.unique(np.asarray(phase_labels), return_counts=True)
    total = max(int(counts.sum()), 1)
    for label, count in zip(labels, counts):
        lines.append(f"| {int(label)} | {int(count)} | {int(count) / total:.3f} |")

    # Sample mask statistics
    sample_mask = summary.get("sample_mask", {})
    if sample_mask and sample_mask.get("generated"):
        lines.extend([
            "## Sample Mask",
            "",
            f"- Source: `{sample_mask.get('source', 'adf')}`",
            f"- Sample pixels: `{sample_mask.get('sample_pixels', '?')}`",
            f"- Background pixels: `{sample_mask.get('background_pixels', '?')}`",
            f"- Sample fraction: `{sample_mask.get('sample_fraction', '?')}`",
            "",
            "Pixels outside the sample mask are marked as background "
            "(label `-1`) and excluded from diffraction-class clustering.",
            "",
        ])

    lines.extend(["", "## Key PNG Outputs", ""])
    for key in [
        "fingerprint_class_labels_annotated",
        "fingerprint_class_labels",
        "cluster_cleaned_labels",
        "cluster_mean_radial_profiles",
        "cluster_virtual_image_statistics",
        "cluster_vs_orientation_heatmap",
        "ring_2_over_ring_1",
        "ring_3_over_ring_1",
        "ring_3_over_ring_2",
        "roi_candidates_overlay",
        "sample_mask",
        "sample_mask_overlay_adf",
        "virtual_bf",
        "virtual_adf",
        "virtual_haadf",
        "com_x",
        "com_y",
        "mean_diffraction",
        "mean_radial_profile",
        "orientation_index",
        "orientation_score",
    ]:
        if key in png:
            png_path = Path(png[key])
            try:
                link = png_path.relative_to(base_dir).as_posix()
            except ValueError:
                link = png_path.as_posix()
            lines.append(f"- {key}: [{png_path.name}]({link})")

    diagnostics = outputs.get("diagnostics", {}) if isinstance(outputs.get("diagnostics"), dict) else {}
    lines.extend(["", "## Cluster Diagnostics", ""])
    for title_text, path_text in [
        ("Cluster virtual-image statistics", diagnostics.get("cluster_summary_csv")),
        ("Connected-component cleanup", (diagnostics.get("connected_components") or {}).get("connected_components_csv") if isinstance(diagnostics.get("connected_components"), dict) else None),
        ("Cluster vs orientation", (diagnostics.get("cluster_vs_orientation") or {}).get("csv") if isinstance(diagnostics.get("cluster_vs_orientation"), dict) else None),
        ("K-sweep metrics", (diagnostics.get("k_sweep") or {}).get("metrics_csv") if isinstance(diagnostics.get("k_sweep"), dict) else None),
    ]:
        table = _csv_to_markdown(path_text, base_dir=base_dir, max_rows=12)
        if table:
            lines.extend([f"### {title_text}", "", table, ""])

    roi_csv = (diagnostics.get("roi_outputs") or {}).get("csv") if isinstance(diagnostics.get("roi_outputs"), dict) else None
    roi_table = _csv_to_markdown(roi_csv, base_dir=base_dir, max_rows=12)
    if roi_table:
        lines.extend(["## Stage 2 ROI Candidates", "", roi_table, ""])

    ring_maps = diagnostics.get("ring_ratio_maps") if isinstance(diagnostics.get("ring_ratio_maps"), dict) else {}
    if ring_maps:
        lines.extend(["## Ring Ratio Map Arrays", ""])
        for name, path_text in ring_maps.items():
            lines.append(f"- {name}: `{path_text}`")
        lines.append("")

    lines.extend(
        [
            "",
            "## Output Directories",
            "",
            f"- Virtual images: `{outputs.get('virtual', 'not generated')}`",
            f"- Fingerprints: `{outputs.get('fingerprints', 'not generated')}`",
            f"- Fingerprint classes: `{outputs.get('fingerprint_classes', outputs.get('phase_screening', 'not generated'))}`",
            f"- Cluster diagnostics: `{outputs.get('cluster_diagnostics', 'not generated')}`",
            f"- ROI candidates: `{outputs.get('roi_candidates', 'not generated')}`",
            f"- Orientation: `{outputs.get('orientation', 'not generated')}`",
            f"- PNG previews: `{Path(next(iter(png.values()))).parent.as_posix() if png else 'not generated'}`",
            "",
            "## Interpretation Notes",
            "",
            "- **Fingerprint-class labels are unsupervised radial-profile clusters, not final crystallographic phase maps.** Formal phase assignment requires Stage-2 validation.",
            "- If clusters strongly follow HAADF / ADF intensity, they may reflect thickness, Z-contrast, or mass-thickness effects — not phase.",
            "- If clusters strongly follow COM-x / COM-y gradients, they may reflect bend, diffraction shift, or beam-center drift — not phase.",
            "- If clusters mainly differ in ring ratios and radial peak positions, they are more plausible **phase candidates**, but still require indexing validation.",
            "- **Stage-2 workflow**: Select ROI candidates from the most promising fingerprint classes, then run py4DSTEM / pyxem template matching or Bragg indexing against candidate CIFs for confirmed phase mapping.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_html_report(summary: dict[str, Any], phase_labels: np.ndarray, *, base_dir: Path) -> str:
    project = summary.get("project", {})
    outputs = summary.get("outputs", {})
    png = outputs.get("png", {}) if isinstance(outputs.get("png"), dict) else {}
    title = str(project.get("name", "4D-STEM Diffraction-Class Screening Report"))
    body = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}",
        "h1,h2{margin-top:1.4em} table{border-collapse:collapse;margin:12px 0;max-width:100%}",
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:right} th:first-child,td:first-child{text-align:left}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}",
        "figure{margin:0} img{max-width:100%;border:1px solid #ddd} figcaption{font-size:13px;color:#555;margin-top:4px}",
        "code{background:#f5f5f5;padding:1px 4px}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    body.extend(_html_qc_banner(summary.get("qc", {})))
    body.extend(_html_data_summary(summary))
    body.extend(_html_label_summary(phase_labels))
    body.extend(_html_sample_mask(summary.get("sample_mask", {})))
    body.extend(_html_provenance(summary.get("provenance", {})))
    body.extend(_html_png_grid(png, base_dir))
    body.extend(_html_diagnostic_tables(outputs.get("diagnostics", {}), base_dir))
    body.extend(
        [
            "<h2>Interpretation Notes</h2>",
            "<ul>",
            "<li><strong>Fingerprint-class labels are unsupervised radial-profile clusters, not crystallographic phase assignments.</strong> Treat them as phase candidates.</li>",
            "<li>Use cluster mean DPs, radial profiles, virtual-image statistics, ring ratios, and orientation consistency to prioritize Stage-2 ROI candidates.</li>",
            "<li><strong>Stage-2:</strong> Select promising ROI candidates, then run py4DSTEM/pyxem Bragg indexing or template matching against candidate CIFs for confirmed phase mapping.</li>",
            "</ul>",
            "</body></html>",
        ]
    )
    return "\n".join(body)


def _html_qc_banner(qc: dict[str, Any]) -> list[str]:
    """Render the QC status banner and caution at the top of the report."""
    if not qc:
        return []
    status = qc.get("stage1_status", "UNKNOWN")
    emoji = {"PASS": "✅", "PASS_WITH_WARNINGS": "⚠️", "FAIL": "❌"}.get(status, "❓")
    bg = {"PASS": "#d4edda", "PASS_WITH_WARNINGS": "#fff3cd", "FAIL": "#f8d7da"}.get(status, "#e2e3e5")
    border = {"PASS": "#28a745", "PASS_WITH_WARNINGS": "#ffc107", "FAIL": "#dc3545"}.get(status, "#6c757d")

    lines = [
        f"<div style=\"background:{bg};border-left:6px solid {border};padding:14px 20px;margin:16px 0;border-radius:4px\">",
        f"<h2 style=\"margin:0 0 6px 0\">Stage 1 QC Status: {emoji} {status}</h2>",
        f"<p style=\"margin:0\">Warnings: {qc.get('n_warnings', '?')} &nbsp;|&nbsp; Critical: {qc.get('n_critical', '?')}</p>",
    ]

    qc_flags = qc.get("flags", [])
    if qc_flags:
        lines.append("<details style=\"margin-top:10px\"><summary><strong>QC Flags</strong></summary>")
        lines.append("<table style=\"margin-top:8px\"><thead><tr><th>Severity</th><th>Code</th><th>Message</th></tr></thead><tbody>")
        for flag in qc_flags:
            sev = flag.get("severity", "info").upper()
            lines.append(
                f"<tr><td><code>{html.escape(sev)}</code></td>"
                f"<td><code>{html.escape(flag.get('code', '?'))}</code></td>"
                f"<td>{html.escape(flag.get('message', ''))}</td></tr>"
            )
        lines.append("</tbody></table></details>")

    lines.append("</div>")

    lines.extend([
        "<blockquote style=\"border-left:4px solid #ffc107;padding:10px 16px;margin:12px 0;background:#fff8e1\">",
        "<strong>⚠ Caution:</strong> The fingerprint-class labels are generated by unsupervised "
        "clustering of radial-fingerprint profiles (PCA/NMF + KMeans). They represent "
        "<strong>radial-profile similarity groups</strong>, not confirmed crystallographic "
        "phases. Treat them as <strong>phase candidates</strong> until validated by "
        "Stage-2 py4DSTEM / pyxem Bragg indexing or template matching.",
        "</blockquote>",
    ])
    return lines


def _html_data_summary(summary: dict[str, Any]) -> list[str]:
    data = summary.get("data_config", {})
    dataset = summary.get("dataset", {})
    shapes = summary.get("shapes", {})
    data_contract = summary.get("data_contract", {})
    provenance = summary.get("provenance", {})
    rows = [
        ("Source", data.get("path", dataset.get("path", "unknown"))),
        ("Backend", dataset.get("source_backend", "unknown")),
        ("Preprocessed shape", dataset.get("shape", "unknown")),
        ("Navigation shape", dataset.get("navigation_shape", "unknown")),
        ("Signal shape", dataset.get("signal_shape", "unknown")),
        ("Radial fingerprints", shapes.get("radial_fingerprints", "unknown")),
        ("Fingerprint labels", shapes.get("fingerprint_class_labels", "unknown")),
        ("Orientation preview", shapes.get("orientation_index", "unknown")),
        ("Axis order", data_contract.get("axis_order", "nav_y_nav_x_q_y_q_x")),
        ("BBox order", data_contract.get("bbox_order", "y0_y1_x0_x1")),
        ("Centre order", data_contract.get("center_order", "y_x")),
        ("Pipeline version", provenance.get("pipeline_version", "unknown")),
        ("Git commit", provenance.get("git_commit", "unknown")),
        ("Config hash", provenance.get("config_hash", "n/a")),
    ]
    lines = ["<h2>Data And Settings</h2>", "<table><tbody>"]
    for key, value in rows:
        lines.append(f"<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(str(value))}</code></td></tr>")
    lines.append("</tbody></table>")
    return lines


def _html_label_summary(phase_labels: np.ndarray) -> list[str]:
    labels, counts = np.unique(np.asarray(phase_labels), return_counts=True)
    total = max(int(counts.sum()), 1)
    lines = ["<h2>Fingerprint Classes (Phase Candidates)</h2>", "<table><thead><tr><th>Cluster</th><th>Pixels</th><th>Fraction</th></tr></thead><tbody>"]
    for label, count in zip(labels, counts):
        lines.append(f"<tr><td>{int(label)}</td><td>{int(count)}</td><td>{int(count) / total:.3f}</td></tr>")
    lines.append("</tbody></table>")
    return lines


def _html_sample_mask(sample_mask: dict[str, Any]) -> list[str]:
    """Render sample-mask statistics as an HTML section."""
    if not sample_mask or not sample_mask.get("generated"):
        return []
    lines = [
        "<h2>Sample Mask</h2>",
        "<table><tbody>",
        f"<tr><th>Sample pixels</th><td><code>{sample_mask.get('sample_pixels', '?')}</code></td></tr>",
        f"<tr><th>Background pixels</th><td><code>{sample_mask.get('background_pixels', '?')}</code></td></tr>",
        f"<tr><th>Sample fraction</th><td><code>{sample_mask.get('sample_fraction', '?')}</code></td></tr>",
        "</tbody></table>",
        "<p>Pixels outside the sample mask are marked as background "
        "(label <code>-1</code>) and excluded from diffraction-class "
        "clustering.</p>",
    ]
    return lines


def _html_provenance(provenance: dict[str, Any]) -> list[str]:
    """Render provenance as an HTML section."""
    if not provenance:
        return []
    rows = [
        ("Pipeline version", provenance.get("pipeline_version")),
        ("Git commit", provenance.get("git_commit")),
        ("Config path", provenance.get("config_path")),
        ("Config hash", provenance.get("config_hash")),
        ("Input path", provenance.get("input_path")),
        ("Input file size", _human_size(provenance.get("input_file_size"))),
        ("Input file mtime", provenance.get("input_file_mtime")),
        ("Python version", provenance.get("python_version")),
        ("Platform", provenance.get("platform")),
        ("Random seed", provenance.get("random_seed")),
        ("Start time", provenance.get("start_time")),
        ("End time", provenance.get("end_time")),
    ]
    packages = provenance.get("packages", {})
    lines = ["<h2>Provenance</h2>", "<table><tbody>"]
    for key, value in rows:
        lines.append(
            f"<tr><th>{html.escape(str(key))}</th>"
            f"<td><code>{html.escape(str(value) if value is not None else 'n/a')}</code></td></tr>"
        )
    lines.append("</tbody></table>")
    if packages:
        lines.append("<h3>Package Versions</h3>")
        lines.append("<table><tbody>")
        for name in sorted(packages):
            version = packages.get(name)
            lines.append(
                f"<tr><th>{html.escape(name)}</th>"
                f"<td><code>{html.escape(str(version) if version is not None else 'not installed')}</code></td></tr>"
            )
        lines.append("</tbody></table>")
    return lines


def _html_png_grid(png: dict[str, Any], base_dir: Path) -> list[str]:
    keys = [
        "fingerprint_class_labels_annotated",
        "cluster_cleaned_labels",
        "cluster_mean_radial_profiles",
        "cluster_virtual_image_statistics",
        "cluster_vs_orientation_heatmap",
        "ring_2_over_ring_1",
        "ring_3_over_ring_1",
        "roi_candidates_overlay",
        "sample_mask",
        "sample_mask_overlay_adf",
        "mean_dp_with_center_marker",
        "orientation_score_masked",
        "k_sweep_metrics",
    ]
    lines = ["<h2>Visual Outputs</h2>", "<div class=\"grid\">"]
    for key in keys:
        if key not in png:
            continue
        path = Path(png[key])
        try:
            link = path.relative_to(base_dir).as_posix()
        except ValueError:
            link = path.as_posix()
        lines.append(
            "<figure>"
            f"<img src=\"{html.escape(link)}\" alt=\"{html.escape(key)}\">"
            f"<figcaption>{html.escape(key)}</figcaption>"
            "</figure>"
        )
    lines.append("</div>")
    return lines


def _html_diagnostic_tables(diagnostics: Any, base_dir: Path) -> list[str]:
    if not isinstance(diagnostics, dict):
        return []
    items = [
        ("Cluster virtual-image statistics", diagnostics.get("cluster_summary_csv")),
        ("Connected-component cleanup", (diagnostics.get("connected_components") or {}).get("connected_components_csv") if isinstance(diagnostics.get("connected_components"), dict) else None),
        ("Cluster vs orientation", (diagnostics.get("cluster_vs_orientation") or {}).get("csv") if isinstance(diagnostics.get("cluster_vs_orientation"), dict) else None),
        ("Stage 2 ROI candidates", (diagnostics.get("roi_outputs") or {}).get("csv") if isinstance(diagnostics.get("roi_outputs"), dict) else None),
        ("K-sweep metrics", (diagnostics.get("k_sweep") or {}).get("metrics_csv") if isinstance(diagnostics.get("k_sweep"), dict) else None),
    ]
    lines = ["<h2>Diagnostic Tables</h2>"]
    for title, path_text in items:
        table = _csv_to_html(path_text, max_rows=12)
        if table:
            lines.extend([f"<h3>{html.escape(title)}</h3>", table])
    return lines


def _csv_to_markdown(path_text: Any, *, base_dir: Path, max_rows: int) -> str:
    if not path_text:
        return ""
    path = Path(str(path_text))
    if not path.exists():
        return ""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ""
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    if len(rows) > max_rows:
        try:
            link = path.relative_to(base_dir).as_posix()
        except ValueError:
            link = path.as_posix()
        tail = [""] * len(keys)
        tail[0] = "..."
        if len(tail) > 1:
            tail[1] = f"see [{path.name}]({link})"
        lines.append("| " + " | ".join(tail) + " |")
    return "\n".join(lines)


def _csv_to_html(path_text: Any, *, max_rows: int) -> str:
    if not path_text:
        return ""
    path = Path(str(path_text))
    if not path.exists():
        return ""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ""
    keys = list(rows[0])
    lines = ["<table><thead><tr>"]
    lines.extend(f"<th>{html.escape(key)}</th>" for key in keys)
    lines.extend(["</tr></thead><tbody>"])
    for row in rows[:max_rows]:
        lines.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in keys) + "</tr>")
    if len(rows) > max_rows:
        lines.append(f"<tr><td colspan=\"{len(keys)}\">Showing {max_rows} of {len(rows)} rows. Full CSV: {html.escape(path.name)}</td></tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


_FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "=": ["00000", "11111", "00000", "00000", "11111", "00000", "00000"],
}


def _human_size(size: Any) -> str:
    """Format a byte count as a human-readable string."""
    if size is None:
        return "n/a"
    size = int(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024:
            return f"{size} {unit}"
        size //= 1024
    return f"{size} TiB"


def _render_package_versions(packages: dict[str, Any]) -> str:
    """Render optional package versions as a Markdown list."""
    if not packages:
        return "Package versions: not recorded"
    items = [f"- `{name}`: `{version or 'not installed'}`" for name, version in sorted(packages.items())]
    return "### Package Versions\n\n" + "\n".join(items)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
