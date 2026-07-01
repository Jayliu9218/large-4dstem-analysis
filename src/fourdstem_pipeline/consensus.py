"""Consensus/conflict maps for Stage 2B and Stage 2C evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


CONSENSUS_LABELS = {
    0: "MIXED_OR_NOT_INDEXABLE",
    1: "AGREED_Ti_bcc",
    2: "AGREED_Ti_hcp",
    3: "PY4DSTEM_ONLY_Ti_bcc",
    4: "PY4DSTEM_ONLY_Ti_hcp",
    5: "PYXEM_ONLY_Ti_bcc",
    6: "PYXEM_ONLY_Ti_hcp",
    7: "AMBIGUOUS_BOTH",
    8: "CONFLICT_BCC_HCP",
}


def run_consensus(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Build a consensus map from standardised Stage 2B and Stage 2C manifests."""

    cfg, config_path = _load_config(config)
    output_dir = Path(cfg.get("output_dir") or _default_output_dir(cfg))
    output_dir.mkdir(parents=True, exist_ok=True)

    stage2b_manifest = _resolve_manifest(cfg.get("stage2b_manifest") or cfg.get("stage2b_results"))
    stage2c_manifest = _resolve_manifest(cfg.get("stage2c_manifest") or cfg.get("stage2c_results"))

    errors: list[dict[str, Any]] = []
    if stage2b_manifest is None or not stage2b_manifest.exists():
        errors.append({"stage": "consensus", "error": "Missing Stage 2B standard manifest."})
    if stage2c_manifest is None or not stage2c_manifest.exists():
        errors.append({"stage": "consensus", "error": "Missing Stage 2C standard manifest."})
    if errors:
        return _write_summary(output_dir, config_path, "skipped", cfg, errors=errors)

    try:
        b_manifest = json.loads(stage2b_manifest.read_text(encoding="utf-8"))
        c_manifest = json.loads(stage2c_manifest.read_text(encoding="utf-8"))
        b = _load_standard_arrays(stage2b_manifest.parent, b_manifest)
        c = _load_standard_arrays(stage2c_manifest.parent, c_manifest)
        if b["phase_index"].shape != c["phase_index"].shape:
            raise ValueError(f"Stage 2B/2C map shapes differ: {b['phase_index'].shape} vs {c['phase_index'].shape}")
        consensus = _build_consensus_map(b, c)
    except Exception as exc:
        return _write_summary(output_dir, config_path, "failed", cfg, errors=[{"stage": "consensus", "error": str(exc)}])

    npz_path = output_dir / "consensus_phase_map.npz"
    np.savez_compressed(npz_path, consensus_label=consensus)
    npy_path = output_dir / "consensus_label.npy"
    np.save(npy_path, consensus)
    counts = _label_counts(consensus)
    report_path = _write_conflict_report(output_dir, counts)
    final_report_path = _write_final_interpretation_report(output_dir, counts)
    gallery_path = _write_representative_overlay_gallery(output_dir, b_manifest)

    return _write_summary(
        output_dir,
        config_path,
        "success",
        cfg,
        errors=[],
        extra={
            "consensus_npz": str(npz_path),
            "consensus_label_path": str(npy_path),
            "conflict_report": str(report_path),
            "final_interpretation_report": str(final_report_path),
            "representative_overlay_gallery": str(gallery_path) if gallery_path else None,
            "label_counts": counts,
            "labels": CONSENSUS_LABELS,
        },
    )


def _build_consensus_map(stage2b: dict[str, np.ndarray], stage2c: dict[str, np.ndarray]) -> np.ndarray:
    b_phase = stage2b["phase_index"].astype(np.int16)
    c_phase = stage2c["phase_index"].astype(np.int16)
    b_high = stage2b["high_confidence_mask"].astype(bool)
    c_high = stage2c["high_confidence_mask"].astype(bool)
    b_amb = stage2b["ambiguous_mask"].astype(bool)
    c_amb = stage2c["ambiguous_mask"].astype(bool)

    out = np.zeros(b_phase.shape, dtype=np.int16)
    out[(b_high & c_high) & (b_phase == 0) & (c_phase == 0)] = 1
    out[(b_high & c_high) & (b_phase == 1) & (c_phase == 1)] = 2
    out[(b_high & ~c_high) & (b_phase == 0)] = 3
    out[(b_high & ~c_high) & (b_phase == 1)] = 4
    out[(~b_high & c_high) & (c_phase == 0)] = 5
    out[(~b_high & c_high) & (c_phase == 1)] = 6
    out[(b_amb & c_amb) | (~b_high & ~c_high & (b_phase >= 0) & (c_phase >= 0))] = 7
    out[(b_high & c_high) & (b_phase != c_phase)] = 8
    out[(b_phase < 0) | (c_phase < 0)] = 0
    return out


def _load_standard_arrays(base: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = manifest.get("arrays") or {}
    required = ["phase_index", "ambiguous_mask", "high_confidence_mask"]
    loaded: dict[str, np.ndarray] = {}
    for key in required:
        value = arrays.get(key)
        if not value:
            raise ValueError(f"Manifest is missing required array path {key!r}.")
        loaded[key] = np.load(base / str(value), allow_pickle=False)
    return loaded


def _write_summary(
    output_dir: Path,
    config_path: Path | None,
    status: str,
    cfg: dict[str, Any],
    *,
    errors: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "schema_version": "consensus-map-v1",
        "status": status,
        "config_path": str(config_path) if config_path else None,
        "output_dir": str(output_dir),
        "inputs": {
            "stage2b_manifest": cfg.get("stage2b_manifest") or cfg.get("stage2b_results"),
            "stage2c_manifest": cfg.get("stage2c_manifest") or cfg.get("stage2c_results"),
        },
        "errors": errors if errors else None,
    }
    if extra:
        summary.update(extra)
    path = output_dir / "consensus_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _write_conflict_report(output_dir: Path, counts: dict[str, int]) -> Path:
    lines = ["# Consensus / Conflict Report", ""]
    total = sum(counts.values())
    for key, label in CONSENSUS_LABELS.items():
        count = counts.get(str(key), 0)
        frac = count / max(total, 1)
        lines.append(f"- {key}: {label} - {count} pixels ({frac:.2%})")
    path = output_dir / "conflict_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_final_interpretation_report(output_dir: Path, counts: dict[str, int]) -> Path:
    total = sum(counts.values())
    lines = [
        "# Stage 2D Consensus Evidence Report",
        "",
        "Final interpretation is based on py4DSTEM Bragg-vector evidence plus independent pyxem polar-template evidence.",
        "",
        "先证明 Bragg peaks 和 q calibration 是可信的，再谈 Ti-bcc / Ti-hcp phase identification。否则继续优化匹配阈值没有意义。",
        "",
        "| Consensus class | Fraction | Pixels |",
        "| --- | ---: | ---: |",
    ]
    for key, label in CONSENSUS_LABELS.items():
        count = counts.get(str(key), 0)
        lines.append(f"| `{label}` | {count / max(total, 1):.2%} | {count} |")
    lines.extend([
        "",
        "## Interpretation Rules",
        "",
        "- `AGREED_Ti_bcc` and `AGREED_Ti_hcp` are the only consensus-confirmed classes.",
        "- Backend-only classes remain candidate evidence and should be reviewed in the representative overlay gallery.",
        "- `CONFLICT_BCC_HCP`, `AMBIGUOUS_BOTH`, and `MIXED_OR_NOT_INDEXABLE` are not acceptable final phase calls.",
        "",
    ])
    path = output_dir / "stage2d_consensus_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_representative_overlay_gallery(output_dir: Path, stage2b_manifest: dict[str, Any]) -> Path | None:
    evidence = stage2b_manifest.get("roi_evidence") or []
    if not evidence:
        return None
    categories = [
        ("agreed bcc", lambda r: _phase_contains(r, "bcc") and r.get("indexability_tier") == "INDEXABLE"),
        ("agreed hcp", lambda r: _phase_contains(r, "hcp") and r.get("indexability_tier") == "INDEXABLE"),
        ("py4DSTEM-only hcp", lambda r: _phase_contains(r, "hcp") and r.get("mapping_confidence") == "HIGH_CONFIDENCE"),
        ("pyxem-only hcp", lambda r: False),
        ("conflict bcc/hcp", lambda r: "/" in str(r.get("candidate_phase") or "") or r.get("phase_call") == "AMBIGUOUS"),
        ("ambiguous", lambda r: r.get("phase_call") == "AMBIGUOUS"),
        ("not indexable", lambda r: r.get("indexability_tier") == "NOT_INDEXABLE"),
    ]
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Representative DP-template Overlay Gallery</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#1f2937}section{margin-bottom:28px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d1d5db;padding:8px;vertical-align:top}img{max-width:220px;max-height:220px;object-fit:contain}.muted{color:#6b7280}</style>",
        "</head><body>",
        "<h1>Representative DP-template Overlay Gallery</h1>",
        "<p>Each row links the raw/mean DP evidence, detected Bragg peak overlay, template overlay, score, margin, and q residual when available.</p>",
    ]
    used: set[str] = set()
    for title, predicate in categories:
        rows = [r for r in evidence if str(r.get("name")) not in used and predicate(r)]
        if rows:
            used.add(str(rows[0].get("name")))
        lines.append(f"<section><h2>{title}</h2>")
        if not rows:
            lines.append("<p class='muted'>No representative ROI available for this category.</p></section>")
            continue
        lines.append("<table><tr><th>ROI</th><th>Raw DP</th><th>Detected Bragg peaks</th><th>Template overlay</th><th>Metrics</th></tr>")
        for row in rows[:3]:
            paths = row.get("paths") or {}
            lines.append(
                "<tr>"
                f"<td>{_html_escape(str(row.get('name', '')))}</td>"
                f"<td>{_gallery_cell(paths.get('raw_dp'))}</td>"
                f"<td>{_gallery_cell(paths.get('detected_bragg_peaks'))}</td>"
                f"<td>{_gallery_cell(paths.get('template_overlay'))}</td>"
                "<td>"
                f"phase={_html_escape(str(row.get('candidate_phase')))}<br>"
                f"score={_html_escape(str(row.get('score')))}<br>"
                f"margin={_html_escape(str(row.get('margin')))}<br>"
                f"q residual={_html_escape(str(row.get('q_residual')))}<br>"
                f"indexability={_html_escape(str(row.get('indexability_score')))} / {_html_escape(str(row.get('indexability_tier')))}"
                "</td></tr>"
            )
        lines.append("</table></section>")
    lines.append("</body></html>")
    path = output_dir / "representative_dp_template_overlay_gallery.html"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _phase_contains(row: dict[str, Any], needle: str) -> bool:
    return needle in str(row.get("candidate_phase") or row.get("phase_call") or "").lower()


def _gallery_cell(path_value: Any) -> str:
    path = str(path_value or "")
    if not path:
        return "<span class='muted'>missing</span>"
    if path.lower().endswith((".png", ".jpg", ".jpeg")):
        return f"<img src='{_html_escape(path)}' alt='overlay'>"
    return f"<code>{_html_escape(path)}</code>"


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _label_counts(arr: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(arr, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _load_config(config: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config, (str, Path)):
        path = Path(config)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Consensus config must be a YAML mapping, got {type(raw).__name__}.")
        if "consensus" in raw:
            return copy.deepcopy(raw.get("consensus") or {}), path
        return raw, path
    return copy.deepcopy(config), None


def _resolve_manifest(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _default_output_dir(cfg: dict[str, Any]) -> Path:
    stage2c = cfg.get("stage2c_manifest") or cfg.get("stage2c_results")
    if stage2c:
        return Path(str(stage2c)).parent.parent / "consensus"
    return Path("outputs") / "consensus"
