"""Command-line interface for the 4D-STEM analysis pipeline.

``fourdstem-run``
    Execute the full Stage-1 workflow.
``fourdstem-dry-run``
    Validate configuration and estimate resources without loading data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Entry points (registered in pyproject.toml [project.scripts])
# ---------------------------------------------------------------------------


def run() -> None:
    """``fourdstem-run`` — execute the Stage-1 workflow."""
    parser = _base_parser()
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity  [default: INFO]",
    )
    args = parser.parse_args()

    from .logging import configure_pipeline_logging
    from .workflow import run_workflow

    configure_pipeline_logging(level=args.log_level)
    result = run_workflow(config=args.config, log_level=args.log_level)

    if result.errors:
        print(f"\nWorkflow finished with {len(result.errors)} error(s).", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nWorkflow finished successfully.")


def dry_run() -> None:
    """``fourdstem-dry-run`` — validate config and estimate resources."""
    parser = _base_parser()
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        default=False,
        help="Also print the dry-run result as JSON to stdout.",
    )
    args = parser.parse_args()
    result = _perform_dry_run(args.config)

    # --- Human-readable output -------------------------------------------
    _print_dry_run(result)

    # --- JSON output -----------------------------------------------------
    if args.json_output:
        print(json.dumps(_jsonable(result), indent=2))

    # --- Save dry_run_summary.json ---------------------------------------
    output_dir = result.get("output_dir")
    if output_dir:
        out_path = Path(output_dir) / "dry_run_summary.json"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(_jsonable(result), indent=2, default=str), encoding="utf-8",
            )
            print(f"\nDry-run summary saved to {out_path}")
        except OSError as exc:
            print(f"\nCould not save dry-run summary: {exc}", file=sys.stderr)

    # Exit code
    if result["status"] == "FAIL":
        sys.exit(1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="4D-STEM Stage-1 analysis pipeline",
    )
    p.add_argument(
        "--config",
        default="configs/default_workflow.yaml",
        help="Path to YAML workflow configuration  [default: configs/default_workflow.yaml]",
    )
    return p


def _perform_dry_run(config_path: str) -> dict[str, Any]:
    """Run all dry-run checks and return a result dict."""
    checks: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    # Resolve config path relative to repo root
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        cfg_path = repo_root / cfg_path

    # ── 1. Load config ──────────────────────────────────────────────────
    try:
        from .config import load_workflow_config, resolve_data_config

        cfg = load_workflow_config(str(cfg_path))
        checks.append(_ok("config_loaded", f"Config loaded from {cfg_path}"))
        _warn_unknown_keys(cfg, checks)
    except FileNotFoundError:
        return _fail("config_missing", f"Config file not found: {cfg_path}", checks)
    except Exception as exc:
        return _fail("config_error", f"Failed to load config: {exc}", checks)

    # ── 2. Input file ───────────────────────────────────────────────────
    data_cfg = resolve_data_config(cfg.get("data", {}))
    input_path = data_cfg.get("path", "")

    if input_path and input_path not in ("synthetic://demo",):
        input_file = Path(input_path)
        if not input_file.exists():
            return _fail("input_missing", f"Input file not found: {input_path}", checks)
        checks.append(_ok("input_exists", f"Input file exists: {input_path}"))

        file_size = input_file.stat().st_size
        if file_size == 0:
            checks.append(_warn("input_empty", "Input file is empty (0 bytes)."))
        elif file_size < 1024:
            checks.append(_warn("input_small", f"Input file is very small ({file_size} B)."))
        else:
            checks.append(_ok("input_size", f"File size: {_human_size(file_size)}"))
    elif input_path == "synthetic://demo":
        checks.append(_ok("input_synthetic", "Using synthetic demo dataset (16x16x64x64)."))
    else:
        checks.append(_warn("input_unspecified", "No data path or directory specified; falling back to synthetic://demo."))

    # ── 3. Scan / detector shape ────────────────────────────────────────
    scan_shape = data_cfg.get("scan_shape")
    detector_shape = data_cfg.get("detector_shape")
    preprocess_cfg = cfg.get("preprocess", {})

    if scan_shape:
        ny, nx = int(scan_shape[0]), int(scan_shape[1])
        if ny <= 0 or nx <= 0:
            checks.append(_warn("scan_shape_invalid", f"scan_shape {scan_shape} has non-positive values."))
        else:
            checks.append(_ok("scan_shape", f"Scan shape: [{ny}, {nx}]"))
    else:
        checks.append(_info("scan_shape_unknown", "scan_shape not specified; cannot estimate navigation size."))

    if detector_shape:
        sy, sx = int(detector_shape[0]), int(detector_shape[1])
        if sy <= 0 or sx <= 0:
            checks.append(_warn("det_shape_invalid", f"detector_shape {detector_shape} has non-positive values."))
        else:
            checks.append(_ok("det_shape", f"Detector shape: [{sy}, {sx}]"))
    else:
        checks.append(_info("det_shape_unknown", "detector_shape not specified; cannot estimate signal size."))

    # ── 4. Shape estimation ─────────────────────────────────────────────
    shape_est = _estimate_output_shape(scan_shape, detector_shape, preprocess_cfg)
    checks.append(_ok("shape_estimate", _format_shape_estimate(shape_est)))

    # ── 5. Chunk / block estimation ─────────────────────────────────────
    block_shape = _resolve_block_shape(cfg, data_cfg)
    chunk_est = _estimate_chunks(shape_est, block_shape)
    checks.append(_ok("chunk_estimate", _format_chunk_estimate(chunk_est)))

    # ── 6. Memory estimate ──────────────────────────────────────────────
    mem_est = _estimate_memory(shape_est, block_shape, data_cfg.get("lazy", True))
    checks.append(_ok("memory_estimate", _format_memory_estimate(mem_est)))

    # ── 7. Output directory ─────────────────────────────────────────────
    project_cfg = cfg.get("project", {})
    output_dir = Path(project_cfg.get("output_dir", "outputs"))

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        checks.append(_ok("output_dir_writable", f"Output directory is writable: {output_dir}"))
    except OSError as exc:
        return _fail("output_dir_unwritable", f"Cannot write to output directory {output_dir}: {exc}", checks)

    # ── 8. Existing results ─────────────────────────────────────────────
    existing = _check_existing_results(output_dir)
    if existing:
        checks.append(
            _warn(
                "existing_results",
                f"Output directory contains {len(existing)} previous output(s) "
                f"that may be overwritten: {', '.join(existing[:5])}"
                + ("..." if len(existing) > 5 else ""),
                evidence={"existing_files": existing},
            )
        )
    else:
        checks.append(_ok("output_dir_clean", "Output directory is empty (no prior results)."))

    # ── 9. Enabled stages ───────────────────────────────────────────────
    stages = _resolve_stages(cfg)
    checks.append(_ok("stages", f"Enabled stages ({len(stages)}): {', '.join(stages)}"))

    # ── 10. Sample mask config ──────────────────────────────────────────
    sample_mask_cfg = cfg.get("sample_mask", {})
    if sample_mask_cfg.get("enabled", True):
        source = sample_mask_cfg.get("source", "adf")
        checks.append(
            _ok(
                "sample_mask_enabled",
                f"Sample mask enabled (source={source}, "
                f"percentile={sample_mask_cfg.get('percentile', 15)}, "
                f"min_size={sample_mask_cfg.get('min_size', 100)}).",
            )
        )

    # ── Aggregate ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_error = sum(1 for c in checks if c["status"] == "ERROR")

    if n_error > 0:
        overall = "FAIL"
    elif n_warn > 0:
        overall = "OK_WITH_WARNINGS"
    else:
        overall = "OK"

    return {
        "status": overall,
        "config_path": str(cfg_path),
        "input_path": input_path or "synthetic://demo",
        "output_dir": str(output_dir),
        "scan_shape": list(scan_shape) if scan_shape else None,
        "detector_shape": list(detector_shape) if detector_shape else None,
        "preprocess": {
            "q_crop": preprocess_cfg.get("q_crop"),
            "q_bin": int(preprocess_cfg.get("q_bin", 1)),
            "r_bin": int(preprocess_cfg.get("r_bin", 1)),
        },
        "estimated_shape": shape_est,
        "block_shape": list(block_shape),
        "chunk_estimate": chunk_est,
        "memory_estimate": mem_est,
        "stages": stages,
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed, 3),
    }


# ── Check helpers ────────────────────────────────────────────────────────────


def _ok(code: str, message: str) -> dict[str, Any]:
    return {"status": "OK", "code": code, "message": message}


def _warn(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "WARN", "code": code, "message": message, **extra}


def _info(code: str, message: str) -> dict[str, Any]:
    return {"status": "INFO", "code": code, "message": message}


def _fail(code: str, message: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a FAIL result with the given check appended to *checks*."""
    checks.append({"status": "ERROR", "code": code, "message": message})
    return {
        "status": "FAIL",
        "config_path": None,
        "input_path": None,
        "output_dir": None,
        "scan_shape": None,
        "detector_shape": None,
        "preprocess": {},
        "estimated_shape": None,
        "block_shape": None,
        "chunk_estimate": {},
        "memory_estimate": {},
        "stages": [],
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": 0,
    }


# ── Shape / chunk / memory estimation ────────────────────────────────────────


def _estimate_output_shape(
    scan_shape: list | tuple | None,
    detector_shape: list | tuple | None,
    preprocess_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Estimate the preprocessed 4-D shape."""
    if not scan_shape or not detector_shape:
        return None
    ny, nx = int(scan_shape[0]), int(scan_shape[1])
    sy, sx = int(detector_shape[0]), int(detector_shape[1])

    q_crop = preprocess_cfg.get("q_crop")
    if q_crop and len(q_crop) == 4:
        qy0, qy1, qx0, qx1 = [int(v) for v in q_crop]
        sy = max(1, qy1 - qy0)
        sx = max(1, qx1 - qx0)

    q_bin = max(1, int(preprocess_cfg.get("q_bin", 1)))
    r_bin = max(1, int(preprocess_cfg.get("r_bin", 1)))

    sy_out = math.ceil(sy / q_bin)
    sx_out = math.ceil(sx / q_bin)
    ny_out = math.ceil(ny / r_bin)
    nx_out = math.ceil(nx / r_bin)

    return {
        "input": [ny, nx, sy, sx],
        "output": [ny_out, nx_out, sy_out, sx_out],
        "nav_output": [ny_out, nx_out],
        "sig_output": [sy_out, sx_out],
    }


def _resolve_block_shape(cfg: dict[str, Any], data_cfg: dict[str, Any]) -> tuple[int, int]:
    """Resolve the navigation block shape from config or defaults."""
    block_shape = cfg.get("block_shape")
    if block_shape is not None:
        return (max(1, int(block_shape[0])), max(1, int(block_shape[1])))
    chunks = data_cfg.get("chunks", {})
    if isinstance(chunks, dict):
        nav_chunks = chunks.get("navigation", (8, 8))
    else:
        nav_chunks = chunks[:2] if chunks else (8, 8)
    return (max(1, int(nav_chunks[0])), max(1, int(nav_chunks[1])))


def _estimate_chunks(
    shape_est: dict[str, Any] | None,
    block_shape: tuple[int, int],
) -> dict[str, Any]:
    """Estimate the number of navigation blocks."""
    if shape_est is None:
        return {"block_shape": list(block_shape), "n_chunks": None}
    nav = shape_est["nav_output"]
    by, bx = block_shape
    ny, nx = nav
    n_chunks = math.ceil(ny / by) * math.ceil(nx / bx)
    return {
        "block_shape": list(block_shape),
        "n_chunks": n_chunks,
        "blocks_per_dim": [math.ceil(ny / by), math.ceil(nx / bx)],
    }


def _estimate_memory(
    shape_est: dict[str, Any] | None,
    block_shape: tuple[int, int],
    lazy: bool,
) -> dict[str, Any]:
    """Estimate memory usage."""
    if shape_est is None:
        return {"mode": "unknown"}

    out = shape_est["output"]
    total_pixels = out[0] * out[1] * out[2] * out[3]
    total_gb = total_pixels * 4 / (1024**3)  # float32

    by, bx = block_shape
    sy, sx = shape_est["sig_output"]
    chunk_pixels = by * bx * sy * sx
    chunk_mb = chunk_pixels * 4 / (1024**2)

    if lazy:
        mode = "lazy (per-chunk)"
        peak_gb = max(chunk_mb / 1024, 0.001)  # at least 1 MB
    else:
        mode = "eager (full dataset)"
        peak_gb = total_gb

    return {
        "mode": mode,
        "total_dataset_gb": round(total_gb, 3),
        "per_chunk_mb": round(chunk_mb, 2),
        "peak_memory_gb": round(peak_gb, 3),
    }


# ── Existing results check ───────────────────────────────────────────────────


def _check_existing_results(output_dir: Path) -> list[str]:
    """Return list of existing output files that would be overwritten."""
    markers = [
        "workflow_summary.json",
        "stage1_summary.json",
        "provenance.json",
        "qc_summary.json",
    ]
    existing: list[str] = []
    try:
        for marker in markers:
            if (output_dir / marker).exists():
                existing.append(marker)
    except OSError:
        pass
    return existing


# ── Unknown keys warning ─────────────────────────────────────────────────────


def _warn_unknown_keys(cfg: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    """Detect unknown top-level keys and emit warnings."""
    known = {
        "project", "data", "preprocess", "geometry", "virtual_images",
        "phase_screening", "orientation", "roi_bragg", "sample_mask",
    }
    unknown = sorted(set(cfg) - known)
    for key in unknown:
        checks.append(_warn("unknown_section", f"Unknown top-level config section: '{key}' — will be ignored."))


# ── Stage list ───────────────────────────────────────────────────────────────


def _resolve_stages(cfg: dict[str, Any]) -> list[str]:
    """Return the ordered list of enabled stages."""
    stages = ["load", "preprocess", "virtual_images", "sample_mask", "fingerprints",
              "fingerprint_classes", "orientation_preview"]
    roi_cfg = cfg.get("roi_bragg", {})
    if roi_cfg.get("enabled", False):
        stages.append("roi_bragg")
    stages.extend(["png_exports", "diagnostics", "provenance", "qc", "report"])
    return stages


# ── Formatting helpers ────────────────────────────────────────────────────────


def _format_shape_estimate(est: dict[str, Any] | None) -> str:
    if est is None:
        return "Cannot estimate — provide scan_shape and detector_shape in data config."
    inp = est["input"]
    out = est["output"]
    nav = est["nav_output"]
    sig = est["sig_output"]
    return (
        f"Input {inp} -> r_bin x q_bin -> Output {out} "
        f"(nav {nav}, sig {sig})"
    )


def _format_chunk_estimate(est: dict[str, Any]) -> str:
    n = est.get("n_chunks")
    if n is None:
        return f"Block shape {est['block_shape']} — n_chunks: unknown"
    return (
        f"Block shape {est['block_shape']} -> "
        f"{n} navigation chunk(s) "
        f"({est['blocks_per_dim'][0]}x{est['blocks_per_dim'][1]} grid)"
    )


def _format_memory_estimate(est: dict[str, Any]) -> str:
    mode = est.get("mode", "unknown")
    if mode == "unknown":
        return "Cannot estimate — provide scan_shape and detector_shape."
    return (
        f"{mode}: ~{est['total_dataset_gb']:.3f} GB total, "
        f"~{est['per_chunk_mb']:.1f} MB/chunk, "
        f"~{est['peak_memory_gb']:.3f} GB peak"
    )


def _human_size(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


# ── Pretty-printer ───────────────────────────────────────────────────────────


def _print_dry_run(result: dict[str, Any]) -> None:
    """Print the dry-run result in a human-readable format."""
    status = result["status"]
    label = {"OK": "[OK]", "OK_WITH_WARNINGS": "[OK/WARN]", "FAIL": "[FAIL]"}.get(status, "[?]")

    print()
    print("=" * 64)
    print(f"  DRY RUN  {label}  {status}")
    print("=" * 64)
    print()

    if result.get("input_path"):
        _safe_print(f"  Input:       {result['input_path']}")
    if result.get("scan_shape"):
        _safe_print(f"  Scan shape:  {result['scan_shape']}")
    if result.get("detector_shape"):
        _safe_print(f"  Det. shape:  {result['detector_shape']}")
    print()

    prep = result.get("preprocess", {})
    print("  Preprocess:")
    _safe_print(f"    q_crop: {prep.get('q_crop')}")
    _safe_print(f"    q_bin:  {prep.get('q_bin', 1)}")
    _safe_print(f"    r_bin:  {prep.get('r_bin', 1)}")
    if result.get("estimated_shape"):
        est = result["estimated_shape"]
        _safe_print(f"    Est. output shape: {est['output']}")
    print()

    stages = result.get("stages", [])
    if stages:
        _safe_print(f"  Stages ({len(stages)}):")
        for s in stages:
            _safe_print(f"    - {s}")
        print()

    chunk = result.get("chunk_estimate", {})
    if chunk.get("n_chunks"):
        _safe_print(f"  Chunks:      {chunk['n_chunks']}  (block {chunk['block_shape']}, {chunk['blocks_per_dim']} grid)")
        print()

    mem = result.get("memory_estimate", {})
    if mem.get("mode") and mem["mode"] != "unknown":
        _safe_print(f"  Memory:      {mem.get('mode')}")
        _safe_print(f"    Total:    ~{mem['total_dataset_gb']:.3f} GB")
        _safe_print(f"    Per chunk: ~{mem['per_chunk_mb']:.1f} MB")
        _safe_print(f"    Peak:     ~{mem['peak_memory_gb']:.3f} GB")
        print()

    _safe_print(f"  Output:      {result.get('output_dir', 'unknown')}")
    _safe_print(f"  Elapsed:     {result.get('elapsed_s', 0):.3f} s")
    print()

    # Print checks
    checks = result.get("checks", [])
    status_label = {"OK": "v", "WARN": "!", "ERROR": "X", "INFO": "i"}
    for c in checks:
        sl = status_label.get(c["status"], "?")
        _safe_print(f"  {sl}  [{c['code']}]  {c['message']}")

    print()
    if status == "OK":
        print("  DRY RUN OK -- ready to run the workflow.")
    elif status == "OK_WITH_WARNINGS":
        print("  DRY RUN OK (with warnings) -- review warnings above before running.")
    else:
        print("  DRY RUN FAILED -- fix the errors above before running.")
    print()


def _safe_print(text: str) -> None:
    """Print *text*, replacing characters that can't be encoded by the terminal."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace non-encodable characters with '?'
        encoded = text.encode(sys.stdout.encoding or "ascii", errors="replace")
        sys.stdout.buffer.write(encoded + b"\n")


# ── JSON serialisation ───────────────────────────────────────────────────────


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return str(value)
