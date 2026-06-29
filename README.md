# large-4dstem-analysis

End-to-end 4D-STEM analysis pipeline for large datasets: from raw data loading
through unsupervised fingerprint-class screening to **evidence-based
crystallographic phase validation with ambiguity detection**.

> **Design principle:** If the diffraction evidence is insufficient to
> distinguish phases, the pipeline reports `AMBIGUOUS / LOW_CONFIDENCE`
> rather than forcing a phase label.  Correlation scores, peak-position
> residuals, and negative controls all feed a hybrid validation score;
> near-degenerate impostors (e.g. TiO₂-rutile with Ti-hcp) are caught
> and reported openly.

**What you get:**
- **Stage 1** — Virtual images (BF/ADF/HAADF/COM/rings), radial fingerprints,
  unsupervised fingerprint classes, orientation preview, sample mask, ROI
  candidates, QC diagnostics (10 checks), markdown + HTML report (~40 PNGs).
- **Stage 2A** — Per-ROI py4DSTEM Bragg disk detection with beam-centre
  calibration cascade, central-exclusion peak filtering, cluster/background
  validation, Bragg QC metrics, tabular peak output (Parquet), and a report
  labelling which ROIs are ready for indexing.
- **Stage 2B (v4)** — CIF→kinematic template generation with space-group
  extinction filtering, multi-zone-axis orientation sweep, physical reciprocal
  calibration, **peak-position residual analysis**, **hybrid validation
  scoring** (correlation + matched-observable fraction + q-residual),
  **ambiguity detection** (reports `AMBIGUOUS` when evidence is degenerate),
  negative-control CIF validation, 4-tier confidence reporting
  (`HIGH_CONFIDENCE` / `MEDIUM_CONFIDENCE` / `LOW_CONFIDENCE` / `UNINDEXED`),
  score-sign QC check, ROI-level top-k phase/orientation evidence,
  experimental-template peak overlays, radial q-profile validation,
  parameter stability sweep, cluster-level phase match map, and an interactive
  PNG gallery.

**Real-data verification** (Ti, 512×512×256×256 detector, 34 GB MIB, `bin_q=2`):

| Stage | Output | Result |
|-------|--------|--------|
| 1 | Fingerprint classes | 4 clusters on 256×256 nav (`r_bin=2`) |
| 2A | Bragg detection | 75,080 total peaks across 11 ROIs (`minRelativeIntensity=0.10`, central exclusion), QC PASS |
| 2B (v4) | Phase validation | ROI-level top-k phase/orientation evidence with radial q gating; ambiguous/low-confidence calls are retained instead of forced labels |

> **Current status (v4):** `Stage 2B-v4` is an ambiguity-aware, evidence-based
> candidate phase reporter — not a confirmed phase mapper.  The limiting factor
> is diffraction evidence at `bin_q=2` (128×128 effective detector), not the
> algorithm.  See the [Real-Data Results](#real-data-results-ti-34-gb-mib)
> section for the full pipeline evolution and interpretation guidance.

---

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Three-Stage Workflow](#three-stage-workflow)
  - [Stage 1 — Fingerprint-Class Screening](#stage-1--fingerprint-class-screening)
  - [Stage 2A — ROI Bragg Detection](#stage-2a--roi-bragg-detection)
  - [Stage 2B — Crystallographic Indexing](#stage-2b--crystallographic-indexing)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Output Directory Structure](#output-directory-structure)
- [Coordinate Conventions](#coordinate-conventions)
- [Config Presets](#config-presets)
- [Real-Data Results](#real-data-results-ti-34-gb-mib)
- [Package Structure](#package-structure)
- [Data Contracts](#data-contracts-between-stages)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Installation

### Prerequisites

- **Conda environment `large-4dstem`** — provides py4DSTEM, HyperSpy, scikit-learn,
  and all other dependencies.  All commands below assume the environment is active.

### Setup

```bash
# 1. Activate the environment
conda activate large-4dstem

# 2. Install this package in editable mode
pip install -e .

# 3. Verify
python -m fourdstem_pipeline.cli dry_run --config configs/default_workflow.yaml
```

### Environment location

| Platform | Path |
|----------|------|
| Windows (user) | `C:\Users\<user>\.conda\envs\large-4dstem` |
| Linux / macOS | `<conda_root>/envs/large-4dstem` |

> **Windows note:** If you see `Windows Error 0xc06d007f` when importing
> py4DSTEM, this is a known OpenBLAS/threadpoolctl compatibility issue.
> Workaround: `pip install threadpoolctl==3.5.0` in the `large-4dstem`
> environment, or set `SET OPENBLAS_CORETYPE=Haswell` before running.

---

## Quick Start

```bash
# 1. Validate config — no data loaded, ~1 second
python -m fourdstem_pipeline.cli dry_run --config configs/default_workflow.yaml

# 2. Stage 1 on synthetic demo dataset (16×16×64×64)
python -m fourdstem_pipeline.cli run --config configs/default_workflow.yaml

# 3. Full 3-stage pipeline on real data (Ti example)
python -m fourdstem_pipeline.cli run --config configs/0617_4d_stage1_enhanced.yaml
python -m fourdstem_pipeline.cli stage2 --config configs/stage2_roi_bragg.yaml
python -m fourdstem_pipeline.cli stage2b --config configs/stage2_indexing.yaml

# 4. Run tests
python -m pytest tests/ -v
# Expected: ~49 passed, 1 skipped (real-data smoke test)
```

---

## Three-Stage Workflow

### Stage 1 — Fingerprint-Class Screening

```bash
python -m fourdstem_pipeline.cli run --config configs/0617_4d_stage1_enhanced.yaml
```

**What it does:**
1. Loads 4D-STEM data (MIB/HyperSpy, .npy/.npz, or synthetic)
2. Lazy preprocessing: `q_crop`, `q_bin`, `r_bin`
3. Virtual images: BF, ADF, HAADF, COM-x, COM-y, ring images
4. Sample mask (percentile threshold + morphological cleaning)
5. Radial fingerprints (per-pixel radial intensity profiles)
6. Unsupervised fingerprint-class screening (PCA + NMF + KMeans)
7. Orientation preview (low-resolution COM-angle proxy)
8. ROI candidates from: connected components, boundaries, orientation
   extremes, intensity anomalies
9. QC checks → `qc_summary.json` (PASS / PASS_WITH_WARNINGS / FAIL)
10. Markdown + HTML report with ~40 diagnostic PNGs

**Key outputs:**

| File | Content |
|------|---------|
| `stage1_summary.json` | Canonical manifest bridging to Stage 2 |
| `report.html` | Human-readable report with inline PNGs |
| `qc_summary.json` | QC flags and PASS/FAIL verdict |
| `virtual/virtual_images.npz` | BF, ADF, HAADF, ring, COM maps |
| `fingerprints/radial_fingerprints.npy` | Per-pixel radial profiles |
| `fingerprint_classes/fingerprint_class_labels.npy` | Class labels (NOT crystallographic phases) |
| `orientation/orientation_index.npy` | Per-pixel orientation index |
| `roi_candidates/roi_candidates.yaml` | ROI proposals with bboxes and rationale |
| `png/` | ~40 diagnostic PNGs |

---

### Stage 2A — ROI Bragg Detection

```bash
python -m fourdstem_pipeline.cli stage2 --config configs/stage2_roi_bragg.yaml
```

**What it does:**
1. Loads the Stage 1 manifest and ROI candidates
2. Loads original 4D-STEM data via py4DSTEM (`import_file`)
3. Converts bboxes from binned (Stage 1) to raw scan coordinates
4. Extracts per-ROI 4D sub-cubes with navigation thinning (`thin_r`)
5. Runs py4DSTEM `find_Bragg_disks()` with detector binning (`bin_q`)
6. Beam-centre provenance cascade: Stage 1 COM → py4DSTEM calibration →
   detector-centre fallback
7. Bragg peak QC: centre-zone, edge, duplicate fractions, radial stats
8. Validates ROIs against fingerprint labels and sample mask
9. Saves per-ROI tabular peaks as `bragg_peaks.parquet`
10. Per-ROI PNGs: mean DP, Bragg vector map, Bragg overlay, radius histogram
11. `stage2_report.html` with [READY]/[REVIEW]/[SKIP] verdicts
12. `stage2_benchmark.json` with per-ROI timing

**Key outputs:**

| File | Content |
|------|---------|
| `stage2_summary.json` | Aggregate results with `stage1_bbox` + `raw_bbox` |
| `stage2_qc_summary.json` | QC flags: NO_BRAGG_PEAKS, HIGH_BACKGROUND_ROIS, etc. |
| `stage2_report.html` | Per-ROI table with verdicts and Bragg QC |
| `stage2_benchmark.json` | Per-ROI extraction time, Bragg time, data size |
| `stage2_gallery.html` | Interactive PNG gallery (detail + cross-ROI comparison) |
| `roi_<name>/bragg_vector_map.npy` | Bragg peak vote histogram |
| `roi_<name>/bragg_summary.json` | Full per-ROI metadata + QC |
| `roi_<name>/bragg_peaks.parquet` | Tabular peaks: scan_y/x, qy/x, intensity, snr |
| `roi_<name>/roi_data.npy` | 4D sub-cube (only when `save_roi_data: true`) |

**Key parameters:** `thin_r`, `bin_q`, `corr_power`, `edge_boundary`,
`min_relative_intensity`, `max_num_peaks`, `min_peak_spacing`, `subpixel`,
`cuda`, `max_rois`, `save_roi_data`

> **`save_roi_data` note:** When `false` (default), the large 4D sub-cube is
> skipped. Stage 2B falls back to `bragg_vector_map.npy` for template matching,
> which produces usable but lower-quality correlation scores. Set to `true` if
> you need full mean-DP-based matching for publication-quality results.

---

### Stage 2B — Crystallographic Indexing (v3)

```bash
python -m fourdstem_pipeline.cli stage2b --config configs/stage2_indexing.yaml
# Parameter stability sweep:
python scripts/run_stage2b_sweep.py --config configs/stage2_indexing.yaml
```

**What it does:**
1. Filters ROIs via `is_roi_ready_for_indexing()` (peaks>0, bg≤50%,
   sample coverage>0, beam calibrated)
2. Parses CIF files for lattice parameters and space group
3. Generates kinematic template stacks from the reciprocal lattice:
   - Multi-zone-axis orthographic projection (5 axes: [001], [100], [110], [111], [112])
   - In-plane orientation sweep (`orientation_step_deg`)
   - Space-group extinction filtering (P6₃/mmc, Im-3m, Fm-3m, Fd-3m, R-3m, P4₂/mnm)
   - Gaussian spot rendering (`peak_sigma_px` controls CBED disk width)
   - Intensity ≈ 1/|q|^power kinematic proxy
   - Physical reciprocal calibration (`reciprocal_pixels_per_inv_angstrom`)
4. Matches ROI mean DPs against templates via normalized correlation,
   reporting best (candidate, zone axis, orientation) triplet per candidate
5. **Peak-position residual analysis** — reconstructs template peak
   coordinates from hkl/qxy metadata, extracts measured peaks from the
   Bragg vector map via local-maximum detection, performs greedy
   closest-pair matching with cKDTree
6. **Hybrid validation scoring** — combines correlation (35%) +
   matched-observable template fraction (40%) + q-residual (15%) +
   unexplained experimental fraction (10%)
7. **v4 phase/orientation evidence** keeps top-k template matches per phase,
   computes radial q support before interpreting 2D orientation matches, and
   reports separate phase, orientation, and mapping confidence fields.
7. **Ambiguity detection** — reports `AMBIGUOUS` when hybrid scores of
   competing candidates are within 0.08 and matched fractions are < 20%,
   or when the correlation winner differs from the peak-matching winner
8. **Negative-control validation** — scores all candidates (including
   wrong-CIF controls: TiO₂-rutile, Ni-fcc, Al-fcc, Fe-bcc, Ti-hcp-wrong-a)
   and checks that the best candidate meaningfully outperforms them
9. **4-tier confidence:** `HIGH_CONFIDENCE` / `MEDIUM_CONFIDENCE` /
   `LOW_CONFIDENCE` / `UNINDEXED` — incorporates peak-matching evidence
   in addition to correlation score and margin
10. **Score-sign QC check** — emits `FAIL` when all template correlations
    are negative (anti-correlation with data)
11. **Experimental-template peak overlay** per ROI: mean diffraction pattern
    with matched experimental peaks in green, unexplained experimental peaks
    in red, and unmatched template peaks in blue. This is the primary
    diagnostic for q calibration, template orientation, peak detection, or
    wrong-candidate failures.
12. **Radial q-profile validation** per ROI before 2D orientation matching:
    radial integration of the experimental DP with expected candidate/template
    q-bands, so basic d-spacing support can be checked first.
13. Renders an **EBSD-style phase match map** with ambiguous regions shown
    as distinct candidate-group labels (e.g. "Ti-hcp / TiO₂-rutile")
14. Updates `stage2_gallery.html` with a Global Overview section and the
    per-ROI diagnostic overlays

**Key outputs (v4 schema):**

| File | Content |
|------|---------|
| `stage2_indexing_summary.json` | Phase call, top-k evidence, radial support, phase/orientation/mapping confidence, score-sign QC (schema v4) |
| `phase_match_map.png` | EBSD-style phase overview — ambiguous ROIs shown as candidate-group labels |
| `sweep_summary.json` | (from sweep script) Stability matrix across parameter grid |
| `templates/<candidate>_template_stack.npy` | Full orientation template stack (float32), per-zone hkls/qxy persisted |
| `templates/<candidate>_template_metadata.json` | Cell, HKLs, qxy coords, projections, beam centre, extinction stats |
| `roi_<name>/template_best_match.png` | Best-matching kinematic template |
| `roi_<name>/template_match_overlay.png` | Mean DP + template peaks (green) |
| `roi_<name>/experimental_template_peak_overlay.png` | Mean DP diagnostic: green=matched experimental peaks, red=unexplained experimental peaks, blue=unmatched template peaks |
| `roi_<name>/radial_q_profile_validation.png` | 1D radial q-profile validation against expected template q-bands before 2D orientation matching |
| `roi_<name>/phase_orientation_topk.json` | Ranked top-k template matches grouped by phase, with radial support and phase/orientation margins |
| `roi_<name>/correlation_vs_angle.png` | Correlation vs. in-plane angle |

**Key parameters:** `max_index`, `zone_axes`, `orientation_step_deg`,
`peak_sigma_px`, `reciprocal_pixels_per_inv_angstrom`, `intensity_power`,
`space_group` (per-candidate override for extinction rules)

---

## CLI Reference

| Command | Module form | Purpose |
|---------|-------------|---------|
| `fourdstem-run` | `python -m fourdstem_pipeline.cli run` | Stage 1 screening |
| `fourdstem-dry-run` | `python -m fourdstem_pipeline.cli dry_run` | Pre-flight config validation |
| `fourdstem-stage2` | `python -m fourdstem_pipeline.cli stage2` | Stage 2A ROI Bragg detection |
| `fourdstem-stage2b` | `python -m fourdstem_pipeline.cli stage2b` | Stage 2B crystallographic indexing |
| `fourdstem-bin-export` | `python -m fourdstem_pipeline.cli bin-export` | Bin raw data (R_bin, Q_bin) → EMD/H5 |
| `fourdstem-crop-export` | `python -m fourdstem_pipeline.cli crop-export` | Crop navigation dimensions → EMD/H5 |
| `fourdstem-stage2b-sweep` | `python scripts/run_stage2b_sweep.py` | Parameter stability sweep (v3) |

All accept `--config <path>` and `--log-level DEBUG|INFO|WARNING|ERROR`.

### `dry_run` — pre-flight validation

Without loading data, validates: config parse, input file existence/size,
scan & detector shapes, estimated output shape/chunks/memory, writable
output directory, unknown config keys, existing results. Use `--json`
for machine-readable output.

### `bin-export` — bin raw data and export to EMD/H5

Load a large 4D-STEM dataset (MIB or H5/EMD), apply navigation and/or
detector binning via py4DSTEM, and save the compressed result as a
standards-compliant EMD 1.0 file.  Avoids re-importing and re-binning
on every pipeline run.

```powershell
# MIB: R_bin=4 (512→128 nav), Q_bin=2 (256→128 detector)
python -m fourdstem_pipeline.cli bin-export `
    --input "data/scan.mib" `
    --output "data/binned.h5" `
    --r-bin 4 --q-bin 2 --scan-shape 512 512

# Already-binned H5: further Q_bin only
python -m fourdstem_pipeline.cli bin-export `
    --input "data/binned.h5" `
    --output "data/binned_q2.h5" `
    --q-bin 2
```

| Flag | Purpose |
|------|---------|
| `--input` | Raw data path (`.mib`, `.h5`, `.hdf5`, `.emd`) |
| `--output` | Output path (`.h5` appended if no suffix) |
| `--r-bin` | Navigation binning factor (default 1) |
| `--q-bin` | Detector binning factor (default 1) |
| `--mem` | py4DSTEM memory mode: `MEMMAP` (default) or `RAM` |
| `--scan-shape` | Raw nav shape `ny nx` for MIB import |

### `crop-export` — crop navigation and export to EMD/H5

Load raw data, extract a rectangular sub-region from the navigation
dimensions (detector unchanged), and export as EMD/H5.  Typical use:
extract a 64×64 region from a 512×512 scan for fast screening.

```powershell
# 512×512×256×256 → 64×64×256×256 (detector unchanged)
python -m fourdstem_pipeline.cli crop-export `
    --input "data/scan.mib" `
    --output "data/cropped.h5" `
    --nav-crop 0 64 0 64 --scan-shape 512 512
```

```powershell
python -m fourdstem_pipeline.cli crop-export `
    --input "data/0617-4d/1_512x512_ss15.63nm_0.55ms_c2 50um_CL91mm_0.75mrad_spot7_0.022nA_GL3_mag12500k_12b 0913.mib" `
    --output "data/0617-4d/1_0_64_0_64.h5" `
    --nav-crop 0 64 0 64 --scan-shape 512 512
```

| Flag | Purpose |
|------|---------|
| `--input` | Raw data path |
| `--output` | Output path |
| `--nav-crop` | Crop region `y0 y1 x0 x1` in nav pixels (y1/x1 exclusive) |
| `--mem` | py4DSTEM memory mode |
| `--scan-shape` | Raw nav shape for MIB import |

---

## Configuration

### Stage 1 (`configs/*.yaml`) — key sections

```yaml
project:
  name: my_experiment
  output_dir: outputs/my_run

data:
  path: synthetic://demo          # Or path/to/file.mib
  lazy: true
  cache: outputs/cache
  scan_shape: [512, 512]
  detector_shape: [256, 256]
  chunks:
    navigation: [8, 8]
    signal: [64, 64]

preprocess:
  q_crop: null                    # [qy0, qy1, qx0, qx1]
  q_bin: 1
  r_bin: 1

geometry:
  center: null                    # [y, x] beam centre override
  radial_bins: 48

virtual_images:
  masks:
    bf:   {inner_radius: 0,  outer_radius: 8}
    adf:  {inner_radius: 10, outer_radius: 22}
    haadf:{inner_radius: 22, outer_radius: 31}

phase_screening:
  n_components: 3
  n_clusters: 3
  method: pca_nmf_cluster
  candidate_phases: []

orientation:
  preview_binning: [2, 2]
  roi: [4, 12, 4, 12]
  confidence_threshold: 0.05

sample_mask:
  enabled: true
  source: adf
  percentile: 15
  fill_holes: true
  min_size: 100
  background_label: -1
```

### Stage 2A (`configs/stage2_roi_bragg.yaml`)

```yaml
stage1_dir: outputs/my_run
output_dir: null                  # null = <stage1_dir>/stage2/roi_bragg/
roi_source: roi_candidates        # Or path to YAML
max_rois: null
scan_shape: null                  # null = nav_shape × r_bin

thin_r: 2
bin_q: 2
mem: MEMMAP

# py4DSTEM Bragg detection
corr_power: 1.0
sigma_cc: 1
edge_boundary: 10
min_relative_intensity: 0.10      # 0.10–0.15 recommended (0.05 admits central-beam tails)
min_peak_spacing: 4
subpixel: poly
max_num_peaks: 70
cuda: false

central_exclusion_radius: 30      # zero vmap within 30 px (binned) of beam centre
save_roi_data: true               # required for full mean-DP template matching
```

### Stage 2B (`configs/stage2_indexing.yaml`) — v3

```yaml
stage2_dir: outputs/my_run/stage2/roi_bragg
output_dir: null

template_generation:
  max_index: 4
  zone_axes:                       # multi-zone-axis mode
    - [0, 0, 1]
    - [1, 0, 0]
    - [1, 1, 0]
    - [1, 1, 1]
    - [1, 1, 2]
  orientation_step_deg: 5
  peak_sigma_px: 5.0
  # Physical calibration (200 kV, L=91 mm, pixel ~6.5 um):
  reciprocal_pixels_per_inv_angstrom: 55.9
  intensity_power: 2.0

matching:
  top_k_per_phase: 5
  radial_gate_enabled: true
  radial_min_support: 0.25
  phase_margin_threshold: 0.08
  orientation_margin_threshold: 0.05

candidate_cifs:
  - name: Ti_hcp
    phase: Ti-hcp
    path: data/0617-4d/Ti-hcp.cif
    space_group: 194               # P6_3/mmc — overrides CIF P1 for extinctions
  - name: Ti_bcc
    phase: Ti-bcc
    path: data/0617-4d/Ti-bcc.cif
    space_group: 229               # Im-3m
  # --- Negative controls (P0 validation) ---
  - name: TiO2_rutile
    phase: TiO2-rutile
    path: data/0617-4d/TiO2_rutile.cif
    space_group: 136
  - name: Ni_fcc
    phase: Ni-fcc
    path: data/0617-4d/Ni_fcc.cif
    space_group: 225
  - name: Al_fcc
    phase: Al-fcc
    path: data/0617-4d/Al_fcc.cif
    space_group: 225
  - name: Fe_bcc
    phase: Fe-bcc
    path: data/0617-4d/Fe_bcc.cif
    space_group: 229
  - name: Ti_hcp_wrong_a
    phase: Ti-hcp-wrong
    path: data/0617-4d/Ti_hcp_wrong_a.cif
    space_group: 194
```

---

## Output Directory Structure

```
outputs/<run>/
├── stage1_summary.json              # Canonical manifest (Stage 1 → 2 bridge)
├── workflow_summary.json            # Full pipeline metadata
├── report.md / report.html          # Stage 1 report
├── qc_summary.json                  # QC flags + PASS/FAIL verdict
├── provenance.json                  # Git commit, versions, timestamps
├── data_contract.json               # Axis/bbox/centre conventions
├── preprocess_info.json             # Crop/bin params, shapes
│
├── virtual/                         # BF, ADF, HAADF, COM, ring images
├── fingerprints/                    # Radial profiles + axis
├── fingerprint_classes/             # Labels, cluster stats, mean profiles
├── orientation/                     # Orientation index + confidence maps
├── 00_preprocess/                   # Beam centre, sample mask .npy
├── png/                             # ~40 diagnostic PNGs
├── roi_candidates/                  # ROI proposals (YAML + CSV)
├── 05_cluster_diagnostics/          # Per-cluster mean DPs, profiles, ring ratios
│
├── stage2/roi_bragg/                # ── Stage 2A ──
│   ├── stage2_summary.json          # Aggregate results
│   ├── stage2_qc_summary.json       # QC flags
│   ├── stage2_report.md / .html     # Human-readable verdicts
│   ├── stage2_benchmark.json        # Per-ROI timing + sizes
│   ├── stage2_gallery.html          # Interactive PNG gallery
│   ├── provenance.json
│   └── roi_<name>/
│       ├── roi_data.npy             # 4D sub-cube (save_roi_data: true only)
│       ├── bragg_vector_map.npy     # Bragg peak vote histogram
│       ├── bragg_summary.json       # Full per-ROI metadata + QC
│       ├── bragg_peaks.parquet      # Tabular peaks (scan_y/x, qy/x, int, snr)
│       ├── mean_dp.png / bragg_vector_map.png / bragg_overlay.png
│       ├── bragg_peak_radius_histogram.png
│       ├── template_best_match.png  # (from Stage 2B)
│       ├── template_match_overlay.png
│       ├── experimental_template_peak_overlay.png  # Stage 2B matched/unexplained/unmatched peak diagnostic
│       ├── radial_q_profile_validation.png         # Stage 2B 1D q-profile validation
│       └── correlation_vs_angle.png
│
│   └── stage2b_indexing/            # ── Stage 2B v3 ──
│       ├── stage2_indexing_summary.json   # Phase call, hybrid score, peak residuals, confidence
│       ├── phase_match_map.png           # EBSD-style phase overview (ambiguous = candidate-group labels)
│       ├── phase_match_legend.png        # Phase + ambiguous-group legend
│       ├── stage2b_sweep/                # (from sweep script)
│       │   └── sweep_summary.json        # Stability matrix across parameter grid
│       └── templates/
│           ├── <candidate>_template_stack.npy     # Template stack (per-zone hkls/qxy persisted)
│           └── <candidate>_template_metadata.json # Cell, HKLs, qxy, extinctions, space group
```

---

## Coordinate Conventions

All stages follow a unified contract (`data_contract.json`):

| Convention | Order | Example |
|------------|-------|---------|
| Axis order | `nav_y, nav_x, q_y, q_x` | `data[ny, nx, qy, qx]` |
| BBox / ROI | `y0, y1, x0, x1` | `[10, 20, 30, 40]` |
| Centre | `y, x` (row, column) | `[cy, cx]` |

- **Stage 1 bboxes** are in binned (preprocessed) navigation coordinates.
- **Stage 2A raw bboxes** are in original scan coordinates (`× r_bin`).
- **Beam centre** is `[qy, qx]` in detector pixels. Stage 2A converts
  the Stage 1 COM estimate from preprocessed to raw detector coordinates
  using `q_crop` and `q_bin`.

---

## Config Presets

| File | Purpose |
|------|---------|
| `default_workflow.yaml` | Synthetic smoke-test (16×16×64×64) |
| `0617_4d_workflow.yaml` | Ti: 128×128 screening, `r_bin=4` |
| `0617_4d_workflow_rbin1.yaml` | Ti: full 512×512 nav |
| `0617_4d_workflow_rbin2.yaml` | Ti: 256×256 nav |
| `0617_4d_stage1_enhanced.yaml` | Ti: enhanced QC + fingerprint-class screening |
| `1_R4Q2_workflow.yaml` | Ti: H5 data 128×128×128×128, `r_bin=2`, Ti-bcc/hcp candidates |
| `stage2_roi_bragg.yaml` | Stage 2A Bragg detection (`save_roi_data: true`, `minRelativeIntensity: 0.10`, central exclusion) |
| `1_R4Q2_stage2_roi_bragg.yaml` | Stage 2A for 1_R4Q2 (128×128, `thin_r=1`, `bin_q=2`) |
| `stage2_smoke_test.yaml` | Stage 2A smoke test (`max_rois: 1`) |
| `stage2_indexing.yaml` | Stage 2B v3 indexing (7 CIFs, 5 zone axes, physical calibration, space-group extinctions) |
| `1_R4Q2_stage2_indexing.yaml` | Stage 2B for 1_R4Q2 (Ti-bcc + Ti-hcp only, auto-scale calibration) |

---

## Real-Data Results (Ti, 34 GB MIB)

```
Data:     512×512 scan × 256×256 detector, 0.75 mrad, 26,000× mag, CL 91 mm
Config:   0617_4d_stage1_enhanced.yaml (r_bin=2, q_crop=[16,240,16,240], q_bin=2)
Stage 2A: stage2_roi_bragg.yaml (thin_r=2, bin_q=2, minRelativeIntensity=0.10, central_exclusion_radius=30, save_roi_data=true)
Stage 2B: stage2_indexing.yaml (7 candidates, 5 zone axes, reciprocal scale 55.9 px/Å, space-group extinctions)
```

### Pipeline evolution across optimisation stages

| Metric | Initial (`save_roi_data=false`, single [001], auto-scale) | +mean DP +multi-zone +cal | +extinctions +cleanBragg | **v3: +hybrid +negative controls +ambiguity** |
|--------|-----------------------------------------------------------|---------------------------|-------------------------|----------------------------------------------|
| Match scores | −0.07 to −0.03 ❌ | +0.50 to +0.58 | +0.51 to +0.59 | +0.53 to +0.62 |
| Score margins | 0.0004–0.007 | 0.042–0.177 | 0.086–0.177 | 0.020–0.076 |
| ROIs medium+ | 0/11 | 6/11 | 11/11 | 0/11 |
| Phase call | 10× Ti-hcp, 1× Ti-bcc | 11× Ti-hcp [100] | 11× Ti-hcp [100] | **11× AMBIGUOUS** |
| Phase confidence | low | low–medium | medium | **LOW_CONFIDENCE** |
| Negative controls | — | — | — | TiO₂-rutile wins correlation (0.54–0.62 vs Ti-hcp 0.53–0.54); Ti-hcp wins peak matching 3:1 |
| Candidates | 2 | 2 | 2 | **7** (Ti-hcp, Ti-bcc, TiO₂-rutile, Ni-fcc, Al-fcc, Fe-bcc, Ti-hcp-wrong-a) |
| Templates | 144 | 720 | 720 | **2,520** |
| Zone axes | [001] only | 5 axes | 5 axes | 5 axes |
| Bragg peaks | 90,638 | 90,638 | 75,080 | 75,080 |
| Detector calibration | auto-fit | 55.9 px/Å (computed) | 55.9 px/Å | 55.9 px/Å |
| Space-group extinctions | — | — | SG 194/229 | SG 136/194/225/229 |
| Score-sign QC | — | — | — | PASS |
| Schema version | v2 | v2 | v2 | **v3** |

### Current v3 interpretation

```text
All 11 ROIs are crystallographically consistent with a Ti-hcp-like /
TiO₂-rutile-like diffraction geometry, but the available 128×128 binned
detector data (bin_q=2) are insufficient for confident phase separation.
The pipeline therefore assigns AMBIGUOUS / LOW_CONFIDENCE rather than
forcing a phase label.
```

**Why this is correct:** TiO₂-rutile (a=4.593 Å) has nearly the same `a`
lattice parameter as Ti-hcp (a=4.567 Å).  On the [100] zone axis, the
ring patterns are almost indistinguishable by dot-product correlation.
TiO₂-rutile **wins correlation** (0.54–0.62) but Ti-hcp **wins peak
matching** (60–69 matched peaks vs 19–25).  Neither reaches the 20%
observable-template threshold at `bin_q=2`.  The hybrid validation score
correctly identifies this as a degenerate case.

**The pipeline catches three critical failure modes:**
1. Correlation-only false positives (TiO₂ out-scores Ti-hcp)
2. Near-degenerate wrong-CIF impostors (similar lattice parameters)
3. Insufficient peak-level evidence (matched fraction < detection limit)

### Recommended next steps for phase confirmation

The algorithm is now honest.  The limiting factor is diffraction evidence:

| Action | Expected impact |
|--------|----------------|
| `bin_q=1` | Full 256×256 detector resolution → more resolvable Bragg peaks |
| Higher-q peak retention | More template peaks become observable |
| Better beam-centre refinement | Lower q-residuals for matched peaks |
| EDS/EELS oxygen check | Chemically rule out TiO₂ |
| SAED/NBED on selected ROIs | Manual validation of candidate phases |

### Template matching notes

For physically meaningful template matching:

1. **Set `save_roi_data: true`** in Stage 2A config. Matching against the actual
   mean DP (vs bragg-vector-map fallback) is the single most impactful change.
2. **Set `reciprocal_pixels_per_inv_angstrom`** from camera length (e.g.
   55.9 px/Å for CL=91 mm, 200 kV, 6.5 μm pixels).
3. **Enable `zone_axes`** (multi-zone-axis mode). Real grains are rarely
   aligned to a single zone axis.
4. **Add negative-control CIFs.** Wrong phases with similar lattice parameters
   (e.g. TiO₂-rutile vs Ti-hcp) can produce comparable correlation scores.
   Peak-residual analysis and hybrid validation catch these impostors.
5. **Run the parameter stability sweep** (`scripts/run_stage2b_sweep.py`) to
   verify that the phase call is stable across `peak_sigma_px`,
   `orientation_step_deg`, and `reciprocal_pixels_per_inv_angstrom`.
6. **Check `score_sign_qc`** in the summary — `FAIL` means all templates are
   anti-correlated with the data (likely `save_roi_data: false` or wrong
   zone axes).

---

## Package Structure

```
src/fourdstem_pipeline/        # 27 modules
├── cli.py                     # CLI entry points (run, dry_run, stage2, stage2b)
├── workflow.py                # Stage 1 orchestration
├── stage2.py                  # Stage 2A orchestration
├── roi_bragg.py               # py4DSTEM Bragg disk detection + QC + central exclusion
├── indexing.py                # Stage 2B v3: CIF→template→hybrid matching→ambiguity detection
├── export.py                  # PNG writer, reports, phase maps
├── export_stage2.py           # Stage 2A report, benchmark, gallery
├── contracts.py               # DataContract, Stage1Manifest, indexing gate
├── config.py                  # YAML loading, validation, defaults
├── loaders.py                 # Synthetic, NumPy/NPZ, HyperSpy/MIB backends
├── preprocess.py              # Lazy q_crop/q_bin/r_bin
├── preprocess_raw.py          # Raw-data bin & crop → EMD/H5 export
├── virtual.py                 # BF/ADF/HAADF/COM/ring computation
├── fingerprints.py            # Radial fingerprint profiles
├── phase.py                   # PCA + NMF + KMeans screening
├── orientation.py             # COM-angle orientation preview
├── sample_mask.py             # Percentile mask + morphological cleaning
├── diagnostics.py             # Diagnostics orchestration
├── diagnostics_cluster.py     # Per-cluster DPs, profiles, ring ratios, K-sweep
├── diagnostics_spatial.py     # Beam centre, connected components, ROI candidates
├── qc.py                      # 10 QC checks → PASS/WARN/FAIL
├── provenance.py              # Git + package version tracking
├── dataset.py                 # DatasetHandle dataclass
├── array_utils.py             # Chunked array helpers, ROI parsing
├── masks.py                   # Annular detector mask generation
├── synthetic.py               # Synthetic 4D-STEM demo generator
└── logging.py                 # Structured pipeline logging

scripts/
└── run_stage2b_sweep.py       # Parameter stability sweep for Stage 2B v3
```

---

## Data Contracts Between Stages

| Contract | File | Purpose |
|----------|------|---------|
| Coordinate conventions | `data_contract.json` | Axis/bbox/centre order |
| Stage 1 → 2A | `stage1_summary.json` | Shapes, paths, r_bin, q_crop, q_bin, QC status |
| Stage 1 → 2A | `roi_candidates.yaml` | ROI name, bbox (binned coords), cluster, reason |
| Stage 2A per-ROI | `bragg_summary.json` | Both bboxes, beam centre, peaks, validation |
| Stage 2A aggregate | `stage2_summary.json` | All ROI results + provenance |
| Stage 2A → 2B gate | `is_roi_ready_for_indexing()` | Shared filter: peaks>0, bg≤50%, sample>0, beam calibrated |
| Stage 2B output (v3) | `stage2_indexing_summary.json` | Phase call, candidate group, hybrid score, peak residuals, confidence tier, score-sign QC (schema v3) |

---

## Troubleshooting

### Environment

All commands require the `large-4dstem` conda environment:

```powershell
conda activate large-4dstem
```

### Python not found / wrong version (Windows)

The Windows Store Python stub doesn't work in non-interactive shells.
Use the `large-4dstem` conda environment:

```powershell
conda activate large-4dstem
python -m fourdstem_pipeline.cli dry_run --config configs/default_workflow.yaml
```

### py4DSTEM: `Windows Error 0xc06d007f`

This is a known OpenBLAS / threadpoolctl compatibility issue on Windows.
Fix inside the `large-4dstem` environment:

```powershell
conda activate large-4dstem
pip install threadpoolctl==3.5.0
# Or set the environment variable:
set OPENBLAS_CORETYPE=Haswell
```

### CLI commands not found

Use the module form — works identically without PATH configuration:

```bash
python -m fourdstem_pipeline.cli run --config configs/default_workflow.yaml
python -m fourdstem_pipeline.cli stage2 --config configs/stage2_roi_bragg.yaml
python -m fourdstem_pipeline.cli stage2b --config configs/stage2_indexing.yaml
```

### Stage 2A: py4DSTEM not installed

```bash
pip install git+https://github.com/py4dstem/py4DSTEM.git@dev
```

### Stage 2A: "Cannot determine data file path"

Set `data_path` in the Stage 2 config, or ensure the Stage 1 output's
`provenance.json` has a valid `input_path`.

### Stage 2B: "0 candidate CIF(s)"

The `candidate_cifs` list in the config is empty or commented out.
Point it to your CIF files:

```yaml
candidate_cifs:
  - name: my_phase
    phase: my_phase
    path: data/cifs/my_phase.cif
```

### Stage 2B: "cannot reshape array" error

Stage 2B templates are generated at `sig_shape` (pre-Q-bin) and may not
match the data resolution. Ensure `bin_q` in the Stage 2A config matches
what was used during extraction. The pipeline auto-detects Q-binning
from `stage2_summary.json` parameters.

### Stage 2B: low or negative match scores

1. Set `save_roi_data: true` in Stage 2A for full mean-DP matching
   (the bragg-vector-map fallback produces noisier correlations).
2. Tune `peak_sigma_px` to match your convergence angle (3–6 for CBED).
3. Set `reciprocal_pixels_per_inv_angstrom` if camera length is known.
4. Use `zone_axes` for multi-zone-axis coverage — real grains may not be
   aligned to `[0,0,1]`.

### Stage 2B: all ROIs report AMBIGUOUS / LOW_CONFIDENCE

This is **by design** when evidence is insufficient.  Check:
- `roi_<name>/experimental_template_peak_overlay.png` first. Many red peaks
  indicate experimental peaks the template cannot explain; many blue peaks
  indicate expected template peaks not observed experimentally.
- `roi_<name>/radial_q_profile_validation.png` before trusting 2D orientation
  matching. If the 1D q-bands do not align with experimental radial peaks,
  the candidate phase, reciprocal calibration, or peak extraction is suspect.
- `score_sign_qc` in the summary — if `FAIL`, fix the root causes above first.
- `matched_observable_template_fraction` — if < 0.20 for all candidates,
  the detector resolution (bin_q) or reciprocal calibration may be limiting.
- `candidate_group` — identifies which phases are near-degenerate.
  Consider chemical validation (EDS/EELS) or higher-resolution data (bin_q=1).
- Run `scripts/run_stage2b_sweep.py` to check parameter stability.

### Stage 2B: score_sign_qc reports FAIL

All template correlations are negative — templates are anti-correlated with
the data.  The most common cause is matching against `bragg_vector_map.npy`
instead of a mean DP.  Set `save_roi_data: true` in Stage 2A and re-run.
Also check `reciprocal_pixels_per_inv_angstrom` and `zone_axes` settings.

### Binning truncation

When `q_bin` or `r_bin` doesn't evenly divide dimensions, edge pixels are
truncated. The pipeline logs warnings. Choose bin factors that divide evenly.

### Emoji/unicode display

All report verdicts use ASCII labels (`[READY]`, `[REVIEW]`, `[FAIL]`, `[PASS]`)
for maximum terminal/text-editor compatibility.

---

## Development

```bash
conda activate large-4dstem
pip install -e ".[test,large-data,diffraction]"
python -m pytest tests/ -v
# Expected: ~49 passed, 1 skipped
```

### Real-data smoke tests

```bash
# Stage 2A (requires py4DSTEM + real MIB data)
FOURDSTEM_REAL_DATA=1 python -m pytest tests/ -k test_stage2_real_data_smoke -v

# Stage 2B (requires CIF files + Stage 2A output)
python -m fourdstem_pipeline.cli stage2b --config configs/stage2_indexing.yaml
```

### Running individual tests

```bash
python -m pytest tests/test_workflow.py -k "stage2b" -v
python -m pytest tests/test_workflow.py -k "indexing" -v
```
