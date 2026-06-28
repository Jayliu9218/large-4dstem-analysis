# large-4dstem-analysis

End-to-end 4D-STEM analysis pipeline for large datasets: from raw data loading
through unsupervised fingerprint-class screening to crystallographic phase
assignment.  Three-stage design with validated data contracts between stages.

> **Terminology note:** Stage 1 produces *unsupervised diffraction fingerprint
> classes* — not crystallographic phase assignments.  Radial profiles can
> separate thickness, orientation, strain, amorphous/crystalline contrast,
> detector artifacts, and phase changes all at once.  A cluster is a
> contrast group; only Stages 2B/2C assign crystallographic phases.

**What you get:**
- **Stage 1** — Virtual images, radial fingerprints, unsupervised fingerprint
  classes (phase-candidate screening), orientation preview, sample mask, ROI
  candidates, QC diagnostics, markdown + HTML report with ~40 PNG visualisations.
- **Stage 2A** — Per-ROI Bragg disk detection via py4DSTEM with coordinate
  mapping, beam-centre calibration, cluster/background validation, per-ROI
  visualisations, and a human-readable report labelling which ROIs are ready
  for indexing.
- **Stage 2B** — Analytic kinematic template generation from CIF lattice
  parameters, in-plane orientation matching against ROI mean diffraction
  patterns (crystallographic phase assignment), template-score and
  correlation-vs-angle visualisations, and a stage-level indexing summary.

**Real-data verification** (Ti, 512×512×256×256 detector, 34 GB MIB):
| Stage | Output | Result |
|-------|--------|--------|
| 1 | Fingerprint classes | 4 clusters on 256×256 nav (`r_bin=2`) |
| 2A | Bragg detection | 6,750 peaks in `cluster0_core_01`, QC PASS |
| 2B | Template matching | Ti-bcc, score 0.43 (medium), 62° orientation |

All three stages run on the same dataset with validated handoff contracts.

---

## Installation

### Quick: conda (recommended)

```bash
conda env create -f environment.yml
conda activate large-4dstem
pip install -e ".[test,large-data,diffraction]"
```

### Verify

```bash
python -m fourdstem_pipeline.cli dry_run --config configs/default_workflow.yaml
```

All commands use `python -m fourdstem_pipeline.cli <subcommand>` form so they
work even if pip-installed scripts are not on PATH.

---

## Quick Start

```bash
# 1. Validate config — no data loaded, ~1 second
python -m fourdstem_pipeline.cli dry_run --config configs/default_workflow.yaml

# 2. Stage 1 on synthetic demo dataset (16×16×64×64)
python -m fourdstem_pipeline.cli run --config configs/default_workflow.yaml

# 3. Run tests
python -m pytest tests/ -v
# Expected: ~49 passed, 1 skipped (real-data smoke test)
```

---

## Three-Stage Workflow

### Stage 1 — Fingerprint-Class Screening (Phase-Candidate Screening)

```
python -m fourdstem_pipeline.cli run --config configs/0617_4d_stage1_enhanced.yaml
```

**What it does:**
1. Loads 4D-STEM data (MIB via HyperSpy, or .npy/.npz, or synthetic)
2. Applies lazy preprocessing: `q_crop`, `q_bin`, `r_bin`
3. Computes virtual images: BF, ADF, HAADF, COM-x, COM-y, ring images
4. Generates sample mask (percentile threshold + morphological cleaning)
5. Computes radial fingerprints (per-pixel radial intensity profiles)
6. Runs unsupervised fingerprint-class screening (PCA + NMF + KMeans clustering)
7. Runs orientation preview (low-resolution template-free COM-angle proxy)
8. Generates ROI candidates from: connected components, cluster boundaries,
   orientation-score extremes, intensity anomalies
9. Runs QC checks and writes `qc_summary.json` (PASS / PASS_WITH_WARNINGS / FAIL)
10. Produces markdown + HTML report with ~40 diagnostic PNGs

**Key outputs:**
| File | Content |
|------|---------|
| `stage1_summary.json` | Canonical manifest bridging to Stage 2 |
| `report.html` | Human-readable report with inline PNGs |
| `qc_summary.json` | QC flags and PASS/FAIL verdict |
| `virtual/virtual_images.npz` | BF, ADF, HAADF, ring, COM maps |
| `fingerprints/radial_fingerprints.npy` | Per-pixel radial profiles |
| `fingerprint_classes/fingerprint_class_labels.npy` | Unsupervised fingerprint-class labels (NOT crystallographic phases) |
| `roi_candidates/roi_candidates.yaml` | ROI proposals with bboxes and rationale |
| `orientation/orientation_index.npy` | Per-pixel orientation label |
| `png/` | ~40 diagnostic PNGs |

---

### Stage 2A — ROI Bragg Detection

```
python -m fourdstem_pipeline.cli stage2 --config configs/stage2_roi_bragg.yaml
```

**What it does:**
1. Loads the Stage 1 manifest (`stage1_summary.json`)
2. Reads ROI candidates from `roi_candidates.yaml`
3. Loads the original 4D-STEM data via py4DSTEM (`import_file`)
4. Converts ROI bboxes from binned (Stage 1) to raw (original scan) coordinates
5. Extracts per-ROI 4D sub-cubes with navigation thinning (`thin_r`)
6. Runs py4DSTEM `find_Bragg_disks()` on each ROI with detector binning (`bin_q`)
7. Records beam centre provenance (Stage 1 COM → py4DSTEM calibration → fallback)
8. Saves per-ROI tabular Bragg peaks as `bragg_peaks.parquet` (scan_y, scan_x, qy, qx, intensity, snr)
9. Computes Bragg peak QC metrics (centre-zone, edge, duplicate fractions, radial stats)
10. Validates each ROI against fingerprint labels (background fraction) and sample mask
11. Produces per-ROI visualisations: mean DP, Bragg vector map, Bragg overlay, radius histogram
12. Generates `stage2_report.md`/`.html` with indexing-ready verdicts and Bragg QC table
13. Generates `stage2_benchmark.json` with per-ROI timing and data sizes

**Key outputs:**
| File | Content |
|------|---------|
| `stage2_summary.json` | Aggregate results with both `stage1_bbox` and `raw_bbox` |
| `stage2_qc_summary.json` | QC flags: NO_BRAGG_PEAKS, HIGH_BACKGROUND_ROIS, etc. |
| `stage2_report.html` | Per-ROI table with [READY]/[REVIEW]/[SKIP] verdicts |
| `stage2_benchmark.json` | Per-ROI extraction time, Bragg time, data size |
| `roi_<name>/roi_data.npy` | Extracted 4D sub-cube (only when `save_roi_data: true`) |
| `roi_<name>/bragg_vector_map.npy` | Calibrated Bragg peak positions |
| `roi_<name>/bragg_summary.json` | Full per-ROI metadata |
| `roi_<name>/mean_dp.png` | Log-scale mean diffraction pattern |
| `roi_<name>/bragg_vector_map.png` | Bragg peak histogram |
| `roi_<name>/bragg_peaks.parquet` | Tabular Bragg peaks: scan_y, scan_x, qy, qx, intensity, snr |
| `roi_<name>/bragg_peak_radius_histogram.png` | Radial distance histogram for QC |
| `roi_<name>/bragg_overlay.png` | Mean DP + Bragg peaks (green crosses) |

**Configurable parameters:**
- `thin_r`, `bin_q` — navigation/detector binning for extraction
- `corr_power`, `edge_boundary`, `min_relative_intensity`, `max_num_peaks`,
  `min_peak_spacing`, `subpixel`, `cuda` — py4DSTEM Bragg detection
- `max_rois` — cap number of ROIs processed
- `scan_shape` — override raw navigation shape for py4DSTEM import

---

### Stage 2B — Crystallographic Indexing

```
python -m fourdstem_pipeline.cli stage2b --config configs/stage2_indexing.yaml
```

**What it does:**
1. Reads Stage 2A `stage2_summary.json` and filters ROIs via `is_roi_ready_for_indexing()`
2. Parses CIF files for lattice parameters (`_cell_length_*`, `_cell_angle_*`)
3. Generates kinematic template stacks from the reciprocal lattice:
   - Multi-zone-axis orthographic projection (configurable `zone_axes`, or `zone_axis` for single-zone backward compat)
   - In-plane orientation sweep (configurable `orientation_step_deg`)
   - Gaussian spot rendering (`peak_sigma_px` controls disk width)
   - Intensity ≈ `1/|q|^power` kinematic proxy
4. Matches ROI mean diffraction patterns against all candidate templates
   via normalized correlation, tracking the best (candidate, zone axis, in-plane angle) triplet
5. Computes per-ROI phase confidence (high/medium/low) from the best score and
   its margin over the second-best candidate
6. Saves per-candidate template stacks and metadata to `templates/`
7. Produces per-ROI visualisations: best template, template+DP overlay,
   correlation-vs-angle bar chart
8. Writes `stage2_indexing_summary.json` (schema v2) with conservative field
   names: `candidate_phase`, `match_score`, `orientation_candidate_deg`,
   `phase_confidence`, `best_zone_axis`, `score_margin`

**Key outputs:**
| File | Content |
|------|---------|
| `stage2_indexing_summary.json` | Candidate phase, match score, zone axis, phase confidence, match quality (schema v2) |
| `templates/<candidate>_template_stack.npy` | Full orientation template stack (float32) |
| `templates/<candidate>_template_metadata.json` | Cell, HKL list, projection mode, beam centre |
| `roi_<name>/template_best_match.png` | Best-matching template image |
| `roi_<name>/template_match_overlay.png` | Mean DP (gray) + template spots (green) |
| `roi_<name>/correlation_vs_angle.png` | Bar chart of correlation vs. orientation angle |

**Configurable parameters:**
- `max_index` — maximum HKL index for reciprocal spot generation (default 4)
- `zone_axes` — list of zone axes for multi-zone template generation (e.g. `[[0,0,1],[1,0,0]]`). Overrides `zone_axis`.
- `zone_axis` — single zone axis for backward compat. Equivalent to `zone_axes: [[0,0,1]]`.
- `orientation_step_deg` — in-plane rotation step (default 5°)
- `peak_sigma_px` — Gaussian spot width in pixels (default 5.0, tuned for CBED)
- `reciprocal_pixels_per_inv_angstrom` — calibrated scale, or null for auto-fit
- `intensity_power` — kinematic intensity falloff exponent (default 2.0)

---

## CLI Reference

| Command | Module form | Purpose |
|---------|-------------|---------|
| `fourdstem-run` | `python -m fourdstem_pipeline.cli run` | Stage 1 screening |
| `fourdstem-dry-run` | `python -m fourdstem_pipeline.cli dry_run` | Pre-flight config validation |
| `fourdstem-stage2` | `python -m fourdstem_pipeline.cli stage2` | Stage 2A ROI Bragg detection |
| `fourdstem-stage2b` | `python -m fourdstem_pipeline.cli stage2b` | Stage 2B crystallographic candidate screening |

All commands accept `--config <path>` and `--log-level DEBUG|INFO|WARNING|ERROR`.

### `dry_run` — pre-flight validation

Without loading the full dataset, validates:
- Config file exists and parses correctly
- Input file exists and is non-empty
- Scan/detector shapes are valid
- Estimated output shape, chunk count, and peak memory
- Output directory is writable
- No unknown config keys
- Existing results that would be overwritten

Use `--json` to also print a machine-readable summary.

---

## Configuration

### Stage 1 (`configs/*.yaml`)

```yaml
project:
  name: my_experiment
  output_dir: outputs/my_run

data:
  path: synthetic://demo          # Or path/to/file.mib
  lazy: true
  cache: outputs/cache
  scan_shape: [512, 512]          # Required for MIB
  chunks:
    navigation: [8, 8]
    signal: [64, 64]

preprocess:
  q_crop: null                    # [qy0, qy1, qx0, qx1]
  q_bin: 1
  r_bin: 1

geometry:
  center: null                    # [y, x] beam center override
  radial_bins: 48

virtual_images:
  masks:
    bf:   {inner_radius: 0,  outer_radius: 8}
    adf:  {inner_radius: 10, outer_radius: 22}
    haadf:{inner_radius: 22, outer_radius: 31}

# ── Fingerprint-class screening (unsupervised contrast groups; NOT crystallographic phases) ──
phase_screening:
  n_components: 3
  n_clusters: 3
  method: pca_nmf_cluster
  candidate_phases: []              # Optional: score against reference profiles (phase-candidate screening)

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
data_path: null                   # null = use provenance.json
output_dir: null                  # null = <stage1_dir>/stage2/roi_bragg/
roi_source: roi_candidates        # "roi_candidates" or path to YAML
max_rois: null                    # null = all candidates
scan_shape: null                  # null = nav_shape * r_bin

thin_r: 2
bin_q: 2
mem: MEMMAP

corr_power: 1.0
sigma_cc: 1
edge_boundary: 10
min_relative_intensity: 0.05
min_peak_spacing: 4
subpixel: poly
max_num_peaks: 70
cuda: false
save_roi_data: false          # Set to true to keep roi_data.npy (large; needed for Stage 2B template matching)
```

### Stage 2B (`configs/stage2_indexing.yaml`)

```yaml
stage2_dir: outputs/my_run/stage2/roi_bragg
output_dir: null                  # null = <stage2_dir>/stage2b_indexing/

template_generation:
  max_index: 4
  zone_axis: [0, 0, 1]           # backward compat; overridden by zone_axes if set
  # zone_axes:                   # multi-zone mode (uncomment to enable)
  #   - [0, 0, 1]
  #   - [1, 0, 0]
  #   - [1, 1, 0]
  #   - [1, 1, 1]
  #   - [1, 1, 2]
  orientation_step_deg: 5
  peak_sigma_px: 5.0
  reciprocal_pixels_per_inv_angstrom: null
  intensity_power: 2.0

candidate_cifs:
  - name: Ti-bcc
    phase: Ti-bcc
    path: data/0617-4d/Ti-bcc.cif
  - name: Ti-hcp
    phase: Ti-hcp
    path: data/0617-4d/Ti-hcp.cif
```

---

## Output Directory Structure

```
outputs/<run>/
├── stage1_summary.json              # Canonical manifest (Stage 1 → 2 bridge)
├── workflow_summary.json            # Full pipeline metadata
├── report.md / report.html          # Human-readable Stage 1 report
├── qc_summary.json                  # QC flags and PASS/FAIL verdict
├── provenance.json                  # Git commit, versions, timestamps
├── data_contract.json               # Axis / bbox / center conventions
├── preprocess_info.json             # Crop/bin params, shapes
│
├── virtual/                         # Virtual images (BF, ADF, HAADF, COM, rings)
├── fingerprints/                    # Radial profiles + axis
├── fingerprint_classes/             # Class labels, cluster stats, mean profiles
├── orientation/                     # Orientation index + confidence maps
├── 00_preprocess/                   # Beam centre estimate, sample mask .npy
├── png/                             # ~40 diagnostic PNGs
├── roi_candidates/                  # ROI proposals (YAML + CSV)
├── 05_cluster_diagnostics/          # Per-cluster mean DPs, profiles
│
├── stage2/roi_bragg/                # ── Stage 2A ──
│   ├── stage2_summary.json
│   ├── stage2_qc_summary.json
│   ├── stage2_report.md / .html
│   ├── stage2_benchmark.json
│   ├── provenance.json
│   └── roi_<name>/
│       ├── roi_data.npy             # 4D sub-cube (only when save_roi_data: true)
│       ├── bragg_peaks.parquet      # Tabular Bragg peaks (scan_y/x, qy/x, intensity, snr)
│       ├── bragg_vector_map.npy     # Bragg peak vote histogram
│       ├── bragg_summary.json       # Full per-ROI metadata (incl. Bragg QC)
│       ├── mean_dp.png              # Log-scale mean DP
│       ├── bragg_vector_map.png     # Bragg peak histogram
│       ├── bragg_overlay.png        # Mean DP + Bragg peaks
│       ├── bragg_peak_radius_histogram.png  # Radial distance histogram
│       ├── template_best_match.png  # (from Stage 2B)
│       ├── template_match_overlay.png
│       └── correlation_vs_angle.png
│
│   └── stage2b_indexing/            # ── Stage 2B ──
│       ├── stage2_indexing_summary.json
│       └── templates/
│           ├── <candidate>_template_stack.npy
│           └── <candidate>_template_metadata.json
```

---

## Coordinate Conventions

All stages follow a unified data contract (`data_contract.json`):

| Convention | Order | Example |
|------------|-------|---------|
| Axis order | `nav_y, nav_x, q_y, q_x` | `data[ny, nx, qy, qx]` |
| BBox / ROI | `y0, y1, x0, x1` | `[10, 20, 30, 40]` |
| Centre | `y, x` (row, column) | `[cy, cx]` |

- **Stage 1 ROI bboxes** are in binned (preprocessed) navigation coordinates.
- **Stage 2A raw bboxes** are in original scan coordinates (multiplied by `r_bin`).
- **Beam centre** is recorded as `[qy, qx]` in detector pixels.  Stage 2A
  converts the Stage 1 COM estimate from preprocessed to raw detector
  coordinates using `q_crop` and `q_bin`.

---

## Config Presets

| File | Purpose |
|------|---------|
| `default_workflow.yaml` | Synthetic smoke-test (16×16×64×64) |
| `0617_4d_workflow.yaml` | Ti data: 128×128 screening, `r_bin=4` |
| `0617_4d_workflow_rbin1.yaml` | Ti data: full 512×512 nav |
| `0617_4d_workflow_rbin2.yaml` | Ti data: 256×256 nav, `r_bin=2` |
| `0617_4d_stage1_enhanced.yaml` | Ti data: enhanced QC, fingerprint-class screening |
| `stage2_roi_bragg.yaml` | Stage 2A Bragg detection |
| `stage2_smoke_test.yaml` | Stage 2A smoke test (`max_rois: 1`) |
| `stage2_indexing.yaml` | Stage 2B indexing with CIF candidates |

---

## Real-Data Results (Ti, 34 GB MIB)

```
Data:     512×512 scan × 256×256 detector, 0.75 mrad, 26,000× mag
Config:   0617_4d_stage1_enhanced.yaml (r_bin=2, q_crop=[16,240,16,240], q_bin=2)
```

| Stage | Metric | Value |
|-------|--------|-------|
| **1** | Fingerprint classes | 4 |
| **1** | Navigation shape | 256×256 |
| **1** | Signal shape (preprocessed) | 112×112 |
| **1** | QC status | PASS_WITH_WARNINGS |
| **1** | ROI candidates | 10 |
| | | |
| **2A** | ROI processed | `cluster0_core_01` |
| **2A** | Raw bbox | `[0, 128, 272, 400]` |
| **2A** | ROI nav shape | 64×64 |
| **2A** | Bragg peaks detected | 6,750 |
| **2A** | Beam centre source | `stage1_com` |
| **2A** | Beam centre (raw) | `[126.8, 125.7]` |
| **2A** | Background fraction | 0.0% |
| **2A** | QC status | PASS |
| **2A** | Extraction time | 0.42 s |
| **2A** | Bragg detection time | 4.65 s |
| **2A** | Total elapsed | 10.7 s |
| | | |
| **2B** | Candidates | Ti-bcc (cubic, a=3.25 Å), Ti-hcp (hex, a=4.57 Å, c=2.83 Å) |
| **2B** | Templates generated | 360 (180 per candidate, 2° step) |
| **2B** | **Candidate phase** | **Ti-bcc** |
| **2B** | **Match score** | **0.433 (medium)** |
| **2B** | **Orientation candidate** | **62.0°** |
| **2B** | Auto-scale | 57.7 px/Å (auto-fit to detector) |
| **2B** | Elapsed | 8.2 s |

### Parameter Tuning Notes

`peak_sigma_px` controls the Gaussian spot width in the kinematic template.
The default of 5.0 was calibrated against this dataset:

| `peak_sigma_px` | Score | Quality |
|-----------------|-------|---------|
| 1.2 | 0.17 | low |
| 3.0 | 0.34 | low |
| 4.0 | 0.40 | medium |
| **5.0** | **0.43** | **medium** |
| 6.0 | 0.43 | medium |

The 62° orientation is stable across sigma values and orientation step sizes
(5° → 2° → 1°), confirming a real crystallographic signal.

---

## Package Structure

```
src/fourdstem_pipeline/
├── cli.py                  # CLI: run, dry_run, stage2, stage2b
├── workflow.py             # Stage 1 orchestration
├── stage2.py               # Stage 2A orchestration
├── roi_bragg.py            # py4DSTEM Bragg disk detection
├── indexing.py             # Stage 2B CIF-to-template matching
├── export.py               # PNG writer, Stage 1 report (md + html)
├── export_stage2.py        # Stage 2A report + benchmark
├── contracts.py            # DataContract, Stage1Manifest, is_roi_ready_for_indexing
├── config.py               # YAML config loading + validation
├── loaders.py              # synthetic, NumPy, NPZ, HyperSpy/MIB
├── preprocess.py           # lazy q_crop, q_bin, r_bin
├── virtual.py              # BF/ADF/HAADF/COM/ring computation
├── masks.py                # annular mask generation
├── fingerprints.py         # radial fingerprint profiles
├── phase.py                # PCA + NMF + KMeans fingerprint-class screening
├── orientation.py          # orientation preview (COM-angle proxy)
├── sample_mask.py          # sample/vacuum mask + label masking
├── diagnostics.py          # cluster diagnostics orchestration
├── diagnostics_cluster.py  # per-cluster mean DPs, profiles, stats
├── diagnostics_spatial.py  # beam centre, connected components, ROI candidates
├── qc.py                   # quality control checks
├── provenance.py           # runtime dependency + provenance reporting
├── synthetic.py            # synthetic 4D-STEM demo generator
├── dataset.py              # DatasetHandle dataclass
├── array_utils.py          # ROI parsing, normalisation, sub-pixel
└── logging.py              # structured logging (FOURDSTEM_LOG_LEVEL)
```

---

## Data Contracts Between Stages

| Contract | File | Purpose |
|----------|------|---------|
| Coordinate conventions | `data_contract.json` | Axis/bbox/centre order |
| Stage 1 → 2A | `stage1_summary.json` | Shapes, paths, r_bin, q_crop, q_bin, QC status |
| Stage 1 → 2A | `roi_candidates.yaml` | ROI name, bbox (binned coords), cluster, reason |
| Stage 2A per-ROI | `bragg_summary.json` | Both bboxes, beam centre, peaks, params, validation |
| Stage 2A aggregate | `stage2_summary.json` | All ROI results + provenance |
| Stage 2A → 2B gate | `is_roi_ready_for_indexing()` | Shared filter: peaks>0, bg≤50%, sample>0, beam calibrated |
| Stage 2B output | `stage2_indexing_summary.json` | Candidate phase, match score, zone axis, phase confidence, match quality (schema v2) |

---

## Troubleshooting

### Python not found (Windows)

The Windows Store Python stub does not work in non-interactive shells.
Use Miniconda Python or activate the environment first:

```powershell
conda activate large-4dstem
python -m fourdstem_pipeline.cli dry_run --config configs/default_workflow.yaml
```

### CLI commands not found

Use the module form — works identically and doesn't require PATH configuration:

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

Either set `data_path` in the Stage 2 config, or ensure the Stage 1 output's
`provenance.json` has a valid `input_path`.

### Stage 2B: low template scores

1. Verify `peak_sigma_px` matches your convergence angle (try 3–6 for CBED).
2. Set `reciprocal_pixels_per_inv_angstrom` if you know the camera length.
3. Use `zone_axes` to try multiple zone axes — real grains may not be on `[0,0,1]`.
4. Check that `beam_center_yx` in `bragg_summary.json` matches the detector shape.

### Binning truncation

When `q_bin` or `r_bin` doesn't evenly divide dimensions, edge pixels are
truncated. The pipeline logs warnings. Choose bin factors that divide evenly.

### Emoji display

All report verdicts use ASCII labels (`[READY]`, `[REVIEW]`, `[FAIL]`, `[PASS]`)
for maximum compatibility across terminals and text editors.

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
