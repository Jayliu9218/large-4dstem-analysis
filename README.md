# large-4dstem-analysis

End-to-end 4D-STEM analysis pipeline for large datasets: from raw data loading
through unsupervised fingerprint-class screening to crystallographic phase
assignment.  Three-stage design with validated data contracts between stages.

> **Terminology:** Stage 1 produces *unsupervised diffraction fingerprint classes* —
> not crystallographic phase assignments.  Radial profiles can separate thickness,
> orientation, strain, amorphous/crystalline contrast, detector artifacts, and
> phase changes.  A cluster is a contrast group; only Stage 2B assigns
> crystallographic phase candidates.

**What you get:**
- **Stage 1** — Virtual images (BF/ADF/HAADF/COM/rings), radial fingerprints,
  unsupervised fingerprint classes, orientation preview, sample mask, ROI
  candidates, QC diagnostics (10 checks), markdown + HTML report (~40 PNGs).
- **Stage 2A** — Per-ROI py4DSTEM Bragg disk detection with beam-centre
  calibration cascade, cluster/background validation, Bragg QC metrics,
  tabular peak output (Parquet), and a report labelling which ROIs are
  ready for indexing.
- **Stage 2B** — CIF→kinematic template generation, multi-zone-axis orientation
  sweep, normalized correlation matching, phase confidence tiering,
  EBSD-style phase match map, and an interactive PNG gallery.

**Real-data verification** (Ti, 512×512×256×256 detector, 34 GB MIB):

| Stage | Output | Result |
|-------|--------|--------|
| 1 | Fingerprint classes | 4 clusters on 256×256 nav (`r_bin=2`) |
| 2A | Bragg detection | 6,750 peaks in `cluster0_core_01`, QC PASS |
| 2B | Template matching | Ti-hcp / Ti-bcc candidates, scores −0.03 to −0.07 (low), phase confidence: low |

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

All commands use `python -m fourdstem_pipeline.cli <subcommand>` and work even
if pip-installed scripts are not on PATH.

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

### Stage 2B — Crystallographic Indexing

```bash
python -m fourdstem_pipeline.cli stage2b --config configs/stage2_indexing.yaml
```

**What it does:**
1. Filters ROIs via `is_roi_ready_for_indexing()` (peaks>0, bg≤50%,
   sample coverage>0, beam calibrated)
2. Parses CIF files for lattice parameters (`_cell_length_*`, `_cell_angle_*`)
3. Generates kinematic template stacks from the reciprocal lattice:
   - Multi-zone-axis orthographic projection
   - In-plane orientation sweep (`orientation_step_deg`)
   - Gaussian spot rendering (`peak_sigma_px` controls CBED disk width)
   - Intensity ≈ 1/|q|^power kinematic proxy
4. Matches ROI patterns against templates via normalized correlation,
   reporting best (candidate, zone axis, orientation) triplet
5. Computes phase confidence (high/medium/low) from best score and
   its margin over the second-best **competing** candidate
6. Renders an **EBSD-style phase match map** — navigation-space overview
   with ROIs coloured by matched phase, boundaries, and legend
7. Saves per-ROI match PNGs: best template, template+data overlay,
   correlation-vs-angle chart
8. Updates `stage2_gallery.html` with a Global Overview section

**Key outputs:**

| File | Content |
|------|---------|
| `stage2_indexing_summary.json` | Phase, score, zone axis, orientation, confidence (schema v2) |
| `phase_match_map.png` | EBSD-style nav-space phase overview with legend |
| `templates/<candidate>_template_stack.npy` | Full orientation template stack (float32) |
| `templates/<candidate>_template_metadata.json` | Cell, HKL list, projection, beam centre |
| `roi_<name>/template_best_match.png` | Best-matching kinematic template |
| `roi_<name>/template_match_overlay.png` | Mean DP + template peaks (green) |
| `roi_<name>/correlation_vs_angle.png` | Correlation vs. in-plane angle |

**Key parameters:** `max_index`, `zone_axes` (multi-zone) / `zone_axis`
(single-zone backward compat), `orientation_step_deg`, `peak_sigma_px`,
`reciprocal_pixels_per_inv_angstrom` (null = auto-fit), `intensity_power`

---

## CLI Reference

| Command | Module form | Purpose |
|---------|-------------|---------|
| `fourdstem-run` | `python -m fourdstem_pipeline.cli run` | Stage 1 screening |
| `fourdstem-dry-run` | `python -m fourdstem_pipeline.cli dry_run` | Pre-flight config validation |
| `fourdstem-stage2` | `python -m fourdstem_pipeline.cli stage2` | Stage 2A ROI Bragg detection |
| `fourdstem-stage2b` | `python -m fourdstem_pipeline.cli stage2b` | Stage 2B crystallographic indexing |

All accept `--config <path>` and `--log-level DEBUG|INFO|WARNING|ERROR`.

### `dry_run` — pre-flight validation

Without loading data, validates: config parse, input file existence/size,
scan & detector shapes, estimated output shape/chunks/memory, writable
output directory, unknown config keys, existing results. Use `--json`
for machine-readable output.

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
min_relative_intensity: 0.05
min_peak_spacing: 4
subpixel: poly
max_num_peaks: 70
cuda: false

save_roi_data: false              # Set true for full-DP template matching
```

### Stage 2B (`configs/stage2_indexing.yaml`)

```yaml
stage2_dir: outputs/my_run/stage2/roi_bragg
output_dir: null                  # null = <stage2_dir>/stage2b_indexing/

template_generation:
  max_index: 4
  zone_axis: [0, 0, 1]           # backward compat
  # zone_axes:                   # multi-zone mode (uncomment to enable)
  #   - [0, 0, 1]
  #   - [1, 0, 0]
  #   - [1, 1, 0]
  #   - [1, 1, 1]
  #   - [1, 1, 2]
  orientation_step_deg: 5
  peak_sigma_px: 5.0
  reciprocal_pixels_per_inv_angstrom: null   # null = auto-fit to detector
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
│       └── correlation_vs_angle.png
│
│   └── stage2b_indexing/            # ── Stage 2B ──
│       ├── stage2_indexing_summary.json
│       ├── phase_match_map.png      # EBSD-style phase overview
│       ├── phase_match_legend.png   # Phase legend
│       └── templates/
│           ├── <candidate>_template_stack.npy
│           └── <candidate>_template_metadata.json
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
| `stage2_roi_bragg.yaml` | Stage 2A Bragg detection |
| `stage2_smoke_test.yaml` | Stage 2A smoke test (`max_rois: 1`) |
| `stage2_indexing.yaml` | Stage 2B indexing with Ti CIF candidates |

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
| **2A** | ROIs processed | 10 |
| **2A** | Bragg peaks (cluster0) | 6,750 |
| **2A** | Beam centre source | `stage1_com` |
| **2A** | Beam centre (raw) | `[126.8, 125.7]` |
| **2A** | QC status | PASS |
| **2A** | Elapsed | 10.7 s |
| | | |
| **2B** | Candidates | Ti-bcc (cubic, a=3.25 Å), Ti-hcp (hex, a=4.57 Å, c=2.83 Å) |
| **2B** | Templates | 144 (72 per candidate, 5° step, [0,0,1] zone axis) |
| **2B** | Match scores | −0.069 to −0.028 (low; *negative* — templates anti-correlated with data) |
| **2B** | Score margins | 0.0004–0.007 (below random-noise floor of ~0.008 for 5 ROIs) |
| **2B** | Phase confidence | low for all 11 ROIs (single zone axis, `save_roi_data: false`) |
| **2B** | Phase distribution | 10/11 Ti-hcp, 1/11 Ti-bcc (not crystallographic proof; see caveats below) |
| **2B** | Auto-scale | ~55–58 px/Å (auto-fit to detector radius; **no camera-length calibration**) |
| **2B** | Elapsed | ~8 s |

> **⚠️ Caveat:** These results are from `save_roi_data: false` (bragg-vector-map
> fallback) with a single [001] zone axis and auto-fit reciprocal scale. Under
> these conditions template matching produces **negative normalized correlations**
> — the templates anti-correlate with the Bragg vector map because the map is
> dominated by the central-beam falloff. Score margins between Ti-hcp and Ti-bcc
> are smaller than the expected random-noise floor (σ ≈ 0.008), making the two
> phases statistically indistinguishable. **Treat the phase map as a candidate
> distribution for method development, not as crystallographic evidence.**

### Template matching notes

`peak_sigma_px` controls Gaussian spot width — tune to match your convergence
angle (3–6 px for typical CBED). With the default of 5.0 px and the
bragg-vector-map fallback (`save_roi_data: false`), template correlations are
negative and non-discriminative (scores −0.07 to −0.03, margins < 0.008). For
physically meaningful matching:

1. **Set `save_roi_data: true`** in Stage 2A config for full mean-DP correlation.
   This is the single most impactful change — matching against the actual mean
   diffraction pattern instead of the Bragg vector map yields physically
   meaningful (positive) correlation scores.
2. **Set `reciprocal_pixels_per_inv_angstrom`** if camera length is calibrated
   (e.g. CL91mm at this convergence angle). Without it, templates are
   auto-scaled to the detector radius and ring positions have no physical basis.
3. Enable multi-zone-axis mode (`zone_axes`) in Stage 2B config for
   discriminative score margins between competing phases.

---

## Package Structure

```
src/fourdstem_pipeline/        # 28 modules
├── cli.py                     # CLI entry points (run, dry_run, stage2, stage2b)
├── workflow.py                # Stage 1 orchestration
├── stage2.py                  # Stage 2A orchestration
├── roi_bragg.py               # py4DSTEM Bragg disk detection + QC
├── indexing.py                # Stage 2B CIF→template→matching
├── export.py                  # PNG writer, reports, phase maps
├── export_stage2.py           # Stage 2A report, benchmark, gallery
├── contracts.py               # DataContract, Stage1Manifest, indexing gate
├── config.py                  # YAML loading, validation, defaults
├── loaders.py                 # Synthetic, NumPy/NPZ, HyperSpy/MIB backends
├── preprocess.py              # Lazy q_crop/q_bin/r_bin
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
| Stage 2B output | `stage2_indexing_summary.json` | Phase, score, zone axis, orientation, confidence (schema v2) |

---

## Troubleshooting

### Python not found (Windows)

The Windows Store Python stub doesn't work in non-interactive shells.
Use Miniconda Python or activate the environment first:

```powershell
conda activate large-4dstem
python -m fourdstem_pipeline.cli dry_run --config configs/default_workflow.yaml
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
