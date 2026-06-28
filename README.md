# large-4dstem-analysis

Single non-visual workflow for large 4D-STEM data, focused on:

- Virtual BF/ADF/HAADF and COM imaging
- Radial fingerprints and unsupervised phase screening
- Candidate phase score maps
- Low-resolution full-field orientation preview
- ROI Bragg-disk detection via py4DSTEM (Stage 2A)
- Sample mask generation for vacuum exclusion

The codebase is intentionally small: one package, three CLI entry points, YAML
configs, and focused tests. Outputs are NumPy arrays, JSON metadata, and PNG
previews — no Jupyter notebooks required.

---

## Installation

```bash
# Clone and install in editable mode with test dependencies
git clone <repo-url> && cd large-4dstem-analysis
pip install -e ".[test]"

# For real MIB/HDF5 data, also install the large-data stack
pip install -e ".[large-data]"

# For Stage 2A ROI Bragg detection
pip install -e ".[diffraction]"

# Or install everything at once
pip install -e ".[test,large-data,diffraction]"
```

> **Windows note**: If `python` doesn't work in bash, use the full path to your
> Miniconda Python (e.g. `C:/ProgramData/miniconda3/python.exe`). The Windows
> Store Python stub at `%LOCALAPPDATA%/Microsoft/WindowsApps/python.exe` does
> not work in non-interactive shells.

After installation, three commands become available:

```bash
fourdstem-run       --config configs/your_config.yaml    # Stage 1 screening
fourdstem-dry-run   --config configs/your_config.yaml    # Validate without loading data
fourdstem-stage2    --config configs/stage2_roi_bragg.yaml  # Stage 2A Bragg detection
```

---

## Quick Start

```bash
# 1. Validate your config — no data loaded, ~1 second
fourdstem-dry-run --config configs/default_workflow.yaml

# 2. Run Stage 1 on the synthetic demo dataset
fourdstem-run --config configs/default_workflow.yaml

# 3. Run tests to verify everything works
python -m pytest tests/ -v
```

The default config runs on `synthetic://demo` (a 16×16×64×64 toy dataset).
Point `data.path` at a real `.mib`, `.npy`, or `.npz` file when ready.

### Real data workflow (end-to-end)

```bash
# Step 0: Dry-run to check config and estimate memory
fourdstem-dry-run --config configs/0617_4d_workflow.yaml

# Step 1: Stage 1 screening
fourdstem-run --config configs/0617_4d_workflow.yaml

# Step 2: Inspect the HTML report to pick ROIs
#   → outputs/<run>/report.html

# Step 3: Run Stage 2A Bragg detection on those ROIs
fourdstem-stage2 --config configs/stage2_roi_bragg.yaml
```

---

## CLI Reference

### `fourdstem-run` — Stage 1 screening

```
fourdstem-run --config configs/default_workflow.yaml [--log-level DEBUG|INFO|WARNING|ERROR]
```

Loads a 4D-STEM dataset, preprocesses, computes virtual images
(BF/ADF/HAADF/COM), radial fingerprints, unsupervised phase screening,
orientation preview, sample mask, QC checks, and writes a Markdown+HTML report.

Returns exit code 0 on success, 1 if any stage errors.

### `fourdstem-dry-run` — pre-flight validation

```
fourdstem-dry-run --config configs/default_workflow.yaml [--json]
```

Without loading the full dataset, validates:
- Config file exists and parses correctly
- Input file exists and is non-empty
- Scan/detector shapes are valid
- Estimated output shape, chunk count, and peak memory
- Output directory is writable
- No unknown config keys
- Existing results that would be overwritten

Use `--json` to also print a machine-readable summary to stdout. A
`dry_run_summary.json` is always saved to the output directory.

Returns exit code 0 on OK/OK_WITH_WARNINGS, 1 on FAIL.

### `fourdstem-stage2` — Stage 2A ROI Bragg detection

```
fourdstem-stage2 --config configs/stage2_roi_bragg.yaml [--log-level DEBUG|INFO|WARNING|ERROR]
```

Consumes a Stage-1 output directory, loads the validated `Stage1Manifest`,
extracts per-ROI sub-cubes, runs py4DSTEM `find_Bragg_disks()` on each, and
produces a `stage2_summary.json` + `stage2_qc_summary.json`.

Requires `py4DSTEM>=0.14` (`pip install -e ".[diffraction]"`).

Returns exit code 0 if all ROIs succeed, 1 if any fail.

---

## Configuration

All pipeline behavior is driven by a single YAML file. Here's the full
structure:

```yaml
project:                          # Output naming
  name: my_experiment
  output_dir: outputs/my_run

data:                             # Input data source
  path: synthetic://demo          # Or path/to/file.mib
  lazy: true                      # Lazy/chunked loading
  cache: outputs/cache
  scan_shape: [512, 512]          # Required for MIB files
  chunks:
    navigation: [8, 8]
    signal: [64, 64]

preprocess:                       # Data reduction
  q_crop: null                    # [qy0, qy1, qx0, qx1] or null
  q_bin: 1                        # Detector binning
  r_bin: 1                        # Navigation binning

geometry:
  center: null                    # [y, x] beam center override
  radial_bins: 48                 # Number of radial bins

virtual_images:
  masks:
    bf:   {inner_radius: 0,  outer_radius: 8}
    adf:  {inner_radius: 10, outer_radius: 22}
    haadf:{inner_radius: 22, outer_radius: 31}

phase_screening:
  n_components: 3
  n_clusters: 3
  method: pca_nmf_cluster
  candidate_phases:
    - name: phase_alpha
      reference_profile: null     # .npy path or null

orientation:
  preview_binning: [2, 2]
  roi: [4, 12, 4, 12]           # [y0, y1, x0, x1] — y1>y0, x1>x0
  confidence_threshold: 0.05

sample_mask:
  enabled: true
  source: adf
  method: percentile
  percentile: 15
  fill_holes: true
  min_size: 100
  background_label: -1

roi_bragg:                        # Optional Stage 1 inline Bragg
  enabled: false                  # Set true to run during Stage 1
  roi: [192, 320, 192, 320]
  thin_r: 2
  bin_q: 2
  mem: MEMMAP
```

### Stage 2A configuration

```yaml
# configs/stage2_roi_bragg.yaml
stage1_dir: outputs/my_run        # Path to Stage-1 output
# data_path: data/file.mib        # Optional: override data path
output_dir: null                  # null = <stage1_dir>/stage2/roi_bragg/
roi_source: roi_candidates        # "roi_candidates" or path to YAML
max_rois: null                    # Cap ROIs (null = all)

thin_r: 2                         # Navigation thinning
bin_q: 2                          # Detector binning
mem: MEMMAP                       # "MEMMAP" or "RAM"

# py4DSTEM Bragg disk parameters
corr_power: 1.0
sigma_cc: 1
edge_boundary: 10
min_relative_intensity: 0.05
min_peak_spacing: 4
subpixel: poly
max_num_peaks: 70
cuda: false
```

---

## Output Directory Structure

After a Stage 1 run:

```
outputs/<run>/
├── stage1_summary.json          # Canonical manifest (Stage 1 → 2 bridge)
├── workflow_summary.json        # Full pipeline metadata
├── data_contract.json           # Axis / bbox / center conventions
├── provenance.json              # Git commit, package versions, timestamps
├── qc_summary.json              # QC flags and PASS/FAIL verdict
├── preprocess_info.json         # Crop/bin params, shapes
├── report.md                    # Human-readable summary
├── report.html                  # HTML report with inline PNGs
├── dry_run_summary.json         # (if dry-run was used)
│
├── virtual/
│   ├── virtual_images.npz       # BF, ADF, HAADF, ring images, COM maps
│   └── com_x.npy, com_y.npy
│
├── fingerprints/
│   ├── radial_fingerprints.npy  # Per-pixel radial profiles
│   └── radial_axis.npy          # Radial bin centers
│
├── fingerprint_classes/
│   ├── fingerprint_class_labels.npy        # Per-pixel class labels
│   ├── cluster_summary.csv                # Per-cluster statistics
│   └── cluster_mean_radial_profiles.npy   # Mean profile per class
│
├── orientation/
│   ├── orientation_index.npy    # Per-pixel orientation label
│   └── orientation_score.npy    # Per-pixel orientation confidence
│
├── sample_mask/
│   ├── sample_mask.npy          # Boolean sample mask
│   └── sample_mask_stats.csv    # Coverage statistics
│
├── png/                         # Visual diagnostics (~35-40 PNGs)
│   ├── virtual_bf.png, virtual_adf.png, virtual_haadf.png
│   ├── mean_dp_log.png, max_diffraction.png
│   ├── mean_radial_profile.png
│   ├── fingerprint_class_labels_annotated.png
│   ├── cluster_mean_dp_*.png, cluster_mean_radial_profiles_*.png
│   ├── orientation_index.png, orientation_score.png
│   ├── roi_candidates_overlay.png
│   ├── sample_mask.png, sample_mask_overlay_adf.png
│   └── ... (ring ratios, COM maps, candidate scores, etc.)
│
├── roi_candidates/
│   └── roi_candidates.yaml      # ROI proposals with bboxes and rationale
│
└── 05_cluster_diagnostics/      # Per-cluster diagnostics (arrays only)
    ├── cluster_*_mean_dp.npy
    └── cluster_*_radial_profiles.npy
```

After a Stage 2A run:

```
outputs/<run>/stage2/roi_bragg/
├── stage2_summary.json
├── stage2_qc_summary.json
├── provenance.json
│
└── roi_<name>/
    ├── roi_data.npy             # Extracted 4D sub-cube
    ├── bragg_vector_map.npy     # Calibrated Bragg peak positions
    └── bragg_summary.json       # Peak count, detection params
```

---

## Coordinate Conventions

The pipeline follows a single unified data contract (`data_contract.json`):

- **Axis order**: `nav_y, nav_x, q_y, q_x` (navigation rows, navigation columns,
  diffraction rows, diffraction columns).
- **BBox / ROI order**: `y0, y1, x0, x1` (row start, row end, column start,
  column end). Python slice semantics: `y0` is inclusive, `y1` is exclusive.
- **Center order**: `y, x` (row first, column second). The geometric center of a
  detector of shape `(H, W)` is `((H-1)/2, (W-1)/2)`.

All array indexing and ROI specifications throughout the pipeline use this
convention. When configuring `orientation.roi` or `roi_bragg.roi`, ensure
`y1 > y0` and `x1 > x0` — zero-area ROIs are rejected with a clear error.

---

## Troubleshooting

### "python not found" or empty output in bash (Windows)

`which python` may point to the Windows Store stub at
`%LOCALAPPDATA%/Microsoft/WindowsApps/python.exe`, which does not work in
non-interactive shells. Use your Miniconda Python explicitly:

```bash
C:/ProgramData/miniconda3/python.exe -m pytest tests/ -v
C:/ProgramData/miniconda3/python.exe -m fourdstem_pipeline.cli run --config ...
```

Or set up an alias in `~/.bashrc`:
```bash
alias python='C:/ProgramData/miniconda3/python.exe'
```

### Orientation ROI with zero area

If the pipeline reports `ORIENTATION_ROI_INVALID` in `qc_summary.json`, the
configured `orientation.roi` has zero or negative height/width. For example:

```yaml
orientation:
  roi: [64, 64, 192, 192]   # ERROR: y1 == y0, zero height
```

The fix is to ensure `y1 > y0` and `x1 > x0`:

```yaml
orientation:
  roi: [64, 192, 64, 192]   # Correct: [y0, y1, x0, x1]
```

### Orientation preview skipped

Orientation preview can fail for several reasons:
- Zero-area ROI (see above).
- The configured ROI falls entirely outside the navigation shape after binning.
- Template matching failed because no candidate phases with valid templates
  were provided.

When orientation is skipped, orientation-dependent diagnostics (reliability
maps, ROI candidates based on orientation scores) are not generated, but all
other diagnostics and the report are still produced.

### Stage 2A: "py4DSTEM not installed"

```bash
pip install -e ".[diffraction]"
```

Or directly: `pip install py4DSTEM>=0.14`

### Stage 2A: "Cannot determine data file path"

The Stage 2 config needs to find the original MIB file. Either:
- Set `data_path` in `configs/stage2_roi_bragg.yaml`, or
- Ensure `provenance.json` in the Stage 1 output has a valid `input_path`.

### Binning truncation

When `q_bin` or `r_bin` doesn't evenly divide the detector or navigation
dimensions, the edge pixels are silently truncated. The pipeline now logs
warnings when this happens. To avoid data loss, choose bin factors that
divide your dimensions evenly.

### Emoji display (mojibake) in generated reports

The pipeline avoids Unicode emoji in generated Markdown and HTML. QC status is
rendered as ASCII labels (`[PASS]`, `[WARN]`, `[FAIL]`, `[N/A]`) for maximum
compatibility across platforms, terminals, and text editors.

---

## Large Data Policy

The default workflow avoids materializing a full 4D dataset. Virtual images,
radial fingerprints, and orientation previews are computed in navigation blocks.
Expensive orientation matching defaults to a downsampled preview plus ROI
refinement.

For real MIB data, install the optional large-data stack. MIB loading goes
through HyperSpy/RosettaSciIO, and `backend: hyperspy_pyxem` asks the loader to
register pyxem diffraction semantics when pyxem is available while keeping the
block-wise fallback path intact.

```bash
pip install -e ".[large-data]"
```

`load_dataset()` will then try HyperSpy lazy loading first for `.mib` files.
Use `preprocess.q_crop`, `preprocess.q_bin`, and `preprocess.r_bin` to reduce
large data before phase screening.

For the current `data/0617-4d` folder, start with:

```bash
fourdstem-run --config configs/0617_4d_workflow.yaml
```

That preset chooses one 512×512 MIB scan from the directory, crops the central
diffraction region, bins diffraction 2×, and bins scan navigation 4× for a
128×128 first-pass screening workflow.

---

## Development

```bash
# Install with all optional dependencies
pip install -e ".[test,large-data,diffraction]"

# Run tests
python -m pytest tests/ -v

# Run with debug logging
fourdstem-run --config configs/default_workflow.yaml --log-level DEBUG
```

### Running without pip install

If you prefer not to install the package, use the script entry point:

```bash
python scripts/run_workflow.py --config configs/default_workflow.yaml
```

Or run the CLI module directly:

```bash
python -m fourdstem_pipeline.cli run --config configs/default_workflow.yaml
```

### Package structure

```
src/fourdstem_pipeline/
├── cli.py              # CLI entry points (fourdstem-run, -dry-run, -stage2)
├── workflow.py         # Stage 1 orchestration
├── stage2.py           # Stage 2A orchestration
├── roi_bragg.py        # py4DSTEM Bragg disk detection per ROI
├── loaders.py          # synthetic, NumPy, NPZ, HyperSpy data loading
├── preprocess.py       # lazy q_crop, q_bin, r_bin reduction
├── virtual.py          # virtual BF/ADF/HAADF/COM image computation
├── masks.py            # annular mask generation
├── fingerprints.py     # radial fingerprint computation
├── phase.py            # unsupervised phase screening (PCA+NMF+clustering)
├── orientation.py      # orientation preview
├── sample_mask.py      # sample/vacuum mask generation
├── diagnostics.py      # cluster diagnostics, ring ratios, statistics
├── diagnostics_cluster.py   # per-cluster visualizations
├── diagnostics_spatial.py   # spatial diagnostics
├── qc.py               # quality control checks
├── provenance.py       # runtime dependency reporting
├── export.py           # report generation (Markdown, HTML, PNG writing)
├── contracts.py        # DataContract and Stage1Manifest
├── config.py           # YAML config loading and validation
├── dataset.py          # DatasetHandle dataclass
└── logging.py          # structured logging
```

---

## Framework

- **Data loading**: `loaders.py` — synthetic, NumPy, NPZ, or HyperSpy-backed.
- **Preprocessing**: `preprocess.py` — lazy `q_crop`, `q_bin`, and `r_bin` reduction.
- **Workflow**: `workflow.py` — single orchestration path for all Stage 1 stages.
- **Stage 2A**: `stage2.py` + `roi_bragg.py` — py4DSTEM Bragg disk detection per ROI.
- **Contract**: `contracts.py` — `DataContract` coordinate conventions and `Stage1Manifest` bridge validation.

### Config presets

| File | Purpose |
|---|---|
| `configs/default_workflow.yaml` | Synthetic smoke-test (16×16×64×64) |
| `configs/0617_4d_workflow.yaml` | Real-data: 128×128 screening with 4× r_bin |
| `configs/0617_4d_workflow_rbin1.yaml` | Real-data: full 512×512, no navigation binning |
| `configs/0617_4d_workflow_rbin2.yaml` | Real-data: 256×256 with 2× r_bin |
| `configs/0617_4d_stage1_enhanced.yaml` | Real-data: enhanced QC with sample mask |
| `configs/stage2_roi_bragg.yaml` | Stage 2A Bragg detection configuration |
