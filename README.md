# large-4dstem-analysis

Unified non-visual 4D-STEM analysis pipeline:

1. Stage 1 fingerprint-class screening
2. Stage 2A ROI Bragg detection
3. Stage 2B crystallographic candidate indexing
4. Optional Stage 2C pyxem pattern-matching validation
5. Optional consensus/conflict mapping

The normal path is now a single YAML file and a single command.

## Install

```bash
conda activate large-4dstem
pip install -e .
```

On Windows, avoid the Microsoft Store Python stub by activating the conda
environment before running commands.

## Quick Start

Run all enabled stages:

```bash
fourdstem-pipeline --config configs/pipeline.yaml
```

Equivalent module form:

```bash
python -m fourdstem_pipeline.cli pipeline --config configs/pipeline.yaml
```

For a lightweight local smoke run that executes Stage 1 only:

```bash
fourdstem-pipeline --config configs/pipeline_smoke.yaml
```

## Unified Config

`configs/pipeline.yaml` is the canonical configuration. It contains:

| Section | Purpose |
| --- | --- |
| `pipeline` | Enabled stages and aggregate pipeline output directory |
| `project` / `data` / `preprocess` | Stage 1 input and preprocessing |
| `geometry` / `virtual_images` | Stage 1 detector geometry and virtual images |
| `phase_screening` / `orientation` / `sample_mask` | Stage 1 analysis options |
| `stage2a` | ROI Bragg extraction and py4DSTEM disk-detection parameters |
| `stage2b` | CIF template generation, matching, and candidate phases |
| `stage2c` | pyxem/HyperSpy polar pattern-matching validation inputs and QC |
| `consensus` | Optional agreement/conflict fusion between Stage 2B and Stage 2C |

Choose stages with:

```yaml
pipeline:
  stages: [stage1, stage2a, stage2b]
```

To consume an existing pyxem validation NPZ from
`scripts/pyxem_hyperspy_ti_phase_orientation.py`, set:

```yaml
pipeline:
  stages: [stage1, stage2a, stage2b, stage2c]

stage2c:
  input:
    results_npz: results/pyxem_roi/pyxem_ti_phase_orientation_results.npz
```

Examples:

```yaml
pipeline:
  stages: [stage1]
```

```yaml
pipeline:
  stages: [stage1, stage2a]
```

Intermediate paths are injected automatically:

- Stage 2A receives `stage1_dir` from the Stage 1 output directory.
- Stage 2B receives `stage2_dir` from the Stage 2A output directory.
- Stage 2C receives Stage 2A/2B directories and writes a consensus-ready
  `stage2c_manifest.json`.

## Outputs

The unified runner writes:

| Output | Purpose |
| --- | --- |
| `pipeline_summary.json` | Per-stage status, output paths, errors, and skip reasons |
| Stage 1 output dir | `stage1_summary.json`, QC, report, virtual images, fingerprints, ROI candidates |
| Stage 2A output dir | `stage2_summary.json`, Bragg maps, ROI summaries, benchmark, gallery |
| Stage 2B output dir | `stage2_indexing_summary.json`, phase/orientation evidence, reports |
| Stage 2C output dir | `stage2c_summary.json`, standardised pyxem arrays, validation manifest |

Default output layout:

```text
outputs/<run>/
  stage1_summary.json
  pipeline/pipeline_summary.json
  stage2/roi_bragg/
    stage2_summary.json
    stage2b_indexing/
      stage2_indexing_summary.json
    stage2c_pyxem_validation/
      stage2c_summary.json
      stage2c_manifest.json
```

Stage 2B and Stage 2C are different evidence branches. Stage 2B evaluates
Bragg-vector crystallographic evidence; Stage 2C evaluates full-pattern polar
template matching. The consensus layer reports agreement, ambiguity, and
conflict instead of forcing one final phase map.

## Single-Stage Debugging

The old stage-specific commands are still available and can read the unified
config:

```bash
fourdstem-run --config configs/pipeline.yaml
fourdstem-stage2 --config configs/pipeline.yaml
fourdstem-stage2b --config configs/pipeline.yaml
fourdstem-stage2c --config configs/pipeline.yaml
```

The module form also works:

```bash
python -m fourdstem_pipeline.cli run --config configs/pipeline.yaml
python -m fourdstem_pipeline.cli stage2 --config configs/pipeline.yaml
python -m fourdstem_pipeline.cli stage2b --config configs/pipeline.yaml
python -m fourdstem_pipeline.cli stage2c --config configs/pipeline.yaml
```

## CLI Reference

| Command | Description |
| --- | --- |
| `fourdstem-pipeline` | Run the unified multi-stage pipeline |
| `fourdstem-run` | Run Stage 1 only |
| `fourdstem-dry-run` | Validate Stage 1 config and estimate resources |
| `fourdstem-stage2` | Run Stage 2A ROI Bragg detection |
| `fourdstem-stage2b` | Run Stage 2B indexing |
| `fourdstem-stage2c` | Standardise pyxem validation outputs |
| `fourdstem-bin-export` | Bin raw data and export to EMD/H5 |
| `fourdstem-crop-export` | Crop navigation dimensions and export to EMD/H5 |

## Development

Run tests from an environment with Python and pytest installed:

```bash
python -m pytest tests/test_workflow.py -q
```

If `python` exits immediately on Windows, activate the `large-4dstem` conda
environment first.

## Coordinate Conventions

All stages use:

| Concept | Order |
| --- | --- |
| 4D data axes | `nav_y, nav_x, q_y, q_x` |
| ROI bbox | `y0, y1, x0, x1` |
| Point/center | `y, x` |

Stage 1 ROI candidates are in preprocessed navigation coordinates. Stage 2A
converts them to raw scan coordinates using `r_bin`.
