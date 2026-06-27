# large-4dstem-analysis

Single non-visual workflow for large 4D-STEM data, focused on:

- virtual BF/ADF/HAADF and COM imaging
- radial fingerprints and unsupervised phase screening
- candidate phase score maps
- low-resolution full-field orientation preview plus optional ROI Bragg detection

The codebase is intentionally small: one package, one default config, one CLI
entrypoint, and focused tests. Outputs are NumPy arrays and JSON metadata, not
figures or notebooks.

## Quick Start

```powershell
python -m pip install -e ".[test]"
python scripts/run_workflow.py --config configs/default_workflow.yaml
python -m unittest discover -s tests -v
```

The default config runs fully on `synthetic://demo`. Point `data.path` at a real
`.mib`, `.npy`, or `.npz` dataset when data are available.

## Stage 1 Outputs

A successful Stage 1 run produces the following key outputs in the configured
`output_dir`:

| Output | Path | Description |
| --- | --- | --- |
| Workflow summary | `workflow_summary.json` | Full pipeline metadata, shapes, and paths |
| Stage 1 manifest | `stage1_summary.json` | Canonical Stage-1 to Stage-2 interface file |
| QC summary | `qc_summary.json` / `qc_summary.md` | Quality-control flags and status |
| Data contract | `data_contract.json` | Axis order, bbox convention, centre convention |
| Preprocess info | `preprocess_info.json` | Crop/bin parameters and resulting shapes |
| Provenance | `provenance.json` | Git commit, config hash, package versions, timestamps |
| Virtual images | `virtual/virtual_images.npz` | BF, ADF, HAADF, ring images and COM maps |
| Radial fingerprints | `fingerprints/radial_fingerprints.npy` | Per-pixel radial profiles |
| Radial axis | `fingerprints/radial_axis.npy` | Radial bin centres (pixels) |
| Fingerprint classes | `fingerprint_classes/fingerprint_class_labels.npy` | Unsupervised phase-candidate label map |
| Orientation preview | `orientation/orientation_index.npy` | Binned orientation index map |
| Cluster diagnostics | `05_cluster_diagnostics/` | Cluster mean DPs, radial profiles, statistics, ring ratios |
| ROI candidates | `roi_candidates/roi_candidates.yaml` | Stage-2 ROI proposals with bboxes and rationale |
| Report | `report.md` / `report.html` | Human-readable summary with inline images |
| PNG previews | `png/` | Label maps, profiles, overlays for quick visual inspection |

## Coordinate Conventions

The pipeline follows a single unified data contract (`data_contract.json`):

- **Axis order**: `nav_y, nav_x, q_y, q_x` (navigation rows, navigation columns,
  diffraction rows, diffraction columns).
- **BBox / ROI order**: `y0, y1, x0, x1` (row start, row end, column start,
  column end). Python slice semantics: `y0` is inclusive, `y1` is exclusive.
- **Centre order**: `y, x` (row first, column second). The geometric centre of a
  detector of shape `(H, W)` is `((H-1)/2, (W-1)/2)`.

All array indexing and ROI specifications throughout the pipeline use this
convention. When configuring `orientation.roi` or `roi_bragg.roi`, ensure
`y1 > y0` and `x1 > x0` — zero-area ROIs are rejected with a clear error.

## Troubleshooting

### Orientation ROI with zero area

If the pipeline reports `ORIENTATION_ROI_INVALID` in `qc_summary.md`, the
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
- The configured ROI falls entirely outside the navigation shape after binning
  and clamping.
- Template matching failed because no candidate phases with valid templates
  were provided.

When orientation is skipped, orientation-dependent diagnostics (orientation
reliability maps, ROI candidates based on orientation scores) are not
generated, but all other diagnostics and the report are still produced.

### Emoji display (mojibake) in generated reports

The pipeline avoids Unicode emoji in generated Markdown and HTML. QC status is
rendered as ASCII labels (`[PASS]`, `[WARN]`, `[FAIL]`, `[N/A]`) for maximum
compatibility across platforms, terminals, and text editors.

## Framework

- `src/fourdstem_pipeline/loaders.py`: load synthetic, NumPy, NPZ, or HyperSpy-backed data.
- `src/fourdstem_pipeline/preprocess.py`: lazy `q_crop`, `q_bin`, and `r_bin` reduction.
- `src/fourdstem_pipeline/workflow.py`: single orchestration path for all stages.
- `configs/default_workflow.yaml`: synthetic smoke-test workflow configuration.
- `configs/0617_4d_workflow.yaml`: real-data preset for `data/0617-4d`.
- `scripts/run_workflow.py`: command-line entrypoint.

## Large Data Policy

The default workflow avoids materializing a full 4D dataset. Virtual images,
radial fingerprints, and orientation previews are computed in navigation blocks,
and expensive orientation matching defaults to downsampled preview plus ROI
refinement.

For real MIB data, install the optional large-data stack. MIB loading goes
through HyperSpy/RosettaSciIO, and `backend: hyperspy_pyxem` asks the loader to
register pyxem diffraction semantics when pyxem is available while keeping the
block-wise fallback path intact.

```powershell
python -m pip install -e ".[large-data]"
```

`load_dataset()` will then try HyperSpy lazy loading first for `.mib` files.
Use `preprocess.q_crop`, `preprocess.q_bin`, and `preprocess.r_bin` to reduce
large data before phase screening. The optional `roi_bragg` stage is disabled by
default and requires py4DSTEM when enabled:

```powershell
python -m pip install -e ".[diffraction]"
```

For the current `data/0617-4d` folder, start with:

```powershell
python scripts/run_workflow.py --config configs/0617_4d_workflow.yaml
```

That preset chooses one 512x512 MIB scan from the directory, crops the central
diffraction region, bins diffraction 2x, and bins scan navigation 4x for a
128x128 first-pass screening workflow.
