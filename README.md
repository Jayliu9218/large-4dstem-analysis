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
