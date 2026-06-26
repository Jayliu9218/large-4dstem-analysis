# large-mib-4dstem-analysis

Notebook-first workflow for large 4D-STEM data, focused on:

- virtual BF/ADF/HAADF and COM imaging
- radial fingerprints and unsupervised phase screening
- candidate phase score maps
- low-resolution full-field orientation preview plus ROI refinement

The code is designed so the notebook and future desktop workbench call the same
Python interfaces in `src/fourdstem_pipeline`.

## Quick Start

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

Open `notebooks/01_full_workflow.ipynb` for the full workflow. It runs with
synthetic data by default, then can be pointed at a real `.mib`, `.npy`, or
`.npz` dataset through `configs/default_workflow.yaml`.

## Large Data Policy

The default workflow avoids materializing a full 4D dataset. Virtual images,
radial fingerprints, and orientation previews are computed in navigation blocks,
and expensive orientation matching defaults to downsampled preview plus ROI
refinement.

For real MIB data, install the optional large-data stack:

```powershell
python -m pip install -e ".[large-data]"
```

`load_dataset()` will then try HyperSpy lazy loading first for `.mib` files.
