from __future__ import annotations

import unittest
import shutil
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fourdstem_pipeline import (
    apply_preprocess,
    build_annular_masks,
    compute_radial_fingerprints,
    compute_virtual_images,
    load_dataset,
    run_orientation_preview,
    run_workflow,
    screen_phases,
)


WORKSPACE_TMP = Path(__file__).resolve().parents[1] / "test_outputs"
WORKSPACE_TMP.mkdir(exist_ok=True)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = WORKSPACE_TMP / self._testMethodName
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def test_synthetic_loader_reports_navigation_and_signal_shapes(self):
        dataset = load_dataset("synthetic://demo")
        self.assertEqual(dataset.navigation_shape, (16, 16))
        self.assertEqual(dataset.signal_shape, (64, 64))
        self.assertEqual(dataset.describe()["source"], "synthetic")
        self.assertEqual(dataset.describe()["source_backend"], "synthetic")

    def test_numpy_loaders_do_not_require_pyxem(self):
        data = np.zeros((3, 4, 8, 10), dtype=np.float32)
        npy_path = self.output_dir / "scan.npy"
        npz_path = self.output_dir / "scan.npz"
        np.save(npy_path, data)
        np.savez_compressed(npz_path, data=data + 1)

        npy = load_dataset(npy_path, backend="numpy")
        npz = load_dataset(npz_path, backend="numpy")

        self.assertEqual(npy.navigation_shape, (3, 4))
        self.assertEqual(npy.signal_shape, (8, 10))
        self.assertEqual(npy.describe()["source_backend"], "numpy")
        self.assertEqual(npz.navigation_shape, (3, 4))
        self.assertEqual(npz.signal_shape, (8, 10))
        self.assertEqual(npz.describe()["source_backend"], "numpy")

    def test_virtual_images_have_navigation_shape(self):
        dataset = load_dataset("synthetic://demo")
        masks = build_annular_masks(
            dataset.signal_shape,
            {
                "bf": {"inner_radius": 0, "outer_radius": 8},
                "adf": {"inner_radius": 10, "outer_radius": 22},
            },
        )
        result = compute_virtual_images(dataset, masks, output_dir=self.output_dir)
        self.assertEqual(result.images["bf"].shape, dataset.navigation_shape)
        self.assertEqual(result.images["adf"].shape, dataset.navigation_shape)
        self.assertEqual(result.com_x.shape, dataset.navigation_shape)
        self.assertEqual(result.mean_diffraction.shape, dataset.signal_shape)
        self.assertTrue((self.output_dir / "virtual_bf.npy").exists())

    def test_radial_fingerprints_and_phase_screening(self):
        dataset = load_dataset("synthetic://demo")
        fingerprints = compute_radial_fingerprints(dataset, {"center": None}, 24, output_dir=self.output_dir)
        self.assertEqual(fingerprints.profiles.shape, dataset.navigation_shape + (24,))

        reference_a = fingerprints.profiles[:4, :4].mean(axis=(0, 1)).tolist()
        reference_b = fingerprints.profiles[-4:, -4:].mean(axis=(0, 1)).tolist()
        result = screen_phases(
            fingerprints,
            candidate_phases=[
                {"name": "alpha", "reference_profile": reference_a},
                {"name": "beta", "reference_profile": reference_b},
            ],
            n_components=3,
            n_clusters=3,
            output_dir=self.output_dir,
        )
        self.assertEqual(result.labels.shape, dataset.navigation_shape)
        self.assertEqual(result.representative_profiles.shape[1], 24)
        self.assertEqual(set(result.candidate_scores), {"alpha", "beta"})
        self.assertEqual(result.low_confidence_mask.shape, dataset.navigation_shape)

    def test_orientation_preview_roi_path(self):
        dataset = load_dataset("synthetic://demo")
        result = run_orientation_preview(
            dataset,
            binning=(2, 2),
            roi=(4, 4, 12, 12),
            confidence_threshold=0.0,
            output_dir=self.output_dir,
        )
        self.assertEqual(result.orientation_index.shape, (4, 4))
        self.assertEqual(result.phase_label.shape, (4, 4))
        self.assertEqual(result.score.shape, (4, 4))
        self.assertTrue(np.all(result.score >= 0))
        self.assertTrue((self.output_dir / "orientation_index.npy").exists())

    def test_preprocess_changes_navigation_and_signal_shapes(self):
        dataset = load_dataset("synthetic://demo")
        preprocessed = apply_preprocess(dataset, q_crop=[8, 56, 4, 60], q_bin=2, r_bin=4)
        self.assertEqual(preprocessed.navigation_shape, (4, 4))
        self.assertEqual(preprocessed.signal_shape, (24, 28))
        block = preprocessed.data[:2, :2, :, :]
        self.assertEqual(block.shape, (2, 2, 24, 28))

    def test_run_workflow_synthetic_config(self):
        config = {
            "project": {"name": "test", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True, "chunks": {"navigation": [8, 8], "signal": [64, 64]}},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 24},
            "virtual_images": {
                "masks": {
                    "bf": {"inner_radius": 0, "outer_radius": 8},
                    "adf": {"inner_radius": 10, "outer_radius": 22},
                }
            },
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 3, "n_clusters": 3, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 4, 12, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
        }
        result = run_workflow(config)
        self.assertEqual(result.dataset.navigation_shape, (16, 16))
        self.assertTrue((self.output_dir / "workflow_summary.json").exists())
        self.assertTrue((self.output_dir / "01_virtual_images" / "virtual_bf.npy").exists())
        self.assertTrue((self.output_dir / "02_fingerprints" / "radial_fingerprints.npy").exists())
        self.assertTrue((self.output_dir / "03_diffraction_classes" / "diffraction_class_labels.npy").exists())
        self.assertTrue((self.output_dir / "04_orientation_preview" / "orientation_index.npy").exists())
        self.assertTrue((self.output_dir / "05_cluster_diagnostics" / "cluster_summary.csv").exists())
        self.assertTrue((self.output_dir / "06_roi_candidates" / "roi_candidates.yaml").exists())

    def test_run_workflow_resolves_directory_input(self):
        data_dir = self.output_dir / "data"
        data_dir.mkdir()
        yy, xx = np.indices((16, 16))
        data = np.empty((4, 4, 16, 16), dtype=np.float32)
        for iy in range(4):
            for ix in range(4):
                data[iy, ix] = np.exp(-0.5 * ((np.sqrt((yy - 8) ** 2 + (xx - 8) ** 2) - 3 - iy) / 1.5) ** 2)
        np.save(data_dir / "scan_b.npy", data + 1)
        np.save(data_dir / "scan_a.npy", data)
        config = {
            "project": {"name": "test", "output_dir": str(self.output_dir / "run")},
            "data": {"backend": "hyperspy_pyxem", "directory": str(data_dir), "pattern": "*.npy", "index": 0, "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": None, "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
        }
        result = run_workflow(config)
        self.assertTrue(result.dataset.metadata["path"].endswith("scan_a.npy"))
        self.assertTrue((self.output_dir / "run" / "workflow_summary.json").exists())
        summary = json.loads((self.output_dir / "run" / "workflow_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["data_config"]["backend"], "hyperspy_pyxem")
        self.assertEqual(summary["dataset"]["source_backend"], "hyperspy_pyxem")


if __name__ == "__main__":
    unittest.main()
