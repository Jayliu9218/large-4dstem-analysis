from __future__ import annotations

import unittest
import shutil
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fourdstem_pipeline import (
    build_annular_masks,
    compute_radial_fingerprints,
    compute_virtual_images,
    load_dataset,
    run_orientation_preview,
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


if __name__ == "__main__":
    unittest.main()
