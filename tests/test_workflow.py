from __future__ import annotations

import unittest
import shutil
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fourdstem_pipeline import (
    Stage1Manifest,
    Stage1ManifestLoadError,
    apply_preprocess,
    build_annular_masks,
    compute_radial_fingerprints,
    compute_virtual_images,
    load_dataset,
    load_roi_candidates,
    load_stage1_manifest,
    run_orientation_preview,
    run_stage1_diagnostics,
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
            roi=(4, 12, 4, 12),
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
                    "ring_1": {"inner_radius": 8, "outer_radius": 16},
                    "ring_2": {"inner_radius": 16, "outer_radius": 28},
                    "ring_3": {"inner_radius": 28, "outer_radius": 31},
                }
            },
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 3, "n_clusters": 3, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        result = run_workflow(config)
        self.assertEqual(result.dataset.navigation_shape, (16, 16))
        self.assertTrue((self.output_dir / "workflow_summary.json").exists())
        self.assertTrue((self.output_dir / "stage1_summary.json").exists())
        self.assertTrue((self.output_dir / "data_contract.json").exists())
        self.assertTrue((self.output_dir / "preprocess_info.json").exists())
        self.assertTrue((self.output_dir / "virtual" / "virtual_bf.npy").exists())
        self.assertTrue((self.output_dir / "virtual" / "virtual_images.npz").exists())
        self.assertTrue((self.output_dir / "fingerprints" / "radial_fingerprints.npy").exists())
        self.assertTrue((self.output_dir / "fingerprints" / "radial_axis.npy").exists())
        self.assertTrue((self.output_dir / "fingerprint_classes" / "fingerprint_class_labels.npy").exists())
        self.assertTrue((self.output_dir / "fingerprint_classes" / "fingerprint_class_labels_cleaned.npy").exists())
        self.assertTrue((self.output_dir / "fingerprint_classes" / "cluster_summary.csv").exists())
        self.assertTrue((self.output_dir / "fingerprint_classes" / "cluster_mean_radial_profiles.npy").exists())
        self.assertTrue((self.output_dir / "orientation" / "orientation_index.npy").exists())
        self.assertTrue((self.output_dir / "05_cluster_diagnostics" / "cluster_summary.csv").exists())
        self.assertTrue((self.output_dir / "05_cluster_diagnostics" / "cluster_cleaned_labels.npy").exists())
        self.assertTrue((self.output_dir / "05_cluster_diagnostics" / "ring_2_over_ring_1.npy").exists())
        self.assertTrue((self.output_dir / "05_cluster_diagnostics" / "cluster_vs_orientation.csv").exists())
        self.assertTrue((self.output_dir / "roi_candidates" / "roi_candidates.yaml").exists())
        self.assertTrue((self.output_dir / "report.html").exists())

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
            "sample_mask": {"enabled": False},
        }
        result = run_workflow(config)
        self.assertTrue(result.dataset.metadata["path"].endswith("scan_a.npy"))
        self.assertTrue((self.output_dir / "run" / "workflow_summary.json").exists())
        summary = json.loads((self.output_dir / "run" / "workflow_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["data_config"]["backend"], "hyperspy_pyxem")
        self.assertEqual(summary["dataset"]["source_backend"], "hyperspy_pyxem")


    def test_parse_roi_rejects_zero_area(self):
        """parse_roi raises ValueError when ROI has zero height or width."""
        from fourdstem_pipeline.array_utils import parse_roi

        # Direct zero area (y1 == y0)
        with self.assertRaises(ValueError):
            parse_roi([64, 64, 192, 256], shape=(512, 512))

        # Zero width (x1 == x0)
        with self.assertRaises(ValueError):
            parse_roi([64, 128, 192, 192], shape=(512, 512))

        # Both zero
        with self.assertRaises(ValueError):
            parse_roi([64, 64, 192, 192], shape=(512, 512))

        # Clamped to zero (all values beyond shape)
        with self.assertRaises(ValueError):
            parse_roi([600, 700, 600, 700], shape=(512, 512))

        # Valid ROI should not raise
        y_s, x_s = parse_roi([64, 128, 192, 256], shape=(512, 512))
        self.assertEqual(y_s, slice(64, 128))
        self.assertEqual(x_s, slice(192, 256))

    def test_run_workflow_skips_zero_area_orientation_roi(self):
        """Workflow skips orientation with info flag when ROI has zero area."""
        config = {
            "project": {"name": "test_zero_roi", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True, "chunks": {"navigation": [8, 8], "signal": [64, 64]}},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 24},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 8}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [64, 64, 192, 192], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        result = run_workflow(config)
        # Orientation should be None
        self.assertIsNone(result.orientation)
        # Workflow should still complete with diagnostics
        self.assertTrue((self.output_dir / "workflow_summary.json").exists())
        self.assertTrue((self.output_dir / "report.md").exists())
        summary = json.loads((self.output_dir / "workflow_summary.json").read_text(encoding="utf-8"))
        # QC should include the ORIENTATION_ROI_INVALID info flag
        qc = summary.get("qc", {})
        flags = qc.get("flags", [])
        self.assertTrue(
            any(f.get("code") == "ORIENTATION_ROI_INVALID" for f in flags),
            f"Expected ORIENTATION_ROI_INVALID flag, got: {[f.get('code') for f in flags]}",
        )

    def test_diagnostics_resilient_to_missing_orientation(self):
        """run_stage1_diagnostics preserves partial results when orientation is None."""
        from fourdstem_pipeline.orientation import OrientationResult
        import numpy as np

        dataset = load_dataset("synthetic://demo")
        fingerprints = compute_radial_fingerprints(dataset, {"center": None}, 8, output_dir=self.output_dir)
        phase = screen_phases(fingerprints, n_components=2, n_clusters=2, output_dir=self.output_dir / "classes")
        masks = build_annular_masks(
            dataset.signal_shape,
            {"bf": {"inner_radius": 0, "outer_radius": 4}, "adf": {"inner_radius": 4, "outer_radius": 8}},
        )
        virtual = compute_virtual_images(dataset, masks, output_dir=self.output_dir)

        # With valid orientation — all outputs present
        orientation = run_orientation_preview(
            dataset, binning=(2, 2), roi=(4, 12, 4, 12),
            confidence_threshold=0.0, output_dir=self.output_dir / "orient",
        )
        result = run_stage1_diagnostics(
            dataset, fingerprints, phase, virtual, orientation,
            output_dir=self.output_dir / "diag",
            png_dir=self.output_dir / "diag_png",
            block_shape=(8, 8),
            confidence_threshold=0.0,
        )
        self.assertIn("cluster_diagnostics", result)
        self.assertIn("cluster_summary_csv", result)
        self.assertIn("beam", result)
        self.assertIn("connected_components", result)
        self.assertIn("orientation_reliability", result)
        self.assertIn("roi_outputs", result)
        self.assertIn("cluster_vs_orientation", result)
        self.assertNotIn("_errors", result)

        # With orientation=None — core outputs present, orientation-dep ones empty
        result_none = run_stage1_diagnostics(
            dataset, fingerprints, phase, virtual, None,
            output_dir=self.output_dir / "diag_none",
            png_dir=self.output_dir / "diag_none_png",
            block_shape=(8, 8),
            confidence_threshold=0.0,
        )
        self.assertIn("cluster_diagnostics", result_none)
        self.assertIn("beam", result_none)
        self.assertEqual(result_none.get("orientation_reliability"), {})
        self.assertEqual(result_none.get("roi_outputs"), {})
        self.assertEqual(result_none.get("cluster_vs_orientation"), {})

    def test_qc_summary_and_report_use_ascii_labels(self):
        """QC summary and report Markdown contain only ASCII labels, no emoji."""
        from fourdstem_pipeline.qc import QCResult, QCFlag, save_qc_summary

        for status, expected_label in [
            ("PASS", "[PASS]"),
            ("PASS_WITH_WARNINGS", "[WARN]"),
            ("FAIL", "[FAIL]"),
        ]:
            result = QCResult(
                stage1_status=status,
                n_warnings=1 if status != "PASS" else 0,
                n_critical=1 if status == "FAIL" else 0,
                flags=[QCFlag(severity="info", code="TEST", message="Test flag.")],
            )
            _, md_path = save_qc_summary(self.output_dir / f"qc_{status}", result)
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn(expected_label, md_text)
            for emoji_char in ["✅", "⚠", "❌", "❓"]:
                self.assertNotIn(emoji_char, md_text,
                                 f"Emoji {repr(emoji_char)} found in QC markdown for {status}")

        # Verify the workflow report Markdown also avoids emoji
        config = {
            "project": {"name": "test_ascii", "output_dir": str(self.output_dir / "run_ascii")},
            "data": {"path": "synthetic://demo", "lazy": True, "chunks": {"navigation": [8, 8], "signal": [64, 64]}},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 24},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 8}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        wf_result = run_workflow(config)
        report_text = wf_result.report_path.read_text(encoding="utf-8")
        for emoji_char in ["✅", "⚠", "❌", "❓"]:
            self.assertNotIn(emoji_char, report_text,
                             f"Emoji {repr(emoji_char)} found in report Markdown")

    def test_no_legacy_diffraction_class_terminology(self):
        """Generated Markdown uses 'fingerprint-class' not 'diffraction-class'."""
        from fourdstem_pipeline.qc import QCResult, QCFlag, save_qc_summary

        # Check QC markdown
        result = QCResult(
            stage1_status="PASS",
            n_warnings=0,
            n_critical=0,
            flags=[QCFlag(
                severity="warning", code="BEAM_CENTER_OFFSET",
                message="Radial fingerprints and fingerprint-class labels may be biased.",
                evidence={},
            )],
        )
        _, md_path = save_qc_summary(self.output_dir / "qc_term", result)
        md_text = md_path.read_text(encoding="utf-8")
        self.assertNotIn("diffraction-class", md_text)

        # Run a full workflow and check the report
        config = {
            "project": {"name": "test_term", "output_dir": str(self.output_dir / "run_term")},
            "data": {"path": "synthetic://demo", "lazy": True, "chunks": {"navigation": [8, 8], "signal": [64, 64]}},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 24},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 8}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        wf_result = run_workflow(config)
        report_text = wf_result.report_path.read_text(encoding="utf-8")
        self.assertNotIn("diffraction-class", report_text)


    # ------------------------------------------------------------------
    # Provenance / dependency reporting
    # ------------------------------------------------------------------

    def test_provenance_reports_scikit_learn_not_sklearn(self):
        """provenance.json reports 'scikit-learn' (the package name), not 'sklearn'."""
        config = {
            "project": {"name": "test_prov", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        prov_path = self.output_dir / "provenance.json"
        self.assertTrue(prov_path.exists())
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        packages = prov.get("packages", {})
        # scikit-learn should be reported (not sklearn)
        self.assertIn("scikit-learn", packages)
        self.assertNotIn("sklearn", packages)
        self.assertIsNotNone(packages["scikit-learn"],
                             "scikit-learn should report a version when installed")

    def test_report_has_runtime_dependency_availability_section(self):
        """Both MD and HTML reports use 'Runtime Dependency Availability' not 'Package Versions'."""
        config = {
            "project": {"name": "test_dep", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        result = run_workflow(config)

        md_text = result.report_path.read_text(encoding="utf-8")
        self.assertIn("Runtime Dependency Availability", md_text)
        self.assertNotIn("Package Versions", md_text)

        html_path = result.report_path.with_suffix(".html")
        html_text = html_path.read_text(encoding="utf-8")
        self.assertIn("Runtime Dependency Availability", html_text)
        self.assertNotIn("Package Versions", html_text)

    def test_summary_includes_dependency_fields(self):
        """workflow_summary.json includes 'dependencies' with pyxem/py4DSTEM info."""
        config = {
            "project": {"name": "test_dep2", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        summary_path = self.output_dir / "workflow_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        deps = summary.get("dependencies", {})
        self.assertIn("pyxem_available", deps)
        self.assertIn("pyxem_signal_type", deps)
        self.assertIn("py4DSTEM_used", deps)
        self.assertIn("source_backend", deps)
        # py4DSTEM_used should be False since roi_bragg is disabled
        self.assertFalse(deps["py4DSTEM_used"])

        # stage1_summary.json should also include dependencies
        s1_path = self.output_dir / "stage1_summary.json"
        s1 = json.loads(s1_path.read_text(encoding="utf-8"))
        self.assertIn("dependencies", s1)

    def test_provenance_pyxem_reports_not_installed(self):
        """When pyxem is not installed, provenance reports it as null/not-installed."""
        from fourdstem_pipeline.provenance import _installed_packages

        packages = _installed_packages()
        self.assertIn("pyxem", packages)
        # pyxem may or may not be installed in test env, but key should exist
        if packages["pyxem"] is None:
            # Verify the report renders "not installed" label
            from fourdstem_pipeline.export import _render_package_versions
            rendered = _render_package_versions(packages)
            self.assertIn("not installed", rendered)

    # ------------------------------------------------------------------
    # Stage1Manifest contract validation
    # ------------------------------------------------------------------

    def test_stage1_manifest_loads_valid_output(self):
        """Stage1Manifest.load succeeds on a valid Stage-1 output directory."""
        config = {
            "project": {"name": "test_manifest", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        manifest = Stage1Manifest.load(self.output_dir)
        self.assertEqual(manifest.run_name, "test_manifest")
        self.assertEqual(manifest.nav_shape, [16, 16])
        self.assertEqual(manifest.sig_shape, [64, 64])
        self.assertEqual(manifest.qc_status, "PASS")
        self.assertTrue(manifest.labels_path.exists())
        self.assertTrue(manifest.roi_candidates_path.exists())
        self.assertTrue(manifest.virtual_images_path.exists())
        self.assertTrue(manifest.fingerprints_path.exists())
        self.assertTrue(manifest.radial_axis_path.exists())
        self.assertTrue(manifest.data_contract_path.exists())
        self.assertTrue(manifest.provenance_path.exists())
        self.assertTrue(manifest.qc_summary_path.exists())
        # Convenience accessors
        self.assertTrue(manifest.has_orientation)
        self.assertTrue(manifest.qc_passed)

    def test_stage1_manifest_missing_file_raises(self):
        """Stage1Manifest.load raises when stage1_summary.json is absent."""
        empty_dir = self.output_dir / "empty"
        empty_dir.mkdir()
        with self.assertRaises(Stage1ManifestLoadError) as ctx:
            Stage1Manifest.load(empty_dir)
        self.assertIn("not found", str(ctx.exception))

    def test_stage1_manifest_missing_required_key_raises(self):
        """Stage1Manifest.load raises when required keys are missing."""
        manifest_dir = self.output_dir / "bad_manifest"
        manifest_dir.mkdir()
        bad = {
            "run_name": "test",
            "nav_shape": [16, 16],
            # Missing sig_shape, labels_path, etc.
        }
        (manifest_dir / "stage1_summary.json").write_text(
            json.dumps(bad), encoding="utf-8"
        )
        with self.assertRaises(Stage1ManifestLoadError) as ctx:
            Stage1Manifest.load(manifest_dir)
        self.assertIn("missing required keys", str(ctx.exception))

    def test_stage1_manifest_fail_qc_raises(self):
        """Stage1Manifest.load raises when qc_status is FAIL."""
        manifest_dir = self.output_dir / "fail_qc"
        manifest_dir.mkdir()
        # Write a minimal manifest with FAIL status
        fail_manifest = {
            "run_name": "test_fail",
            "nav_shape": [16, 16],
            "sig_shape": [64, 64],
            "qc_status": "FAIL",
            "labels_path": "labels.npy",
            "roi_candidates_path": "rois.yaml",
            "virtual_images_path": "virtual.npz",
            "fingerprints_path": "fprints.npy",
            "radial_axis_path": "axis.npy",
            "data_contract_path": "dc.json",
            "provenance_path": "prov.json",
            "qc_summary_path": "qc.json",
        }
        # Create dummy required files with valid content
        for key in ["labels_path", "roi_candidates_path", "virtual_images_path",
                     "fingerprints_path", "radial_axis_path",
                     "provenance_path", "qc_summary_path"]:
            p = manifest_dir / fail_manifest[key]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        # data_contract.json must have valid bbox_order
        (manifest_dir / "dc.json").write_text(
            json.dumps({"bbox_order": "y0_y1_x0_x1"})
        )
        (manifest_dir / "stage1_summary.json").write_text(
            json.dumps(fail_manifest), encoding="utf-8"
        )
        with self.assertRaises(Stage1ManifestLoadError) as ctx:
            Stage1Manifest.load(manifest_dir)
        self.assertIn("FAIL", str(ctx.exception))

    def test_stage1_manifest_missing_file_on_disk_raises(self):
        """Stage1Manifest.load raises when a required file path does not exist."""
        manifest_dir = self.output_dir / "missing_file"
        manifest_dir.mkdir()
        ok_manifest = {
            "run_name": "test",
            "nav_shape": [16, 16],
            "sig_shape": [64, 64],
            "qc_status": "PASS",
            "labels_path": "labels.npy",
            "roi_candidates_path": "rois.yaml",
            "virtual_images_path": "virtual.npz",
            "fingerprints_path": "fprints.npy",
            "radial_axis_path": "axis.npy",
            "data_contract_path": "dc.json",
            "provenance_path": "prov.json",
            "qc_summary_path": "qc.json",
        }
        # Create only some of the required files (missing labels.npy)
        for key in ["roi_candidates_path", "virtual_images_path",
                     "fingerprints_path", "radial_axis_path", "data_contract_path",
                     "provenance_path", "qc_summary_path"]:
            p = manifest_dir / ok_manifest[key]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        # labels.npy intentionally not created
        (manifest_dir / "stage1_summary.json").write_text(
            json.dumps(ok_manifest), encoding="utf-8"
        )
        with self.assertRaises(Stage1ManifestLoadError) as ctx:
            Stage1Manifest.load(manifest_dir)
        self.assertIn("does not exist", str(ctx.exception))

    def test_load_stage1_manifest_convenience(self):
        """load_stage1_manifest is a convenience wrapper."""
        config = {
            "project": {"name": "test_conv", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        manifest = load_stage1_manifest(self.output_dir)
        self.assertIsInstance(manifest, Stage1Manifest)
        self.assertEqual(manifest.run_name, "test_conv")

    def test_stage1_summary_paths_are_relative(self):
        """All path values in stage1_summary.json are relative (POSIX style)."""
        config = {
            "project": {"name": "test_rel", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        s1_path = self.output_dir / "stage1_summary.json"
        s1 = json.loads(s1_path.read_text(encoding="utf-8"))
        for key in ["labels_path", "roi_candidates_path", "virtual_images_path",
                     "fingerprints_path", "radial_axis_path", "data_contract_path",
                     "provenance_path", "qc_summary_path"]:
            val = s1[key]
            self.assertIsNotNone(val, f"{key} should not be None")
            # Paths should use forward slashes (POSIX)
            self.assertNotIn("\\", val, f"{key} should use forward slashes: {val}")
            # Should not be absolute
            self.assertFalse(
                val.startswith("/") or (len(val) > 1 and val[1] == ":"),
                f"{key} should be relative, got: {val}",
            )

    def test_roi_candidates_loadable(self):
        """ROI candidates YAML can be loaded with load_roi_candidates."""
        config = {
            "project": {"name": "test_roi_load", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)
        yaml_path = self.output_dir / "roi_candidates" / "roi_candidates.yaml"
        self.assertTrue(yaml_path.exists())
        rois = load_roi_candidates(yaml_path)
        self.assertIsInstance(rois, list)
        self.assertGreater(len(rois), 0)
        for roi in rois:
            self.assertIn("name", roi)
            self.assertIn("bbox", roi)
            self.assertIn("center", roi)
            self.assertIn("size", roi)
            # bbox should be [y0, y1, x0, x1]
            bbox = roi["bbox"]
            self.assertEqual(len(bbox), 4)
            self.assertLess(bbox[0], bbox[1], f"y0 < y1 violated: {bbox}")
            self.assertLess(bbox[2], bbox[3], f"x0 < x1 violated: {bbox}")

    # ------------------------------------------------------------------
    # Stage 2A tests (no py4DSTEM required)
    # ------------------------------------------------------------------

    def test_stage2_config_validation(self):
        """Stage 2 config must contain stage1_dir."""
        from fourdstem_pipeline.stage2 import _load_stage2_config

        cfg_dir = self.output_dir / "cfg"
        cfg_dir.mkdir()
        bad_path = cfg_dir / "bad.yaml"
        bad_path.write_text("not_a_dir: true", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            _load_stage2_config(bad_path)
        self.assertIn("stage1_dir", str(ctx.exception))

    def test_stage2_missing_py4dstem_errors_clearly(self):
        """Stage 2 run raises ImportError with a clear message when py4DSTEM is missing."""
        import sys
        from unittest.mock import patch
        from fourdstem_pipeline.stage2 import run_stage2

        # Create a valid Stage-1 output
        config = {
            "project": {"name": "test_s2_err", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)

        # Create a real data file so _resolve_data_path succeeds
        import numpy as np
        data_path = str(self.output_dir / "dummy.npy")
        np.save(data_path, np.zeros((16, 16, 64, 64), dtype=np.float32))

        stage2_cfg = {
            "stage1_dir": str(self.output_dir),
            "output_dir": str(self.output_dir / "stage2"),
            "max_rois": 1,
            "thin_r": 2,
            "bin_q": 2,
            "data_path": data_path,
        }

        # Simulate py4DSTEM not being installed
        with patch.dict(sys.modules, {"py4DSTEM": None}):
            # Also remove any cached import
            for mod_key in list(sys.modules):
                if "py4DSTEM" in mod_key or "py4dstem" in mod_key:
                    sys.modules.pop(mod_key, None)
            with self.assertRaises(ImportError) as ctx:
                run_stage2(stage2_cfg)
            self.assertIn("py4DSTEM", str(ctx.exception))

    def test_stage2_manifest_validation_required(self):
        """Stage 2 rejects a missing stage1_summary.json."""
        from fourdstem_pipeline.stage2 import run_stage2
        from fourdstem_pipeline.contracts import Stage1ManifestLoadError

        empty_dir = self.output_dir / "no_s1"
        empty_dir.mkdir()
        stage2_cfg = {
            "stage1_dir": str(empty_dir),
            "output_dir": str(self.output_dir / "stage2"),
        }
        with self.assertRaises(Stage1ManifestLoadError):
            run_stage2(stage2_cfg)

    def test_stage2_run_creates_output_structure(self):
        """Stage 2 with mocked py4DSTEM creates expected output structure."""
        import sys
        from unittest.mock import MagicMock, patch
        import numpy as np

        # Create valid Stage-1 output first
        config = {
            "project": {"name": "test_s2_struct", "output_dir": str(self.output_dir)},
            "data": {"path": "synthetic://demo", "lazy": True},
            "preprocess": {"q_crop": None, "q_bin": 1, "r_bin": 1},
            "geometry": {"center": None, "radial_bins": 8},
            "virtual_images": {"masks": {"bf": {"inner_radius": 0, "outer_radius": 4}}},
            "phase_screening": {"method": "pca_nmf_cluster", "n_components": 2, "n_clusters": 2, "candidate_phases": []},
            "orientation": {"preview_binning": [2, 2], "roi": [4, 12, 4, 12], "confidence_threshold": 0.0},
            "roi_bragg": {"enabled": False},
            "sample_mask": {"enabled": False},
        }
        run_workflow(config)

        # Mock py4DSTEM to avoid needing the real package
        mock_py4dstem = MagicMock()
        mock_py4dstem.import_file.return_value = MagicMock(
            data=np.zeros((16, 16, 64, 64), dtype=np.float32),
            calibration=MagicMock(),
        )
        mock_dc = MagicMock()
        mock_dc.bin_Q.return_value = mock_dc
        mock_dc.data = np.zeros((4, 4, 32, 32), dtype=np.float32)
        mock_bragg = MagicMock()
        mock_hist = MagicMock()
        mock_hist.data = np.zeros((64, 64), dtype=np.float32)
        mock_bragg.histogram.return_value = mock_hist
        mock_dc.find_Bragg_disks.return_value = mock_bragg
        mock_py4dstem.DataCube.return_value = mock_dc

        with patch.dict(sys.modules, {"py4DSTEM": mock_py4dstem}):
            from fourdstem_pipeline.stage2 import run_stage2

            stage2_cfg = {
                "stage1_dir": str(self.output_dir),
                "output_dir": str(self.output_dir / "stage2_mock"),
                "max_rois": 1,
                "thin_r": 2,
                "bin_q": 2,
                "data_path": str(self.output_dir / "dummy.mib"),
            }
            # Create a dummy data file so path resolution works
            (self.output_dir / "dummy.mib").write_text("fake mib data")

            result = run_stage2(stage2_cfg)
            self.assertIsNotNone(result)
            self.assertEqual(len(result.roi_results), 1)
            self.assertIsNone(result.roi_results[0].error)

        # Check output structure
        s2_dir = self.output_dir / "stage2_mock"
        self.assertTrue((s2_dir / "stage2_summary.json").exists())
        self.assertTrue((s2_dir / "stage2_qc_summary.json").exists())
        self.assertTrue((s2_dir / "provenance.json").exists())

        summary = json.loads((s2_dir / "stage2_summary.json").read_text(encoding="utf-8"))
        self.assertIn("roi_results", summary)
        self.assertEqual(len(summary["roi_results"]), 1)
        r = summary["roi_results"][0]
        self.assertIn("n_bragg_peaks", r)
        self.assertIn("bragg_vector_map_path", r)

        qc = json.loads((s2_dir / "stage2_qc_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(qc["n_rois_total"], 1)
        self.assertEqual(qc["n_rois_success"], 1)
        self.assertEqual(qc["n_rois_failed"], 0)


if __name__ == "__main__":
    unittest.main()
