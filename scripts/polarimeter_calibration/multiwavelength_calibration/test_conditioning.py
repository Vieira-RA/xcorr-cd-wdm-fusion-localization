#!/usr/bin/env python3
"""Hardware-free tests for the calibration-matrix conditioning diagnostics."""
import csv
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from calibration import calibrate_csv_pair
from conditioning import (
    IDEAL_TETRAHEDRON_MATRIX,
    analyze_calibration_matrix,
    classify_conditioning,
    row_orthogonality,
    spectral_condition,
)
from synthetic import simulate_fused_acquisition, true_calibration_matrix

# The paper's unbalanced tetrahedral guess (Section "DOP Calibration"):
# rows are mutually orthogonal, but row 0 is sqrt(3) times shorter than rows 1-3.
PAPER_TETRAHEDRON_MATRIX = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [3.0, -1.0, -1.0, -1.0],
        [0.0, 2.0 * np.sqrt(2.0), -np.sqrt(2.0), -np.sqrt(2.0)],
        [0.0, 0.0, np.sqrt(6.0), -np.sqrt(6.0)],
    ]
)


class ConditioningTests(unittest.TestCase):
    def test_ideal_tetrahedron_is_perfectly_conditioned(self):
        report = analyze_calibration_matrix(IDEAL_TETRAHEDRON_MATRIX)
        self.assertAlmostEqual(report["condition"], 1.0, places=12)
        self.assertAlmostEqual(report["condition_row_scaled"], 1.0, places=12)
        self.assertAlmostEqual(report["condition_polarization"], 1.0, places=12)
        self.assertAlmostEqual(report["row_orthogonality"], 1.0, places=12)
        self.assertAlmostEqual(report["row_norm_ratio"], 1.0, places=12)
        self.assertEqual(report["rank"], 4)
        self.assertEqual(report["status"], "excellent")
        self.assertEqual(report["issues"], [])

    def test_paper_tetrahedron_is_well_conditioned_after_row_scaling(self):
        report = analyze_calibration_matrix(PAPER_TETRAHEDRON_MATRIX)
        # Raw condition number is dominated by the 2 vs 2*sqrt(3) row-norm ratio.
        self.assertAlmostEqual(report["condition"], np.sqrt(3.0), places=10)
        self.assertAlmostEqual(report["row_norm_ratio"], np.sqrt(3.0), places=10)
        # The geometry itself is ideal: orthogonal rows.
        self.assertAlmostEqual(report["condition_row_scaled"], 1.0, places=10)
        self.assertAlmostEqual(report["condition_polarization"], 1.0, places=10)
        self.assertAlmostEqual(report["row_orthogonality"], 1.0, places=10)
        self.assertEqual(report["status"], "excellent")

    def test_row_scale_does_not_change_scale_invariant_metrics(self):
        matrix = true_calibration_matrix(1550.0, seed=4)
        baseline = analyze_calibration_matrix(matrix)

        scaled = matrix.copy()
        scaled[0, :] *= 1.0e-3
        shifted = analyze_calibration_matrix(scaled)

        self.assertGreater(shifted["condition"], baseline["condition"] * 100.0)
        self.assertAlmostEqual(
            shifted["condition_row_scaled"], baseline["condition_row_scaled"], places=9
        )
        self.assertAlmostEqual(
            shifted["condition_polarization"], baseline["condition_polarization"], places=9
        )
        self.assertAlmostEqual(
            shifted["row_orthogonality"], baseline["row_orthogonality"], places=9
        )

    def test_weak_detector_channel_is_flagged(self):
        # One tap 50x weaker than the others: still full rank, but noisy.
        matrix = IDEAL_TETRAHEDRON_MATRIX @ np.diag([1.0, 1.0, 1.0, 0.02])
        report = analyze_calibration_matrix(matrix)
        self.assertEqual(report["rank"], 4)
        self.assertGreater(report["condition"], 40.0)
        self.assertGreater(report["condition_row_scaled"], 40.0)
        self.assertEqual(report["status"], "poor")
        self.assertLess(report["row_orthogonality"], 0.1)

    def test_mild_channel_imbalance_is_still_acceptable(self):
        matrix = IDEAL_TETRAHEDRON_MATRIX @ np.diag([1.0, 1.0, 1.0, 0.5])
        report = analyze_calibration_matrix(matrix)
        self.assertLess(report["condition_row_scaled"], 2.5)
        self.assertIn(report["status"], ("excellent", "ok"))

    def test_degenerate_matrix_is_flagged(self):
        matrix = IDEAL_TETRAHEDRON_MATRIX.copy()
        matrix[3, :] = matrix[2, :]  # collinear detector projection
        report = analyze_calibration_matrix(matrix)
        self.assertEqual(report["rank"], 3)
        self.assertFalse(np.isfinite(report["condition"]))
        self.assertEqual(report["row_orthogonality"], 0.0)
        self.assertEqual(report["status"], "poor")
        self.assertTrue(report["issues"])

    def test_nearly_collinear_rows_are_flagged_before_rank_loss(self):
        matrix = IDEAL_TETRAHEDRON_MATRIX.copy()
        matrix[3, :] = matrix[2, :] + 1.0e-3 * matrix[3, :]
        report = analyze_calibration_matrix(matrix)
        self.assertEqual(report["rank"], 4)
        self.assertGreater(report["condition_polarization"], 100.0)
        self.assertEqual(report["status"], "poor")
        self.assertLess(report["row_orthogonality"], 0.05)

    def test_classify_conditioning_tiers(self):
        self.assertEqual(classify_conditioning(1.0, 1.0), "excellent")
        self.assertEqual(classify_conditioning(1.4, 1.2), "excellent")
        self.assertEqual(classify_conditioning(2.0, 1.2), "ok")
        self.assertEqual(classify_conditioning(5.0, 1.2), "warn")
        self.assertEqual(classify_conditioning(50.0, 1.2), "poor")
        self.assertEqual(classify_conditioning(float("inf"), 1.0), "poor")
        self.assertEqual(classify_conditioning(1.0, 2.5, max_condition=2.0), "warn")
        self.assertEqual(classify_conditioning(1.0, 1.8, max_condition=2.0), "ok")
        # A gate stricter than the "excellent" tier must take precedence.
        self.assertEqual(classify_conditioning(1.1, 1.2, max_condition=1.03), "warn")
        self.assertEqual(classify_conditioning(1.0, 1.02, max_condition=1.03), "excellent")
        self.assertEqual(classify_conditioning(1.0, 1.4, max_condition=1.5), "excellent")
        self.assertEqual(classify_conditioning(1.0, 1.6, max_condition=1.5), "warn")

    def test_strict_gate_is_reflected_in_matrix_status(self):
        matrix = true_calibration_matrix(1550.0, seed=3)
        loose = analyze_calibration_matrix(matrix, max_condition=3.0)
        self.assertGreater(loose["condition_row_scaled"], 1.0)
        strict = analyze_calibration_matrix(matrix, max_condition=1.0001)
        self.assertEqual(loose["status"], "excellent")
        self.assertNotEqual(strict["status"], "excellent")
        self.assertEqual(strict["max_condition"], 1.0001)

    def test_rejects_malformed_matrices(self):
        with self.assertRaises(ValueError):
            analyze_calibration_matrix(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            analyze_calibration_matrix(np.full((4, 4), np.nan))
        # A zero matrix is finite but singular: infinite condition, not an error.
        self.assertEqual(spectral_condition(np.zeros((4, 4))), float("inf"))
        self.assertEqual(row_orthogonality(IDEAL_TETRAHEDRON_MATRIX), 1.0)

    def test_conditioning_is_reported_in_calibration_metrics(self):
        detector_csv, stokes_csv, truth = _synthetic_pair(1550.0, n_sops=24, seed=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = calibrate_csv_pair(
                detector_csv, stokes_csv, temp_dir, wavelength_nm=1550.0,
                validation_fraction=0.25, seed=5,
            )
            conditioning = result["metrics"]["conditioning"]
            self.assertIn("status", conditioning)
            self.assertIn("condition_row_scaled", conditioning)
            self.assertEqual(conditioning["rank"], 4)
            self.assertLess(conditioning["condition_row_scaled"], 1.5)
            # The fitted matrix must match the conditioning of the truth.
            truth_report = analyze_calibration_matrix(truth)
            self.assertAlmostEqual(
                conditioning["condition_row_scaled"],
                truth_report["condition_row_scaled"],
                places=2,
            )
            self.assertAlmostEqual(
                conditioning["row_orthogonality"], truth_report["row_orthogonality"], places=2
            )
            # The matrix written to disk (%.12g) is the one that was analyzed.
            saved = np.loadtxt(result["matrix_path"], delimiter=",")
            np.testing.assert_allclose(saved, result["C"], rtol=1e-11, atol=1e-6)


class ConditioningReportTests(unittest.TestCase):
    def test_report_writes_csv_json_and_flags_poor_matrices(self):
        try:
            from conditioning_report import main as report_main
        except ImportError:  # pragma: no cover
            self.skipTest("conditioning_report not importable")

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "mwcal_test")
            for wavelength, matrix in (
                (1540.0, true_calibration_matrix(1540.0, seed=1)),
                (1550.0, true_calibration_matrix(1550.0, seed=1)),
            ):
                lambda_dir = os.path.join(run_dir, "lambda_{:.3f}nm".format(wavelength))
                os.makedirs(lambda_dir)
                np.savetxt(
                    os.path.join(lambda_dir, "calibration_{:.3f}nm.csv".format(wavelength)),
                    matrix,
                    delimiter=",",
                )

            exit_code = report_main(
                ["--run-dir", run_dir, "--output-dir", run_dir, "--max-condition", "3.0",
                 "--quiet"]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "conditioning_report.csv")))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "conditioning_report.json")))

            with open(os.path.join(run_dir, "conditioning_report.csv"), newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["status"] for row in rows}, {"excellent"})
            self.assertEqual(
                sorted(float(row["wavelength_nm"]) for row in rows), [1540.0, 1550.0]
            )

            # Poison one matrix: two collinear rows must fail the gate.
            bad = true_calibration_matrix(1550.0, seed=1).copy()
            bad[3, :] = bad[2, :]
            np.savetxt(
                os.path.join(run_dir, "lambda_1550.000nm", "calibration_1550.000nm.csv"),
                bad,
                delimiter=",",
            )
            exit_code = report_main(
                ["--run-dir", run_dir, "--output-dir", run_dir, "--max-condition", "3.0",
                 "--quiet"]
            )
            self.assertEqual(exit_code, 1)

    def test_scan_mode_collects_matrices_from_several_runs(self):
        try:
            from conditioning_report import main as report_main
        except ImportError:  # pragma: no cover
            self.skipTest("conditioning_report not importable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run_name, offset in (("mwcal_a", 0.0), ("mwcal_b", 1.0)):
                for wavelength in (1530.0, 1560.0):
                    lambda_dir = root / run_name / "lambda_{:.3f}nm".format(wavelength)
                    lambda_dir.mkdir(parents=True)
                    np.savetxt(
                        lambda_dir / "calibration_{:.3f}nm.csv".format(wavelength),
                        true_calibration_matrix(wavelength + offset, seed=2),
                        delimiter=",",
                    )
            # A reference matrix without a wavelength in its name.
            np.savetxt(
                root / "calibration_matrix_reference.csv",
                IDEAL_TETRAHEDRON_MATRIX,
                delimiter=",",
            )

            out_dir = root / "report"
            exit_code = report_main(
                ["--scan", str(root), "--output-dir", str(out_dir), "--quiet",
                 "--max-condition", "3.0"]
            )
            self.assertEqual(exit_code, 0)
            with open(out_dir / "conditioning_report.csv", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            unknown = [row for row in rows if row["wavelength_nm"] == ""]
            self.assertEqual(len(unknown), 1)
            self.assertTrue(unknown[0]["matrix_file"].endswith("calibration_matrix_reference.csv"))


def _synthetic_pair(wavelength_nm, n_sops=24, seed=2):
    temp_dir = tempfile.mkdtemp()
    simulated = simulate_fused_acquisition(
        output_dir=temp_dir, wavelength_nm=wavelength_nm, n_sops=n_sops, seed=seed
    )
    return simulated["detector_csv"], simulated["stokes_csv"], simulated["truth_matrix"]


class SyntheticAcquisitionTests(unittest.TestCase):
    def test_simulated_data_recovers_the_truth_matrix(self):
        detector_csv, stokes_csv, truth = _synthetic_pair(1550.0, n_sops=30, seed=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = calibrate_csv_pair(
                detector_csv, stokes_csv, temp_dir, wavelength_nm=1550.0,
                validation_fraction=0.25, seed=11,
            )
        relative_error = np.linalg.norm(result["C"] - truth) / np.linalg.norm(truth)
        self.assertLess(relative_error, 0.02)
        self.assertLess(result["metrics"]["validation"]["angular_mean_deg"], 1.5)

    def test_truth_matrix_varies_smoothly_with_wavelength(self):
        first = true_calibration_matrix(1530.0, seed=8)
        second = true_calibration_matrix(1531.0, seed=8)
        far = true_calibration_matrix(1610.0, seed=8)
        near_change = np.linalg.norm(second - first) / np.linalg.norm(first)
        far_change = np.linalg.norm(far - first) / np.linalg.norm(first)
        self.assertLess(near_change, 0.02)
        self.assertGreater(far_change, 5.0 * near_change)

    def test_synthetic_truth_is_realistically_but_not_perfectly_conditioned(self):
        for wavelength in (1525.0, 1550.0, 1575.0, 1625.0):
            report = analyze_calibration_matrix(
                true_calibration_matrix(wavelength, seed=3)
            )
            self.assertEqual(report["rank"], 4)
            # Non-ideal geometry, but nothing that should trip the default gate.
            self.assertGreater(report["condition_row_scaled"], 1.0)
            self.assertLess(report["condition_row_scaled"], 3.0)
            self.assertIn(report["status"], ("excellent", "ok"))
            # Raw cond is inflated by the S0 row scale, as on the real device.
            self.assertGreater(report["row_norm_ratio"], 2.0)

    def test_synthetic_pair_matches_parent_csv_format(self):
        detector_csv, stokes_csv, truth = _synthetic_pair(1550.0, n_sops=7, seed=1)
        self.assertEqual(truth.shape, (4, 4))
        with open(detector_csv, newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["index", "ch1", "ch2", "ch3", "ch4"])
        self.assertEqual(len(rows), 8)
        with open(stokes_csv, newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["index", "S0_uW", "S1_uW", "S2_uW", "S3_uW", "DOP"])
        self.assertEqual(len(rows), 8)


if __name__ == "__main__":
    unittest.main()
