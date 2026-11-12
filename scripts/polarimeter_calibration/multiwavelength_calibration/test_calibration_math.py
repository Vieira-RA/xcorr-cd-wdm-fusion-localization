#!/usr/bin/env python3
"""Hardware-free tests for the multiwavelength calibration math and CSV reader."""
import csv
import os
import tempfile
import unittest

import numpy as np

from calibration import calibrate_arrays, calibrate_csv_pair, load_fused_csv


class CalibrationMathTests(unittest.TestCase):
    def test_recovers_linear_matrix_and_validates_held_out_rows(self):
        rng = np.random.default_rng(5)
        detector_matrix = rng.normal(size=(4, 40))
        true_matrix = np.array([
            [2.0, 0.3, 0.1, 0.2],
            [0.1, 1.5, 0.2, 0.0],
            [0.2, 0.1, 1.7, 0.1],
            [0.0, 0.2, 0.1, 1.3],
        ])
        reference = true_matrix @ detector_matrix

        result = calibrate_arrays(
            detector_matrix, reference, validation_fraction=0.25, seed=2
        )

        np.testing.assert_allclose(result["C"], true_matrix, rtol=1e-11, atol=1e-11)
        self.assertEqual(result["metrics"]["detector_training_rank"], 4)
        self.assertLess(result["metrics"]["validation"]["relative_residual"], 1e-10)
        self.assertEqual(result["validation_indices"].size, 10)

    def test_csv_reader_checks_alignment_and_loads_expected_shapes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detector_path = os.path.join(temp_dir, "detector.csv")
            stokes_path = os.path.join(temp_dir, "stokes.csv")
            with open(detector_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "ch1", "ch2", "ch3", "ch4"])
                for index in range(1, 9):
                    writer.writerow([index, index, index + 1, index + 2, index + 3])
            with open(stokes_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "S0_uW", "S1_uW", "S2_uW", "S3_uW", "DOP"])
                for index in range(1, 9):
                    writer.writerow([index, 10, 1, 2, 3, 0.374])

            D, S, indices = load_fused_csv(detector_path, stokes_path)
            self.assertEqual(D.shape, (4, 8))
            self.assertEqual(S.shape, (4, 8))
            np.testing.assert_array_equal(indices, np.arange(1, 9))

    def test_calibrate_csv_pair_writes_compatible_matrix_and_metrics(self):
        rng = np.random.default_rng(9)
        detector_matrix = rng.normal(size=(4, 32))
        true_matrix = np.array([
            [2.0, 0.1, 0.2, 0.0],
            [0.1, 1.4, 0.0, 0.1],
            [0.0, 0.2, 1.6, 0.1],
            [0.1, 0.0, 0.2, 1.8],
        ])
        reference = true_matrix @ detector_matrix
        with tempfile.TemporaryDirectory() as temp_dir:
            detector_path = os.path.join(temp_dir, "detector.csv")
            stokes_path = os.path.join(temp_dir, "stokes.csv")
            with open(detector_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "ch1", "ch2", "ch3", "ch4"])
                for index, row in enumerate(detector_matrix.T, start=1):
                    writer.writerow([index, *row.tolist()])
            with open(stokes_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "S0_uW", "S1_uW", "S2_uW", "S3_uW", "DOP"])
                for index, row in enumerate(reference.T, start=1):
                    writer.writerow([index, *row.tolist(), 1.0])

            result = calibrate_csv_pair(
                detector_path, stokes_path, temp_dir, wavelength_nm=1550.0,
                validation_fraction=0.25, seed=3,
            )
            saved_matrix = np.loadtxt(result["matrix_path"], delimiter=",")
            np.testing.assert_allclose(saved_matrix, true_matrix, rtol=1e-11, atol=1e-11)
            self.assertTrue(os.path.isfile(result["metrics_path"]))
            self.assertTrue(os.path.isfile(result["validation_path"]))

    def test_rank_deficient_training_data_is_rejected(self):
        # Linearly dependent channels, but every row is distinct (so this is a
        # genuine rank problem, not a stale capture).
        sweep = np.linspace(0.0, 1.0, 12)
        detector_matrix = np.vstack([sweep, 2 * sweep, 3 * sweep, 0.5 * sweep]) + 1.0
        reference = np.vstack([np.ones(12), sweep, sweep ** 2, sweep ** 3])
        with self.assertRaisesRegex(ValueError, "rank"):
            calibrate_arrays(detector_matrix, reference, validation_fraction=0.25)


if __name__ == "__main__":
    unittest.main()


class StaleCaptureTests(unittest.TestCase):
    def test_repeated_detector_vectors_are_rejected(self):
        """A stale scope capture repeats the same reading: rank 4 but meaningless."""
        import numpy as np
        from calibration import calibrate_arrays

        base = np.array([[0.02], [0.018], [0.019], [0.006]])
        D = np.repeat(base, 40, axis=1)          # 40 identical columns
        S = np.random.default_rng(0).normal(size=(4, 40))
        with self.assertRaises(ValueError) as context:
            calibrate_arrays(D, S, validation_fraction=0.25)
        self.assertIn("stale acquisition", str(context.exception))

    def test_partially_stale_data_is_also_rejected(self):
        import numpy as np
        from calibration import calibrate_arrays

        rng = np.random.default_rng(1)
        D = rng.normal(size=(4, 30)) * 1e-2
        D[:, 10:] = D[:, :1]                     # two thirds frozen
        S = rng.normal(size=(4, 30))
        with self.assertRaises(ValueError):
            calibrate_arrays(D, S, validation_fraction=0.25)

    def test_healthy_data_still_passes(self):
        import numpy as np
        from calibration import calibrate_arrays

        rng = np.random.default_rng(2)
        D = rng.normal(size=(4, 40))
        true = np.array([[2.0, .3, .1, .2], [.1, 1.5, .2, .0], [.2, .1, 1.7, .1], [.0, .2, .1, 1.3]])
        result = calibrate_arrays(D, true @ D, validation_fraction=0.25)
        self.assertEqual(result["metrics"]["detector_training_rank"], 4)
