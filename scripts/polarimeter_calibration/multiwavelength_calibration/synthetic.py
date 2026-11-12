#!/usr/bin/env python3
"""Synthetic fused-acquisition data for offline rehearsal of the lambda sweep.

This module exists so the whole multiwavelength *pipeline* (sweep -> fit ->
conditioning -> summary -> npz -> report) can be exercised without touching the
laser, the scope, the PM1000 or the EPS1000.  It writes the same two index-
aligned CSVs that ``fused_acquisition_controlled.py`` produces, so every
downstream step runs unmodified.

The generator is deliberately simple and *smooth in wavelength*:

    C_true(lambda) = scale * diag(row_gain(lambda)) @ (H4 @ R(lambda))

* ``H4`` is the balanced tetrahedral (Hadamard) matrix, so the rows start out
  mutually orthogonal and equally long.
* ``R(lambda)`` is a smooth random orthogonal rotation of the detector basis,
  which preserves row norms and orthogonality (a Stokes-frame rotation).
* ``row_gain`` adds a slow per-row gain ripple, and ``s0_scale`` makes the S0
  row weaker than S1..S3 to mimic the real microwatt-per-volt unit mismatch.

The result therefore has a realistic, slightly wavelength-dependent conditioning
(row-scaled condition number close to 1, raw condition number inflated by the
S0/S1..S3 row-norm ratio), which is exactly the regime the real instrument
shows.  Detector voltages are then produced by inverting the matrix, so the
fitted ``C`` can be compared against the known truth.
"""
from __future__ import annotations

import csv
import datetime
import os
from typing import Any, Dict, Optional

import numpy as np

try:  # works as a module inside the package and as a flat script import
    from .conditioning import IDEAL_TETRAHEDRON_MATRIX
except ImportError:  # pragma: no cover - script-style execution
    from conditioning import IDEAL_TETRAHEDRON_MATRIX


def _smooth_rotation(wavelength_nm: float, seed: int, amplitude: float) -> np.ndarray:
    """Smooth, deterministic-in-wavelength orthogonal 4x4 matrix."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(3, 4, 4))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    periods = np.array([420.0, 280.0, 190.0], dtype=float)

    mix = np.eye(4)
    for index in range(3):
        argument = 2.0 * np.pi * float(wavelength_nm) / periods[index] + phases[index]
        mix = mix + amplitude * np.sin(argument) * directions[index]

    left, _, right = np.linalg.svd(mix)
    return left @ right


def true_calibration_matrix(
    wavelength_nm: float,
    seed: int = 0,
    scale: float = 3000.0,
    s0_scale: float = 0.25,
    row_spread: float = 0.10,
    rotation_amplitude: float = 0.10,
    skew: float = 0.25,
    skew_period_nm: float = 130.0,
) -> np.ndarray:
    """Ground-truth ``C(lambda)`` (mapping detector volts to Stokes microwatts).

    Parameters mirror the docstring above; defaults give detector voltages of a
    few millivolts for ~100 uW of optical power, i.e. the same order as the real
    acquisition at 20 mV/div.  ``skew`` breaks the exact row orthogonality a
    little, so the generated matrices have a realistic small conditioning spread
    (row-scaled condition number a few percent above 1) instead of the perfect
    tetrahedral value.
    """
    rng = np.random.default_rng(seed + 1000)
    row_phases = rng.uniform(0.0, 2.0 * np.pi, size=4)
    row_periods = np.array([260.0, 210.0, 150.0, 120.0], dtype=float)
    skew_direction = rng.normal(size=(4, 4))
    skew_direction /= np.linalg.norm(skew_direction)
    skew_phase = rng.uniform(0.0, 2.0 * np.pi)

    row_gain = 1.0 + row_spread * np.sin(
        2.0 * np.pi * float(wavelength_nm) / row_periods + row_phases
    )
    row_gain[0] *= float(s0_scale)

    rotation = _smooth_rotation(wavelength_nm, seed, rotation_amplitude)
    basis = IDEAL_TETRAHEDRON_MATRIX @ rotation
    if skew:
        # Left multiplication by a near-identity matrix is a Stokes-frame
        # rotation, the physically expected deviation for a referenced
        # calibration; it breaks exact row orthogonality by a few percent.
        modulation = 0.55 + 0.45 * np.sin(
            2.0 * np.pi * float(wavelength_nm) / float(skew_period_nm) + skew_phase
        )
        basis = (np.eye(4) + float(skew) * modulation * skew_direction) @ basis
    return float(scale) * np.diag(row_gain) @ basis


def _random_polarized_stokes(rng: np.random.Generator, power_uw: float, power_spread: float):
    """Fully polarized SOP with random orientation on the Poincare sphere."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    s0 = float(power_uw) * (1.0 + float(power_spread) * (rng.random() - 0.5))
    return np.array([s0, *(direction * s0)], dtype=float)


def simulate_fused_acquisition(
    output_dir: str,
    wavelength_nm: float,
    n_sops: int,
    seed: int = 0,
    detector_noise_rel: float = 2.0e-3,
    stokes_noise_rel: float = 2.0e-3,
    power_uw: float = 100.0,
    power_spread: float = 0.06,
    timestamp: Optional[str] = None,
    matrix_scale: float = 3000.0,
    s0_scale: float = 0.25,
) -> Dict[str, Any]:
    """Write a synthetic detector/Stokes CSV pair and return their paths.

    Returns a dict with ``detector_csv``, ``stokes_csv``, ``truth_matrix`` and
    ``truth_csv`` (the ground-truth matrix is also saved for comparison).
    """
    if n_sops < 5:
        raise ValueError("n_sops must be at least 5 for a held-out split.")

    rng = np.random.default_rng(int(seed) + 7919)
    matrix = true_calibration_matrix(
        wavelength_nm, seed=int(seed), scale=matrix_scale, s0_scale=s0_scale
    )
    inverse_matrix = np.linalg.pinv(matrix)

    stokes = np.column_stack(
        [
            _random_polarized_stokes(rng, power_uw, power_spread)
            for _ in range(int(n_sops))
        ]
    )
    detector_clean = inverse_matrix @ stokes

    # Absolute detector noise scaled to the typical signal (scope noise floor),
    # plus relative PM1000 noise on the reference Stokes vector.
    detector_sigma = float(detector_noise_rel) * float(np.mean(np.abs(detector_clean)))
    detector = detector_clean + detector_sigma * rng.normal(size=detector_clean.shape)
    stokes_measured = stokes + float(stokes_noise_rel) * power_uw * rng.normal(
        size=stokes.shape
    )

    stamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    detector_csv = os.path.join(output_dir, "detector_readings_{}.csv".format(stamp))
    stokes_csv = os.path.join(output_dir, "stokes_{}.csv".format(stamp))
    truth_csv = os.path.join(
        output_dir, "truth_calibration_{:.3f}nm.csv".format(float(wavelength_nm))
    )

    with open(detector_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "ch1", "ch2", "ch3", "ch4"])
        for column in range(detector.shape[1]):
            writer.writerow([column + 1, *detector[:, column].tolist()])

    with open(stokes_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "S0_uW", "S1_uW", "S2_uW", "S3_uW", "DOP"])
        for column in range(stokes_measured.shape[1]):
            values = stokes_measured[:, column]
            s0 = values[0]
            dop = float(np.linalg.norm(values[1:4]) / s0) if s0 else 0.0
            writer.writerow([column + 1, *values.tolist(), dop])

    # Headerless numeric 4x4, same convention as calibration_<wl>nm.csv.
    np.savetxt(truth_csv, matrix, delimiter=",", fmt="%.12g")

    return {
        "detector_csv": detector_csv,
        "stokes_csv": stokes_csv,
        "truth_csv": truth_csv,
        "truth_matrix": matrix,
        "truth_condition_row_scaled_hint": "see conditioning.analyze_calibration_matrix",
        "detector_noise_volts": detector_sigma,
        "wavelength_nm": float(wavelength_nm),
    }
