#!/usr/bin/env python3
"""Cross-wavelength validation: how usable is C(lambda_i) on data taken at lambda_j?

A per-wavelength calibration only answers "how well does C fit the data it was
fitted on". The interesting question for a wavelength sweep is how far a single
matrix can be stretched: apply C(lambda_i) to detector data acquired at
lambda_j and compare the reconstructed Stokes vector with the PM1000 reference.

The result is a square matrix of angular errors (in degrees on the Poincare
sphere). Its diagonal is the in-sample fit (optimistic); the off-diagonal entries
say how much a calibration degrades when the wavelength moves. That is the
number that decides the wavelength step of a sweep, and whether interpolation
between measured wavelengths is worth adding.

Usage
-----
    python cross_wavelength_validation.py                    # newest runs/mwcal_*
    python cross_wavelength_validation.py --run-dir runs/mwcal_20261112_151258
    python cross_wavelength_validation.py --run-dir <run> --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_run(run_dir: Path) -> List[Dict[str, Any]]:
    """Collect {wavelength, C, D, S} for every completed wavelength of a run."""
    entries: List[Dict[str, Any]] = []
    for matrix_path in sorted(run_dir.glob("*/calibration_*nm.csv")):
        if matrix_path.name.startswith("truth_"):
            continue
        lambda_dir = matrix_path.parent
        tag = os.path.basename(str(matrix_path)).replace("calibration_", "").replace("nm.csv", "")
        try:
            wavelength = float(tag)
        except ValueError:
            continue
        detector_files = sorted(lambda_dir.glob("detector_readings_*.csv"))
        stokes_files = sorted(lambda_dir.glob("stokes_*.csv"))
        if not detector_files or not stokes_files:
            print("Skipping {} (no raw pair)".format(lambda_dir), file=sys.stderr)
            continue
        detector = np.genfromtxt(detector_files[-1], delimiter=",", skip_header=1)
        stokes = np.genfromtxt(stokes_files[-1], delimiter=",", skip_header=1)
        if detector.shape[0] != stokes.shape[0]:
            print("Skipping {} (row mismatch)".format(lambda_dir), file=sys.stderr)
            continue
        entries.append(
            {
                "wavelength_nm": wavelength,
                "C": np.loadtxt(str(matrix_path), delimiter=","),
                "D": detector[:, 1:5].T,
                "S": stokes[:, 1:5].T,
            }
        )
    entries.sort(key=lambda item: item["wavelength_nm"])
    return entries


def _angular_error_deg(reference: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """Angle between the polarization parts of two (4, M) Stokes arrays."""
    a = reference[1:4]
    b = reconstructed[1:4]
    na = np.linalg.norm(a, axis=0)
    nb = np.linalg.norm(b, axis=0)
    valid = (na > 0) & (nb > 0)
    cosine = np.full(na.shape, np.nan)
    cosine[valid] = np.sum(a[:, valid] * b[:, valid], axis=0) / (na[valid] * nb[valid])
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _normalized_rmse(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    s0r = reference[0]
    s0p = reconstructed[0]
    valid = (np.abs(s0r) > 0) & (np.abs(s0p) > 0)
    if not np.any(valid):
        return float("nan")
    r = reference[:, valid] / s0r[valid]
    p = reconstructed[:, valid] / s0p[valid]
    return float(np.sqrt(np.mean((p - r) ** 2)))


def cross_validate(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    wavelengths = [entry["wavelength_nm"] for entry in entries]
    n = len(entries)
    mean_angle = np.full((n, n), np.nan)
    max_angle = np.full((n, n), np.nan)
    norm_rmse = np.full((n, n), np.nan)
    for i, calibration in enumerate(entries):
        predicted = calibration["C"] @ entries[i]["D"]
        for j, measurement in enumerate(entries):
            reconstructed = calibration["C"] @ measurement["D"]
            angles = _angular_error_deg(measurement["S"], reconstructed)
            mean_angle[i, j] = float(np.nanmean(angles))
            max_angle[i, j] = float(np.nanmax(angles))
            norm_rmse[i, j] = _normalized_rmse(measurement["S"], reconstructed)
    return {
        "wavelengths_nm": wavelengths,
        "mean_angle_deg": mean_angle,
        "max_angle_deg": max_angle,
        "normalized_stokes_rmse": norm_rmse,
    }


def _print_matrix(title: str, wavelengths: List[float], matrix: np.ndarray, fmt: str) -> None:
    print("\n" + title)
    print("rows = matrix C(lambda_i), columns = data taken at lambda_j")
    header = "{:>10s}".format("C \\ data") + "".join(
        "{:>9.0f}".format(value) for value in wavelengths
    )
    print(header)
    for i, value in enumerate(wavelengths):
        cells = "".join(fmt.format(matrix[i, j]) for j in range(len(wavelengths)))
        print("{:>10.0f}{}".format(value, cells))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", default=None, help="sweep directory (default: newest)")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--csv", default=None, help="write the angle matrix to this CSV")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        candidates = sorted(Path(args.runs_root).glob("mwcal_*"))
        if not candidates:
            print("No run directory found under {}".format(args.runs_root), file=sys.stderr)
            return 2
        run_dir = candidates[-1]
    print("Run: {}".format(run_dir))

    entries = _load_run(run_dir)
    if len(entries) < 2:
        print("Need at least two calibrated wavelengths; found {}.".format(len(entries)),
              file=sys.stderr)
        return 2

    result = cross_validate(entries)
    wavelengths = result["wavelengths_nm"]

    _print_matrix("Mean angular error [deg]", wavelengths, result["mean_angle_deg"], "{:>9.2f}")
    _print_matrix("Worst-case angular error [deg]", wavelengths, result["max_angle_deg"], "{:>9.2f}")
    _print_matrix("Normalized Stokes RMSE", wavelengths, result["normalized_stokes_rmse"], "{:>9.4f}")

    # Summary: how much does using the *nearest* calibration cost?
    off_diagonal = []
    matrix = result["mean_angle_deg"]
    for i in range(len(wavelengths)):
        for j in range(len(wavelengths)):
            if i != j:
                off_diagonal.append((abs(wavelengths[i] - wavelengths[j]), matrix[i, j]))
    if off_diagonal:
        diagonal = float(np.nanmean(np.diag(matrix)))
        print("\nIn-sample (diagonal) mean error : {:.2f} deg".format(diagonal))
        for distance in sorted({dist for dist, _ in off_diagonal}):
            values = [value for dist, value in off_diagonal if dist == distance]
            print("  {:>5.0f} nm away : {:.2f} deg mean ({:.2f} deg worst of {} pairs)".format(
                distance, float(np.mean(values)), float(np.max(values)), len(values)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
