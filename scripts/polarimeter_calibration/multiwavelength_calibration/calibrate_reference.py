#!/usr/bin/env python3
"""Fit and validate a single wavelength's referenced calibration matrix.

Example:
    python calibrate_reference.py \
        --detector ../runs/lambda_1550/detector_readings_....csv \
        --stokes ../runs/lambda_1550/stokes_....csv \
        --wavelength-nm 1550 --output-dir ../runs/lambda_1550
"""
from __future__ import annotations

import argparse
import os
import sys

from calibration import calibrate_csv_pair
from conditioning import format_conditioning


_CONDITIONING_EXIT = {"excellent": 0, "ok": 0, "warn": 0, "poor": 1}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, help="detector_readings_*.csv from the scope")
    parser.add_argument("--stokes", required=True, help="stokes_*.csv from the PM1000")
    parser.add_argument("--wavelength-nm", type=float, required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--max-condition",
        type=float,
        default=3.0,
        help="gate on the worse scale-invariant condition number (default 3.0)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = calibrate_csv_pair(
            detector_csv=args.detector,
            stokes_csv=args.stokes,
            output_dir=args.output_dir,
            wavelength_nm=args.wavelength_nm,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            max_condition=args.max_condition,
        )
    except Exception as exc:
        print("Calibration failed: {}".format(exc), file=sys.stderr)
        return 2

    print("\nCalibration matrix C(lambda) [Stokes uW <- detector volts]:")
    print(result["C"])
    print("\nHeld-out validation metrics:")
    for name, value in result["metrics"]["validation"].items():
        print("  {:28s} {}".format(name, value))
    print("\nTraining detector rank: {}".format(result["metrics"]["detector_training_rank"]))
    print("Training detector condition: {}".format(
        result["metrics"]["detector_training_condition"]
    ))
    print("Calibration matrix condition: {}".format(
        result["metrics"]["calibration_matrix_condition"]
    ))

    conditioning = result["metrics"]["conditioning"]
    print("\nConditioning (ideal tetrahedron = 1.0):")
    print(format_conditioning(conditioning))
    for issue in conditioning["issues"]:
        print("  - {}".format(issue))

    print("\nWrote:")
    print("  {}".format(os.path.abspath(result["matrix_path"])))
    print("  {}".format(os.path.abspath(result["metrics_path"])))
    print("  {}".format(os.path.abspath(result["validation_path"])))
    return _CONDITIONING_EXIT.get(conditioning["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
