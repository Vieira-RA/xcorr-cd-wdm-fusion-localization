#!/usr/bin/env python3
"""Acquire and calibrate the tilted-FBG polarimeter across wavelengths.

For each wavelength this script:
  1. Sets the tunable laser (manual control is the safe default).
  2. Runs the existing referenced acquisition workflow, which collects stable
     EPS1000 SOPs, four oscilloscope detector voltages, and PM1000 Stokes data.
  3. Fits C(lambda), reports held-out validation, and checkpoints a summary.

No laser API is assumed. Configure the HTTP adapter only after confirming the
specific laser's endpoints and request/response format.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from calibration import calibrate_csv_pair
from conditioning import format_conditioning
from laser_client import LaserController
from synthetic import simulate_fused_acquisition

SUMMARY_FIELDS = [
    "status",
    "requested_wavelength_nm",
    "readback_wavelength_nm",
    "readback_source",
    "laser_output",
    "n_measurements",
    "n_training",
    "n_validation",
    "detector_training_rank",
    "detector_training_condition",
    "calibration_matrix_condition",
    "calibration_matrix_condition_row_scaled",
    "polarization_condition",
    "row_orthogonality",
    "conditioning_status",
    "simulated",
    "validation_relative_residual",
    "validation_normalized_stokes_rmse",
    "validation_dop_mae",
    "validation_dop_error_std",
    "validation_angular_mean_deg",
    "validation_angular_std_deg",
    "validation_angular_max_deg",
    "matrix_file",
    "error",
]


def _wavelength_tag(wavelength_nm: float) -> str:
    return "{:.3f}".format(float(wavelength_nm)).replace(".", "p")


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Top-level config must be a JSON object.")
    wavelengths = config.get("wavelengths_nm")
    if not isinstance(wavelengths, list) or not wavelengths:
        raise ValueError("Set a non-empty wavelengths_nm list in the JSON config.")
    wavelengths = [float(wavelength) for wavelength in wavelengths]
    if any(not math.isfinite(wavelength) or wavelength <= 0 for wavelength in wavelengths):
        raise ValueError("All wavelengths_nm values must be positive finite numbers.")
    if len(set(wavelengths)) != len(wavelengths):
        raise ValueError("wavelengths_nm contains duplicate values.")
    config["wavelengths_nm"] = wavelengths
    n_sops = int(config.get("n_sops", 30))
    if n_sops < 5:
        raise ValueError("n_sops must be at least 5 (4 fit states plus held-out validation).")
    config["n_sops"] = n_sops

    max_condition = float(config.get("max_condition", 3.0))
    if not math.isfinite(max_condition) or max_condition <= 1.0:
        raise ValueError("max_condition must be a finite number greater than 1.")
    config["max_condition"] = max_condition

    config["simulate"] = bool(config.get("simulate", False))
    config["fail_on_poor_conditioning"] = bool(
        config.get("fail_on_poor_conditioning", False)
    )
    config["synthetic"] = _synthetic_settings(config.get("synthetic"))

    laser = config.get("laser", {})
    if not isinstance(laser, dict):
        raise ValueError("The 'laser' config block must be a JSON object.")
    mode = str(laser.get("mode", "manual")).lower()
    if mode not in ("manual", "http", "cobrite", "scpi"):
        raise ValueError(
            "laser.mode must be 'manual', 'cobrite' or 'http' (got {!r}).".format(mode)
        )
    laser["mode"] = "cobrite" if mode == "scpi" else mode
    cobrite_block = laser.get("cobrite")
    if cobrite_block is not None and not isinstance(cobrite_block, dict):
        raise ValueError("laser.cobrite must be a JSON object.")
    if isinstance(cobrite_block, dict) and cobrite_block.get("power_dbm") is not None:
        requested_power = float(cobrite_block["power_dbm"])
        if requested_power > 10.0:
            raise ValueError(
                "laser.cobrite.power_dbm = {:.2f} dBm exceeds the 10 dBm lab limit "
                "(the laser must never run above 10 dBm).".format(requested_power)
            )
    config["laser"] = laser

    scope_block = config.get("scope")
    if scope_block is not None:
        if not isinstance(scope_block, dict):
            raise ValueError("The 'scope' config block must be a JSON object.")
        allowed = {"ip", "transport", "resource", "timeout_s"}
        unknown = [key for key in scope_block if key not in allowed]
        if unknown:
            raise ValueError("Unknown scope setting(s): {}".format(", ".join(sorted(unknown))))
        if scope_block.get("transport") is not None and str(scope_block["transport"]).lower() not in (
            "auto", "vxi11", "pyvisa"
        ):
            raise ValueError("scope.transport must be 'auto', 'vxi11' or 'pyvisa'.")
        config["scope"] = dict(scope_block)

    pm_block = config.get("pm1000")
    if pm_block is not None:
        if not isinstance(pm_block, dict):
            raise ValueError("The 'pm1000' config block must be a JSON object.")
        allowed = {"wavelength_register", "wavelength_encoding"}
        unknown = [key for key in pm_block if key not in allowed]
        if unknown:
            raise ValueError("Unknown pm1000 setting(s): {}".format(", ".join(sorted(unknown))))
        register = pm_block.get("wavelength_register")
        if register is not None:
            register = int(register)
            if register < 0 or register > 511:
                raise ValueError("pm1000.wavelength_register must be 0..511 (offset from 512).")
            if register in (list(range(0, 18)) + [45, 92, 226]):
                raise ValueError(
                    "pm1000.wavelength_register {} collides with ATE/Stokes/trigger registers."
                    .format(register)
                )
            pm_block["wavelength_register"] = register
        encoding = str(pm_block.get("wavelength_encoding", "thz0.01"))
        if encoding not in ("thz0.01", "thz0.001", "ghz", "nm", "nm0.1", "nm0.01"):
            raise ValueError("Unsupported pm1000.wavelength_encoding {!r}.".format(encoding))
        pm_block["wavelength_encoding"] = encoding
        config["pm1000"] = pm_block

    scales = config.get("ch_scales")
    if scales is not None:
        if isinstance(scales, (int, float)):
            scales = [float(scales)] * 4
        if not isinstance(scales, list) or len(scales) != 4:
            raise ValueError("ch_scales must be a number or a list of four V/div values.")
        values = [float(value) for value in scales]
        if any(not math.isfinite(value) or value <= 0.0 or value > 10.0 for value in values):
            raise ValueError("Each ch_scales value must be in (0, 10] V/div.")
        config["ch_scales"] = values
    return config


def _apply_laser_overrides(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Let the command line override the laser endpoint without editing JSON."""
    laser = config.setdefault("laser", {})
    if getattr(args, "laser_mode", None):
        mode = str(args.laser_mode).lower()
        laser["mode"] = "cobrite" if mode == "scpi" else mode
    if getattr(args, "laser_host", None) or getattr(args, "laser_address", None):
        block = laser.setdefault("cobrite", {})
        if not isinstance(block, dict):
            raise ValueError("laser.cobrite must be a JSON object.")
        if args.laser_host:
            block["host"] = args.laser_host
        if args.laser_address:
            block["address"] = args.laser_address
    return config


_SYNTHETIC_KEYS = (
    "detector_noise_rel",
    "stokes_noise_rel",
    "power_uw",
    "power_spread",
    "matrix_scale",
    "s0_scale",
)


def _synthetic_settings(settings: Any) -> Dict[str, float]:
    """Validate the optional ``synthetic`` block used by ``--simulate``."""
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise ValueError("The 'synthetic' config block must be a JSON object.")
    unknown = [key for key in settings if key not in _SYNTHETIC_KEYS]
    if unknown:
        raise ValueError(
            "Unknown synthetic setting(s): {}. Allowed: {}.".format(
                ", ".join(sorted(unknown)), ", ".join(_SYNTHETIC_KEYS)
            )
        )
    return {key: float(value) for key, value in settings.items()}


def _write_summary(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _find_acquisition_pair(output_dir: Path):
    detector_files = sorted(output_dir.glob("detector_readings_*.csv"))
    stokes_files = sorted(output_dir.glob("stokes_*.csv"))
    if len(detector_files) != 1 or len(stokes_files) != 1:
        raise RuntimeError(
            "Expected exactly one detector/stokes CSV pair in {}; found {} and {}."
            .format(output_dir, len(detector_files), len(stokes_files))
        )
    detector = detector_files[0]
    stokes = stokes_files[0]
    detector_stamp = detector.name[len("detector_readings_"):-len(".csv")]
    stokes_stamp = stokes.name[len("stokes_"):-len(".csv")]
    if detector_stamp != stokes_stamp:
        raise RuntimeError("Detector and PM1000 CSV timestamps do not match.")
    return detector, stokes


def _metric_record(wavelength_nm: float, laser_meta: Dict[str, Any], result: Dict[str, Any],
                   simulated: bool = False) -> Dict[str, Any]:
    metrics = result["metrics"]
    validation = metrics["validation"]
    conditioning = metrics["conditioning"]
    return {
        "status": "ok",
        "requested_wavelength_nm": wavelength_nm,
        "readback_wavelength_nm": laser_meta.get("readback_nm"),
        "readback_source": laser_meta.get("readback_source"),
        "laser_output": laser_meta.get("output"),
        "n_measurements": metrics["n_measurements"],
        "n_training": metrics["n_training"],
        "n_validation": metrics["n_validation"],
        "detector_training_rank": metrics["detector_training_rank"],
        "detector_training_condition": metrics["detector_training_condition"],
        "calibration_matrix_condition": metrics["calibration_matrix_condition"],
        "calibration_matrix_condition_row_scaled": conditioning["condition_row_scaled"],
        "polarization_condition": conditioning["condition_polarization"],
        "row_orthogonality": conditioning["row_orthogonality"],
        "conditioning_status": conditioning["status"],
        "simulated": bool(simulated),
        "validation_relative_residual": validation["relative_residual"],
        "validation_normalized_stokes_rmse": validation["normalized_stokes_rmse"],
        "validation_dop_mae": validation["dop_mae"],
        "validation_dop_error_std": validation["dop_error_std"],
        "validation_angular_mean_deg": validation["angular_mean_deg"],
        "validation_angular_std_deg": validation["angular_std_deg"],
        "validation_angular_max_deg": validation["angular_max_deg"],
        "matrix_file": str(result["matrix_path"]),
        "error": "",
    }


def _save_combined(output_dir: Path, records: List[Dict[str, Any]]) -> Optional[Path]:
    successful = [
        record for record in records if record.get("status") in ("ok", "warn")
    ]
    if not successful:
        return None
    wavelengths = np.asarray(
        [float(record["requested_wavelength_nm"]) for record in successful], dtype=float
    )
    matrices = []
    for record in successful:
        matrix_path = Path(str(record["matrix_file"]))
        matrix = np.loadtxt(str(matrix_path), delimiter=",")
        if matrix.shape != (4, 4):
            raise ValueError("Expected 4x4 calibration matrix at {}".format(matrix_path))
        matrices.append(matrix)
    output_path = output_dir / "all_calibration_matrices.npz"
    np.savez_compressed(
        str(output_path),
        wavelengths_nm=wavelengths,
        calibration_matrices=np.stack(matrices, axis=0),
        equation=np.asarray("S_uW = C @ D_volts"),
    )
    return output_path


def _scope_environment(config: Dict[str, Any]) -> Dict[str, str]:
    """Environment exported to the acquisition subprocess so it can find the scope."""
    environment = dict(os.environ)
    block = config.get("scope") or {}
    if block.get("ip"):
        environment["PM_SCOPE_IP"] = str(block["ip"])
    if block.get("transport"):
        environment["PM_SCOPE_TRANSPORT"] = str(block["transport"]).lower()
    if block.get("resource"):
        environment["PM_SCOPE_RESOURCE"] = str(block["resource"])
    pm_block = config.get("pm1000") or {}
    if pm_block.get("wavelength_register") is not None:
        environment["PM1000_WL_REGISTER"] = str(pm_block["wavelength_register"])
    if pm_block.get("wavelength_encoding"):
        environment["PM1000_WL_ENCODING"] = str(pm_block["wavelength_encoding"])
    return environment


def _describe_laser_target(config: Dict[str, Any]) -> str:
    laser = config.get("laser", {})
    mode = str(laser.get("mode", "manual")).lower()
    if mode != "cobrite":
        return mode
    block = laser.get("cobrite") or {}
    if str(block.get("transport", "tcp")).lower() == "serial":
        return "cobrite SCPI over serial {} (port {})".format(
            block.get("com_port", "?"), block.get("address", "1,1,1")
        )
    host = block.get("host", laser.get("host", "cobrite.local"))
    port = block.get("port", laser.get("port", 2000))
    address = block.get("address", laser.get("address", "1,1,1"))
    return "cobrite SCPI {}:{} port {}".format(host, port, address)


def _print_plan(config: Dict[str, Any], config_path: Path, acquisition_script: Path) -> None:
    simulate = bool(config.get("simulate", False))
    print("Multiwavelength calibration plan (dry run; no hardware contacted)")
    print("  Config              : {}".format(config_path.resolve()))
    print("  Data source         : {}".format(
        "SYNTHETIC (offline rehearsal; no instruments)" if simulate
        else "hardware via {} ({})".format(
            acquisition_script.resolve().name,
            config.get("acquisition_python") or "this interpreter",
        )
    ))
    print("  Laser control       : {}".format(
        "skipped (simulate)" if simulate else _describe_laser_target(config)
    ))
    print("  Scope endpoint      : {}".format(
        "default (198.192.1.1 over VXI-11)"
        if not (config.get("scope") or {}).get("ip") and not (config.get("scope") or {}).get("resource")
        else "{}{}".format(
            (config.get("scope") or {}).get("ip") or "",
            " via " + str((config.get("scope") or {}).get("transport")) if (config.get("scope") or {}).get("transport") else "",
        )
    ))
    pm_block = config.get("pm1000") or {}
    print("  PM1000 wavelength   : {}".format(
        "not set remotely (register not configured)"
        if pm_block.get("wavelength_register") is None
        else "register {} ({}), follows every wavelength".format(
            pm_block["wavelength_register"], pm_block.get("wavelength_encoding", "thz0.01"))
    ))
    print("  Scope scales        : {}".format(
        "default (20 mV/div per channel in the acquisition script)"
        if not config.get("ch_scales")
        else ", ".join("CH{:.0f}={:.1f} mV/div".format(i + 1, value * 1e3)
                     for i, value in enumerate(config["ch_scales"]))
    ))
    print("  SOPs per wavelength : {}".format(config["n_sops"]))
    print("  Wavelengths (nm)    : {}".format(", ".join(
        "{:.3f}".format(value) for value in config["wavelengths_nm"]
    )))
    print("  Validation fraction : {:.3f}".format(
        float(config.get("validation_fraction", 0.25))
    ))
    print("  Conditioning gate   : {:.3f} (row-scaled and polarization cond)".format(
        float(config.get("max_condition", 3.0))
    ))
    if not simulate:
        print("\nFor each wavelength, the existing hardware script will acquire stable EPS1000 SOPs, ")
        print("oscilloscope detector means, and PM1000 reference Stokes values.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.example.json")),
        help="JSON configuration (default: config.example.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without contacting instruments or the laser.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Rehearse the whole sweep on synthetic data instead of driving the instruments.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override config output_root (useful for simulated rehearsals).",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="skip the interactive operator-checklist prompt (non-interactive runs)",
    )
    parser.add_argument(
        "--laser-mode",
        choices=("manual", "cobrite", "scpi", "http"),
        default=None,
        help="Override laser.mode from the config.",
    )
    parser.add_argument(
        "--laser-host", default=None, help="Override the CoBrite host/IP (cobrite mode)."
    )
    parser.add_argument(
        "--laser-address",
        default=None,
        help="Override the CoBrite port address, e.g. '1,1,1' (cobrite mode).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = _load_config(config_path)
        config = _apply_laser_overrides(config, args)
    except Exception as exc:
        print("Invalid configuration: {}".format(exc), file=sys.stderr)
        return 2

    config_dir = config_path.parent
    acquisition_path = Path(config.get(
        "acquisition_script", "../fused_acquisition_controlled.py"
    ))
    if not acquisition_path.is_absolute():
        acquisition_path = (config_dir / acquisition_path).resolve()

    if args.dry_run:
        _print_plan(config, config_path, acquisition_path)
        return 0

    simulate = bool(args.simulate or config.get("simulate", False))

    if not simulate and not acquisition_path.is_file():
        print("Acquisition script not found: {}".format(acquisition_path), file=sys.stderr)
        return 2

    acquisition_python = config.get("acquisition_python") or sys.executable
    if not Path(acquisition_python).is_file():
        print("acquisition_python does not exist: {}".format(acquisition_python), file=sys.stderr)
        return 2

    output_root_setting = args.output_root or config.get("output_root", "runs")
    output_root = Path(output_root_setting)
    if not output_root.is_absolute():
        output_root = (config_dir / output_root).resolve()
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / (("mwcal_sim_" if simulate else "mwcal_") + run_stamp)
    run_dir.mkdir(parents=True, exist_ok=False)

    validation_fraction = float(config.get("validation_fraction", 0.25))
    seed = int(config.get("random_seed", 17))
    continue_on_error = bool(config.get("continue_on_error", False))
    max_condition = float(config.get("max_condition", 3.0))
    fail_on_poor = bool(config.get("fail_on_poor_conditioning", False))
    synthetic_settings = dict(config.get("synthetic") or {})
    laser_config = config.get("laser", {})
    laser = LaserController(laser_config)
    records: List[Dict[str, Any]] = []
    failed = False

    if simulate:
        print("\nSIMULATION: synthetic detector/Stokes data will be generated for every")
        print("wavelength. No laser, oscilloscope, PM1000 or EPS1000 will be contacted.")

    checklist = config.get("operator_checklist", [])
    if checklist and not simulate and not args.yes:
        print("\nBefore acquisition, verify:")
        for item in checklist:
            print("  [ ] {}".format(item))
        input("Complete the checklist, then press Enter to begin the sweep.")

    laser_info = None
    if not simulate and str(laser_config.get("mode", "manual")).lower() == "cobrite":
        try:
            laser_info = laser.describe()
        except Exception as exc:
            print("Cannot reach the CoBrite laser: {}".format(exc), file=sys.stderr)
            print("Fix the connection (or use --laser-mode manual), then retry.", file=sys.stderr)
            laser.close()
            return 2
        print("\nLaser: {}".format(laser_info.get("idn")))
        limits = laser_info.get("limits_nm") or [float("nan"), float("nan")]
        print("  port {} range {:.3f}-{:.3f} nm, setpoint {:.2f} dBm, actual {} dBm, output {}".format(
            laser_info.get("address"),
            limits[0] if limits else float("nan"),
            limits[1] if limits else float("nan"),
            float(laser_info.get("power_setpoint_dbm") or float("nan")),
            "n/a" if laser_info.get("actual_power_dbm") is None
            else "{:.2f}".format(laser_info["actual_power_dbm"]),
            "ON" if laser_info.get("output_state") == 1 else "OFF",
        ))
        out_of_range = [
            wavelength for wavelength in config["wavelengths_nm"]
            if limits and not limits[0] <= wavelength <= limits[1]
        ]
        if out_of_range:
            print("\nWavelength(s) outside the laser range: {}".format(
                ", ".join("{:.3f}".format(value) for value in out_of_range)
            ), file=sys.stderr)

    with open(run_dir / "sweep_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "simulated": simulate,
                "config_file": str(config_path),
                "acquisition_script": str(acquisition_path),
                "acquisition_python": acquisition_python,
                "wavelengths_nm": config["wavelengths_nm"],
                "n_sops": config["n_sops"],
                "validation_fraction": validation_fraction,
                "random_seed": seed,
                "max_condition": max_condition,
                "fail_on_poor_conditioning": fail_on_poor,
                "ch_scales_v_per_div": config.get("ch_scales"),
                "scope": config.get("scope"),
                "pm1000": config.get("pm1000"),
                "laser_mode": "simulated" if simulate else laser_config.get("mode", "manual"),
                "laser": laser_info,
                "synthetic": synthetic_settings if simulate else None,
            },
            handle,
            indent=2,
            sort_keys=True,
            default=float,
        )
        handle.write("\n")

    try:
        for wavelength_nm in config["wavelengths_nm"]:
            wavelength_dir = run_dir / ("lambda_{}nm".format(_wavelength_tag(wavelength_nm)))
            wavelength_dir.mkdir(parents=True, exist_ok=False)
            print("\n" + "=" * 72)
            print("Wavelength {:.6f} nm".format(wavelength_nm))
            print("Output directory: {}".format(wavelength_dir))

            try:
                if simulate:
                    laser_meta = {
                        "requested_nm": wavelength_nm,
                        "readback_nm": wavelength_nm,
                        "readback_source": "simulated",
                        "output": laser_config.get("output"),
                    }
                else:
                    laser_meta = laser.set_and_wait(
                        wavelength_nm, output=laser_config.get("output")
                    )
                print("Laser read-back: {} nm ({})".format(
                    laser_meta.get("readback_nm"), laser_meta.get("readback_source")
                ))

                truth_matrix = None
                truth_csv = None
                if simulate:
                    simulated = simulate_fused_acquisition(
                        output_dir=str(wavelength_dir),
                        wavelength_nm=wavelength_nm,
                        n_sops=config["n_sops"],
                        seed=seed + int(round(wavelength_nm * 100.0)),
                        **synthetic_settings
                    )
                    detector_csv = Path(simulated["detector_csv"])
                    stokes_csv = Path(simulated["stokes_csv"])
                    truth_matrix = simulated["truth_matrix"]
                    truth_csv = simulated["truth_csv"]
                    print("Simulated {} SOPs (detector noise {:.3g} V, truth in {})".format(
                        config["n_sops"], simulated["detector_noise_volts"],
                        Path(truth_csv).name
                    ))
                else:
                    # The parent script manages the Tektronix, PM1000, and EPS1000
                    # using the already-tested single-wavelength workflow.
                    command = [
                        acquisition_python, str(acquisition_path), str(config["n_sops"]),
                        "--wavelength-nm", "{:.4f}".format(float(wavelength_nm)),
                    ]
                    if config.get("ch_scales"):
                        command += [
                            "--ch-scales",
                            ",".join("{:g}".format(value) for value in config["ch_scales"]),
                        ]
                    print("Running acquisition: {}".format(" ".join(command)))
                    subprocess.run(
                        command, cwd=str(wavelength_dir), check=True,
                        env=_scope_environment(config),
                    )
                    detector_csv, stokes_csv = _find_acquisition_pair(wavelength_dir)

                result = calibrate_csv_pair(
                    detector_csv=str(detector_csv),
                    stokes_csv=str(stokes_csv),
                    output_dir=str(wavelength_dir),
                    wavelength_nm=wavelength_nm,
                    validation_fraction=validation_fraction,
                    seed=seed,
                    max_condition=max_condition,
                )
                conditioning = result["metrics"]["conditioning"]
                if fail_on_poor and conditioning["status"] == "poor":
                    raise RuntimeError(
                        "calibration matrix is poorly conditioned (row-scaled cond {:.3f}, "
                        "polarization cond {:.3f})".format(
                            conditioning["condition_row_scaled"],
                            conditioning["condition_polarization"],
                        )
                    )

                record = _metric_record(
                    wavelength_nm, laser_meta, result, simulated=simulate
                )
                record["matrix_file"] = str(Path(result["matrix_path"]).resolve())
                records.append(record)

                metadata = {
                    "requested_wavelength_nm": wavelength_nm,
                    "laser": laser_meta,
                    "simulated": simulate,
                    "ch_scales_v_per_div": config.get("ch_scales"),
                "scope": config.get("scope"),
                "pm1000": config.get("pm1000"),
                    "detector_csv": str(detector_csv.resolve()),
                    "stokes_csv": str(stokes_csv.resolve()),
                    "calibration_matrix": str(Path(result["matrix_path"]).resolve()),
                    "conditioning": conditioning,
                    "metrics": result["metrics"],
                }

                truth_error = None
                if truth_matrix is not None:
                    fitted = np.asarray(result["C"], dtype=float)
                    truth_error = float(
                        np.linalg.norm(fitted - truth_matrix) / np.linalg.norm(truth_matrix)
                    )
                    metadata["truth_csv"] = str(Path(truth_csv).resolve())
                    metadata["truth_matrix"] = np.asarray(truth_matrix).tolist()
                    metadata["fit_to_truth_relative_error"] = truth_error

                with open(wavelength_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2, sort_keys=True, default=float)
                    handle.write("\n")

                print("Held-out relative residual: {:.6g}".format(
                    result["metrics"]["validation"]["relative_residual"]
                ))
                print("Held-out mean angular error: {:.4f} deg".format(
                    result["metrics"]["validation"]["angular_mean_deg"]
                ))
                if truth_error is not None:
                    print("Fit vs synthetic truth: {:.4g} relative error".format(truth_error))
                print("Conditioning of C({:.3f} nm):".format(wavelength_nm))
                print(format_conditioning(conditioning, indent="  "))
                if conditioning["status"] in ("warn", "poor"):
                    print("  WARNING: conditioning status is {}.".format(
                        conditioning["status"].upper()
                    ), file=sys.stderr)
                    for issue in conditioning["issues"]:
                        print("    - {}".format(issue), file=sys.stderr)
            except Exception as exc:
                failed = True
                records.append({
                    "status": "failed",
                    "requested_wavelength_nm": wavelength_nm,
                    "readback_wavelength_nm": "",
                    "readback_source": "",
                    "laser_output": laser_config.get("output", ""),
                    "simulated": bool(simulate),
                    "error": "{}: {}".format(type(exc).__name__, exc),
                })
                print("FAILED at {:.6f} nm: {}".format(wavelength_nm, exc), file=sys.stderr)
                _write_summary(run_dir / "summary.csv", records)
                if not continue_on_error:
                    break
            else:
                _write_summary(run_dir / "summary.csv", records)
    finally:
        laser.close()

    _write_summary(run_dir / "summary.csv", records)
    combined_path = _save_combined(run_dir, records)
    print("\nRun directory: {}".format(run_dir))
    print("Summary:       {}".format(run_dir / "summary.csv"))
    if combined_path is not None:
        print("Matrices:      {}".format(combined_path))

    report = None
    if combined_path is not None or any(
        record.get("status") in ("ok", "warn") for record in records
    ):
        try:
            from conditioning_report import main as conditioning_report_main

            conditioning_report_main(
                [
                    "--run-dir", str(run_dir),
                    "--max-condition", "{:.6f}".format(max_condition),
                    "--output-dir", str(run_dir),
                    "--plot",
                ]
            )
        except SystemExit:
            pass
        except Exception as exc:  # pragma: no cover - reporting must never break a run
            print("Conditioning report failed: {}".format(exc), file=sys.stderr)

    conditioning_statuses = [
        record.get("conditioning_status")
        for record in records
        if record.get("conditioning_status")
    ]
    if conditioning_statuses:
        worst = max(
            conditioning_statuses,
            key=lambda value: {"excellent": 0, "ok": 1, "warn": 2, "poor": 3}.get(value, 4),
        )
        print("\nWorst conditioning status across the sweep: {}".format(worst.upper()))

    if failed:
        print("One or more wavelengths failed; see summary.csv and per-wavelength logs.", file=sys.stderr)
        return 1
    if "poor" in conditioning_statuses:
        print("At least one calibration matrix is poorly conditioned; see conditioning_report.csv.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
