#!/usr/bin/env python3
"""Report the conditioning of every calibration matrix produced by a sweep.

Usage examples
--------------
Newest sweep under ``./runs`` (default)::

    python conditioning_report.py

A specific sweep, or the combined archive::

    python conditioning_report.py --run-dir runs/mwcal_20250101_120000
    python conditioning_report.py --npz runs/mwcal_20250101_120000/all_calibration_matrices.npz

Loose matrix files (wavelength inferred from ``calibration_<wl>nm.csv``)::

    python conditioning_report.py ../calibration_matrix_reference.csv --wavelength 1550

The exit code is 0 only when every matrix passes the gate (default 3.0 on the
worse of the row-scaled and polarization condition numbers), so this can be used
as a go/no-go check at the end of a sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # works as a module inside the package and as a flat script import
    from .conditioning import analyze_calibration_matrix, format_conditioning
except ImportError:  # pragma: no cover - script-style execution
    from conditioning import analyze_calibration_matrix, format_conditioning

REPORT_FIELDS = [
    "wavelength_nm",
    "status",
    "condition",
    "condition_row_scaled",
    "condition_polarization",
    "row_orthogonality",
    "row_norm_ratio",
    "rank",
    "matrix_file",
    "issues",
]

_STATUS_ORDER = {"excellent": 0, "ok": 1, "warn": 2, "poor": 3, "unknown": 4}


def _wavelength_from_filename(path: Path) -> Optional[float]:
    stem = path.stem
    # Formats used in the repo: calibration_1550.000nm.csv, lambda_1550p000nm, ...
    match = re.search(r"(\d+)p(\d+)nm$", stem)
    if match:
        return float("{}.{}".format(match.group(1), match.group(2)))
    match = re.search(r"(\d+(?:\.\d+)?)nm$", stem)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(str(path), delimiter=",")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("{} does not contain a 4x4 matrix (shape {}).".format(path, matrix.shape))
    return matrix


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted(
        [path for path in root.glob("mwcal_*") if path.is_dir()],
        key=lambda path: path.name,
    )
    if not candidates:
        raise SystemExit(
            "No sweep directory found under {} (expected runs/mwcal_*).".format(root)
        )
    return candidates[-1]


def collect_from_run_dir(run_dir: Path) -> List[Tuple[Optional[float], Path, np.ndarray]]:
    matrices: List[Tuple[Optional[float], Path, np.ndarray]] = []
    known = [
        path
        for path in sorted(run_dir.glob("*/calibration_*nm.csv"))
        if not path.name.startswith("truth_")
    ]
    if known:
        for path in known:
            wavelength = _wavelength_from_filename(path)
            if wavelength is None:
                continue
            matrices.append((wavelength, path, _load_matrix(path)))
        return matrices

    # Fall back to the combined archive (identical content, single file).
    archive = run_dir / "all_calibration_matrices.npz"
    if archive.is_file():
        return collect_from_npz(archive)
    raise SystemExit("No calibration_*nm.csv or all_calibration_matrices.npz in {}".format(run_dir))


def collect_from_npz(path: Path) -> List[Tuple[Optional[float], Path, np.ndarray]]:
    payload = np.load(str(path), allow_pickle=False)
    try:
        wavelengths = np.asarray(payload["wavelengths_nm"], dtype=float)
        matrices = np.asarray(payload["calibration_matrices"], dtype=float)
    except KeyError as exc:
        raise SystemExit("{} is missing key {}.".format(path, exc))
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise SystemExit("Unexpected matrix block shape {} in {}.".format(matrices.shape, path))
    if matrices.shape[0] != wavelengths.size:
        raise SystemExit("wavelengths_nm and calibration_matrices disagree in {}.".format(path))
    return [(float(wavelength), path, matrices[index]) for index, wavelength in enumerate(wavelengths)]


def collect_by_scan(root: Path) -> List[Tuple[Optional[float], Path, np.ndarray]]:
    """Recursively find every calibration matrix under ``root``.

    This is the "check every matrix I have ever produced" mode: it picks up all
    per-wavelength matrices from every sweep plus the single-wavelength
    ``calibration_matrix_reference.csv``.  Matrices whose filename carries no
    wavelength are reported with an unknown wavelength (``None``).
    """
    if not root.is_dir():
        raise SystemExit("Scan root is not a directory: {}".format(root))
    paths = sorted(
        {path for pattern in ("calibration_*nm.csv", "calibration_matrix_reference.csv")
         for path in root.rglob(pattern)}
    )
    if not paths:
        raise SystemExit(
            "No calibration_*nm.csv or calibration_matrix_reference.csv found under {}.".format(
                root
            )
        )
    return [( _wavelength_from_filename(path), path, _load_matrix(path)) for path in paths ]


def analyze_all(
    inputs: Sequence[Tuple[Optional[float], Path, np.ndarray]],
    max_condition: float = 3.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for wavelength, source, matrix in inputs:
        report = analyze_calibration_matrix(matrix, max_condition=max_condition)
        rows.append(
            {
                "wavelength_nm": None if wavelength is None else float(wavelength),
                "matrix_file": str(source),
                "matrix": matrix,
                "report": report,
            }
        )
    rows.sort(key=lambda row: (row["wavelength_nm"] is None, row["wavelength_nm"] or 0.0))
    return rows


def _print_table(rows: Sequence[Dict[str, Any]], max_condition: float) -> None:
    header = (
        "{:>12s}  {:>10s}  {:>11s}  {:>12s}  {:>11s}  {:>10s}  {:>5s}".format(
            "lambda [nm]",
            "cond(C)",
            "cond_rowsc.",
            "cond_polariz",
            "orthogon.",
            "row_ratio",
            "rank",
        )
    )
    print("\nCalibration matrix conditioning (S = C @ D, ideal = 1.0)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        report = row["report"]
        wavelength_text = (
            "n/a" if row["wavelength_nm"] is None
            else "{:.3f}".format(row["wavelength_nm"])
        )
        print(
            "{:>12s}  {:>10.3f}  {:>11.3f}  {:>12.3f}  {:>11.4f}  {:>10.3f}  {:>5d}  {}".format(
                wavelength_text,
                report["condition"],
                report["condition_row_scaled"],
                report["condition_polarization"],
                report["row_orthogonality"],
                report["row_norm_ratio"],
                report["rank"],
                report["status"].upper(),
            )
        )
    print("-" * len(header))
    print(
        "Gate: row-scaled and polarization condition numbers must stay below {:.3f}.".format(
            float(max_condition)
        )
    )


def _summarize(rows: Sequence[Dict[str, Any]], max_condition: float) -> int:
    worst_status = "excellent"
    offenders: List[str] = []
    for row in rows:
        status = row["report"]["status"]
        if _STATUS_ORDER.get(status, 4) > _STATUS_ORDER.get(worst_status, 0):
            worst_status = status
        if status in ("warn", "poor"):
            label = (
                "unknown wavelength" if row["wavelength_nm"] is None
                else "{:.3f} nm".format(row["wavelength_nm"])
            )
            offenders.append("{} ({})".format(label, status))

    finite = [row["report"] for row in rows]
    if finite:
        worst_row_scaled = max(report["condition_row_scaled"] for report in finite)
        worst_polarization = max(report["condition_polarization"] for report in finite)
        print(
            "Worst row-scaled cond = {:.3f}, worst polarization cond = {:.3f} over {} matrices.".format(
                worst_row_scaled, worst_polarization, len(rows)
            )
        )

    if not offenders:
        print(
            "PASS: all {} matrices are well conditioned against the {:.3f} gate.".format(
                len(rows), float(max_condition)
            )
        )
        return 0

    print("ATTENTION: {} of {} matrices need a look: {}".format(
        len(offenders), len(rows), ", ".join(offenders)
    ))
    print("Worst status: {}".format(worst_status.upper()))
    for row in rows:
        if row["report"]["status"] in ("warn", "poor"):
            label = (
                "unknown wavelength" if row["wavelength_nm"] is None
                else "{:.3f} nm".format(row["wavelength_nm"])
            )
            print("\n{} ({})".format(label, row["matrix_file"]))
            print(format_conditioning(row["report"], indent="    "))
            for issue in row["report"]["issues"]:
                print("    - {}".format(issue))
    return 1


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            report = row["report"]
            writer.writerow(
                {
                    "wavelength_nm": row["wavelength_nm"],
                    "status": report["status"],
                    "condition": report["condition"],
                    "condition_row_scaled": report["condition_row_scaled"],
                    "condition_polarization": report["condition_polarization"],
                    "row_orthogonality": report["row_orthogonality"],
                    "row_norm_ratio": report["row_norm_ratio"],
                    "rank": report["rank"],
                    "matrix_file": row["matrix_file"],
                    "issues": "; ".join(report["issues"]),
                }
            )


def _write_json(path: Path, rows: Sequence[Dict[str, Any]], max_condition: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": float(max_condition),
        "n_matrices": len(rows),
        "matrices": [
            {
                "wavelength_nm": row["wavelength_nm"],
                "matrix_file": row["matrix_file"],
                "matrix": np.asarray(row["matrix"]).tolist(),
                "conditioning": row["report"],
            }
            for row in rows
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _plot(path: Path, rows: Sequence[Dict[str, Any]], max_condition: float) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        print("matplotlib is not available; skipping the conditioning plot.")
        return None

    wavelengths = [row["wavelength_nm"] for row in rows]
    if any(wavelength is None for wavelength in wavelengths):
        print("Skipping the conditioning plot: some matrices have no wavelength in their name.")
        return None
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        wavelengths,
        [row["report"]["condition_row_scaled"] for row in rows],
        "o-",
        label="row-scaled cond (ideal 1.0)",
        color="tab:blue",
    )
    axis.plot(
        wavelengths,
        [row["report"]["condition_polarization"] for row in rows],
        "s--",
        label="polarization cond (D -> S1,S2,S3)",
        color="tab:green",
    )
    axis.plot(
        wavelengths,
        [row["report"]["condition"] for row in rows],
        "^:",
        label="raw cond(C) (includes S0 row scale)",
        color="tab:orange",
    )
    axis.axhline(1.0, color="0.5", linewidth=1.0, linestyle="-", label="ideal (1.0)")
    axis.axhline(float(max_condition), color="tab:red", linewidth=1.0, linestyle="--",
                 label="gate ({:.2f})".format(float(max_condition)))
    axis.set_xlabel("wavelength [nm]")
    axis.set_ylabel("condition number")
    axis.set_yscale("log")
    axis.set_title("Calibration matrix conditioning across the sweep")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrices", nargs="*", help="explicit 4x4 calibration CSV files")
    parser.add_argument("--run-dir", help="sweep output directory (runs/mwcal_*)")
    parser.add_argument("--scan", help="recursively report on every calibration matrix under this path")
    parser.add_argument("--npz", help="all_calibration_matrices.npz to report on")
    parser.add_argument("--runs-root", default="runs", help="where to look for the newest sweep")
    parser.add_argument("--wavelength", type=float, action="append",
                        help="wavelength in nm for each explicit matrix (repeatable)")
    parser.add_argument("--max-condition", type=float, default=3.0,
                        help="gate on the worse scale-invariant condition number (default 3.0)")
    parser.add_argument("--output-dir", help="where to write conditioning_report.{csv,json}")
    parser.add_argument("--plot", nargs="?", const="conditioning_vs_wavelength.png", default=None,
                        help="also write a condition-number-vs-wavelength plot")
    parser.add_argument("--quiet", action="store_true", help="only print the verdict line")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.npz:
        inputs = collect_from_npz(Path(args.npz).expanduser())
        default_output = Path(args.npz).expanduser().parent
    elif args.scan:
        root = Path(args.scan).expanduser()
        inputs = collect_by_scan(root)
        print("Scanned {} matrix file(s) under {}".format(len(inputs), root))
        # A scan spans several runs, so report into the working directory
        # unless the caller picks a location explicitly.
        default_output = Path.cwd()
    elif args.run_dir:
        inputs = collect_from_run_dir(Path(args.run_dir).expanduser())
        default_output = Path(args.run_dir).expanduser()
    elif args.matrices:
        wavelengths = list(args.wavelength or [])
        inputs = []
        for index, name in enumerate(args.matrices):
            path = Path(name).expanduser()
            wavelength = (
                wavelengths[index]
                if index < len(wavelengths)
                else _wavelength_from_filename(path)
            )
            if wavelength is None:
                wavelength = float(index + 1)
            inputs.append((float(wavelength), path, _load_matrix(path)))
        default_output = Path(args.matrices[0]).expanduser().parent
    else:
        run_dir = _latest_run_dir(Path(args.runs_root).expanduser())
        print("Reporting on newest sweep: {}".format(run_dir))
        inputs = collect_from_run_dir(run_dir)
        default_output = run_dir

    if not inputs:
        raise SystemExit("No calibration matrices found.")

    rows = analyze_all(inputs, max_condition=float(args.max_condition))

    if not args.quiet:
        _print_table(rows, float(args.max_condition))

    output_dir = (
        Path(args.output_dir).expanduser() if args.output_dir else default_output
    )
    csv_path = output_dir / "conditioning_report.csv"
    json_path = output_dir / "conditioning_report.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, rows, float(args.max_condition))

    if args.plot:
        plot_path = output_dir / args.plot
        if _plot(plot_path, rows, float(args.max_condition)):
            print("\nConditioning plot: {}".format(plot_path))

    if not args.quiet:
        print("\nWrote:\n  {}\n  {}".format(csv_path, json_path))
        print()

    return _summarize(rows, float(args.max_condition))


if __name__ == "__main__":
    sys.exit(main())
