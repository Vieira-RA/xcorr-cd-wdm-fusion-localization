#!/usr/bin/env python3
"""Numerical tools for referenced, wavelength-specific FBG polarimeter calibration.

The convention matches ``referenced_calibration_lsq.py``:

    S = C @ D
    C = S @ pinv(D)

where columns of D are detector voltages [CH1..CH4] and columns of S are the
PM1000 reference Stokes vectors [S0..S3] in microwatts.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:  # works both as ``import calibration`` and ``multiwavelength_calibration.calibration``
    from .conditioning import analyze_calibration_matrix
    from .conditioning import IDEAL_TETRAHEDRON_MATRIX  # noqa: F401  (re-exported)
except ImportError:  # pragma: no cover - script-style execution
    from conditioning import analyze_calibration_matrix
    from conditioning import IDEAL_TETRAHEDRON_MATRIX  # noqa: F401  (re-exported)

DETECTOR_COLUMNS = ("ch1", "ch2", "ch3", "ch4")
STOKES_COLUMNS = ("S0_uW", "S1_uW", "S2_uW", "S3_uW")


def _read_numeric_csv(path: str, required_columns: Iterable[str]) -> Dict[str, np.ndarray]:
    """Read named numeric columns from a CSV, with useful errors for bad files."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: {}".format(path))
        missing = [name for name in required_columns if name not in reader.fieldnames]
        if missing:
            raise ValueError("{} is missing columns: {}".format(path, ", ".join(missing)))
        columns = {name: [] for name in reader.fieldnames}
        for row_number, row in enumerate(reader, start=2):
            for name in reader.fieldnames:
                value = row.get(name, "")
                try:
                    columns[name].append(float(value))
                except (TypeError, ValueError):
                    raise ValueError(
                        "Non-numeric value in {} row {}, column {!r}: {!r}".format(
                            path, row_number, name, value
                        )
                    )
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def load_fused_csv(detector_csv: str, stokes_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load index-aligned output from ``fused_acquisition_controlled.py``.

    Returns ``(D, S, indices)`` with D and S shaped (4, M).
    """
    detector = _read_numeric_csv(detector_csv, ("index",) + DETECTOR_COLUMNS)
    stokes = _read_numeric_csv(stokes_csv, ("index",) + STOKES_COLUMNS)

    n_detector = len(detector["index"])
    n_stokes = len(stokes["index"])
    if n_detector != n_stokes:
        raise ValueError(
            "Detector/reference row count mismatch: {} vs {}".format(n_detector, n_stokes)
        )
    if n_detector < 5:
        raise ValueError("At least 5 SOP measurements are needed for a held-out validation.")

    detector_index = detector["index"]
    stokes_index = stokes["index"]
    if not np.array_equal(detector_index, stokes_index):
        raise ValueError("Detector and PM1000 index columns do not match.")

    D = np.vstack([detector[name] for name in DETECTOR_COLUMNS])
    S = np.vstack([stokes[name] for name in STOKES_COLUMNS])
    if not np.all(np.isfinite(D)) or not np.all(np.isfinite(S)):
        raise ValueError("Input data contains NaN or infinite values.")
    return D, S, detector_index


def _dop(stokes: np.ndarray) -> np.ndarray:
    s0 = stokes[0]
    polarized_power = np.linalg.norm(stokes[1:4], axis=0)
    result = np.full(s0.shape, np.nan, dtype=float)
    valid = np.abs(s0) > np.finfo(float).eps
    result[valid] = polarized_power[valid] / np.abs(s0[valid])
    return result


def _safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _safe_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.std(finite)) if finite.size else float("nan")


def evaluate_stokes(reference: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """Compute reference/reconstruction metrics for arrays shaped (4, M)."""
    if reference.shape != reconstructed.shape or reference.ndim != 2 or reference.shape[0] != 4:
        raise ValueError("Stokes arrays must have matching shape (4, M).")

    difference = reconstructed - reference
    relative_residual = float(
        np.linalg.norm(difference) / max(np.linalg.norm(reference), np.finfo(float).eps)
    )

    # Normalized Stokes error is useful when absolute optical power varies.
    ref_s0 = reference[0]
    pred_s0 = reconstructed[0]
    valid_s0 = (np.abs(ref_s0) > np.finfo(float).eps) & (
        np.abs(pred_s0) > np.finfo(float).eps
    )
    if np.any(valid_s0):
        ref_normalized = reference[:, valid_s0] / ref_s0[valid_s0]
        pred_normalized = reconstructed[:, valid_s0] / pred_s0[valid_s0]
        normalized_stokes_rmse = float(np.sqrt(np.mean((pred_normalized - ref_normalized) ** 2)))
    else:
        normalized_stokes_rmse = float("nan")

    dop_reference = _dop(reference)
    dop_reconstructed = _dop(reconstructed)
    dop_error = dop_reconstructed - dop_reference

    reference_pol = reference[1:4]
    reconstructed_pol = reconstructed[1:4]
    ref_norm = np.linalg.norm(reference_pol, axis=0)
    pred_norm = np.linalg.norm(reconstructed_pol, axis=0)
    valid_pol = (ref_norm > np.finfo(float).eps) & (pred_norm > np.finfo(float).eps)
    if np.any(valid_pol):
        cosine = np.sum(reference_pol[:, valid_pol] * reconstructed_pol[:, valid_pol], axis=0)
        cosine /= ref_norm[valid_pol] * pred_norm[valid_pol]
        angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        angle_mean = float(np.mean(angles))
        angle_std = float(np.std(angles))
        angle_max = float(np.max(angles))
    else:
        angle_mean = angle_std = angle_max = float("nan")

    return {
        "relative_residual": relative_residual,
        "normalized_stokes_rmse": normalized_stokes_rmse,
        "reference_dop_mean": _safe_mean(dop_reference),
        "reference_dop_std": _safe_std(dop_reference),
        "reconstructed_dop_mean": _safe_mean(dop_reconstructed),
        "reconstructed_dop_std": _safe_std(dop_reconstructed),
        "dop_mae": _safe_mean(np.abs(dop_error)),
        "dop_error_std": _safe_std(dop_error),
        "angular_mean_deg": angle_mean,
        "angular_std_deg": angle_std,
        "angular_max_deg": angle_max,
    }


def _condition_number(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size < 4 or singular_values[-1] <= np.finfo(float).eps * singular_values[0]:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def calibrate_arrays(
    D: np.ndarray,
    S: np.ndarray,
    validation_fraction: float = 0.25,
    seed: int = 17,
    max_condition: float = 3.0,
) -> Dict[str, Any]:
    """Fit C, report held-out metrics, and return a final all-data fit.

    The random split is used only to estimate generalization performance. The
    returned ``C`` is refit on all observations after validation, matching the
    usual calibration workflow and producing the matrix intended for later use.
    """
    D = np.asarray(D, dtype=float)
    S = np.asarray(S, dtype=float)
    if D.ndim != 2 or D.shape[0] != 4:
        raise ValueError("D must have shape (4, M), one row per scope channel.")
    if S.ndim != 2 or S.shape[0] != 4:
        raise ValueError("S must have shape (4, M), ordered S0..S3.")
    if D.shape[1] != S.shape[1]:
        raise ValueError("D and S must contain the same number of SOP measurements.")
    if D.shape[1] < 5:
        raise ValueError("At least 5 SOP measurements are needed for a held-out validation.")
    if not np.all(np.isfinite(D)) or not np.all(np.isfinite(S)):
        raise ValueError("D or S contains NaN or infinite values.")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5.")

    n = D.shape[1]
    # A stale scope capture repeats the same detector vector for every SOP: the
    # matrix still looks full rank (rank 4) but the fit is meaningless, so this
    # must be an error, not a warning. See DELAY_BETWEEN_S in the acquisition.
    distinct = np.unique(np.round(D, 12).T, axis=0).shape[0]
    if distinct < 5 or distinct < 0.6 * n:
        raise ValueError(
            "Only {} distinct detector vectors out of {} measurements: the oscilloscope "
            "returned a stale acquisition (RUN/STOP race). Increase DELAY_BETWEEN_S in "
            "fused_acquisition_controlled.py and re-acquire this wavelength.".format(
                distinct, n
            )
        )
    n_validation = max(1, int(round(n * validation_fraction)))
    n_training = n - n_validation
    if n_training < 4:
        raise ValueError("The training subset must contain at least 4 SOPs.")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    validation_indices = np.sort(permutation[:n_validation])
    training_indices = np.sort(permutation[n_validation:])

    D_train = D[:, training_indices]
    S_train = S[:, training_indices]
    rank = int(np.linalg.matrix_rank(D_train))
    if rank < 4:
        raise ValueError(
            "Training detector matrix has rank {} (need 4); collect more diverse SOPs.".format(rank)
        )

    C_training = S_train @ np.linalg.pinv(D_train)
    S_validation = S[:, validation_indices]
    S_validation_predicted = C_training @ D[:, validation_indices]
    validation_metrics = evaluate_stokes(S_validation, S_validation_predicted)

    S_training_predicted = C_training @ D_train
    training_metrics = evaluate_stokes(S_train, S_training_predicted)

    # Final operational matrix uses all acquired calibration states, as in the
    # existing single-wavelength calibration script.
    C_all = S @ np.linalg.pinv(D)
    conditioning = analyze_calibration_matrix(C_all, max_condition=max_condition)
    metrics: Dict[str, Any] = {
        "n_measurements": int(n),
        "n_training": int(training_indices.size),
        "n_validation": int(validation_indices.size),
        "detector_training_rank": rank,
        "detector_training_condition": _condition_number(D_train),
        "calibration_matrix_condition": _condition_number(C_all),
        "training": training_metrics,
        "validation": validation_metrics,
        "conditioning": conditioning,
        "conditioning_training": analyze_calibration_matrix(
            C_training, max_condition=max_condition
        ),
    }
    return {
        "C": C_all,
        "C_training": C_training,
        "metrics": metrics,
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "validation_reference": S_validation,
        "validation_predicted": S_validation_predicted,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def calibrate_csv_pair(
    detector_csv: str,
    stokes_csv: str,
    output_dir: str,
    wavelength_nm: float,
    validation_fraction: float = 0.25,
    seed: int = 17,
    max_condition: float = 3.0,
) -> Dict[str, Any]:
    """Calibrate a saved fused-acquisition pair and write matrix/metrics files."""
    D, S, indices = load_fused_csv(detector_csv, stokes_csv)
    result = calibrate_arrays(
        D,
        S,
        validation_fraction=validation_fraction,
        seed=seed,
        max_condition=max_condition,
    )
    os.makedirs(output_dir, exist_ok=True)

    wavelength_tag = "{:.3f}".format(float(wavelength_nm))
    matrix_path = os.path.join(output_dir, "calibration_{}nm.csv".format(wavelength_tag))
    metrics_path = os.path.join(output_dir, "metrics_{}nm.json".format(wavelength_tag))
    validation_path = os.path.join(output_dir, "validation_{}nm.csv".format(wavelength_tag))

    # Headerless numeric 4x4 CSV stays compatible with np.loadtxt and the
    # existing calibration-matrix consumers.
    np.savetxt(matrix_path, result["C"], delimiter=",", fmt="%.12g")

    validation = result["validation_reference"]
    predicted = result["validation_predicted"]
    validation_dop = _dop(validation)
    predicted_dop = _dop(predicted)
    ref_pol = validation[1:4]
    pred_pol = predicted[1:4]
    ref_norm = np.linalg.norm(ref_pol, axis=0)
    pred_norm = np.linalg.norm(pred_pol, axis=0)
    angles = np.full(ref_norm.shape, np.nan, dtype=float)
    valid = (ref_norm > np.finfo(float).eps) & (pred_norm > np.finfo(float).eps)
    if np.any(valid):
        cosines = np.sum(ref_pol[:, valid] * pred_pol[:, valid], axis=0)
        cosines /= ref_norm[valid] * pred_norm[valid]
        angles[valid] = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))

    validation_indices = result["validation_indices"]
    with open(validation_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "index", "S0_ref_uW", "S1_ref_uW", "S2_ref_uW", "S3_ref_uW",
            "S0_pred_uW", "S1_pred_uW", "S2_pred_uW", "S3_pred_uW",
            "DOP_ref", "DOP_pred", "angular_error_deg",
        ])
        for column, data_index in enumerate(validation_indices):
            writer.writerow([
                int(indices[data_index]),
                *validation[:, column].tolist(),
                *predicted[:, column].tolist(),
                validation_dop[column],
                predicted_dop[column],
                angles[column],
            ])

    payload = {
        "wavelength_nm": float(wavelength_nm),
        "detector_csv": os.path.abspath(detector_csv),
        "stokes_csv": os.path.abspath(stokes_csv),
        "matrix_file": os.path.basename(matrix_path),
        "validation_file": os.path.basename(validation_path),
        "equation": "S_uW = C @ D_volts",
        "max_condition": float(max_condition),
        "conditioning_status": result["metrics"]["conditioning"]["status"],
        "metrics": result["metrics"],
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")

    result["matrix_path"] = matrix_path
    result["metrics_path"] = metrics_path
    result["validation_path"] = validation_path
    result["wavelength_nm"] = float(wavelength_nm)
    return result
