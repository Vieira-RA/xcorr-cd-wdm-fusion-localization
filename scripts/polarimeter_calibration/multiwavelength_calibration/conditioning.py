#!/usr/bin/env python3
"""Conditioning diagnostics for the 4x4 polarimeter calibration matrices.

The operational map is ``S = C @ D``, so detector noise enters the reconstructed
Stokes vector amplified by the singular-value spread of ``C``.  For a tilted-FBG
polarimeter the ideal reference matrix is tetrahedral: four mutually orthogonal
detector projections of equal strength, i.e. a Hadamard matrix whose rows are
orthogonal and equally long.  That ideal has ``condition == 1``, which is the
baseline every metric below measures departures from.

Three numbers are reported, because the raw spectral condition number alone is
misleading for this device:

``condition``
    Spectral condition number ``sigma_max / sigma_min`` of the full 4x4 matrix.
    Row 0 (S0) and rows 1-3 (S1..S3) come from different unit conversions
    (microwatt-per-volt for power versus polarization), so a large value can be
    produced purely by the S0 row being much weaker than the others.  It is kept
    for continuity with the single-wavelength calibration script.

``condition_row_scaled``
    The same quantity after dividing every row by its norm.  This is invariant
    to unit changes and per-row gain, and measures how far the four detector
    projections are from orthogonal: ``1`` is ideal, ``sqrt(3) ~ 1.73`` is the
    paper's unbalanced tetrahedral guess (row norms 2 vs 2*sqrt(3)), and values
    above ~3 mean a few Stokes directions are much noisier than the rest.

``condition_polarization``
    Spectral condition number of the 3x4 map ``D -> (S1, S2, S3)``.  Because the
    operational target is the *normalized* SOP, this is the number that governs
    angular error on the Poincare sphere; the S0 scale does not enter it.

``row_orthogonality``
    ``volume / product(row_norms)`` for the rows (Gram-determinant based, so it
    works for the 3x4 polarization block too).  It is ``1`` for orthogonal rows
    and tends to ``0`` as two rows become collinear; it is the geometric
    counterpart of ``condition_row_scaled``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

# Balanced ideal: rows are mutually orthogonal and of equal norm, so both
# ``condition`` and ``condition_row_scaled`` are exactly 1.
IDEAL_TETRAHEDRON_MATRIX = 0.5 * np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ],
    dtype=float,
)

# Status tiers, applied to the worse of ``condition_row_scaled`` and
# ``condition_polarization``.  The ideal value is 1.0.
CONDITIONING_EXCELLENT = 1.5
CONDITIONING_WARNING = 10.0

#: Minimal set of scalars copied into run summaries and reports.
CONDITIONING_SUMMARY_KEYS = (
    "condition",
    "condition_row_scaled",
    "condition_polarization",
    "row_orthogonality",
    "row_norm_ratio",
    "status",
)


def _as_matrix(matrix: Any, name: str = "calibration matrix") -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError("{} must be 2-D, got shape {}.".format(name, array.shape))
    if array.size == 0:
        raise ValueError("{} is empty.".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains NaN or infinite values.".format(name))
    return array


def singular_values(matrix: Any) -> np.ndarray:
    """Singular values of ``matrix`` in descending order."""
    array = _as_matrix(matrix)
    values = np.linalg.svd(array, compute_uv=False)
    return np.sort(np.asarray(values, dtype=float))[::-1]


def spectral_condition(matrix: Any) -> float:
    """``sigma_max / sigma_min``; ``inf`` for a numerically rank-deficient matrix."""
    values = singular_values(matrix)
    if values.size == 0 or values[0] <= 0.0:
        return float("inf")
    if values[-1] <= np.finfo(float).eps * values[0]:
        return float("inf")
    return float(values[0] / values[-1])


def row_norms(matrix: Any) -> np.ndarray:
    array = _as_matrix(matrix)
    return np.linalg.norm(array, axis=1)


def row_scaled(matrix: Any) -> np.ndarray:
    """Rows divided by their norms (unit-row version of the matrix)."""
    array = _as_matrix(matrix)
    norms = np.linalg.norm(array, axis=1)
    if not np.all(norms > 0.0):
        raise ValueError("Cannot row-scale a matrix with a zero-norm row.")
    return array / norms[:, None]


def row_orthogonality(matrix: Any) -> float:
    """Gram-determinant based orthogonality ratio in ``[0, 1]`` (``1`` = orthogonal)."""
    array = _as_matrix(matrix)
    norms = np.linalg.norm(array, axis=1)
    if not np.all(norms > 0.0):
        return 0.0
    gram = array @ array.T
    volume_squared = float(np.linalg.det(gram))
    if volume_squared <= 0.0:
        return 0.0
    return float(min(1.0, np.sqrt(volume_squared) / float(np.prod(norms))))


def classify_conditioning(
    condition_row_scaled: float,
    condition_polarization: float,
    max_condition: float = 3.0,
) -> str:
    """Map the scale-invariant condition numbers onto ``excellent/ok/warn/poor``.

    The gate is applied to the worse of the two scale-invariant metrics, since
    both measure the same physical effect (amplification of detector noise).
    """
    values = [float(condition_row_scaled), float(condition_polarization)]
    if any(not np.isfinite(value) for value in values):
        return "poor"
    worst = max(values)
    gate = float(max_condition)
    # A configured gate stricter than the "excellent" tier takes precedence, so
    # an operator can tighten the acceptance criterion below 1.5.
    excellent_threshold = min(CONDITIONING_EXCELLENT, gate)
    if worst <= excellent_threshold:
        return "excellent"
    if worst <= gate:
        return "ok"
    if worst <= max(CONDITIONING_WARNING, gate):
        return "warn"
    return "poor"


def analyze_calibration_matrix(
    matrix: Any,
    max_condition: float = 3.0,
    polarization_rows: Sequence[int] = (1, 2, 3),
) -> Dict[str, Any]:
    """Full conditioning report for one ``C`` (``S = C @ D`` convention).

    Parameters
    ----------
    matrix
        4x4 calibration matrix, rows ``[S0, S1, S2, S3]``, columns
        ``[CH1..CH4]``.
    max_condition
        Gate applied to the worse of ``condition_row_scaled`` and
        ``condition_polarization``.
    polarization_rows
        Row indices mapped by ``condition_polarization`` (default S1..S3).
    """
    array = _as_matrix(matrix)
    if array.shape != (4, 4):
        raise ValueError("Expected a 4x4 calibration matrix, got {}.".format(array.shape))

    values = singular_values(array)
    norms = np.linalg.norm(array, axis=1)
    scaled_values = singular_values(row_scaled(array))

    polarization = array[list(polarization_rows), :]
    polarization_values = singular_values(polarization)

    condition = spectral_condition(array)
    condition_row_scaled = (
        float(scaled_values[0] / scaled_values[-1])
        if scaled_values.size and scaled_values[-1] > np.finfo(float).eps * scaled_values[0]
        else float("inf")
    )
    condition_polarization = (
        float(polarization_values[0] / polarization_values[-1])
        if polarization_values.size
        and polarization_values[-1] > np.finfo(float).eps * polarization_values[0]
        else float("inf")
    )

    issues: List[str] = []
    if np.any(norms <= 0.0):
        issues.append("at least one row has zero norm; the matrix is rank deficient")
    if condition_row_scaled > CONDITIONING_EXCELLENT:
        issues.append(
            "detector projections are not balanced (row-scaled cond {:.3f} > {:.1f})".format(
                condition_row_scaled, CONDITIONING_EXCELLENT
            )
        )
    if condition_polarization > CONDITIONING_EXCELLENT:
        issues.append(
            "polarization path is not balanced (cond {:.3f} > {:.1f})".format(
                condition_polarization, CONDITIONING_EXCELLENT
            )
        )
    if condition_row_scaled > float(max_condition):
        issues.append(
            "row-scaled cond {:.3f} exceeds the configured gate {:.3f}".format(
                condition_row_scaled, float(max_condition)
            )
        )
    if condition_polarization > float(max_condition):
        issues.append(
            "polarization cond {:.3f} exceeds the configured gate {:.3f}".format(
                condition_polarization, float(max_condition)
            )
        )

    return {
        "rank": int(np.linalg.matrix_rank(array)),
        "condition": condition,
        "singular_values": values.tolist(),
        "row_norms": norms.tolist(),
        "row_norm_ratio": float(np.max(norms) / np.min(norms)) if np.min(norms) > 0 else float("inf"),
        "condition_row_scaled": condition_row_scaled,
        "singular_values_row_scaled": scaled_values.tolist(),
        "condition_polarization": condition_polarization,
        "singular_values_polarization": polarization_values.tolist(),
        "row_orthogonality": row_orthogonality(array),
        "row_orthogonality_polarization": row_orthogonality(polarization),
        "gain_min": float(values[-1]),
        "gain_max": float(values[0]),
        "max_condition": float(max_condition),
        "status": classify_conditioning(
            condition_row_scaled, condition_polarization, max_condition=max_condition
        ),
        "issues": issues,
    }


def flatten_conditioning(report: Dict[str, Any], keys: Iterable[str] = CONDITIONING_SUMMARY_KEYS) -> Dict[str, Any]:
    """Pick the scalar summary fields from an ``analyze_calibration_matrix`` result."""
    return {key: report.get(key) for key in keys}


def format_conditioning(report: Dict[str, Any], indent: str = "  ") -> str:
    """Human-readable multi-line rendering of one conditioning report."""
    lines = [
        "{}condition        = {:>10.4f}   (sigma_max/sigma_min of C; row_norm_ratio = {:.3f})".format(
            indent, report["condition"], report["row_norm_ratio"]
        ),
        "{}row-scaled cond  = {:>10.4f}   (scale invariant; 1.0 = ideal tetrahedron)".format(
            indent, report["condition_row_scaled"]
        ),
        "{}polarization cond= {:>10.4f}   (D -> S1,S2,S3; governs normalized-SOP error)".format(
            indent, report["condition_polarization"]
        ),
        "{}row orthogonality= {:>10.4f}   (1.0 = rows mutually orthogonal)".format(
            indent, report["row_orthogonality"]
        ),
        "{}status           = {:>10s}   (rank {}, gate {:.3f})".format(
            indent, report["status"], report["rank"], report["max_condition"]
        ),
    ]
    return "\n".join(lines)
