#!/usr/bin/env python
"""
Referenced polarimeter calibration (least-squares, M > 4 SOPs).

Reads the fused-acquisition output (detector_readings_*.csv from the scope and
stokes_*.csv from the PM1000 reference), builds the over-determined system and
solves for the 4x4 calibration matrix C by least squares:

    C = S @ pinv(D)      so that   S = C @ D

where D is 4xM detector voltages [ch1..ch4] (V) and S is 4xM reference Stokes
[S0,S1,S2,S3] (uW).

Usage:
    python referenced_calibration_lsq.py [detector_csv] [stokes_csv]
If no arguments are given, the most recent detector/stokes pair is auto-detected.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

OUT_CSV = "calibration_matrix_reference.csv"
PLOT_FILE = "poincare_lsq_reconstruction.png"


def find_latest_pair():
    """Return (detector_csv, stokes_csv) for the most recent fused run."""
    pairs = []
    for f in sorted(glob.glob("detector_readings_*.csv")):
        ts = f[len("detector_readings_"):-len(".csv")]
        st = f"stokes_{ts}.csv"
        if os.path.exists(st):
            pairs.append((ts, f, st))
    if not pairs:
        raise FileNotFoundError("No detector_readings_*/stokes_* pair found.")
    pairs.sort()                       # timestamp sorts chronologically
    _, det, st = pairs[-1]
    return det, st


def main():
    if len(sys.argv) == 3:
        detector_csv, stokes_csv = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        detector_csv, stokes_csv = find_latest_pair()
    else:
        print("Usage: python referenced_calibration_lsq.py [detector_csv] [stokes_csv]")
        sys.exit(1)

    print(f"Detector data : {detector_csv}")
    print(f"Reference data: {stokes_csv}")

    # --- load ---
    det = np.genfromtxt(detector_csv, delimiter=",", skip_header=1)   # (M, 5)
    st = np.genfromtxt(stokes_csv, delimiter=",", skip_header=1)      # (M, 6)

    if det.ndim != 2 or det.shape[1] != 5:
        raise ValueError(f"Unexpected detector CSV shape {det.shape}")
    if st.ndim != 2 or st.shape[1] != 6:
        raise ValueError(f"Unexpected stokes CSV shape {st.shape}")
    if det.shape[0] != st.shape[0]:
        raise ValueError("Row count mismatch between detector and stokes files.")
    if not np.allclose(det[:, 0], st[:, 0]):
        raise ValueError("index columns do not match between the two files.")

    D = det[:, 1:5].T     # (4, M): ch1..ch4 in V
    S = st[:, 1:5].T      # (4, M): S0..S3 in uW
    M = D.shape[1]
    print(f"Loaded {M} SOPs.")

    # --- least-squares calibration ---
    C = S @ np.linalg.pinv(D)      # (4, 4)
    S_pred = C @ D                 # (4, M)

    # --- verification ---
    rel_resid = np.linalg.norm(S_pred - S) / np.linalg.norm(S)

    def dop_of(X):
        return np.sqrt(X[1]**2 + X[2]**2 + X[3]**2) / (np.abs(X[0]) + 1e-12)

    dop_meas = dop_of(S)
    dop_pred = dop_of(S_pred)

    u = S[1:4] / (np.linalg.norm(S[1:4], axis=0) + 1e-12)          # (3, M)
    v = S_pred[1:4] / (np.linalg.norm(S_pred[1:4], axis=0) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(np.sum(u * v, axis=0), -1.0, 1.0)))

    print("\n" + "=" * 60)
    print("   CALIBRATION MATRIX C  (detector V -> Stokes uW)")
    print("=" * 60)
    np.set_printoptions(precision=6, suppress=True)
    print(C)

    print("\n" + "=" * 60)
    print("   VERIFICATION")
    print("=" * 60)
    print(f"Relative residual  ||S - C*D|| / ||S|| : {rel_resid:.6e}")
    print(f"DOP (measured)     mean = {np.mean(dop_meas):.6f}, "
          f"std = {np.std(dop_meas):.6f}")
    print(f"DOP (reconstructed) mean = {np.mean(dop_pred):.6f}, "
          f"std = {np.std(dop_pred):.6f}  (SD-DOP)")
    print(f"Angular deviation  mean = {np.mean(ang):.4f} deg, "
          f"max = {np.max(ang):.4f} deg, std = {np.std(ang):.4f} deg")

    # --- save ---
    np.savetxt(OUT_CSV, C, delimiter=",")
    print(f"\nCalibration matrix saved to {OUT_CSV}")

    # --- plot ---
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    uu = np.linspace(0, 2 * np.pi, 60)
    vv = np.linspace(0, np.pi, 60)
    ax.plot_surface(np.outer(np.cos(uu), np.sin(vv)),
                    np.outer(np.sin(uu), np.sin(vv)),
                    np.outer(np.ones_like(uu), np.cos(vv)),
                    color="lightblue", alpha=0.15, edgecolor="none")
    ax.scatter(u[0], u[1], u[2], c="blue", s=15, alpha=0.6, label="Measured (PM1000)")
    ax.scatter(v[0], v[1], v[2], c="red", s=15, alpha=0.6, label="Reconstructed (C·D)")
    ax.set_xlabel("S1")
    ax.set_ylabel("S2")
    ax.set_zlabel("S3")
    ax.set_title(f"Poincaré: measured vs reconstructed ({M} SOPs)")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved to {PLOT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
