#!/usr/bin/env python
"""
Measure a polarization trajectory and show it on the Poincaré sphere.

Two independent paths (no synchronization needed because the trajectory is
periodic):
  1. Scope: read the 4-channel waveform (one long acquisition), apply the
     calibration matrix C, and plot the resulting Stokes trajectory.
  2. PM1000: load a recorded Stokes trajectory from a file (exported from the
     PM1000 GUI) and plot it.

The EPS1000 scrambler is set manually (not controlled here).

Uses the last calibration matrix from calibration_matrix_reference.csv.

The static rotation between the two reference frames is estimated with the
Kabsch algorithm (plus ICP for the unsynchronized correspondence) so the
PM1000 trajectory can be rotated onto the scope trajectory.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import vxi11
from vxi11.vxi11 import Vxi11Exception

# ================= Configuration =================
SCOPE_IP = "198.192.1.1"
CHANNELS = [1, 2, 3, 4]
CH_SCALES = {1: .02, 2: .02, 3: .02, 4: .02}   # V/div - adjust per channel if needed
SCOPE_TIMEOUT = 30.0

HORIZONTAL_SCALE = 150e-6     # s/div -> 1.5 ms window (10 divisions)
RECORD_LENGTH    = 1_000_000    # points -> 10 ns/sample (100 MS/s)

PM_TRAJ_FILE = "PM_trajectory_004.txt"   # PM1000 recorded trajectory (from GUI)
MA_WINDOW = 800               # moving-average window (samples) - same for both

CAL_FILE = "calibration_matrix_reference.csv"
PLOT_FILE = "poincare_trajectory.png"


# ================= Oscilloscope =================
def read_curve(scope):
    """Read the current channel's CURVE? block and return raw int8 samples."""
    scope.write("CURVE?")
    hdr = scope.read_raw(2)
    if hdr[:1] != b"#":
        raise RuntimeError(f"Unexpected CURVE? header: {hdr!r}")
    n_digits = int(hdr[1:2])
    byte_count = int(scope.read_raw(n_digits).decode("ascii"))
    data = scope.read_raw(byte_count)

    saved = scope.timeout
    scope.timeout = 0.5
    try:
        scope.read_raw(1)          # consume optional trailing newline
    except Vxi11Exception:
        pass
    finally:
        scope.timeout = saved

    return np.frombuffer(data, dtype=np.int8).astype(float)


def read_scope_waveforms(scope):
    """Read all 4 channels' full waveforms from the frozen acquisition.

    Returns D of shape (4, N) in volts (rows = ch1..ch4).
    """
    chans = []
    for ch in CHANNELS:
        scope.write(f"DATA:SOURCE CH{ch}")
        y_mult = float(scope.ask("WFMOUTPRE:YMULT?"))
        y_zero = float(scope.ask("WFMOUTPRE:YZERO?"))
        y_off  = float(scope.ask("WFMOUTPRE:YOFF?"))

        raw = read_curve(scope)
        volts = (raw - y_off) * y_mult + y_zero
        chans.append(volts)
    return np.array(chans)


def init_scope():
    """Configure the scope; return (scope, window_s, n_pts)."""
    scope = vxi11.Instrument(SCOPE_IP)
    scope.timeout = SCOPE_TIMEOUT
    print(scope.ask("*IDN?"))

    for ch in CHANNELS:
        scope.write(f"CH{ch}:STATE ON")
        scope.write(f"CH{ch}:SCALE {CH_SCALES[ch]}")
        scope.write(f"CH{ch}:POSITION 0.0")

    scope.write(f"HORIZONTAL:SCALE {HORIZONTAL_SCALE}")
    scope.write(f"HORIZONTAL:RECORDLENGTH {RECORD_LENGTH}")

    scope.write("DATA:ENC SRI")
    scope.write("DATA:WIDTH 1")
    scope.write("WFMOUTPRE:BYT_NR 1")
    scope.write("DATA:START 1")
    scope.write("DATA:STOP 1e10")

    xincr = float(scope.ask("WFMOUTPRE:XINCR?"))
    n_pts = int(scope.ask("WFMOUTPRE:NR_PT?"))
    scope.write(f"DATA:STOP {n_pts}")

    scope.write("TRIGGER:A:MODE AUTO")
    scope.write("ACQUIRE:STOPAFTER RUNSTOP")
    scope.write("ACQUIRE:STATE RUN")
    time.sleep(1.0)

    window_s = xincr * n_pts
    print(f"Scope: xincr = {xincr*1e9:.3f} ns, points = {n_pts}, "
          f"window = {window_s*1e6:.3f} us")
    return scope, window_s, n_pts


# ================= PM1000 trajectory (from file) =================
def load_pm_trajectory(filename):
    """Load a PM1000 recorded trajectory file.

    Format (as exported by the PM1000 GUI):
        # <metadata lines starting with '#', e.g. PowerLeftShift=7>
        time_ns, Power, S1, S2, S3   (comma separated)

    - Power is stored left-shifted by 7 bits  -> S0 = Power / 2**7   (uW)
    - S1..S3 are offset-binary (offset 32768), unit-normalized
      -> s_i = (raw - 32768) / 32768

    Returns S of shape (4, N) with rows [S0, S1, S2, S3] in uW.
    """
    data = np.loadtxt(filename, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(f"Unexpected trajectory file shape {data.shape}")

    S0 = data[:, 1] / 128.0                     # uW (PowerLeftShift = 7)
    s1 = (data[:, 2] - 32768) / 32768.0         # normalized Stokes (unit length)
    s2 = (data[:, 3] - 32768) / 32768.0
    s3 = (data[:, 4] - 32768) / 32768.0

    # Reconstruct raw Stokes in uW so we smooth then re-normalize exactly like
    # the scope path.
    return np.vstack([S0, s1 * S0, s2 * S0, s3 * S0])   # (4, N)


def moving_average(x, w):
    """Box moving average along the last axis (for visualization)."""
    if w <= 1 or w >= x.shape[-1]:
        return x
    kernel = np.ones(w) / w
    return np.stack([np.convolve(row, kernel, mode="same") for row in x])


# ================= Rotation alignment (Kabsch + ICP) =================
def kabsch_rotation(ref, tgt):
    """Optimal proper rotation mapping ``tgt`` onto ``ref`` (Kabsch/Umeyama).

    Solves the absolute-orientation / orthogonal-Procrustes problem: find the
    rotation matrix ``R`` in SO(3) minimizing  ``||R @ tgt - ref||``  for two
    (3, N) point sets with *known* point-to-point correspondence.

    Closed-form SVD solution: with the cross-covariance
        H = tgt @ ref.T = U S V^T ,
    the minimizer is
        R = V diag(1, 1, sign(det(V U^T))) U^T ,
    where the last factor forces det(R) = +1 (a physical rotation, no
    reflection).
    """
    H = tgt @ ref.T
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _rms_cost(ref, tgt, R):
    """Nearest-neighbour RMS residual of ``R @ tgt`` against ``ref``."""
    _, idx = cKDTree(ref.T).query((R @ tgt).T)
    diff = R @ tgt - ref[:, idx]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=0))))


def icp_align(ref, tgt, R0, max_iter=100, tol=1e-12):
    """Iterative Closest Point: refine ``R0`` so that ``R @ tgt ~ ref``.

    Correspondence is re-established each step by nearest-neighbour search and
    the optimal rotation is then solved in closed form with the Kabsch step.
    """
    R = np.asarray(R0, dtype=float)
    for _ in range(max_iter):
        _, idx = cKDTree(ref.T).query((R @ tgt).T)
        R_new = kabsch_rotation(ref[:, idx], tgt)
        if np.linalg.norm(R_new - R) < tol:
            return R_new
        R = R_new
    return R


def _pca_init(ref, tgt):
    """Coarse rotation from principal-axis alignment (ICP initialization).

    Aligns the principal axes (eigenvectors of X @ X.T) of ``tgt`` onto those
    of ``ref``, trying every sign combination and keeping the lowest RMS. This
    gives ICP a good start when the static rotation is large.
    """
    def axes(X):
        _, V = np.linalg.eigh(X @ X.T)      # ascending eigenvalues
        return V[:, ::-1]                   # descending

    Aa = axes(ref)
    Ab = axes(tgt)
    best = None
    sign_combos = [(s1, s2, s3)
                   for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1)]
    for signs in sign_combos:
        S = np.diag(signs)
        R = Aa @ S @ Ab.T
        if np.linalg.det(R) < 0:            # keep a proper rotation
            R = Aa @ S @ np.diag([1.0, 1.0, -1.0]) @ Ab.T
        cost = _rms_cost(ref, tgt, R)
        if best is None or cost < best[1]:
            best = (R, cost)
    return best[0]


def _downsample(x, n):
    """Uniformly decimate a (3, N) array to at most ``n`` columns."""
    x = np.asarray(x, dtype=float)
    if x.shape[1] <= n:
        return x
    idx = np.round(np.linspace(0, x.shape[1] - 1, n)).astype(int)
    return x[:, idx]


def align_rotation(ref, tgt, n_pts=2000, n_restarts=8):
    """Find the static rotation aligning trajectory ``tgt`` onto ``ref``.

    The two trajectories trace the same closed loop up to a static SO(3)
    rotation but are *not* synchronized in time, so correspondence is recovered
    by ICP (nearest-neighbour + Kabsch) rather than by sample index. Robustness
    to large rotations comes from trying the identity, a PCA principal-axis
    guess, and a few random rotations, keeping the lowest RMS.

    Returns ``(R, rms)`` with ``R @ tgt ~ ref`` and ``det(R) = +1``.
    """
    ref_ds = _downsample(ref, n_pts)
    tgt_ds = _downsample(tgt, n_pts)

    inits = [np.eye(3), _pca_init(ref_ds, tgt_ds)]
    for _ in range(n_restarts):
        inits.append(Rotation.random().as_matrix())

    best = None
    for R0 in inits:
        R = icp_align(ref_ds, tgt_ds, R0)
        cost = _rms_cost(ref_ds, tgt_ds, R)
        if best is None or cost < best[1]:
            best = (R, cost)
    return best


# ================= Plotting =================
def _add_sphere(ax):
    """Draw the unit Poincaré sphere on a 3D axis."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    ax.plot_surface(np.outer(np.cos(u), np.sin(v)),
                    np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.ones_like(u), np.cos(v)),
                    color="lightblue", alpha=0.15, edgecolor="none")


def _plot_traj(ax, s, color, label):
    """Plot a (3, N) Stokes trajectory on axis ``ax``."""
    ax.plot(s[0], s[1], s[2], color=color, lw=1, alpha=0.9, label=label)


def _format_axes(ax, title):
    """Apply common labels/title/legend to a Poincaré axis."""
    ax.set_xlabel("s1")
    ax.set_ylabel("s2")
    ax.set_zlabel("s3")
    ax.set_title(title)
    ax.legend()
    ax.set_box_aspect([1, 1, 1])


def plot_poincare(s_scope, s_pm, s_pm_aligned=None):
    """Plot both trajectories on a Poincaré sphere.

    If ``s_pm_aligned`` is provided, a second panel shows the PM1000 trajectory
    after it has been rotated onto the scope trajectory.
    """
    two_panels = s_pm_aligned is not None
    fig = plt.figure(figsize=(9 * (2 if two_panels else 1), 8))

    ax = fig.add_subplot(1, 2 if two_panels else 1, 1, projection="3d")
    _add_sphere(ax)
    _plot_traj(ax, s_scope, "blue", f"Scope (C·D) [{s_scope.shape[1]} pts]")
    _plot_traj(ax, s_pm, "red", f"PM1000 [{s_pm.shape[1]} pts]")
    _format_axes(ax, "Poincaré trajectory: scope vs PM1000 (raw)")

    if two_panels:
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        _add_sphere(ax2)
        _plot_traj(ax2, s_scope, "blue", f"Scope (C·D) [{s_scope.shape[1]} pts]")
        _plot_traj(ax2, s_pm_aligned, "green",
                   f"PM1000 aligned [{s_pm_aligned.shape[1]} pts]")
        _format_axes(ax2, "Aligned (R · PM1000 → scope)")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved to {PLOT_FILE}")
    plt.show()


def main():
    scope = None
    try:
        C = np.loadtxt(CAL_FILE, delimiter=",")
        if C.shape != (4, 4):
            raise ValueError(f"Expected 4x4 calibration matrix, got {C.shape}")
        print(f"Loaded calibration matrix from {CAL_FILE}")

        scope, window_s, n_pts = init_scope()

        # --- Scope trajectory: one frozen acquisition ---
        scope.write("ACQUIRE:STATE STOP")
        D = read_scope_waveforms(scope)      # (4, N) volts
        scope.write("ACQUIRE:STATE RUN")

        S_scope = C @ D                       # (4, N) Stokes in uW
        # smooth the raw S1,S2,S3, then project onto the unit Poincaré sphere
        s_scope_raw = moving_average(S_scope[1:4], MA_WINDOW)
        s_scope_sm = s_scope_raw / (np.linalg.norm(s_scope_raw, axis=0) + 1e-12)
        print(f"Scope trajectory: {n_pts} points over {window_s*1e6:.1f} us")

        # --- PM1000 trajectory: load from file ---
        S_pm = load_pm_trajectory(PM_TRAJ_FILE)   # (4, M) Stokes in uW
        s_pm_raw = moving_average(S_pm[1:4], MA_WINDOW)
        s_pm_sm = s_pm_raw / (np.linalg.norm(s_pm_raw, axis=0) + 1e-12)
        print(f"PM1000 trajectory: {S_pm.shape[1]} points from {PM_TRAJ_FILE}")

        # --- Static rotation: align the PM1000 trajectory onto the scope ---
        R, rms = align_rotation(s_scope_sm, s_pm_sm)
        s_pm_aligned = R @ s_pm_sm
        angle_deg = np.degrees(np.arccos(
            np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
        print(f"Alignment rotation (PM1000 -> scope), RMS = {rms:.4f}:")
        print(R)
        print(f"Rotation angle = {angle_deg:.2f} deg")

        plot_poincare(s_scope_sm, s_pm_sm, s_pm_aligned)

    finally:
        if scope is not None:
            scope.close()
        print("Done.")


if __name__ == "__main__":
    main()
