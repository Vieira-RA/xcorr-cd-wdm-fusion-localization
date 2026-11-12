#!/usr/bin/env python3
"""Poincare-sphere comparison of two polarimeters over a wavelength sweep.

Reads the CSV written by ``sweep_trajectory.py`` (fixed input SOP, wavelength
swept), reconstructs both trajectories and renders two figures:

  before :  the two trajectories as each instrument reports them
  after  :  the FBG trajectory rotated onto the PM1000 one by a Kabsch/Umeyama
            fit (SVD, det = +1), with the recovered rotation and the residual

The FBG trajectory is C(lambda) @ D, where C comes either from a single matrix
(``--matrix``) or from a completed sweep directory (``--run-dir``, nearest
wavelength). Use the per-wavelength matrices for the honest picture; a single
matrix is what you get if you assume one calibration covers the band.

Usage
-----
    python plot_trajectory_alignment.py --csv trajectory_1535.csv \
        --run-dir runs/mwcal_20261112_153706 --outdir figures

    python plot_trajectory_alignment.py --csv trajectory.csv --matrix C.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------- geometry helpers
def unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0, norms, 1.0)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise angle between unit vectors, in degrees."""
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))


def rms_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(angle_deg(a, b) ** 2)))


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimising ||R @ source - target|| (rows are points, det R = +1)."""
    covariance = source.T @ target
    u, _, vt = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    return vt.T @ correction @ u.T


def axis_angle(rotation: np.ndarray) -> Tuple[float, np.ndarray]:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosine)))
    if angle < 1e-9:
        return 0.0, np.array([0.0, 0.0, 1.0])
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    return angle, axis / np.linalg.norm(axis)


# ------------------------------------------------------------------------ data load
def load_trajectory(path: str) -> Dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise SystemExit("empty trajectory file: {}".format(path))
    columns = rows[0].keys()
    data = {name: np.array([float(row[name]) for row in rows]) for name in columns}
    if not {"S1", "S2", "S3", "ch1", "ch2", "ch3", "ch4"} <= set(columns):
        raise SystemExit("{} needs S1..S3 and ch1..ch4 columns".format(path))
    return data


def matrix_for(wavelength: float, run_dir: Optional[Path], single: Optional[np.ndarray],
               tolerance: float) -> np.ndarray:
    if single is not None:
        return single
    if run_dir is None:
        raise SystemExit("give --matrix or --run-dir so C(lambda) is known")
    candidates = []
    for path in sorted(run_dir.glob("*/calibration_*nm.csv")):
        try:
            value = float(os.path.basename(str(path)).replace("calibration_", "").replace("nm.csv", ""))
        except ValueError:
            continue
        candidates.append((abs(value - wavelength), value, path))
    if not candidates:
        raise SystemExit("no calibration_*nm.csv under {}".format(run_dir))
    distance, value, path = min(candidates)
    if distance > tolerance:
        raise SystemExit("nearest matrix is {:.3f} nm, {:.2f} nm away (tolerance {:.2f})".format(
            value, distance, tolerance))
    return np.loadtxt(str(path), delimiter=",")


# -------------------------------------------------------------------------- plotting
def frame_rotation(points: np.ndarray) -> np.ndarray:
    """Rotation that brings the plane of ``points`` onto the screen plane (z normal).

    The SOP trajectory of a birefringent link is a circle about the PMD axis, so
    viewing along that axis turns the trajectory into a large circular arc that
    fills the sphere instead of a tiny segment lost in a corner.
    """
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred)
    normal = vt[-1]
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z)
    s = np.linalg.norm(v)
    c = float(normal @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - c) / s ** 2)


def _draw_sphere(ax) -> None:
    """Light wireframe unit sphere: meridians, parallels and the three great circles."""
    t = np.linspace(0.0, 2.0 * np.pi, 361)
    for elevation in range(-80, 90, 20):
        z = np.sin(np.radians(elevation))
        r = np.cos(np.radians(elevation))
        ax.plot(r * np.cos(t), r * np.sin(t), z * np.ones_like(t),
                color="#aebdd2", lw=0.45, alpha=0.55, zorder=1)
    for azimuth in range(0, 180, 20):
        phi = np.radians(azimuth)
        ax.plot(np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t),
                color="#aebdd2", lw=0.45, alpha=0.55, zorder=1)
    for xs, ys, zs in ((np.cos(t), np.sin(t), 0 * t), (np.cos(t), 0 * t, np.sin(t)),
                       (0 * t, np.cos(t), np.sin(t))):
        ax.plot(xs, ys, zs, color="#8fa1bd", lw=0.8, alpha=0.75, zorder=1)


def _style_axes(ax) -> None:
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_zlim(-1.02, 1.02)
    try:
        ax.set_box_aspect([1, 1, 1], zoom=1.35)
    except TypeError:            # matplotlib < 3.5 has no zoom argument
        ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_xlabel("$S_1'$", labelpad=-6, fontsize=11)
    ax.set_ylabel("$S_2'$", labelpad=-6, fontsize=11)
    ax.set_zlabel("$S_3'$", labelpad=-8, fontsize=11)
    ax.tick_params(labelsize=6, pad=-3)
    ax.set_xticks([-1, 1]); ax.set_yticks([-1, 1]); ax.set_zticks([-1, 1])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor("white")
        pane.pane.set_alpha(0.0)
    ax.view_init(elev=28, azim=-90)


def _render(ax, wavelengths, a, b, label_a, label_b, tie_step=1, colour_map="viridis"):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    norm = plt.Normalize(wavelengths.min(), wavelengths.max())
    cmap = plt.get_cmap(colour_map)
    colours = cmap(norm(wavelengths))
    _draw_sphere(ax)

    if tie_step:
        for i in range(0, len(wavelengths), max(1, tie_step)):
            ax.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], [a[i, 2], b[i, 2]],
                    color="0.55", lw=0.6, alpha=0.7, zorder=2)

    for i in range(len(wavelengths) - 1):
        ax.plot(a[i:i + 2, 0], a[i:i + 2, 1], a[i:i + 2, 2],
                color=colours[i], lw=2.6, alpha=0.95, zorder=4, solid_capstyle="round")
        ax.plot(b[i:i + 2, 0], b[i:i + 2, 1], b[i:i + 2, 2],
                color=colours[i], lw=1.8, alpha=0.95, zorder=5,
                linestyle=(0, (4, 2)))

    ax.scatter(*a.T, s=42, facecolor="white", edgecolor=colours, linewidth=1.2, zorder=6)
    ax.scatter(*b.T, s=26, marker="s", facecolor=colours, edgecolor="white",
               linewidth=0.5, zorder=7)
    ax.scatter(*a[0], s=150, marker="*", color="#c0392b", edgecolor="white",
               linewidth=0.7, zorder=8)

    handles = [
        Line2D([], [], color="#34495e", lw=2.6, marker="o", markerfacecolor="white",
               markersize=7, label=label_a),
        Line2D([], [], color="#34495e", lw=1.8, linestyle=(0, (4, 2)), marker="s",
               markersize=5, label=label_b),
        Line2D([], [], color="#c0392b", lw=0, marker="*", markersize=12,
               label="$\\lambda$ = {:.3f} nm".format(wavelengths[0])),
    ]
    return handles, plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="trajectory.csv")
    parser.add_argument("--run-dir", default=None, help="sweep with calibration_*nm.csv")
    parser.add_argument("--matrix", default=None, help="single 4x4 matrix to use for all lambda")
    parser.add_argument("--matrix-tolerance", type=float, default=0.51)
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_trajectory(args.csv)
    wavelengths = data["wavelength_nm"]
    reference = unit(np.column_stack([data["S1"], data["S2"], data["S3"]]))
    detectors = np.column_stack([data["ch1"], data["ch2"], data["ch3"], data["ch4"]])

    single = np.loadtxt(args.matrix, delimiter=",") if args.matrix else None
    run_dir = Path(args.run_dir) if args.run_dir else None
    stokes_dut = []
    for wavelength, detector in zip(wavelengths, detectors):
        matrix = matrix_for(float(wavelength), run_dir, single, args.matrix_tolerance)
        stokes_dut.append((matrix @ detector)[1:4])
    measurement = unit(np.array(stokes_dut))

    view = frame_rotation(reference)              # screen frame: along the PMD axis
    reference = reference @ view.T
    measurement = measurement @ view.T

    before = rms_deg(reference, measurement)
    rotation = kabsch(measurement, reference)
    aligned = measurement @ rotation.T
    after = rms_deg(reference, aligned)
    angle, axis = axis_angle(rotation)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    span = "{:.2f}-{:.2f} nm".format(wavelengths.min(), wavelengths.max())

    fig = plt.figure(figsize=(11.0, 9.6))
    ax = fig.add_axes([0.01, 0.02, 0.80, 0.86], projection="3d")
    cax = fig.add_axes([0.855, 0.24, 0.022, 0.50])
    handles, mappable = _render(ax, wavelengths, reference, measurement,
                                "PM1000 (reference)", "FBG polarimeter, unaligned", tie_step=4)
    _style_axes(ax)
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, framealpha=0.92)
    bar = fig.colorbar(mappable, cax=cax)
    bar.set_label("wavelength [nm]", fontsize=10)
    ax.set_title("Before alignment — two polarimeters, one fixed SOP\n"
                 "{}   |   RMS angular difference {:.1f}$^\\circ$".format(span, before),
                 fontsize=13, pad=14)
    for name in ("before",):
        fig.savefig(str(outdir / "trajectory_{}.png".format(name)), dpi=args.dpi)
    plt.close(fig)

    fig = plt.figure(figsize=(11.0, 9.6))
    ax = fig.add_axes([0.01, 0.02, 0.80, 0.86], projection="3d")
    cax = fig.add_axes([0.855, 0.24, 0.022, 0.50])
    handles, mappable = _render(ax, wavelengths, reference, aligned,
                                "PM1000 (reference)",
                                "FBG polarimeter, Kabsch-aligned", tie_step=4)
    _style_axes(ax)
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, framealpha=0.92)
    bar = fig.colorbar(mappable, cax=cax)
    bar.set_label("wavelength [nm]", fontsize=10)
    ax.set_title("After Kabsch alignment — the same two trajectories in one frame\n"
                 "rotation {:.1f}$^\\circ$ about ({:+.2f}, {:+.2f}, {:+.2f})   |   "
                 "RMS {:.1f}$^\\circ \\rightarrow$ {:.1f}$^\\circ$".format(
                     angle, *axis, before, after),
                 fontsize=13, pad=14)
    fig.savefig(str(outdir / "trajectory_after.png"), dpi=args.dpi)
    plt.close(fig)

    print("points          : {}".format(len(wavelengths)))
    print("RMS before      : {:.2f} deg".format(before))
    print("Kabsch rotation : {:.1f} deg about ({:+.3f}, {:+.3f}, {:+.3f})".format(angle, *axis))
    print("RMS after       : {:.2f} deg".format(after))
    print("figures         : {}, {}".format(outdir / "trajectory_before.png",
                                           outdir / "trajectory_after.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
