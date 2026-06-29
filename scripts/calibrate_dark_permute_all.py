#!/usr/bin/env python3
"""
Full polarimeter calibration with dark‑offset subtraction,
testing ALL permutations of the four detector channels.
No channel is assumed to be a power monitor; all are treated
as polarisation probes forming a tetrahedron.

Usage:
  1. Place dark trace in ../data/dark_measurement.csv
     (four columns of voltage samples, one per channel, no header)
  2. Place SOP data in ../data/SOP*.csv
  3. Run from paper repo root.
"""

import itertools
import numpy as np
from pathlib import Path
import csv, re

from lightera_POL_calibration import (
    fit_power_row,
    fit_polarization_rows,
    align_slow_axis,
    compute_dop_std,
)
from visualization import plot_poincare_sphere
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DARK_FILE = Path("../data/dark_measurement.csv")
DATA_DIR = Path("../data")          # folder containing SOP*_Ch*.csv files
OUTPUT_DIR = Path("../output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONSTANT_POWER_mW = 1.328           # total optical power (mW)
ALIGN_SLOW_AXIS = True              # align first channel of permuted order to S1

# ----------------------------------------------------------------------
# 1. Load dark offsets
# ----------------------------------------------------------------------
print(f"Loading dark offsets from {DARK_FILE}")
dark_data = np.genfromtxt(DARK_FILE, delimiter=',', skip_header=0)
if np.isnan(dark_data[0]).any():
    dark_data = np.genfromtxt(DARK_FILE, delimiter=',', skip_header=1)

if dark_data.ndim != 2 or dark_data.shape[1] != 4:
    raise ValueError(f"Dark file should have exactly 4 columns; got {dark_data.shape}")

dark_offsets = np.mean(dark_data, axis=0)   # [Ch1, Ch2, Ch3, Ch4]
print("Dark offsets (V):", dark_offsets)

# ----------------------------------------------------------------------
# 2. Load all SOP data
# ----------------------------------------------------------------------
pattern = re.compile(r"^SOP(\d+)_Ch1\.csv$")
sop_numbers = sorted(
    int(pattern.match(f.name).group(1))
    for f in DATA_DIR.glob("SOP*_Ch1.csv") if pattern.match(f.name)
)
N = len(sop_numbers)
print(f"Found {N} SOP measurements.")

def read_channel_mean(filepath):
    voltages = []
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0] == "" and row[1] == "":
                try:
                    voltages.append(float(row[-1]))
                except (ValueError, IndexError):
                    continue
    return np.mean(voltages)

D_raw = np.empty((4, N))
for idx, sop in enumerate(sop_numbers):
    for ch in range(1, 5):
        fpath = DATA_DIR / f"SOP{sop}_Ch{ch}.csv"
        D_raw[ch-1, idx] = read_channel_mean(fpath)

# ----------------------------------------------------------------------
# 3. Subtract dark offsets (per physical channel)
# ----------------------------------------------------------------------
D_dark_corrected = D_raw - dark_offsets.reshape(4, 1)
print("\nAfter dark subtraction:")
for i in range(4):
    print(f"  Ch{i+1}: mean = {np.mean(D_dark_corrected[i,:]):.4f} V, "
          f"std = {np.std(D_dark_corrected[i,:]):.4f} V")

# ----------------------------------------------------------------------
# 4. Test all 24 permutations of the four channels
# ----------------------------------------------------------------------
all_indices = [0, 1, 2, 3]
perms = list(itertools.permutations(all_indices))
print(f"\nTesting all {len(perms)} permutations of the four channels...")

best_sigma = np.inf
best_perm = None
best_C = None
results = []

for perm in perms:
    # Permute rows of the dark‑corrected data
    D_perm = D_dark_corrected[list(perm), :]

    # Stage 1: fit top row (constant power)
    S0 = np.full(N, CONSTANT_POWER_mW)
    c0 = fit_power_row(D_perm, S0)

    # Build tetrahedral guess for this channel order
    # The guess assumes the columns correspond to channels in order 0,1,2,3
    # We build the standard guess and then permute its columns to match.
    eta = np.mean(c0)
    C_guess_std = np.array([
        [4*c0[0], 4*c0[1], 4*c0[2], 4*c0[3]],
        [3*eta,   -eta,    -eta,    -eta],
        [0,        2*eta*np.sqrt(2), -eta*np.sqrt(2), -eta*np.sqrt(2)],
        [0,        0,        eta*np.sqrt(6), -eta*np.sqrt(6)]
    ]) / 4.0

    # The standard guess expects column i to correspond to channel i.
    # Since we permuted the detector rows, we need to apply the same
    # permutation to the columns of the guess matrix.
    # The guess matrix maps detector voltages D to Stokes S = C_guess D.
    # If we permute the rows of D as D_perm = P D, then we need
    # C_guess_perm = C_guess_std P^T to keep S unchanged.
    # (S = C_guess_perm D_perm = C_guess_std P^T P D = C_guess_std D)
    P = np.zeros((4,4))
    for i, j in enumerate(perm):
        P[i, j] = 1
    C_guess = C_guess_std @ P.T

    # Stage 2: fit full matrix
    try:
        C_full = fit_polarization_rows(c0, D_perm, C_guess=C_guess)
    except Exception as e:
        print(f"Permutation {perm} failed: {e}")
        continue

    # Align slow axis: the first channel in the permuted order is aligned to S1
    if ALIGN_SLOW_AXIS:
        C_aligned = align_slow_axis(C_full, detector_index=1)
    else:
        C_aligned = C_full

    sigma_dop = compute_dop_std(C_aligned, D_perm)

    results.append({
        'perm': perm,
        'sigma': sigma_dop,
        'C': C_aligned,
        'D_perm': D_perm
    })
    print(f"Permutation {perm}: σ_DOP = {sigma_dop:.4f}")

    if sigma_dop < best_sigma:
        best_sigma = sigma_dop
        best_perm = perm
        best_C = C_aligned

# ----------------------------------------------------------------------
# 5. Summary
# ----------------------------------------------------------------------
print("\n--- Best permutation ---")
print(f"Permutation {best_perm}: σ_DOP = {best_sigma:.4f}")
if best_sigma < 0.01:
    print("✓ σ_DOP < 1% — calibration successful.")
else:
    print("✗ σ_DOP > 1% — check data quality, dark offset accuracy, or scrambling range.")

np.save(OUTPUT_DIR / "C_calibrated_best_all_perms.npy", best_C)
print("Best calibration matrix saved.")

# ----------------------------------------------------------------------
# 6. Plot Poincaré spheres for a subset of best results
# ----------------------------------------------------------------------
# Sort results by sigma_dop and plot top 6 (or fewer)
results_sorted = sorted(results, key=lambda r: r['sigma'])[:6]
if results_sorted:
    n_plots = len(results_sorted)
    cols = min(3, n_plots)
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows),
                             subplot_kw={'projection': '3d'})
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, res in zip(axes, results_sorted):
        S = res['C'] @ res['D_perm']
        plot_poincare_sphere(S, ax=ax,
                             title=f"Perm {res['perm']}  σ={res['sigma']:.4f}",
                             color='C0', alpha=0.8, s=20)
        ax.quiver(0, 0, 0, 1.2, 0, 0, color='red', linewidth=1)

    for ax in axes[len(results_sorted):]:
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "permutations_all_dark_corrected.png", dpi=150)
    plt.close()
    print("Plot saved to ../output/permutations_all_dark_corrected.png")