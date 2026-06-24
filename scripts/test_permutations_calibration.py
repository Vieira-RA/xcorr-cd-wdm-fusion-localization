#!/usr/bin/env python
"""
test_permutations_calibration.py

Runs the full calibration for each possible detector order (with D0 fixed)
and plots the resulting Stokes vectors on the Poincaré sphere.
This helps identify whether any permutation naturally avoids the collapse
without the need for regularization.
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

# Import your calibration functions (without regularization)
from lightera_POL_calibration import calibrate_polarimeter

# ------------------- CONFIGURATION -------------------
DATA_DIR = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/raw/lightera_POL_calibration_traces/fast_Polarimeter_Callbration/"
OUTPUT_DIR = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONSTANT_POWER = 1.328  # mW
FIXED_INDEX = 0          # We know Ch1 = D0

# ------------------- CSV PARSER (same as before) -------------------
def parse_voltage_from_csv(filename: str) -> float:
    with open(filename, 'r') as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        fields = [field.strip().strip('"') for field in line.strip().split(',')]
        fields = [f for f in fields if f != '']
        if len(fields) == 2:
            try:
                float(fields[0])
                float(fields[1])
                start_idx = i
                break
            except ValueError:
                continue
    if start_idx is None:
        raise ValueError(f"No data rows found in {filename}")
    voltages = []
    for line in lines[start_idx:]:
        fields = [field.strip().strip('"') for field in line.strip().split(',')]
        fields = [f for f in fields if f != '']
        if len(fields) == 2:
            try:
                volt = float(fields[1])
                voltages.append(volt)
            except ValueError:
                continue
    if not voltages:
        raise ValueError(f"No voltage data extracted from {filename}")
    return np.mean(voltages)

def load_sop_data(data_dir: str) -> np.ndarray:
    file_pattern = os.path.join(data_dir, "SOP*_Ch*.csv")
    file_list = glob.glob(file_pattern)
    if not file_list:
        raise FileNotFoundError(f"No files found matching pattern in {data_dir}")
    sop_dict = {}
    for f in file_list:
        basename = os.path.basename(f)
        parts = basename.replace('.csv', '').split('_')
        sop_num = int(parts[0].replace('SOP', ''))
        ch_num = int(parts[1].replace('Ch', ''))
        if sop_num not in sop_dict:
            sop_dict[sop_num] = {}
        sop_dict[sop_num][ch_num] = f
    sop_numbers = sorted(sop_dict.keys())
    print(f"Found {len(sop_numbers)} SOP states.")
    avg_vectors = []
    for sop_num in sop_numbers:
        ch_files = [sop_dict[sop_num][ch] for ch in sorted(sop_dict[sop_num].keys())]
        if len(ch_files) != 4:
            print(f"Warning: SOP {sop_num} has only {len(ch_files)} files, skipping.")
            continue
        channel_avgs = []
        for ch_file in ch_files:
            avg_val = parse_voltage_from_csv(ch_file)
            channel_avgs.append(avg_val)
        avg_vectors.append(channel_avgs)
    D_matrix = np.array(avg_vectors).T
    print(f"D_matrix shape: {D_matrix.shape}")
    return D_matrix

# ------------------- MAIN -------------------
if __name__ == "__main__":
    D_matrix = load_sop_data(DATA_DIR)
    N_sop = D_matrix.shape[1]
    print(f"Number of SOPs: {N_sop}")

    # Generate all permutations where row 0 is fixed (D0)
    other_indices = [1, 2, 3]
    all_perms = []
    for perm_tuple in itertools.permutations(other_indices):
        perm = (FIXED_INDEX,) + perm_tuple
        all_perms.append(perm)

    # We'll store results for each permutation
    results = {}

    # For each permutation, run calibration and store Stokes vectors and metrics
    for perm in all_perms:
        print(f"\nTesting permutation {perm} ...")
        D_perm = D_matrix[perm, :]  # reorder rows

        # Run calibration without regularization
        C_mat, dop = calibrate_polarimeter(D_perm, constant_power_value=CONSTANT_POWER)

        # Compute Stokes vectors
        S = C_mat @ D_perm
        S0, S1, S2, S3 = S[0, :], S[1, :], S[2, :], S[3, :]

        # Metrics
        dop_mean = np.mean(dop)
        dop_std = np.std(dop)
        s0_mean = np.mean(S0)

        results[perm] = {
            'C': C_mat,
            'S': S,
            'dop': dop,
            'dop_mean': dop_mean,
            'dop_std': dop_std,
            's0_mean': s0_mean,
            'S1': S1,
            'S2': S2,
            'S3': S3,
        }

        print(f"  DOP mean: {dop_mean:.6f}, std: {dop_std:.6f}, S0 mean: {s0_mean:.6f}")

    # ------------------- Plotting -------------------
    fig = plt.figure(figsize=(15, 10))
    plot_idx = 1

    for perm, data in results.items():
        ax = fig.add_subplot(2, 3, plot_idx, projection='3d')
        S1, S2, S3 = data['S1'], data['S2'], data['S3']
        # Normalize for sphere plotting
        norm = np.sqrt(S1**2 + S2**2 + S3**2)
        s1_norm, s2_norm, s3_norm = S1/norm, S2/norm, S3/norm
        ax.scatter(s1_norm, s2_norm, s3_norm, c='b', s=10, alpha=0.7)
        ax.set_xlabel('S1')
        ax.set_ylabel('S2')
        ax.set_zlabel('S3')
        ax.set_title(f'Perm {perm}\nDOP std = {data["dop_std"]:.4f}')
        # Wireframe sphere for reference
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "permutation_calibration_sphere.png"), dpi=150)
    print(f"\nPlot saved to {OUTPUT_DIR}/permutation_calibration_sphere.png")

    # Print a summary table
    print("\n=== Summary of Permutation Calibrations ===")
    print("Permutation | DOP mean | DOP std | S0 mean")
    print("--------------------------------------------")
    for perm in sorted(results.keys()):
        d = results[perm]
        print(f"  {perm}    | {d['dop_mean']:.6f} | {d['dop_std']:.6f} | {d['s0_mean']:.6f}")