#!/usr/bin/env python
"""
test_tetrahedral_projection.py

Tests the initial tetrahedral guess matrix on the raw data.
If the points are spread on the Poincaré sphere, the data is fine.
If they collapse, the data itself lacks polarization diversity.
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the helper functions (without the optimizer)
from lightera_POL_calibration import fit_power_row, generate_tetrahedral_guess

# ------------------- CONFIGURATION -------------------
DATA_DIR = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/raw/lightera_POL_calibration_traces/fast_Polarimeter_Callbration/"
OUTPUT_DIR = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/"
PERMUTATION_FILE = os.path.join(OUTPUT_DIR, "best_permutation.npy")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONSTANT_POWER = 1.328  # mW

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
    print("=== Testing Tetrahedral Guess on Raw Data ===")
    print(f"Constant power set to {CONSTANT_POWER} mW")

    # 1. Load data
    D_matrix = load_sop_data(DATA_DIR)
    N_sop = D_matrix.shape[1]

    # 2. Apply best permutation if available
    if os.path.exists(PERMUTATION_FILE):
        best_perm = tuple(np.load(PERMUTATION_FILE).tolist())
        print(f"\nApplying saved permutation: {best_perm}")
        D_matrix = D_matrix[best_perm, :]
    else:
        print("\n⚠️ No permutation file found. Using default order.")

    # 3. Compute the initial tetrahedral guess (without any optimization)
    C0 = fit_power_row(D_matrix, target_power=CONSTANT_POWER)
    C_guess = generate_tetrahedral_guess(C0)

    print("\nInitial Tetrahedral Guess Matrix:")
    print(C_guess)

    # 4. Compute Stokes vectors using ONLY the guess
    S_guess = C_guess @ D_matrix
    S0, S1, S2, S3 = S_guess[0, :], S_guess[1, :], S_guess[2, :], S_guess[3, :]

    # 5. Compute DOP using the guess (should be close to 1)
    dop_guess = np.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-12)
    dop_mean = np.mean(dop_guess)
    dop_std = np.std(dop_guess)

    print("\n--- Performance of Tetrahedral Guess (No Optimization) ---")
    print(f"  DOP Mean: {dop_mean:.8f}")
    print(f"  DOP Std:  {dop_std:.8f}")

    # 6. Check the diversity of the Stokes vectors
    var_s1 = np.var(S1)
    var_s2 = np.var(S2)
    var_s3 = np.var(S3)
    print(f"\nVariance of Stokes components (indicating spread):")
    print(f"  Var(S1) = {var_s1:.6f}")
    print(f"  Var(S2) = {var_s2:.6f}")
    print(f"  Var(S3) = {var_s3:.6f}")

    if var_s1 < 1e-6 and var_s2 < 1e-6 and var_s3 < 1e-6:
        print("\n⚠️ WARNING: Variance is extremely low! The data itself has NO polarization diversity.")
        print("   → The calibration will collapse regardless of the optimizer.")
        print("   → You need to generate more varied SOPs during data collection.")
    else:
        print("\n✅ Variance is significant! The data contains different SOPs.")
        print("   → The optimizer degeneracy (flat DOP-only landscape) is the problem.")
        print("   → Adding regularization will solve it.")

    # 7. Plot the SOPs on the Poincaré sphere
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize for plotting on the unit sphere
    norm = np.sqrt(S1**2 + S2**2 + S3**2)
    s1_norm, s2_norm, s3_norm = S1/norm, S2/norm, S3/norm

    ax.scatter(s1_norm, s2_norm, s3_norm, c='blue', s=30, alpha=0.7, edgecolors='k')
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')
    ax.set_title(f'Tetrahedral Guess Projection\nDOP Std = {dop_std:.6f}')

    # Wireframe sphere for reference
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.15)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "tetrahedral_guess_projection.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    plt.show()  # If you have a display, otherwise it will just save.

    print("\n=== Test Complete ===")