#!/usr/bin/env python
"""
run_final_calibration.py

Performs the final reference‑free calibration after determining the correct detector order.
Uses the best permutation from correct_detector_order.py.
Assumes constant power = 1.328 mW.

Outputs:
- Calibration matrix (4x4) as .npy and .txt
- Validation metrics (mean/std DOP, power recovery)
- 2x2 validation plot
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the calibration functions from your shared library
from lightera_POL_calibration import calibrate_polarimeter

# ------------------- CONFIGURATION -------------------
DATA_DIR = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/raw/lightera_POL_calibration_traces/fast_Polarimeter_Callbration/"
OUTPUT_DIR = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/"
PERMUTATION_FILE = os.path.join(OUTPUT_DIR, "best_permutation.npy")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONSTANT_POWER = 1.328  # mW
FIXED_INDEX = 0          # We know Ch1 = D0

# ------------------- CUSTOM CSV PARSER (same as before) -------------------
def parse_voltage_from_csv(filename: str) -> float:
    """
    Reads a CSV file, finds the first data row with exactly two numeric fields,
    and returns the average of the second column (voltage) from all subsequent rows.
    Handles leading commas, empty fields, quoted empty strings, and scientific notation.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the first line that contains exactly two numeric values (after removing empties)
    start_idx = None
    for i, line in enumerate(lines):
        fields = [field.strip().strip('"') for field in line.strip().split(',')]
        fields = [f for f in fields if f != '']
        if len(fields) == 2:
            try:
                float(fields[0])  # time
                float(fields[1])  # voltage
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

# ------------------- LOAD DATA -------------------
def load_sop_data(data_dir: str) -> np.ndarray:
    """
    Loads all SOP CSV files and returns a (4, N) matrix.
    Uses the custom parser to extract average voltage per channel.
    """
    file_pattern = os.path.join(data_dir, "SOP*_Ch*.csv")
    file_list = glob.glob(file_pattern)

    if not file_list:
        raise FileNotFoundError(f"No files found matching pattern in {data_dir}")

    # Group files by SOP number
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
    print("=== Final Polarimeter Calibration ===")
    print(f"Constant power set to {CONSTANT_POWER} mW")

    # 1. Load raw data (order: Ch1, Ch2, Ch3, Ch4)
    D_matrix = load_sop_data(DATA_DIR)
    N_sop = D_matrix.shape[1]

    # 2. Apply the best detector permutation (if available)
    if os.path.exists(PERMUTATION_FILE):
        best_perm = tuple(np.load(PERMUTATION_FILE).tolist())
        print(f"\nApplying saved permutation: {best_perm}")
        print(f"  => D0 = Ch{best_perm[0]+1}, D1 = Ch{best_perm[1]+1}, D2 = Ch{best_perm[2]+1}, D3 = Ch{best_perm[3]+1}")
        D_matrix = D_matrix[best_perm, :]  # Reorder rows
    else:
        print("\n⚠️ No permutation file found. Assuming order is already correct (Ch1=D0, Ch2=D1, Ch3=D2, Ch4=D3).")

    # 3. Run the full calibration
    print("\nRunning reference‑free calibration...")
    C_matrix, final_dop = calibrate_polarimeter(
        D_matrix,
        constant_power_value=CONSTANT_POWER,
        reg_weight=0.01   # Try 0.01 first; adjust if needed
    )
    # 4. Compute Stokes vectors
    S = C_matrix @ D_matrix
    S0, S1, S2, S3 = S[0, :], S[1, :], S[2, :], S[3, :]

    # 5. Compute statistics
    dop_mean = np.mean(final_dop)
    dop_std  = np.std(final_dop)
    dop_min  = np.min(final_dop)
    dop_max  = np.max(final_dop)

    s0_mean = np.mean(S0)
    s0_std  = np.std(S0)
    s0_min  = np.min(S0)
    s0_max  = np.max(S0)

    print("\n--- Calibration Validation Metrics ---")
    print(f"  Number of SOPs used: {N_sop}")
    print(f"  DOP Mean:   {dop_mean:.6f}")
    print(f"  DOP Std:    {dop_std:.6f}   (target < 0.01)")
    print(f"  DOP Min:    {dop_min:.6f}")
    print(f"  DOP Max:    {dop_max:.6f}")
    print(f"  Power (S0) Mean: {s0_mean:.6f} mW (expected {CONSTANT_POWER})")
    print(f"  Power (S0) Std:  {s0_std:.6f}")

    # 6. Save calibration matrix
    np.save(os.path.join(OUTPUT_DIR, "calibration_matrix.npy"), C_matrix)
    np.savetxt(os.path.join(OUTPUT_DIR, "calibration_matrix.txt"), C_matrix, fmt='%.8f')

    # 7. Save metrics to a text file
    with open(os.path.join(OUTPUT_DIR, "calibration_metrics.txt"), 'w') as f:
        f.write("Final Calibration Metrics (Reference‑Free)\n")
        f.write("===========================================\n")
        f.write(f"Constant power: {CONSTANT_POWER} mW\n")
        f.write(f"Number of SOPs: {N_sop}\n")
        if os.path.exists(PERMUTATION_FILE):
            f.write(f"Applied permutation: {best_perm}\n")
        else:
            f.write("No permutation applied (used default order).\n")
        f.write(f"DOP Mean:   {dop_mean:.6f}\n")
        f.write(f"DOP Std:    {dop_std:.6f}\n")
        f.write(f"DOP Min:    {dop_min:.6f}\n")
        f.write(f"DOP Max:    {dop_max:.6f}\n")
        f.write(f"\nPower (S0) Recovery:\n")
        f.write(f"  Mean: {s0_mean:.6f} mW\n")
        f.write(f"  Std:  {s0_std:.6f} mW\n")
        f.write(f"  Min:  {s0_min:.6f} mW\n")
        f.write(f"  Max:  {s0_max:.6f} mW\n")
        f.write("\nCalibration Matrix (4x4):\n")
        np.savetxt(f, C_matrix, fmt='%.8f')

    # 8. Plot validation figures (2x2 grid)
    fig = plt.figure(figsize=(14, 12))

    # Plot 1: DOP per SOP
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(range(N_sop), final_dop, 'bo', markersize=6)
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Ideal DOP = 1')
    ax1.fill_between(range(N_sop), 0.99, 1.01, color='gray', alpha=0.15, label='±1% tolerance')
    ax1.set_xlabel('Calibration SOP Index')
    ax1.set_ylabel('Computed DOP')
    ax1.set_title(f'DOP per SOP (Std = {dop_std:.4f})')
    ax1.set_ylim([0.97, 1.03])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Histogram of DOP errors
    ax2 = fig.add_subplot(2, 2, 2)
    dop_errors = final_dop - 1.0
    ax2.hist(dop_errors, bins=15, edgecolor='black', alpha=0.7, color='blue')
    ax2.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    ax2.set_xlabel('DOP - 1')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of DOP Errors (Std = {dop_std:.4f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Recovered Power (S0)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(range(N_sop), S0, 'go', markersize=6)
    ax3.axhline(y=CONSTANT_POWER, color='r', linestyle='--', label=f'Ideal Power = {CONSTANT_POWER} mW')
    ax3.set_xlabel('Calibration SOP Index')
    ax3.set_ylabel('Recovered Power (S0) [mW]')
    ax3.set_title(f'Power Recovery (Mean = {s0_mean:.4f} mW, Std = {s0_std:.4f})')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Poincaré Sphere
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    # Normalize Stokes vectors for plotting on the sphere
    norm = np.sqrt(S1**2 + S2**2 + S3**2)
    s1_norm, s2_norm, s3_norm = S1/norm, S2/norm, S3/norm
    ax4.scatter(s1_norm, s2_norm, s3_norm, c='b', s=15, alpha=0.7)
    ax4.set_xlabel('S1')
    ax4.set_ylabel('S2')
    ax4.set_zlabel('S3')
    ax4.set_title('Calibrated SOPs on Poincaré Sphere')
    # Wireframe sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax4.plot_wireframe(x, y, z, color='gray', alpha=0.15)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "calibration_validation.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nValidation plot saved to: {plot_path}")
    plt.close(fig)  # Avoid displaying on headless server

    print("\n=== Calibration completed successfully ===")
    print(f"All results saved in: {OUTPUT_DIR}")