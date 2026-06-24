#!/usr/bin/env python
"""
correct_detector_order.py

Determines the correct physical order of the 3 unknown detector channels (D1, D2, D3)
by brute‑force permutation search using the tetrahedral initial guess.
Assumes:
- Each SOP has 4 separate CSV files: SOP<number>_Ch<1-4>.csv
- Channel 1 (Ch1) corresponds to D0 (known).
- Optical power is constant = 1.328 mW.

Custom CSV parser that handles leading empty fields, quoted empty strings, and scientific notation.
"""

import numpy as np
import glob
import os
from lightera_POL_calibration import find_detector_order

# ------------------- CONFIGURATION -------------------
DATA_DIR = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/raw/lightera_POL_calibration_traces/fast_Polarimeter_Callbration/"
OUTPUT_DIR = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONSTANT_POWER = 1.328  # mW
FIXED_INDEX = 0          # Row 0 is Channel 1 (D0)

# ------------------- CUSTOM CSV PARSER -------------------
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
        # Split by comma, strip quotes and whitespace, remove empty strings
        fields = [field.strip().strip('"') for field in line.strip().split(',')]
        # Remove any empty fields (e.g., from leading commas)
        fields = [f for f in fields if f != '']
        # We need exactly two fields that are numeric
        if len(fields) == 2:
            try:
                float(fields[0])  # time
                float(fields[1])  # voltage
                # Both are numbers -> this is a data row
                start_idx = i
                break
            except ValueError:
                continue

    if start_idx is None:
        raise ValueError(f"No data rows found in {filename}")

    # Now collect all voltage values from the data rows
    voltages = []
    for line in lines[start_idx:]:
        fields = [field.strip().strip('"') for field in line.strip().split(',')]
        fields = [f for f in fields if f != '']
        # Data rows should have exactly two numeric fields
        if len(fields) == 2:
            try:
                volt = float(fields[1])  # second column is voltage
                voltages.append(volt)
            except ValueError:
                # If conversion fails, we've hit a non-data row (shouldn't happen)
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
    D_matrix = load_sop_data(DATA_DIR)

    best_perm, best_distance, all_results = find_detector_order(
        D_matrix,
        fixed_index=FIXED_INDEX,
        constant_power_value=CONSTANT_POWER
    )

    # Compute the AVERAGE distance
    num_sops = D_matrix.shape[1]
    best_avg_distance = best_distance / num_sops
    all_results_avg = {perm: dist / num_sops for perm, dist in all_results.items()}

    print("\n--- Detector Order Search Results ---")
    print("Average Euclidean Distance per SOP for each permutation:")
    for perm, avg_dist in sorted(all_results_avg.items(), key=lambda x: x[1]):
        print(f"  Order {perm}: avg distance = {avg_dist:.8f}")

    print(f"\n✅ Best permutation: {best_perm}")
    print(f"   Corresponds to: D0 = Ch{best_perm[0]+1}, D1 = Ch{best_perm[1]+1}, D2 = Ch{best_perm[2]+1}, D3 = Ch{best_perm[3]+1}")
    print(f"   Achieved minimum average distance = {best_avg_distance:.8f}")

    # Save results
    np.save(os.path.join(OUTPUT_DIR, "best_permutation.npy"), np.array(best_perm))

    with open(os.path.join(OUTPUT_DIR, "detector_order_results.txt"), 'w') as f:
        f.write("Detector Order Search Results (Average Distance Metric)\n")
        f.write("=======================================================\n")
        f.write(f"Constant power: {CONSTANT_POWER} mW\n")
        f.write(f"Fixed D0 index: {FIXED_INDEX} (Channel {FIXED_INDEX+1})\n")
        f.write(f"Number of SOPs: {num_sops}\n")
        f.write(f"Best permutation: {best_perm}\n")
        f.write(f"Best average distance: {best_avg_distance:.8f}\n\n")
        f.write("All permutations (sorted by avg distance):\n")
        for perm, avg_dist in sorted(all_results_avg.items(), key=lambda x: x[1]):
            f.write(f"  {perm}: {avg_dist:.8f}\n")

    print(f"\nResults saved to {OUTPUT_DIR}")