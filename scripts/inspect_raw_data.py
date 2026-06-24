#!/usr/bin/env python
"""
inspect_raw_data.py

Diagnostic script to inspect the raw oscilloscope voltages.
Plots:
- Mean voltage per channel across all SOPs.
- Standard deviation per channel across all SOPs.
- A few sample waveforms to see the signal quality.
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# ------------------- CONFIGURATION -------------------
DATA_DIR = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/raw/lightera_POL_calibration_traces/fast_Polarimeter_Callbration/"
OUTPUT_DIR = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERMUTATION_FILE = os.path.join(OUTPUT_DIR, "best_permutation.npy")

# ------------------- CUSTOM CSV PARSER -------------------
def parse_voltage_from_csv(filename: str) -> np.ndarray:
    """Returns the FULL voltage array (not averaged) for inspection."""
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

    return np.array(voltages)

def load_sop_data_full(data_dir: str) -> dict:
    """Returns all data as a dict: {sop_num: {ch: voltage_array}}."""
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
        sop_dict[sop_num][ch_num] = parse_voltage_from_csv(f)

    return sop_dict

# ------------------- MAIN -------------------
if __name__ == "__main__":
    print("=== Raw Data Inspection ===")
    data_dict = load_sop_data_full(DATA_DIR)

    sop_numbers = sorted(data_dict.keys())
    print(f"Found {len(sop_numbers)} SOP states.")
    print(f"Each SOP has {len(data_dict[sop_numbers[0]])} channels.")
    print(f"Each channel has {len(data_dict[sop_numbers[0]][1])} samples.")

    # Compute statistics across SOPs
    sop_means = {}  # {ch: list_of_means}
    sop_stds = {}   # {ch: list_of_stds}
    sop_peak_to_peak = {}  # {ch: list_of_ptp}

    for ch in [1, 2, 3, 4]:
        means = []
        stds = []
        ptps = []
        for sop in sop_numbers:
            data = data_dict[sop][ch]
            means.append(np.mean(data))
            stds.append(np.std(data))
            ptps.append(np.max(data) - np.min(data))
        sop_means[ch] = np.array(means)
        sop_stds[ch] = np.array(stds)
        sop_peak_to_peak[ch] = np.array(ptps)

    # Apply permutation if available
    if os.path.exists(PERMUTATION_FILE):
        best_perm = tuple(np.load(PERMUTATION_FILE).tolist())
        print(f"\nApplied permutation: {best_perm}")
        print(f"  => D0 = Ch{best_perm[0]+1}, D1 = Ch{best_perm[1]+1}, D2 = Ch{best_perm[2]+1}, D3 = Ch{best_perm[3]+1}")
    else:
        best_perm = (0, 1, 2, 3)
        print("\nNo permutation file found. Using default order.")

    # Print summary statistics for the REORDERED channels
    print("\n--- Voltage Statistics (Averaged Over All SOPs) ---")
    for idx, ch in enumerate([1, 2, 3, 4]):
        mean_over_sops = np.mean(sop_means[ch])
        std_over_sops = np.std(sop_means[ch])
        mean_ptp = np.mean(sop_peak_to_peak[ch])
        print(f"  Ch{ch} (D{best_perm.index(ch-1)}):")
        print(f"    Mean voltage across SOPs: {mean_over_sops:.6f} V")
        print(f"    Std deviation across SOPs: {std_over_sops:.6f} V (this is the POLARIZATION signal!)")
        print(f"    Mean peak-to-peak within trace: {mean_ptp:.6f} V")

    # Plot the mean voltage per SOP for each channel
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, ch in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        ax.plot(sop_numbers, sop_means[ch], 'o-', markersize=6, label=f'Ch{ch}')
        ax.set_xlabel('SOP Number')
        ax.set_ylabel('Mean Voltage (V)')
        ax.set_title(f'Channel {ch} (D{best_perm.index(ch-1)})')
        ax.grid(True, alpha=0.3)
        # Also plot error bars (std within each trace)
        # ax.errorbar(sop_numbers, sop_means[ch], yerr=sop_stds[ch], fmt='o', capsize=3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "raw_voltage_means.png"), dpi=150)
    print(f"\nMean voltage plot saved to {OUTPUT_DIR}raw_voltage_means.png")

    # Plot a sample waveform for each channel (first SOP)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    sample_sop = sop_numbers[0]
    for idx, ch in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        data = data_dict[sample_sop][ch]
        time = np.arange(len(data)) * 1e-8  # Assuming 10 ns sample interval
        ax.plot(time[:1000], data[:1000], linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Ch{ch} (D{best_perm.index(ch-1)}) - First SOP')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_waveforms.png"), dpi=150)
    print(f"Sample waveforms saved to {OUTPUT_DIR}sample_waveforms.png")

    # Calculate the variation relative to the DC offset
    print("\n--- Relative Variation Analysis ---")
    for ch in [1, 2, 3, 4]:
        mean_val = np.mean(sop_means[ch])
        std_across_sops = np.std(sop_means[ch])
        relative_variation = (std_across_sops / mean_val) * 100
        print(f"  Ch{ch}: DC = {mean_val:.6f} V, Polarization variation = {std_across_sops:.6f} V ({relative_variation:.3f}%)")

    print("\n=== Inspection Complete ===")
    print("If the relative variation is < 0.1%, the signal is too weak for calibration.")
    print("You may need to:")
    print("  1. Increase the gain on the photodetectors.")
    print("  2. Use AC coupling on the oscilloscope (to remove DC offset).")
    print("  3. Use a larger polarization scrambling amplitude.")