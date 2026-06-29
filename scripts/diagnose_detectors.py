#!/usr/bin/env python3
"""
Diagnose the raw detector data: check if D1, D2, D3 span a 3D space.
"""

import numpy as np
from pathlib import Path
import re, csv
from scipy.stats import pearsonr

# ----------------------------------------------------------------------
DATA_DIR = Path("/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/raw/"
                "lightera_POL_calibration_traces/fast_Polarimeter_Callbration/")

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

D = np.empty((4, N))
for idx, sop in enumerate(sop_numbers):
    for ch in range(1, 5):
        fpath = DATA_DIR / f"SOP{sop}_Ch{ch}.csv"
        D[ch-1, idx] = read_channel_mean(fpath)

print("\n--- Basic statistics ---")
for i in range(4):
    print(f"Ch{i+1}: mean={np.mean(D[i,:]):.6f} V, "
          f"std={np.std(D[i,:]):.6f} V, "
          f"min={np.min(D[i,:]):.6f} V, "
          f"max={np.max(D[i,:]):.6f} V")

print("\n--- Pairwise Pearson correlations ---")
for i in range(4):
    for j in range(i+1, 4):
        r, _ = pearsonr(D[i,:], D[j,:])
        print(f"Ch{i+1} vs Ch{j+1}: r = {r:.4f}")

# Check rank of covariance matrix of D1-D3
# If the effective rank is 1, the channels are essentially identical.
cov = np.cov(D[1:4, :])   # 3x3 covariance of D1,D2,D3
eigvals = np.linalg.eigvalsh(cov)
print("\nEigenvalues of D1-D3 covariance matrix:")
print(eigvals)
print(f"Condition number: {eigvals[-1]/eigvals[0]:.2e}")

# Plot time series to see if they track each other
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes = axes.flatten()
for i in range(4):
    ax = axes[i]
    ax.plot(range(N), D[i,:], 'o-', markersize=4)
    ax.set_ylabel(f"Ch{i+1} [V]")
    ax.set_xlabel("SOP index")
    ax.set_title(f"Channel {i+1}")
plt.tight_layout()
fig.savefig("../output/diagnostic_channels.png", dpi=150)
plt.close()
print("\nPlot saved to ../output/diagnostic_channels.png")