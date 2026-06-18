# scripts/run_lightera_POL_calibration.py
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lightera_POL_calibration import calibrate_polarimeter  # This imports from your shared library

# ------------------- CONFIGURATION -------------------
DATA_DIR = "../data/raw/lightera_POL_calibration_traces/"  # Folder with your oscilloscope files
OUTPUT_DIR = "../output/lightera_POL_calibration/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File naming pattern: e.g., "sop_001.npy", "sop_002.npy", ...
FILE_PATTERN = "sop_*.npy"  

# ------------------- LOAD DATA -------------------
file_list = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))
print(f"Found {len(file_list)} calibration SOP files.")

if len(file_list) < 20:
    print("WARNING: Fewer than 20 SOPs detected. Paper suggests ~20 for robust convergence.")

avg_vectors = []  # Store the 4-element averaged vector for each SOP
for file_path in file_list:
    # Load the oscilloscope data. Shape: (4, n_time_samples)
    # Adjust the loader if you use .csv (e.g., np.loadtxt)
    data = np.load(file_path)  
    
    # Average over the time axis to get a single 4-element vector per SOP
    avg_vector = np.mean(data, axis=1)  # Shape: (4,)
    avg_vectors.append(avg_vector)

# Stack all vectors into a (4, N) matrix
D_matrix = np.stack(avg_vectors, axis=1)  # Shape: (4, N_sop)
print(f"Calibration matrix D has shape: {D_matrix.shape}")

# ================ NEW: SAVE THE RAW AVERAGED DATA ================
np.save(os.path.join(OUTPUT_DIR, "D_matrix_calibration.npy"), D_matrix)
print(f"Saved averaged D_matrix to {OUTPUT_DIR}D_matrix_calibration.npy")

# ------------------- RUN CALIBRATION -------------------
C_matrix, final_dop = calibrate_polarimeter(D_matrix, constant_power=True)

# Print the resulting calibration matrix
print("\n--- Calibration Matrix (4x4) ---")
print(np.array2string(C_matrix, precision=6, suppress_small=True))

# Compute recovered Stokes vectors
S = C_matrix @ D_matrix
S0, S1, S2, S3 = S[0, :], S[1, :], S[2, :], S[3, :]

# ------------------- STATISTICS -------------------
dop_stats = {
    'mean_dop': np.mean(final_dop),
    'std_dop': np.std(final_dop),
    'max_dop': np.max(final_dop),
    'min_dop': np.min(final_dop)
}
power_stats = {
    'mean_s0': np.mean(S0),
    'std_s0': np.std(S0),
    'max_s0': np.max(S0),
    'min_s0': np.min(S0)
}

print(f"\n--- Calibration Validation Metrics ---")
print(f"  DOP Mean:   {dop_stats['mean_dop']:.6f}")
print(f"  DOP Std:    {dop_stats['std_dop']:.6f}  (Should be < 0.01 for 1% spec)")
print(f"  DOP Min/Max:{dop_stats['min_dop']:.6f} / {dop_stats['max_dop']:.6f}")
print(f"  Power Mean: {power_stats['mean_s0']:.6f} (Should be ~1.0 if constant power)")

# ------------------- PLOTTING (2x2 GRID) -------------------
fig = plt.figure(figsize=(14, 12))

# Plot 1: DOP per calibration point
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(range(len(final_dop)), final_dop, 'bo', markersize=6)
ax1.axhline(y=1.0, color='r', linestyle='--', label='Ideal DOP = 1')
ax1.set_xlabel('Calibration SOP Index')
ax1.set_ylabel('Computed DOP')
ax1.set_title(f'DOP per SOP (Std = {dop_stats["std_dop"]:.4f})')
ax1.set_ylim([0.98, 1.02])  # Zoom in to see the error clearly
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Histogram of DOP errors (same as synthetic validation)
ax2 = fig.add_subplot(2, 2, 2)
dop_errors = final_dop - 1.0
ax2.hist(dop_errors, bins=15, edgecolor='black', alpha=0.7, color='blue')
ax2.axvline(x=0, color='r', linestyle='--', label='Zero Error')
ax2.set_xlabel('DOP - 1')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Distribution of DOP Errors (Std = {dop_stats["std_dop"]:.4f})')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Recovered Power (S0) per point - validates top row fit
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(range(len(S0)), S0, 'go', markersize=6)
ax3.axhline(y=1.0, color='r', linestyle='--', label='Ideal Power = 1')
ax3.set_xlabel('Calibration SOP Index')
ax3.set_ylabel('Recovered Power (S0)')
ax3.set_title(f'Power Recovery (Mean = {power_stats["mean_s0"]:.4f})')
ax3.set_ylim([0.98, 1.02])
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Stokes vectors on the Poincare sphere
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
# Normalize for plotting on the sphere
norm = np.sqrt(S1**2 + S2**2 + S3**2)
s1_norm, s2_norm, s3_norm = S1/norm, S2/norm, S3/norm
ax4.scatter(s1_norm, s2_norm, s3_norm, c='b', s=15, alpha=0.7)
ax4.set_xlabel('S1')
ax4.set_ylabel('S2')
ax4.set_zlabel('S3')
ax4.set_title('Calibrated SOPs on Poincare Sphere')
# Draw a wireframe sphere for reference
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax4.plot_wireframe(x, y, z, color='gray', alpha=0.15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "calibration_validation.png"), dpi=150)
print(f"\nFull validation plot saved to {OUTPUT_DIR}calibration_validation.png")

# ------------------- SAVE NUMERICAL RESULTS -------------------
# Save the calibration matrix
np.save(os.path.join(OUTPUT_DIR, "calibration_matrix.npy"), C_matrix)

# Save comprehensive metrics to a text file (matching synthetic script format)
with open(os.path.join(OUTPUT_DIR, "calibration_metrics.txt"), 'w') as f:
    f.write("Real Oscilloscope Calibration Metrics (Reference-Free)\n")
    f.write("=====================================================\n")
    f.write(f"Number of SOPs used: {len(file_list)}\n")
    f.write(f"DOP Mean:   {dop_stats['mean_dop']:.6f}\n")
    f.write(f"DOP Std:    {dop_stats['std_dop']:.6f}\n")
    f.write(f"DOP Min:    {dop_stats['min_dop']:.6f}\n")
    f.write(f"DOP Max:    {dop_stats['max_dop']:.6f}\n")
    f.write(f"\nPower (S0) Recovery (should be ~1.0):\n")
    f.write(f"  Mean: {power_stats['mean_s0']:.6f}\n")
    f.write(f"  Std:  {power_stats['std_s0']:.6f}\n")
    f.write(f"  Min:  {power_stats['min_s0']:.6f}\n")
    f.write(f"  Max:  {power_stats['max_s0']:.6f}\n")
    f.write("\nCalibration Matrix (4x4):\n")
    np.savetxt(f, C_matrix, fmt='%.8f')

print(f"Metrics and calibration matrix saved to {OUTPUT_DIR}")