#!/usr/bin/env python
"""
validate_calibration.py

Validates the reference-free calibration routine using synthetic data.
Generates noisy detector readings, runs the calibration, and checks:
- DOP standard deviation (should be tiny).
- Recovery of the true Stokes vectors (up to a global rotation).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Ensure the shared library is importable (assuming it's installed in editable mode)
from lightera_POL_calibration import (
    generate_synthetic_calibration_data,
    calibrate_polarimeter,
    dop_residuals
)

# ------------------- CONFIGURATION -------------------
OUTPUT_DIR = "../output/validation/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulation parameters
NUM_SOP = 30          # Number of random SOPs to generate (paper says ~20 is enough)
NOISE_STD = 5e-3      # Realistic noise level for photodetectors (approx 0.5% of max signal)
SEED = 42             # For reproducibility

# ------------------- RUN SIMULATION -------------------
print(f"Generating {NUM_SOP} synthetic SOPs with noise std = {NOISE_STD:.4f}...")
D_matrix, C_true, S_true = generate_synthetic_calibration_data(
    num_sop=NUM_SOP, 
    noise_std=NOISE_STD, 
    seed=SEED
)

print(f"D_matrix shape: {D_matrix.shape} (should be 4 x {NUM_SOP})")

# ------------------- RUN CALIBRATION -------------------
print("\nRunning reference-free calibration...")
C_est, final_dop = calibrate_polarimeter(D_matrix, constant_power=True)

# Compute estimated Stokes vectors using the calibrated matrix
S_est = C_est @ D_matrix

# ------------------- VALIDATION METRICS -------------------
# Metric 1: DOP standard deviation (primary internal metric from the paper)
dop_mean = np.mean(final_dop)
dop_std = np.std(final_dop)
print("\n--- Calibration Quality Metrics ---")
print(f"DOP Mean:   {dop_mean:.6f}")
print(f"DOP Std:    {dop_std:.6f}  (Should be < 0.01 for 1% specification)")
print(f"DOP Max:    {np.max(final_dop):.6f}")
print(f"DOP Min:    {np.min(final_dop):.6f}")

# Metric 2: Check if power was correctly recovered (should be ~1.0)
S0_est = S_est[0, :]
print(f"\nRecovered Power (S0) Mean: {np.mean(S0_est):.6f} (should be 1.0)")
print(f"Recovered Power (S0) Std:  {np.std(S0_est):.6f}")

# Metric 3: Angular deviation from true SOP (up to a global Stokes rotation).
# Since C_est is defined up to a rotation, we find the best rotation matrix (R)
# that aligns the estimated S1,S2,S3 to the true ones.
s1_true, s2_true, s3_true = S_true[1, :], S_true[2, :], S_true[3, :]
s1_est, s2_est, s3_est = S_est[1, :], S_est[2, :], S_est[3, :]

# For 3D vectors, we can solve for the rotation matrix using Procrustes analysis.
# However, since the calibration internally defines S1 as the axis of Detector 1,
# and our C_true aligns with that, the rotation should be minimal.
# We'll just compute the angular error per point.
dot_product = s1_true * s1_est + s2_true * s2_est + s3_true * s3_est
norms_true = np.sqrt(s1_true**2 + s2_true**2 + s3_true**2)
norms_est = np.sqrt(s1_est**2 + s2_est**2 + s3_est**2)
cos_angle = dot_product / (norms_true * norms_est + 1e-12)
angles_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
angles_deg = np.degrees(angles_rad)

print(f"\nAngular Deviation from True SOP:")
print(f"  Mean: {np.mean(angles_deg):.3f} degrees")
print(f"  Std:  {np.std(angles_deg):.3f} degrees")
print(f"  Max:  {np.max(angles_deg):.3f} degrees")

# ------------------- PLOTTING -------------------
fig = plt.figure(figsize=(14, 10))

# Plot 1: DOP per calibration point (should be ~1.0 with small scatter)
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(range(NUM_SOP), final_dop, 'bo', markersize=6)
ax1.axhline(y=1.0, color='r', linestyle='--', label='Ideal DOP = 1')
ax1.set_xlabel('Calibration SOP Index')
ax1.set_ylabel('Computed DOP')
ax1.set_title(f'DOP Recovery (Std = {dop_std:.4f})')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0.98, 1.02])  # Zoom in on the relevant range

# Plot 2: Histogram of DOP errors
ax2 = fig.add_subplot(2, 2, 2)
ax2.hist(final_dop - 1.0, bins=15, edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='r', linestyle='--')
ax2.set_xlabel('DOP - 1')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of DOP Errors')
ax2.grid(True, alpha=0.3)

# Plot 3: Angular deviation histogram
ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(angles_deg, bins=15, edgecolor='black', alpha=0.7, color='green')
ax3.set_xlabel('Angular Deviation (degrees)')
ax3.set_ylabel('Frequency')
ax3.set_title('SOP Angular Recovery Error')
ax3.grid(True, alpha=0.3)

# Plot 4: 3D Poincare sphere with True (red) vs Estimated (blue) SOPs
# This visually confirms the rotation ambiguity (if any) - the shapes should match.
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.scatter(s1_true, s2_true, s3_true, c='red', s=20, label='True SOPs', alpha=0.6)
ax4.scatter(s1_est, s2_est, s3_est, c='blue', s=20, label='Estimated SOPs', alpha=0.6)
ax4.set_xlabel('S1')
ax4.set_ylabel('S2')
ax4.set_zlabel('S3')
ax4.set_title('Poincare Sphere: True vs Estimated SOPs')
ax4.legend()
# Draw a wireframe sphere for reference
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax4.plot_wireframe(x, y, z, color='gray', alpha=0.1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "calibration_validation_synthetic.png"), dpi=150)
print(f"\nValidation plot saved to: {OUTPUT_DIR}calibration_validation_synthetic.png")

# ------------------- SAVE NUMERICAL RESULTS -------------------
np.save(os.path.join(OUTPUT_DIR, "C_true.npy"), C_true)
np.save(os.path.join(OUTPUT_DIR, "C_estimated.npy"), C_est)
np.save(os.path.join(OUTPUT_DIR, "D_matrix_synthetic.npy"), D_matrix)

# Save metrics to a text file
with open(os.path.join(OUTPUT_DIR, "validation_metrics.txt"), 'w') as f:
    f.write("Synthetic Validation Metrics\n")
    f.write("============================\n")
    f.write(f"Number of SOPs: {NUM_SOP}\n")
    f.write(f"Noise Std: {NOISE_STD}\n")
    f.write(f"DOP Mean: {dop_mean:.6f}\n")
    f.write(f"DOP Std: {dop_std:.6f}\n")
    f.write(f"Power (S0) Mean: {np.mean(S0_est):.6f}\n")
    f.write(f"Angular Error Mean (deg): {np.mean(angles_deg):.3f}\n")
    f.write(f"Angular Error Std (deg): {np.std(angles_deg):.3f}\n")

print("\nValidation complete! Check the output folder for results.")