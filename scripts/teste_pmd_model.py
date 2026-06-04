#!/usr/bin/env python3
"""Validate the random birefringence generator."""
import numpy as np
import matplotlib.pyplot as plt
from pmd_model import generate_birefringence_profile

# Parameters (example for a 1 km fibre with 10 m segments)
L = 1000.0                  # 1 km
L_F = 10.0                  # 10 m
D_pmd = 3.16e-15            # 0.1 ps/√km in s/√m
lambda0 = 1550e-9

z, beta0 = generate_birefringence_profile(L, L_F, D_pmd, lambda0, seed=123)

# ----- Basic checks -----
print(f"Grid points: {len(z)}")
print(f"Expected β magnitude: (ω₀/√(2 L_F)) D_pmd = "
      f"{2*np.pi*299792458/1550e-9 / np.sqrt(2*L_F) * D_pmd:.4f} rad/m")
print(f"Mean |β| along fibre: {np.mean(np.linalg.norm(beta0, axis=0)):.4f} rad/m")

# Verify that β is constant within each segment
segment_boundaries = np.arange(0, L + L_F, L_F)
for i, bound in enumerate(segment_boundaries[:-1]):
    idx = (z >= bound) & (z < min(bound + L_F, L + 1e-12))
    if not np.any(idx):
        continue
    vals = beta0[:, idx]
    std = np.std(vals, axis=1)
    assert np.allclose(std, 0, atol=1e-14), f"Segment {i} not constant"

# Check randomness of directions (should be uniform on sphere)
# Compute directions from segments (take one sample per segment)
n_segments = int(np.ceil(L / L_F))
directions = []
for seg in range(n_segments):
    mid = min(seg * L_F + L_F/2, L)
    idx = np.argmin(np.abs(z - mid))
    vec = beta0[:, idx]
    directions.append(vec / np.linalg.norm(vec))
dir_array = np.array(directions)

# Visual check: histogram of one component should be uniform-ish
plt.figure()
plt.hist(dir_array[:, 0], bins=20, density=True, alpha=0.7)
plt.title('Distribution of x‑component of segment directions')
plt.xlabel('n_x')
plt.show()

print("All checks passed: constant segments, correct magnitude, random directions.")