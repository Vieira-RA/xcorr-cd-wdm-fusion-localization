#!/usr/bin/env python3
"""Demonstrate quaternion sign regularization on a time series of
Jones matrices with artificially inserted sign flips."""

import numpy as np
import matplotlib.pyplot as plt
from fiber_propagation import propagate_unitary
from quaternion import jones_to_quaternion, quaternion_to_jones, regularize_signs

# ---------- Generate a continuous time series of U'(t) ----------
beta_const = np.array([1.0, 0.5, 0.2])
L = 1.0
z = np.linspace(0, L, 200)
t_samples = np.linspace(0, 1, 100)  # arbitrary time axis

U_clean = []
for t in t_samples:
    beta_t = np.tile(beta_const[:, np.newaxis], (1, len(z)))  # static for simplicity
    U = propagate_unitary(z, beta_t)
    U_clean.append(U)
U_clean = np.array(U_clean)

# Convert to quaternions
Q_clean = np.array([jones_to_quaternion(U) for U in U_clean])

# ---------- Introduce random sign flips ----------
rng = np.random.default_rng(42)
flip = rng.choice([1, -1], size=len(U_clean), p=[0.7, 0.3])  # 30% flips
U_flipped = flip[:, np.newaxis, np.newaxis] * U_clean
Q_flipped = np.array([jones_to_quaternion(U) for U in U_flipped])

# ---------- Regularize ----------
Q_reg = regularize_signs(Q_flipped)

# ---------- Extract rotation angle φ(t) ----------
def rotation_angle_from_quaternion(q):
    """φ = 2 arccos(q0) for a unit quaternion."""
    q0 = np.clip(q[0], -1.0, 1.0)
    return 2 * np.arccos(q0)

phi_clean = np.array([rotation_angle_from_quaternion(q) for q in Q_clean])
phi_flipped = np.array([rotation_angle_from_quaternion(q) for q in Q_flipped])
phi_reg = np.array([rotation_angle_from_quaternion(q) for q in Q_reg])

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(t_samples, phi_clean, 'k-', linewidth=2, label='Clean')
ax1.plot(t_samples, phi_flipped, 'r.', markersize=3, alpha=0.5, label='With sign flips')
ax1.set_ylabel('Rotation angle φ [rad]')
ax1.set_title('Effect of sign flips before regularization')
ax1.legend()
ax1.grid(True)

ax2.plot(t_samples, phi_clean, 'k-', linewidth=2, label='Clean')
ax2.plot(t_samples, phi_reg, 'b--', linewidth=1.5, label='Regularized')
ax2.set_xlabel('Time index')
ax2.set_ylabel('Rotation angle φ [rad]')
ax2.set_title('After quaternion sign regularization')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('../output/regularization_demo.png', dpi=150)
plt.show()