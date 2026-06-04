#!/usr/bin/env python3
"""Check that propagate_unitary gives the exact result for constant β."""

import numpy as np
from fiber_propagation import propagate_unitary, sigma_x, sigma_y, sigma_z

# ---- Constant birefringence vector ----
beta_const = np.array([1.0, 0.5, 0.2])  # rad/m (arbitrary orientation)
L = 1.0                                  # fibre length [m]
N = 200                                  # number of steps

z = np.linspace(0, L, N + 1)
beta = np.tile(beta_const[:, None], (1, N + 1))  # (3, N+1)

# Numerical solution
U_num = propagate_unitary(z, beta)

# Analytical solution – constant β ⇒ closed form
theta = np.linalg.norm(beta_const) * L
n = beta_const / np.linalg.norm(beta_const)
n_dot_sigma = n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z
U_anal = (
    np.cos(theta / 2) * np.eye(2, dtype=complex)
    - 1j * np.sin(theta / 2) * n_dot_sigma
)

# ---- Checks ----
error = np.max(np.abs(U_num - U_anal))
unitary_residual = np.max(
    np.abs(U_num @ U_num.conj().T - np.eye(2))
)

print(f"Maximum absolute error vs. analytical: {error:.2e}")
print(f"Unitarity residual (max |U U† - I|): {unitary_residual:.2e}")

# Verify that the rotation angle is exactly θ
# Jones rotation angle φ satisfies |Tr U| = 2|cos(φ/2)|
phi = 2 * np.arccos(np.abs(np.trace(U_num)) / 2)
phi_expected = theta
print(f"Rotation angle: numerical={phi:.6f}, expected={phi_expected:.6f}")