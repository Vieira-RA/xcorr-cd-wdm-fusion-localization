#!/usr/bin/env python3
"""Effect of a time‑varying birefringence amplitude – with analytical validation.

β(z,t) = (1 + A·sin(ωt)) · β_const

We compute the Jones matrix numerically and compare the extracted rotation
angle against the exact analytical prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from fiber_propagation import propagate_unitary

# ---------- Physical parameters ----------
beta_const = np.array([1.0, 0.5, 0.2])   # rad/m
L = 1.0                                   # m
N_z = 300                                 # spatial steps

A = 0.15                                  # modulation amplitude
f = 2.0                                   # Hz
ω = 2 * np.pi * f
t_samples = np.linspace(0, 1 / f, 201)    # one period

# ---------- Spatial grid ----------
z = np.linspace(0, L, N_z + 1)

# ---------- Analytical reference ----------
beta0_norm = np.linalg.norm(beta_const)
phi0 = beta0_norm * L                     # static rotation angle
phi_analytical = phi0 * (1 + A * np.sin(ω * t_samples))

# ---------- Pre‑allocate ----------
phi_numerical = np.empty_like(t_samples)
S_out = np.empty((3, len(t_samples)))

# Fixed input Jones vector
J_in = np.array([1.0, 0.0], dtype=complex)

# ---------- Time loop ----------
for idx, t in enumerate(t_samples):
    e = A * np.sin(ω * t)
    beta_t = (1 + e) * beta_const[:, np.newaxis]   # (3,1)
    beta_t = np.tile(beta_t, (1, len(z)))          # (3, Nz+1)

    U = propagate_unitary(z, beta_t)

    # Numerical rotation angle from |trace|
    phi_numerical[idx] = 2 * np.arccos(np.abs(np.trace(U)) / 2.0)

    # Output Stokes
    J_out = U @ J_in
    S_out[0, idx] = np.abs(J_out[0])**2 - np.abs(J_out[1])**2
    S_out[1, idx] = 2 * np.real(J_out[0] * np.conj(J_out[1]))
    S_out[2, idx] = 2 * np.imag(J_out[0] * np.conj(J_out[1]))

# ---------- Validation ----------
error = np.abs(phi_numerical - phi_analytical)
max_err = np.max(error)
print(f"Maximum absolute error |φ_num - φ_ana|: {max_err:.2e} rad")
print(f"Static angle φ₀ = {phi0:.4f} rad")

# ---------- Plotting ----------
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

# Top: rotation angle
axes[0].plot(t_samples, phi_numerical, 'b-', label='Numerical', linewidth=1.5)
axes[0].plot(t_samples, phi_analytical, 'r--', label='Analytical', linewidth=1.5)
axes[0].set_ylabel(r'Rotation angle $\varphi$ [rad]')
axes[0].set_title('Time modulation of the total Jones rotation')
axes[0].legend()
axes[0].grid(True)

# Middle: error
axes[1].plot(t_samples, error, 'k-', linewidth=1.0)
axes[1].set_ylabel('Absolute error [rad]')
axes[1].set_title('Difference |φ_num − φ_ana|')
axes[1].grid(True)

# Bottom: Stokes parameters
axes[2].plot(t_samples, S_out[0], label='$S_1$', color='red')
axes[2].plot(t_samples, S_out[1], label='$S_2$', color='green')
axes[2].plot(t_samples, S_out[2], label='$S_3$', color='blue')
axes[2].set_xlabel('Time [s]')
axes[2].set_ylabel('Normalised Stokes parameters')
axes[2].set_title('Output Stokes vector (input = horizontal)')
axes[2].legend(loc='upper right')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('../output/time_modulation_validated.png', dpi=150)
plt.show()