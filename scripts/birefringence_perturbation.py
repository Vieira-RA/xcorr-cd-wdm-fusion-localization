#!/usr/bin/env python3
"""
Demo: Jones matrix propagation through a 10 km fibre with time‑varying birefringence.

Plots the components of the rotation vector φ(t) at the fibre output.
Also prints the final Jones matrix and the output Stokes vector for an input
linear horizontal polarisation.

Assumes `sop-simulation-lib` is installed in editable mode in the active environment.
"""
import numpy as np
import matplotlib.pyplot as plt

from fiber_propagation import (propagate_time_series,
                               quaternion_to_jones,
                               apply_rotation_to_stokes)

# =========================================================================
# 1. Define the birefringence model β(z, t)  (rad / km)
#    Simple model: large constant part + small time‑sinusoidal perturbation
# =========================================================================
def beta_zt(z: float, t: float) -> np.ndarray:
    """
    Birefringence vector (rad/km) at position z (km) and time t (a.u.).
    """
    # constant background (0.5, 0.2, 0.3) rad/km
    beta_const = np.array([0.5, 0.2, 0.3])
    # time‑varying perturbation: amplitude 0.1 rad/km, period 10 time units
    omega = 2 * np.pi / 10.0
    beta_pert = 0.091 * np.array([np.sin(omega * t),
                                np.cos(omega * t),
                                0.0])
    return beta_const + beta_pert

# =========================================================================
# 2. Simulation parameters
# =========================================================================
L = 10.0        # km
dz = 0.05      # km (1 m) – fine enough for the smooth birefringence model

t_samples = np.linspace(0, 20, 201)   # 201 time points from 0 to 20

# =========================================================================
# 3. Propagate for all times, obtain regularised quaternions and φ(t)
# =========================================================================
print("Propagating...")
from fiber_propagation import propagate_time_series_unwrapped
q_series, phi_unwrapped = propagate_time_series_unwrapped(t_samples, L, dz, beta_zt)
# =========================================================================
# 4. Extract quantities of interest at the last time sample (example)
# =========================================================================
q_last = q_series[-1]
U_last = quaternion_to_jones(q_last)
phi_last = phi_unwrapped[-1]

print("\nFinal time sample (t = {:.1f}):".format(t_samples[-1]))
print("Quaternion (q0,q1,q2,q3) = {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(*q_last))
print("Jones matrix U =")
print(U_last)
print("Rotation vector φ = {:.6f}, {:.6f}, {:.6f}".format(*phi_last))

# Stokes output for a horizontally polarised input (1, 0, 0)
s_in = np.array([1.0, 0.0, 0.0])
s_out = apply_rotation_to_stokes(phi_last, s_in)
print("Input Stokes  (1,0,0) -> Output Stokes = ({:.6f}, {:.6f}, {:.6f})".format(*s_out))

# =========================================================================
# 5. Plot φ(t)
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_samples, phi_unwrapped[:, 0], label=r'$\phi_1$')
ax.plot(t_samples, phi_unwrapped[:, 1], label=r'$\phi_2$')
ax.plot(t_samples, phi_unwrapped[:, 2], label=r'$\phi_3$')
ax.set_xlabel('Time (a.u.)')
ax.set_ylabel('Rotation vector component (rad)')
ax.set_title('Time evolution of the rotation vector at fibre output')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/rotation_vector_vs_time.png", dpi=150)
plt.show()