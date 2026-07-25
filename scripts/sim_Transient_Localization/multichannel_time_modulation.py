"""
Multi‑channel uniform birefringence modulation and full rotation vector extraction.

60 channels equally spaced over 184–196 THz (C+L band) are simulated on the
same fibre. The birefringence is modulated uniformly in time:
    β(z,t,ω) = (1 + A sin(ω_m t)) · (β₀(z) + (ω-ω₀)·β′(z))
The full Jones matrix is computed at each time step and each channel, and the
total rotation vector magnitude |φ(t)| is extracted via quaternion
representation. All magnitude curves are overlaid with a colour map indicating
the channel frequency.

NEW: Poincaré‑sphere plot of output Stokes vectors for a fixed input SOP.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # needed for 3D projection
from tqdm import tqdm

from pmd_model import generate_pmd_waveplates, propagate_unitary
from quaternion import jones_to_quaternion, regularize_signs, quaternion_to_rotation_vector

# ==================== Parameters ====================
# Fibre
L_km = 1.0
L = L_km * 1e3
L_F = 20.0                     # correlation length (m)
D_pmd = 2.5298e-15             # s/√m (0.08 ps/√km)
lambda0 = 1550e-9              # m (centre wavelength)
c = 299792458.0
omega0 = 2.0 * np.pi * c / lambda0   # angular frequency at centre

# Modulation
f_mod = 200.0                  # Hz
omega_mod = 2.0 * np.pi * f_mod
A = 5*3.1e-04                  # amplitude

# Time sampling
n_periods = 5
n_samples_per_period = 50
n_samples = n_periods * n_samples_per_period + 1
t_end = n_periods / f_mod
t_grid = np.linspace(0.0, t_end, n_samples)

# Multi‑channel frequencies
n_channels = 80
f_min = 184e12                 # 184 THz
f_max = 196e12                 # 196 THz
freq_channels = np.linspace(f_min, f_max, n_channels)  # optical frequencies (Hz)
omega_channels = 2.0 * np.pi * freq_channels            # angular frequencies (rad/s)

# ==================== Helper: Jones -> rotation matrix ====================
# Pauli matrices (import from fiber_propagation if you prefer)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = np.stack([sigma_x, sigma_y, sigma_z], axis=0)  # shape (3,2,2)

def jones_to_rotation_matrix(U):
    """
    Convert an SU(2) Jones matrix to a 3x3 rotation matrix (Stokes space).
    R_ij = 0.5 * Tr(σ_i U σ_j U^†).
    """
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = 0.5 * np.real(np.trace(PAULI[i] @ U @ PAULI[j] @ U.conj().T))
    return R

# ==================== Generate static fibre (centre freq) ====================
seed = 84
z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
    L, L_F, D_pmd, lambda0, seed=seed
)

# ==================== Simulate all channels ====================
# Store magnitude of rotation vector and output Stokes vectors
phi_mag_all = np.zeros((n_channels, n_samples))
# Stokes vectors: shape (n_channels, n_samples, 3)
stokes_out = np.zeros((n_channels, n_samples, 3))

# Fixed input Stokes vector (e.g., linear horizontal: [1,0,0])
s_in = np.array([1.0, 0.0, 0.0])

for ch_idx, omega_ch in enumerate(tqdm(omega_channels, desc="Channels")):
    delta_omega = omega_ch - omega0
    beta_base = beta0 + delta_omega * beta_prime

    for t_idx, t in enumerate(t_grid):
        s = 1.0 + A * np.sin(omega_mod * t)
        beta_t = s * beta_base
        U_t = propagate_unitary(z, beta_t)

        # Full rotation vector from Jones
        q = jones_to_quaternion(U_t)
        if t_idx == 0:
            Q_seq = np.zeros((n_samples, 4))
        Q_seq[t_idx] = q

        # Compute output Stokes vector
        R = jones_to_rotation_matrix(U_t)
        stokes_out[ch_idx, t_idx, :] = R @ s_in

    # Regularise signs and convert to rotation vectors
    Q_reg = regularize_signs(Q_seq)
    phi_full = np.array([quaternion_to_rotation_vector(Q_reg[i]) for i in range(n_samples)])
    phi_mag_all[ch_idx, :] = np.linalg.norm(phi_full, axis=1)

# ==================== Plot 1: Magnitude curves ====================
plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0, 1, n_channels))
for ch_idx in range(n_channels):
    plt.plot(t_grid, phi_mag_all[ch_idx, :], color=colors[ch_idx], linewidth=0.5)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=f_min/1e12, vmax=f_max/1e12))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label='Channel frequency (THz)')

plt.xlabel('Time (s)')
plt.ylabel(r'$|\boldsymbol{\phi}(t)|$ (rad)')
plt.title(f'Full rotation vector magnitude – {n_channels} channels across C+L band\n'
          f'(A={A}, f_m={f_mod} Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/multichannel_rotation_magnitude.png', dpi=150)
plt.close()
print("Saved multichannel_rotation_magnitude.png")

# ==================== Plot 2: Poincaré sphere trajectories ====================
# Downsample time to avoid overly dense lines (plot every few points)
stride = max(1, n_samples // 200)   # at most 200 points per channel
idx_plot = np.arange(0, n_samples, stride)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw the unit sphere (wireframe)
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.3, linewidth=0.2)

# Plot trajectories
for ch_idx in range(n_channels):
    # Extract Stokes coordinates (downsampled)
    s1 = stokes_out[ch_idx, idx_plot, 0]
    s2 = stokes_out[ch_idx, idx_plot, 1]
    s3 = stokes_out[ch_idx, idx_plot, 2]
    ax.plot(s1, s2, s3, color=colors[ch_idx], linewidth=0.8, alpha=0.7)

# Add axes lines through origin
ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k--', linewidth=0.5)
ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k--', linewidth=0.5)
ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k--', linewidth=0.5)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-1.1, 1.1)
ax.set_xlabel('S₁')
ax.set_ylabel('S₂')
ax.set_zlabel('S₃')
ax.set_title(f'Output SOP trajectories on Poincaré sphere\n'
             f'Input SOP: horizontal linear, {n_channels} channels')

# Add colorbar to this figure as well
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=f_min/1e12, vmax=f_max/1e12))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, label='Channel frequency (THz)')

plt.tight_layout()
plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/poincare_trajectories.png', dpi=150)
plt.close()
print("Saved poincare_trajectories.png")

# ==================== Statistics ====================
mean_mag = np.mean(phi_mag_all)
std_mag = np.std(phi_mag_all)
peak_to_peak = np.ptp(phi_mag_all)
print(f"Across all channels and times: mean |φ| = {mean_mag:.4f} rad, std = {std_mag:.4f} rad, ptp = {peak_to_peak:.4f} rad")