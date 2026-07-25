"""
Multi‑channel transient (Gaussian pulse) birefringence modulation.

The static fibre is modulated uniformly by a Gaussian pulse of 200 Hz bandwidth.
For each channel in the C+L band (184–196 THz), the full Jones matrix is computed
at each time step, the rotation vector φ(t) extracted, and the output SOP
trajectories plotted on the Poincaré sphere.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from pmd_model import generate_pmd_waveplates, propagate_unitary
from quaternion import jones_to_quaternion, regularize_signs, quaternion_to_rotation_vector

# ==================== Parameters ====================
# Fibre
L_km = .500
L = L_km * 1e3
L_F = 20.0                     # m
D_pmd = 2.5298e-15             # s/√m (0.08 ps/√km)
lambda0 = 1550e-9              # m
c = 299792458.0
omega0 = 2.0 * np.pi * c / lambda0

# Gaussian pulse parameters
bandwidth_Hz = 882           # 200 Hz bandwidth (intensity FWHM)
sigma_t = 0.3748 / bandwidth_Hz   # ≈ 1.874 ms
A_pulse = 10*3.1e-04                # peak modulation amplitude (0.5 %)
t0 = 25e-3                     # pulse centre at 25 ms

# Time grid
t_start = 0.0
t_end = 50e-3                  # 50 ms
dt = 0.05e-3                   # 0.05 ms sampling
n_samples = int((t_end - t_start) / dt) + 1
t_grid = np.linspace(t_start, t_end, n_samples)

# Multi‑channel frequencies
n_channels = 60
f_min = 184e12
f_max = 196e12
freq_channels = np.linspace(f_min, f_max, n_channels)
omega_channels = 2.0 * np.pi * freq_channels

# ==================== Helper: Jones -> rotation matrix ====================
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = np.stack([sigma_x, sigma_y, sigma_z], axis=0)

def jones_to_rotation_matrix(U):
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = 0.5 * np.real(np.trace(PAULI[i] @ U @ PAULI[j] @ U.conj().T))
    return R

# ==================== Generate static fibre ====================
seed = 2
z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
    L, L_F, D_pmd, lambda0, seed=seed
)

# ==================== Simulate all channels ====================
phi_mag_all = np.zeros((n_channels, n_samples))
stokes_out = np.zeros((n_channels, n_samples, 3))
s_in = np.array([1.0, 0.0, 0.0])

# Pre‑compute Gaussian modulation envelope
g = np.exp(-(t_grid - t0)**2 / (2 * sigma_t**2))
s_env = 1.0 + A_pulse * g          # time‑varying scale factor

for ch_idx, omega_ch in enumerate(tqdm(omega_channels, desc="Channels")):
    delta_omega = omega_ch - omega0
    beta_base = beta0 + delta_omega * beta_prime

    for t_idx, t in enumerate(t_grid):
        s = s_env[t_idx]           # use pre‑computed envelope
        beta_t = s * beta_base
        U_t = propagate_unitary(z, beta_t)

        # quaternion storage for regularisation
        q = jones_to_quaternion(U_t)
        if t_idx == 0:
            Q_seq = np.zeros((n_samples, 4))
        Q_seq[t_idx] = q

        # output Stokes vector
        R = jones_to_rotation_matrix(U_t)
        stokes_out[ch_idx, t_idx, :] = R @ s_in

    Q_reg = regularize_signs(Q_seq)
    phi_full = np.array([quaternion_to_rotation_vector(Q_reg[i]) for i in range(n_samples)])
    phi_mag_all[ch_idx, :] = np.linalg.norm(phi_full, axis=1)

# ==================== Plot 1: Rotation magnitude ====================
plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0, 1, n_channels))
for ch_idx in range(n_channels):
    plt.plot(t_grid*1e3, phi_mag_all[ch_idx, :], color=colors[ch_idx], linewidth=0.5)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=f_min/1e12, vmax=f_max/1e12))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label='Channel frequency (THz)')
plt.xlabel('Time (ms)')
plt.ylabel(r'$|\boldsymbol{\phi}(t)|$ (rad)')
plt.title(f'Full rotation vector magnitude – Gaussian pulse\n'
          f'(peak A={A_pulse}, bandwidth={bandwidth_Hz} Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/transient_rotation_magnitude.png', dpi=150)
plt.close()
print("Saved transient_rotation_magnitude.png")

# ==================== Plot 2: Poincaré sphere ====================
stride = max(1, n_samples // 200)
idx_plot = np.arange(0, n_samples, stride)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# wireframe sphere
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 40)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(x, y, z_sphere, color='lightgray', alpha=0.3, linewidth=0.2)

for ch_idx in range(n_channels):
    s1 = stokes_out[ch_idx, idx_plot, 0]
    s2 = stokes_out[ch_idx, idx_plot, 1]
    s3 = stokes_out[ch_idx, idx_plot, 2]
    ax.plot(s1, s2, s3, color=colors[ch_idx], linewidth=0.8, alpha=0.7)

# axes
ax.plot([-1.2,1.2],[0,0],[0,0],'k--',lw=0.5)
ax.plot([0,0],[-1.2,1.2],[0,0],'k--',lw=0.5)
ax.plot([0,0],[0,0],[-1.2,1.2],'k--',lw=0.5)
ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1); ax.set_zlim(-1.1,1.1)
ax.set_xlabel('S₁'); ax.set_ylabel('S₂'); ax.set_zlabel('S₃')
ax.set_title('Output SOP trajectories – Gaussian pulse')

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=f_min/1e12, vmax=f_max/1e12))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, label='Channel frequency (THz)')
plt.tight_layout()
plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/transient_poincare.png', dpi=150)
plt.close()
print("Saved transient_poincare.png")

mean_mag = np.mean(phi_mag_all)
std_mag = np.std(phi_mag_all)
ptp = np.ptp(phi_mag_all)
print(f"Mean |φ| = {mean_mag:.4f} rad, std = {std_mag:.4f} rad, ptp = {ptp:.4f} rad")