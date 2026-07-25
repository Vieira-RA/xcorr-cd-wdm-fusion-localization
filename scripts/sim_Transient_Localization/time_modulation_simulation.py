"""
Simulate uniform birefringence modulation and extract the FULL rotation vector
from the full Jones matrix sequence (static + modulation).

The fibre's time‑varying Jones matrix U(t) is computed for each time step.
The absolute rotation vector φ(t) is obtained via quaternion extraction and
sign‑regularisation, giving the total integrated polarisation rotation.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pmd_model import generate_pmd_waveplates, propagate_unitary
from quaternion import jones_to_quaternion, regularize_signs, quaternion_to_rotation_vector

# ==================== Parameters ====================
L_km = 1.0
L = L_km * 1e3
L_F = 20.0                     # correlation length (m)
D_pmd = 2.5298e-15             # s/√m (0.08 ps/√km)
lambda0 = 1550e-9

f_mod = 100.0                  # Hz
omega_mod = 2.0 * np.pi * f_mod
A = 0.00001                       # modulation amplitude (1 %)

n_periods = 5
n_samples_per_period = 200
n_samples = n_periods * n_samples_per_period + 1
t_end = n_periods / f_mod
t_grid = np.linspace(0.0, t_end, n_samples)

# ==================== Generate static fibre ====================
seed = 89
z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
    L, L_F, D_pmd, lambda0, seed=seed
)

# ==================== Time loop → full Jones matrices ====================
U_all = np.zeros((n_samples, 2, 2), dtype=complex)

for idx, t in tqdm(enumerate(t_grid), total=n_samples, desc="Time modulation"):
    s = 1.0 + A * np.sin(omega_mod * t)
    beta_t = s * beta0
    U_all[idx] = propagate_unitary(z, beta_t)

np.save('U_full_time.npy', U_all)
print("Full Jones matrices saved to 'U_full_time.npy'")

# ==================== Full rotation vectors from full Jones matrices ====================
# Step 1: Convert each U to a quaternion
Q_full = np.array([jones_to_quaternion(U_all[i]) for i in range(n_samples)])

# Step 2: Regularise signs (important: removes sign flips that would cause 2π jumps)
Q_reg = regularize_signs(Q_full)

# Step 3: Convert each regularised quaternion to rotation vector φ
phi_full = np.array([quaternion_to_rotation_vector(Q_reg[i]) for i in range(n_samples)])

# ==================== Plotting ====================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = [r'$\phi_1$', r'$\phi_2$', r'$\phi_3$']
for k in range(3):
    axes[k].plot(t_grid, phi_full[:, k])
    axes[k].set_ylabel(labels[k] + ' (rad)')
    axes[k].grid(True)

axes[-1].set_xlabel('Time (s)')
fig.suptitle(f'Full rotation vector (static + modulation, A={A}, f_m={f_mod} Hz)')
plt.tight_layout()
plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/full_rotation_vector_components.png', dpi=150)
plt.close()
print("Saved full_rotation_vector_components.png")

# Also plot magnitude
phi_mag = np.linalg.norm(phi_full, axis=1)
plt.figure(figsize=(10, 5))
plt.plot(t_grid, phi_mag, 'b-')
plt.xlabel('Time (s)')
plt.ylabel(r'$|\boldsymbol{\phi}(t)|$ (rad)')
plt.title(f'Magnitude of full rotation vector')
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/full_rotation_vector_magnitude.png', dpi=150)
plt.close()
print("Saved full_rotation_vector_magnitude.png")

# Print some statistics of the time variation (e.g., oscillation amplitude)
print(f"Mean |φ| = {np.mean(phi_mag):.6f} rad")
print(f"Std  |φ| = {np.std(phi_mag):.6f} rad")
print(f"Peak-to-peak |φ| = {np.ptp(phi_mag):.6f} rad")