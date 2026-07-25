"""
Simplified multi‑channel transient birefringence modulation with CD time delay.
- Computes output SOP trajectories.
- Rotates each trajectory so its centroid is at the North Pole.
- Uses 2D real (S1, S2) cross‑correlation with parabolic peak fitting
  to recover CD delays.
- Plots rotated S1 and S2 components for all channels.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Shared library imports
from fiber_propagation import propagate_unitary
from pmd_model import generate_pmd_waveplates
from chromatic_dispersion import (
    frequency_to_wavelength,
    relative_channel_delays,
    delay_jones_sequence,
)
from visualization import plot_poincare_sphere
from rotations import jones_to_rotation_matrix, rotate_centroid_to_north_pole
from signal_processing import cross_correlation_2d_fft, parabolic_fit

# ============================================================
# Fibre parameters
# ============================================================
L_km = 0.500
L = L_km * 1e3
L_F = 20.0
D_pmd = 2.5298e-15
lambda0 = 1550e-9
c = 299792458.0
omega0 = 2 * np.pi * c / lambda0

# ============================================================
# Transient parameters
# ============================================================
bandwidth_Hz = 50e3          # 50 kHz
sigma_t = 0.3748 / bandwidth_Hz
A_pulse = 10 * 3.1e-4

# ============================================================
# Time grid
# ============================================================
t_start = 0.0
t_end = 300e-6               # 100 µs
dt = 50e-9                   # 50 ns
n_samples = int((t_end - t_start) / dt) + 1
t_grid = np.linspace(t_start, t_end, n_samples)
t0 = t_end / 2

# ============================================================
# WDM channels
# ============================================================
n_channels = 15
f_min = 184e12
f_max = 196e12
freq_channels = np.linspace(f_min, f_max, n_channels)
omega_channels = 2 * np.pi * freq_channels
wavelengths_nm = frequency_to_wavelength(freq_channels)

# ============================================================
# Generate static fibre profile
# ============================================================
seed = 2
z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
    L, L_F, D_pmd, lambda0, seed=seed
)

# ============================================================
# Gaussian modulation envelope
# ============================================================
g = np.exp(-(t_grid - t0) ** 2 / (2 * sigma_t**2))
s_env = 1.0 + A_pulse * g

# ============================================================
# Compute Jones matrices for all channels and times
# ============================================================
U_all = np.zeros((n_channels, n_samples, 2, 2), dtype=complex)

for ch_idx, omega_ch in enumerate(tqdm(omega_channels, desc="Generating Jones matrices")):
    delta_omega = omega_ch - omega0
    beta_base = beta0 + delta_omega * beta_prime
    for t_idx in range(n_samples):
        beta_t = s_env[t_idx] * beta_base
        U_all[ch_idx, t_idx] = propagate_unitary(z, beta_t)

# ============================================================
# Chromatic dispersion delays
# ============================================================
event_distance_km = 400.0
channel_delays = relative_channel_delays(
    wavelengths_nm,
    event_distance_km,
)
print("\nChromatic-dispersion delays:")
for f, d in zip(freq_channels / 1e12, channel_delays * 1e9):   # ns
    print(f"{f:7.3f} THz   {d:8.3f} ns")

# ============================================================
# Apply CD delays (only once)
# ============================================================
print("\nApplying chromatic-dispersion delays...")
for ch in tqdm(range(n_channels), desc="Applying FFT delay"):
    U_all[ch] = delay_jones_sequence(U_all[ch], dt, channel_delays[ch])

# ============================================================
# Validation checks (unitarity & determinant)
# ============================================================
print("\nChecking delayed Jones matrices...")
max_unitary_error = 0.0
max_det_error = 0.0
for ch in range(n_channels):
    for t in range(n_samples):
        U = U_all[ch, t]
        err_unitary = np.linalg.norm(U.conj().T @ U - np.eye(2))
        if err_unitary > max_unitary_error:
            max_unitary_error = err_unitary
        err_det = abs(np.linalg.det(U) - 1.0)
        if err_det > max_det_error:
            max_det_error = err_det

print(f"Maximum ||UᴴU - I|| = {max_unitary_error:.2e}")
print(f"Maximum |det(U) - 1| = {max_det_error:.2e}")

# ============================================================
# Compute output Stokes vectors for all channels
# ============================================================
s_in = np.array([1.0, 0.0, 0.0])          # horizontal linear input
stokes_out = np.zeros((n_channels, n_samples, 3))

for ch in range(n_channels):
    for t in range(n_samples):
        R = jones_to_rotation_matrix(U_all[ch, t])
        stokes_out[ch, t] = R @ s_in

# ============================================================
# Plot all channel SOP trajectories on the same Poincaré sphere
# ============================================================
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.viridis(np.linspace(0, 1, n_channels))

for ch in range(n_channels):
    S = stokes_out[ch]
    ax.plot(S[:, 0], S[:, 1], S[:, 2],
            color=colors[ch], lw=1.5, alpha=0.8,
            label=f"{wavelengths_nm[ch]:.1f} nm")

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color='gray', alpha=0.2, linewidth=0.5)

ax.set_xlabel('S₁')
ax.set_ylabel('S₂')
ax.set_zlabel('S₃')
ax.set_title('Output SOP trajectories for all WDM channels')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])

handles, labels = ax.get_legend_handles_labels()
if n_channels > 10:
    indices = np.linspace(0, n_channels-1, 5, dtype=int)
    ax.legend([handles[i] for i in indices], [labels[i] for i in indices],
              loc='upper right', fontsize=8)
else:
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/sim_Transient_Localization/sop_trajectories_all_channels.png", dpi=300)
plt.close()

print("\nPlot saved: /home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/sim_Transient_Localization/sop_trajectories_all_channels.png")

# ============================================================
# Centroid rotation to North Pole (per channel)
# ============================================================
stokes_rot = np.zeros_like(stokes_out)
rotation_matrices = []

for ch in range(n_channels):
    S_rot, R = rotate_centroid_to_north_pole(stokes_out[ch])
    stokes_rot[ch] = S_rot
    rotation_matrices.append(R)

# ============================================================
# 2D cross‑correlation with parabolic peak fitting
# ============================================================
ref_idx = n_channels - 1   # channel with delay ≈ 0
ref_2d = np.column_stack([stokes_rot[ref_idx, :, 0],
                          stokes_rot[ref_idx, :, 1]])

estimated_delays = np.zeros(n_channels)
corr_mags = []

for ch in range(n_channels):
    ch_2d = np.column_stack([stokes_rot[ch, :, 0],
                             stokes_rot[ch, :, 1]])
    lags_samples, corr = cross_correlation_2d_fft(ref_2d, ch_2d, normalize=True)

    abs_corr = np.abs(corr)
    corr_mags.append(abs_corr)

    peak_idx = np.argmax(abs_corr)
    delta, _ = parabolic_fit(abs_corr, peak_idx)
    lag_sub_samples = lags_samples[peak_idx] + delta

    estimated_delays[ch] = -lag_sub_samples * dt

# ------------------------------------------------------------
# Debug prints and plots
# ------------------------------------------------------------
print("\n--- Debug peak info (2D real cross‑correlation) ---")
for ch in [0, 7]:
    lag_ns = estimated_delays[ch] * 1e9
    theory_ns = channel_delays[ch] * 1e9
    print(f"Channel {ch}: lag = {lag_ns:.1f} ns, theory = {theory_ns:.1f} ns")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, ch in zip(axes, [0, 7]):
    lags_ns = lags_samples * dt * 1e9
    ax.plot(lags_ns, corr_mags[ch], label=f'Channel {ch}', color='C0')

    peak_lag_ns = estimated_delays[ch] * 1e9
    ax.axvline(x=peak_lag_ns, color='r', linestyle='--', linewidth=2,
               label=f'Detected peak: {peak_lag_ns:.1f} ns')

    theory_ns = channel_delays[ch] * 1e9
    ax.axvline(x=theory_ns, color='g', linestyle=':', linewidth=2,
               label=f'Theory: {theory_ns:.1f} ns')

    ax.set_xlabel('Lag (ns)')
    ax.set_ylabel('Cross-correlation magnitude')
    ax.set_title(f'Channel {ch} (λ = {wavelengths_nm[ch]:.1f} nm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Cross-correlation magnitude (2D real correlation)')
plt.tight_layout()
plt.savefig("/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/sim_Transient_Localization/cross_correlation_debug.png", dpi=300)
plt.close()

print("\nDebug plot saved: /home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/sim_Transient_Localization/cross_correlation_debug.png")

# ------------------------------------------------------------
# Comparison table (ns)
# ------------------------------------------------------------
print("\n" + "="*80)
print("CD delay comparison: Theory vs. Estimated from SOP cross-correlation")
print("="*80)
print(f"{'Ch':>3}  {'λ (nm)':>9}  {'Theory (ns)':>12}  {'Est. (ns)':>12}  {'Error (ns)':>11}")
print("-"*80)

for ch in range(n_channels):
    theory_ns = channel_delays[ch] * 1e9
    est_ns = estimated_delays[ch] * 1e9
    error_ns = est_ns - theory_ns
    print(f"{ch:3d}  {wavelengths_nm[ch]:9.1f}  {theory_ns:12.3f}  {est_ns:12.3f}  {error_ns:11.3f}")

print("="*80)

# ------------------------------------------------------------
# Plot rotated S1 and S2 for all channels
# ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for ch in range(n_channels):
    S1_rot = stokes_rot[ch, :, 0]
    S2_rot = stokes_rot[ch, :, 1]
    ax1.plot(t_grid * 1e6, S1_rot, color=colors[ch], lw=0.8,
             label=f"{wavelengths_nm[ch]:.1f} nm")
    ax2.plot(t_grid * 1e6, S2_rot, color=colors[ch], lw=0.8)

ax1.set_ylabel(r'$S_1^{\rm rot}$')
ax2.set_ylabel(r'$S_2^{\rm rot}$')
ax2.set_xlabel('Time (µs)')

handles, labels = ax1.get_legend_handles_labels()
if n_channels > 10:
    indices = np.linspace(0, n_channels-1, 5, dtype=int)
    ax1.legend([handles[i] for i in indices], [labels[i] for i in indices],
               loc='upper right', fontsize=7)
else:
    ax1.legend(loc='upper right', fontsize=7)

ax1.set_title('Rotated S₁ components (centroid at North Pole)')
ax2.set_title('Rotated S₂ components (centroid at North Pole)')

plt.tight_layout()
plt.savefig("/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/sim_Transient_Localization/rotated_s1_s2_components.png", dpi=300)
plt.close()

print("\nPlot saved: /home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/sim_Transient_Localization/rotated_s1_s2_components.png")
print("Simulation complete.")