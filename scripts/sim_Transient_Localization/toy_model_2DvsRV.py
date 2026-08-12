#!/usr/bin/env python3
"""
Toy model – compare 2D Stokes vs rotation‑vector magnitude for CD localisation.
Fixed conditions: 50 kHz, 80 dB SNR, 15 alternating channels, 500 km event.
Only parabolic‑peak cross‑correlation is used.
Channel pairs with wavelength separation < 15 nm are discarded.
Noise definitions are now fair: Stokes SNR computed on the rotated clean vectors.
"""
import numpy as np
import itertools
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ---- Shared library imports ----
from fiber_propagation import propagate_unitary
from pmd_model import generate_pmd_waveplates
from chromatic_dispersion import (
    frequency_to_wavelength,
    relative_channel_delays,
    delay_jones_sequence,
    integrated_dispersion,
)
from quaternion import (
    jones_to_quaternion,
    regularize_signs,
    quaternion_to_rotation_vector,
    quaternion_rotate_stokes,
)
from signal_processing import parabolic_corr_delay_2d, parabolic_corr_delay_1d
from estimation import iqr_filter, adaptive_weighted_mean
from rotations import rotate_centroid_to_north_pole


# ========== Parameters ==========
BANDWIDTH_HZ = 50e3                    # 50 kHz
SNR_DB = 50.0
EVENT_DIST_KM = 500.0
N_CHANNELS = 15
F_MIN_HZ = 184e12
F_MAX_HZ = 196e12
CHANNEL_SPACING_HZ = 50e9
FS_FACTOR = 30
T_END_FACTOR = 160
A_PULSE = 10 * 3.1e-4
FIBER_SEED = 5
NOISE_SEED = 2
FMAX_FACTOR = 2.0
USE_FILTER = True
MIN_DELTA_LAMBDA_NM = 0   # drop pairs closer than this

# ========== Derived time grid ==========
sigma_t = 0.3748 / BANDWIDTH_HZ
fs = FS_FACTOR * BANDWIDTH_HZ
dt = 1.0 / fs
t_end = T_END_FACTOR * sigma_t
n_samples = int(t_end / dt) + 1
t_grid = np.linspace(0, t_end, n_samples)
t0 = t_end / 2

# ========== Channel grid (alternating) ==========
def generate_alternating_grid(f_min, f_max, n_channels, spacing):
    f_low, f_high = [], []
    f_low.append(f_min)
    f_high.append(f_max)
    k = 1
    while len(f_low) + len(f_high) < n_channels:
        f_high.append(f_max - k * spacing)
        if len(f_low) + len(f_high) < n_channels:
            f_low.append(f_min + k * spacing)
        k += 1
    return np.sort(f_low + f_high)

freq_channels = generate_alternating_grid(F_MIN_HZ, F_MAX_HZ, N_CHANNELS, CHANNEL_SPACING_HZ)
omega_channels = 2 * np.pi * freq_channels
wavelengths_nm = frequency_to_wavelength(freq_channels)

# ========== Fibre profile and CD delays ==========
L_km = 0.5
L = L_km * 1e3
L_F = 20.0
D_pmd = 2.5298e-15
lambda0_nm = 1550
lambda0 = lambda0_nm * 1e-9
c = 299792458.0
omega0 = 2 * np.pi * c / lambda0

z, beta0, beta_prime, _ = generate_pmd_waveplates(L, L_F, D_pmd, lambda0, seed=FIBER_SEED)
channel_delays = relative_channel_delays(wavelengths_nm, EVENT_DIST_KM)

# ========== Pair info ==========
all_pair_indices = list(itertools.combinations(range(N_CHANNELS), 2))
# filter pairs by wavelength separation
delta_lambda = np.array([np.abs(wavelengths_nm[i] - wavelengths_nm[j]) for i, j in all_pair_indices])
keep_mask = delta_lambda >= MIN_DELTA_LAMBDA_NM
pair_indices = [p for p, k in zip(all_pair_indices, keep_mask) if k]
n_pairs = len(pair_indices)
print(f"Keeping {n_pairs} pairs with Δλ ≥ {MIN_DELTA_LAMBDA_NM} nm (out of {len(all_pair_indices)} total).")
integrals_abs = np.array([
    abs(integrated_dispersion(wavelengths_nm[i], wavelengths_nm[j]))
    for i, j in pair_indices
])

# ========== Jones matrices (noiseless) ==========
g = np.exp(-((t_grid - t0)**2) / (2 * sigma_t**2))
s_env = 1.0 + A_PULSE * g

U_all = np.zeros((N_CHANNELS, n_samples, 2, 2), dtype=complex)
for ch_idx, omega_ch in enumerate(omega_channels):
    beta_base = beta0 + (omega_ch - omega0) * beta_prime
    for t_idx in range(n_samples):
        beta_t = s_env[t_idx] * beta_base
        U_all[ch_idx, t_idx] = propagate_unitary(z, beta_t)

for ch in range(N_CHANNELS):
    U_all[ch] = delay_jones_sequence(U_all[ch], dt, channel_delays[ch])

# ========== Clean signals ==========
s_in = np.array([1.0, 0.0, 0.0])

# 2D Stokes (via quaternion rotation)
stokes_clean = np.zeros((N_CHANNELS, n_samples, 3))
for ch in range(N_CHANNELS):
    Q = np.zeros((n_samples, 4))
    for t in range(n_samples):
        Q[t] = jones_to_quaternion(U_all[ch, t])
    Q = regularize_signs(Q)
    stokes_clean[ch] = quaternion_rotate_stokes(Q, s_in)

# Rotation‑vector magnitude (clean)
phi_mag_clean = np.zeros((N_CHANNELS, n_samples))
for ch in range(N_CHANNELS):
    Q = np.zeros((n_samples, 4))
    for t in range(n_samples):
        Q[t] = jones_to_quaternion(U_all[ch, t])
    Q = regularize_signs(Q)
    for t in range(n_samples):
        phi = quaternion_to_rotation_vector(Q[t])
        phi_mag_clean[ch, t] = np.linalg.norm(phi)

# ========== Noise addition (FAIR definition) ==========
rng = np.random.RandomState(NOISE_SEED)
snr_lin = 10**(SNR_DB / 10.0)

# ---- Stokes: SNR defined on the CLEAN ROTATED vectors ----
stokes_clean_rot = np.zeros_like(stokes_clean)
for ch in range(N_CHANNELS):
    S_rot_clean, _ = rotate_centroid_to_north_pole(stokes_clean[ch])
    stokes_clean_rot[ch] = S_rot_clean

peak_power_stokes = np.max(stokes_clean_rot[:,:,0]**2 + stokes_clean_rot[:,:,1]**2, axis=1)
noise_var_stokes = peak_power_stokes / (2.0 * snr_lin)   # factor 2 for the two components

stokes_noisy = np.zeros_like(stokes_clean)
for ch in range(N_CHANNELS):
    noise = np.sqrt(noise_var_stokes[ch]) * rng.randn(n_samples, 3)
    stokes_noisy[ch] = stokes_clean[ch] + noise

# ---- Rotation‑vector: SNR defined on the AC part (DC‑free) ----
phi_ac = phi_mag_clean - np.mean(phi_mag_clean, axis=1, keepdims=True)
peak_power_phi = np.max(phi_ac**2, axis=1)
noise_var_phi = peak_power_phi / snr_lin
phi_noisy = phi_mag_clean + np.sqrt(noise_var_phi[:, None]) * rng.randn(*phi_mag_clean.shape)

# ========== Filtering ==========
nyq = 0.5 / dt
b_lp, a_lp = butter(4, 3*BANDWIDTH_HZ/nyq, btype='low')

if USE_FILTER:
    # Stokes filter per component (before centroid rotation)
    for ch in range(N_CHANNELS):
        for comp in range(3):
            stokes_noisy[ch, :, comp] = filtfilt(b_lp, a_lp, stokes_noisy[ch, :, comp])
    # Rotvec filter
    phi_filt = np.zeros_like(phi_noisy)
    for ch in range(N_CHANNELS):
        phi_filt[ch] = filtfilt(b_lp, a_lp, phi_noisy[ch])
else:
    phi_filt = phi_noisy.copy()

# Rotvec DC removal
for ch in range(N_CHANNELS):
    phi_filt[ch] -= np.mean(phi_filt[ch])

# ========== Stokes centroid rotation (on noisy filtered signals) ==========
stokes_rot = np.zeros_like(stokes_noisy)
for ch in range(N_CHANNELS):
    S_rot, _ = rotate_centroid_to_north_pole(stokes_noisy[ch])
    stokes_rot[ch] = S_rot

# ========== Waveform plot (S1,S2 and |φ| for two extreme channels) ==========
ch_a, ch_b = 0, N_CHANNELS - 1   # smallest and largest wavelength
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# 2D Stokes (rotated, noisy + filtered)
ax1.plot(t_grid * 1e6, stokes_rot[ch_a, :, 0], label=f'S₁ ch{ch_a} ({wavelengths_nm[ch_a]:.1f} nm)')
ax1.plot(t_grid * 1e6, stokes_rot[ch_a, :, 1], label=f'S₂ ch{ch_a}')
ax1.plot(t_grid * 1e6, stokes_rot[ch_b, :, 0], '--', label=f'S₁ ch{ch_b} ({wavelengths_nm[ch_b]:.1f} nm)')
ax1.plot(t_grid * 1e6, stokes_rot[ch_b, :, 1], '--', label=f'S₂ ch{ch_b}')
ax1.set_ylabel('Rotated Stokes component')
ax1.set_title('2D Stokes signals (rotated to North Pole, noisy + filtered)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([400, 800])

# Rotation‑vector magnitude (DC‑free, noisy + filtered)
ax2.plot(t_grid * 1e6, phi_filt[ch_a], label=f'|φ| ch{ch_a}')
ax2.plot(t_grid * 1e6, phi_filt[ch_b], '--', label=f'|φ| ch{ch_b}')
ax2.set_xlabel('Time (µs)')
ax2.set_ylabel('|φ| (rad)')
ax2.set_title('Rotation‑vector magnitude (DC‑free, noisy + filtered)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([400, 800])

plt.tight_layout()
plt.savefig('toy_waveforms.png', dpi=150)
plt.close()
print("Waveform plot saved to toy_waveforms.png")

# ========== All-channels waveform plot (S₁ and |φ|) ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Use a colormap to distinguish the 15 channels
colors = plt.cm.viridis(np.linspace(0, 1, N_CHANNELS))

# Rotated Stokes S₁ (noisy + filtered)
for ch in range(N_CHANNELS):
    ax1.plot(t_grid * 1e6, stokes_rot[ch, :, 0],
             color=colors[ch], lw=0.8, alpha=0.8,
             label=f'{wavelengths_nm[ch]:.1f} nm')
ax1.set_ylabel('Rotated S₁')
ax1.set_title('All channels: rotated Stokes S₁ (after CD delays)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([400, 800])

# Rotation‑vector magnitude (DC‑free, noisy + filtered)
for ch in range(N_CHANNELS):
    ax2.plot(t_grid * 1e6, phi_filt[ch],
             color=colors[ch], lw=0.8, alpha=0.8)
ax2.set_xlabel('Time (µs)')
ax2.set_ylabel('|φ| (rad)')
ax2.set_title('All channels: rotation‑vector magnitude (DC‑free)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([400, 800])

# Add a colorbar to indicate wavelength
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=wavelengths_nm.min(), vmax=wavelengths_nm.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], label='Wavelength (nm)')

plt.tight_layout()
plt.savefig('toy_all_waveforms.png', dpi=150)
plt.close()
print("All-channels waveform plot saved to toy_all_waveforms.png")

# ========== Delay estimation (parabolic only) ==========
delays_2d = np.zeros(n_pairs)
delays_rv = np.zeros(n_pairs)

for idx, (i, j) in enumerate(pair_indices):
    # 2D
    sig_ref_2d = stokes_rot[j, :, :2]   # faster (earlier)
    sig_test_2d = stokes_rot[i, :, :2]  # slower (later)
    delays_2d[idx] = parabolic_corr_delay_2d(sig_ref_2d, sig_test_2d, dt)

    # 1D (rotvec)
    sig_ref_1d = phi_filt[j]            # faster
    sig_test_1d = phi_filt[i]           # slower
    delays_rv[idx] = parabolic_corr_delay_1d(sig_ref_1d, sig_test_1d, dt)

# Convert to distances (km)
dists_2d = delays_2d * 1e12 / integrals_abs
dists_rv = delays_rv * 1e12 / integrals_abs

# ========== IQR filter (factor=1.0) & adaptive weighted mean ==========
mask_2d = iqr_filter(dists_2d, factor=1.0)
mask_rv = iqr_filter(dists_rv, factor=1.0)

final_2d, _ = adaptive_weighted_mean(dists_2d, mask_2d, integrals_abs)
final_rv, _ = adaptive_weighted_mean(dists_rv, mask_rv, integrals_abs)

# ========== Results ==========
print(f"True event distance: {EVENT_DIST_KM:.1f} km")
print(f"2D Stokes final estimate: {final_2d:.1f} km  (inliers: {np.sum(mask_2d)}/{n_pairs})")
print(f"RotVec final estimate:    {final_rv:.1f} km  (inliers: {np.sum(mask_rv)}/{n_pairs})")
print(f"2D absolute error: {abs(final_2d - EVENT_DIST_KM):.1f} km")
print(f"RV absolute error: {abs(final_rv - EVENT_DIST_KM):.1f} km")

# ========== Quick scatter plot ==========
delta_lam_kept = np.array([np.abs(wavelengths_nm[i] - wavelengths_nm[j]) for i, j in pair_indices])
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(delta_lam_kept, dists_2d, s=20, marker='o', label='2D Stokes')
ax.scatter(delta_lam_kept, dists_rv, s=20, marker='s', label='RotVec')
ax.axhline(EVENT_DIST_KM, color='k', linestyle='--', label='True distance')
ax.axhline(final_2d, color='C0', linestyle='-', label=f'2D final: {final_2d:.1f} km')
ax.axhline(final_rv, color='C1', linestyle='-', label=f'RV final: {final_rv:.1f} km')
ax.set_xlabel('Wavelength separation (nm)')
ax.set_ylabel('Estimated distance (km)')
ax.set_title('Parabolic peak – 2D Stokes vs Rotation Vector')
ax.legend()
plt.tight_layout()
plt.savefig('toy_2d_vs_rv.png', dpi=150)
plt.close()
print("Scatter plot saved to toy_2d_vs_rv.png")

# ======== Parameter tuning for the adaptive weighted mean ========
print("\n---- Tuning weighting parameters ----")

# Parameters to sweep
iqr_factors = np.arange(0.5, 4.1, 0.5)          # IQR pre‑filter factor
c_bisquare_vals = np.arange(1.0, 5.1, 0.5)      # Bisquare tuning constant

best_error_2d = np.inf
best_error_rv = np.inf
best_params_2d = (None, None)
best_params_rv = (None, None)

# We will reuse the already computed distance arrays (dists_2d, dists_rv) and integrals_abs.
# For each parameter pair, we apply the IQR filter with the given factor,
# then compute the adaptive weighted mean with the given c_bisquare.

for iqr_factor in iqr_factors:
    for c_bisq in c_bisquare_vals:
        # ---- 2D method ----
        mask_2d = iqr_filter(dists_2d, factor=iqr_factor)
        if np.sum(mask_2d) > 0:
            wmean_2d, _ = adaptive_weighted_mean(dists_2d, mask_2d, integrals_abs,
                                                 c_bisquare=c_bisq)
            err_2d = abs(wmean_2d - EVENT_DIST_KM)
            if err_2d < best_error_2d:
                best_error_2d = err_2d
                best_params_2d = (iqr_factor, c_bisq)

        # ---- RotVec method ----
        mask_rv = iqr_filter(dists_rv, factor=iqr_factor)
        if np.sum(mask_rv) > 0:
            wmean_rv, _ = adaptive_weighted_mean(dists_rv, mask_rv, integrals_abs,
                                                 c_bisquare=c_bisq)
            err_rv = abs(wmean_rv - EVENT_DIST_KM)
            if err_rv < best_error_rv:
                best_error_rv = err_rv
                best_params_rv = (iqr_factor, c_bisq)

print(f"Best 2D parameters: IQR factor = {best_params_2d[0]:.2f}, c = {best_params_2d[1]:.2f}  -> error = {best_error_2d:.2f} km")
print(f"Best RV parameters: IQR factor = {best_params_rv[0]:.2f}, c = {best_params_rv[1]:.2f}  -> error = {best_error_rv:.2f} km")

# Optionally re‑compute final estimates using the best parameters for a clean report
mask_2d_opt = iqr_filter(dists_2d, factor=best_params_2d[0])
final_2d_opt, _ = adaptive_weighted_mean(dists_2d, mask_2d_opt, integrals_abs,
                                         c_bisquare=best_params_2d[1])
mask_rv_opt = iqr_filter(dists_rv, factor=best_params_rv[0])
final_rv_opt, _ = adaptive_weighted_mean(dists_rv, mask_rv_opt, integrals_abs,
                                         c_bisquare=best_params_rv[1])
print(f"Refined 2D estimate: {final_2d_opt:.2f} km  (inliers: {np.sum(mask_2d_opt)}/{n_pairs})")
print(f"Refined RV estimate: {final_rv_opt:.2f} km  (inliers: {np.sum(mask_rv_opt)}/{n_pairs})")