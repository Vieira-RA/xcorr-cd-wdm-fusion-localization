#!/usr/bin/env python3
"""
Toy model – 4D quaternion cross‑correlation vs 2D Stokes.
Noise is added directly to the Jones matrix elements.
The 4D quaternion correlation uses all four components after DC removal.
Plots waveforms of quaternion components and 2D Stokes for the farthest pair.
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
    quaternion_rotate_stokes,
)
from signal_processing import (
    cross_correlation_1d_fft,
    parabolic_corr_delay_2d,
)
from rotations import rotate_centroid_to_north_pole

# ========== Parameters ==========
BANDWIDTH_HZ = 50e3
SNR_DB = 50.0
EVENT_DIST_KM = 500.0
N_CHANNELS = 15               # we only use the farthest pair
F_MIN_HZ = 184e12
F_MAX_HZ = 196e12
CHANNEL_SPACING_HZ = 50e9
FS_FACTOR = 30
T_END_FACTOR = 160
A_PULSE = 50 * 3.1e-4
FIBER_SEED = 5
NOISE_SEED = 2
FMAX_FACTOR = 2.0
USE_FILTER = True

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

# Use only the two farthest channels: idx0 (slowest) and idx_last (fastest)
ch_slow = 0
ch_fast = N_CHANNELS - 1
pair = (ch_slow, ch_fast)
integrals_abs = abs(integrated_dispersion(wavelengths_nm[ch_slow], wavelengths_nm[ch_fast]))

# ========== Jones matrices (noiseless) ==========
g = np.exp(-((t_grid - t0)**2) / (2 * sigma_t**2))
s_env = 1.0 + A_PULSE * g

U_all = np.zeros((N_CHANNELS, n_samples, 2, 2), dtype=complex)
for ch_idx, omega_ch in enumerate(omega_channels):
    beta_base = beta0 + (omega_ch - omega0) * beta_prime
    for t_idx in range(n_samples):
        beta_t = s_env[t_idx] * beta_base
        U_all[ch_idx, t_idx] = propagate_unitary(z, beta_t)

# Apply CD delays
for ch in range(N_CHANNELS):
    U_all[ch] = delay_jones_sequence(U_all[ch], dt, channel_delays[ch])

# Extract clean Jones matrices for the two channels
U_ref_clean = U_all[ch_fast]   # earlier arrival
U_test_clean = U_all[ch_slow]  # later arrival

# ========== Noise addition to Jones matrix elements ==========
rng = np.random.RandomState(NOISE_SEED)
snr_lin = 10**(SNR_DB / 10.0)

# Peak signal power: max over time of ||U||_F^2 (should be about 2 for unitaries, but modulated)
peak_power_Jones = np.max(np.sum(np.abs(U_ref_clean)**2, axis=(1,2)))  # scalar

# Total noise power to add to all 8 real components combined
# We set variance per real component such that total added power = peak_power_Jones / snr_lin.
# There are 8 real components per matrix (4 complex numbers).
noise_var_per_real = peak_power_Jones / (8.0 * snr_lin)

# Add independent Gaussian noise to real and imag parts of each element
U_ref_noisy = U_ref_clean + np.sqrt(noise_var_per_real) * (
    rng.randn(n_samples, 2, 2) + 1j * rng.randn(n_samples, 2, 2))
U_test_noisy = U_test_clean + np.sqrt(noise_var_per_real) * (
    rng.randn(n_samples, 2, 2) + 1j * rng.randn(n_samples, 2, 2))

# ========== Convert to quaternions (clean and noisy) ==========
def jones_sequence_to_quaternions(U_seq):
    n = len(U_seq)
    Q = np.zeros((n, 4))
    for i in range(n):
        Q[i] = jones_to_quaternion(U_seq[i])
    return regularize_signs(Q)

q_ref_clean = jones_sequence_to_quaternions(U_ref_clean)
q_test_clean = jones_sequence_to_quaternions(U_test_clean)
q_ref_noisy = jones_sequence_to_quaternions(U_ref_noisy)
q_test_noisy = jones_sequence_to_quaternions(U_test_noisy)

# ========== DC removal ==========
q_ref_ac = q_ref_noisy - np.mean(q_ref_noisy, axis=0, keepdims=True)
q_test_ac = q_test_noisy - np.mean(q_test_noisy, axis=0, keepdims=True)

# ========== 4D cross‑correlation (sum of per‑component correlations) ==========
lags, _ = cross_correlation_1d_fft(q_ref_ac[:,0], q_test_ac[:,0], normalize=False)
corr_sum = np.zeros_like(lags, dtype=float)
for k in range(4):
    _, c = cross_correlation_1d_fft(q_ref_ac[:,k], q_test_ac[:,k], normalize=False)
    corr_sum += c

# Normalize the summed correlation
corr_sum /= np.sqrt(np.sum(q_ref_ac**2) * np.sum(q_test_ac**2))

# Parabolic peak
peak_idx = np.argmax(np.abs(corr_sum))
if 0 < peak_idx < len(corr_sum)-1:
    yl, yc, yr = np.abs(corr_sum[peak_idx-1]), np.abs(corr_sum[peak_idx]), np.abs(corr_sum[peak_idx+1])
    denom = yl - 2*yc + yr
    if abs(denom) > 1e-15:
        delta = (yl - yr) / (2 * denom)
    else:
        delta = 0.0
else:
    delta = 0.0
delay_quat = (lags[peak_idx] + delta) * dt

# Convert to distance
dist_quat = delay_quat * 1e12 / integrals_abs

# ========== 2D Stokes method for comparison ==========
s_in = np.array([1.0, 0.0, 0.0])

stokes_clean = np.zeros((N_CHANNELS, n_samples, 3))
for ch in range(N_CHANNELS):
    Q = np.zeros((n_samples, 4))
    for t in range(n_samples):
        Q[t] = jones_to_quaternion(U_all[ch, t])
    Q = regularize_signs(Q)
    stokes_clean[ch] = quaternion_rotate_stokes(Q, s_in)

# For fair SNR comparison, define noise on the rotated clean vectors
stokes_clean_rot = np.zeros_like(stokes_clean)
for ch in range(N_CHANNELS):
    S_rot_clean, _ = rotate_centroid_to_north_pole(stokes_clean[ch])
    stokes_clean_rot[ch] = S_rot_clean

peak_power_stokes = np.max(stokes_clean_rot[:,:,0]**2 + stokes_clean_rot[:,:,1]**2, axis=1)
noise_var_stokes = peak_power_stokes / (2.0 * snr_lin)

stokes_noisy = np.zeros_like(stokes_clean)
for ch in range(N_CHANNELS):
    noise = np.sqrt(noise_var_stokes[ch]) * rng.randn(n_samples, 3)
    stokes_noisy[ch] = stokes_clean[ch] + noise

# Filtering
nyq = 0.5 / dt
b_lp, a_lp = butter(4, 3*BANDWIDTH_HZ/nyq, btype='low')
if USE_FILTER:
    for ch in range(N_CHANNELS):
        for comp in range(3):
            stokes_noisy[ch, :, comp] = filtfilt(b_lp, a_lp, stokes_noisy[ch, :, comp])

# Centroid rotation
stokes_rot = np.zeros_like(stokes_noisy)
for ch in range(N_CHANNELS):
    S_rot, _ = rotate_centroid_to_north_pole(stokes_noisy[ch])
    stokes_rot[ch] = S_rot

# 2D parabolic delay
sig_ref_2d = stokes_rot[ch_fast, :, :2]
sig_test_2d = stokes_rot[ch_slow, :, :2]
delay_2d = parabolic_corr_delay_2d(sig_ref_2d, sig_test_2d, dt)
dist_2d = delay_2d * 1e12 / integrals_abs

# ========== Print results ==========
true_delay = channel_delays[ch_slow] - channel_delays[ch_fast]  # positive
true_dist = EVENT_DIST_KM

print(f"True event distance: {true_dist:.1f} km")
print(f"True delay: {true_delay*1e12:.2f} ps")
print(f"Quaternion 4D estimate: {dist_quat:.1f} km  (err {dist_quat-true_dist:.1f} km)")
print(f"2D Stokes estimate:    {dist_2d:.1f} km  (err {dist_2d-true_dist:.1f} km)")

# ========== Waveform plots ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# -- Quaternion components (reference channel, noisy, DC‑free) --
ax = axes[0, 0]
for k, label in enumerate(['q0','q1','q2','q3']):
    ax.plot(t_grid*1e6, q_ref_ac[:, k], label=label)
ax.set_title('Quaternion components (ref, DC‑free, noisy)')
ax.set_xlabel('Time (µs)')
ax.set_xlim([400, 800])
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# -- Quaternion components (test channel) --
ax = axes[0, 1]
for k, label in enumerate(['q0','q1','q2','q3']):
    ax.plot(t_grid*1e6, q_test_ac[:, k], label=label)
ax.set_title('Quaternion components (test, DC‑free, noisy)')
ax.set_xlabel('Time (µs)')
ax.set_xlim([400, 800])
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# -- 2D Stokes (ref) --
ax = axes[1, 0]
ax.plot(t_grid*1e6, stokes_rot[ch_fast, :, 0], label='S1 ref')
ax.plot(t_grid*1e6, stokes_rot[ch_fast, :, 1], label='S2 ref')
ax.set_title('Rotated Stokes (ref, noisy+filtered)')
ax.set_xlabel('Time (µs)')
ax.set_xlim([400, 800])
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# -- 2D Stokes (test) --
ax = axes[1, 1]
ax.plot(t_grid*1e6, stokes_rot[ch_slow, :, 0], label='S1 test')
ax.plot(t_grid*1e6, stokes_rot[ch_slow, :, 1], label='S2 test')
ax.set_title('Rotated Stokes (test, noisy+filtered)')
ax.set_xlabel('Time (µs)')
ax.set_xlim([400, 800])
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('toy_waveforms_quat_vs_stokes.png', dpi=150)
plt.close()

# ========== Correlation plot ==========
fig, ax = plt.subplots(figsize=(8,4))
lags_us = lags * dt * 1e6
ax.plot(lags_us, np.abs(corr_sum), label='4D quaternion correlation')
ax.axvline(delay_quat*1e6, color='r', linestyle='--', label=f'Est delay: {delay_quat*1e12:.1f} ps')
ax.axvline(true_delay*1e6, color='g', linestyle=':', label=f'True delay: {true_delay*1e12:.1f} ps')
ax.set_xlabel('Lag (µs)')
ax.set_ylabel('Normalized correlation')
ax.set_title('4D quaternion cross-correlation')
ax.legend()
plt.tight_layout()
plt.savefig('toy_quat_corr.png', dpi=150)
plt.close()

print("Plots saved.")