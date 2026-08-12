#!/usr/bin/env python3
"""
Toy model – 3D quaternion, 2D Stokes, and rotation‑vector magnitude comparison.
All methods use identical SNR definitions (peak AC power) and the same noise
realisation seed. The farthest channel pair is analysed with parabolic
cross‑correlation.  The 2D Stokes estimator now uses a bounded lag window
to avoid spurious peaks.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Shared library imports
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
from signal_processing import (
    cross_correlation_1d_fft,
    parabolic_corr_delay_2d,   # kept for other methods, replaced for 2D Stokes
    parabolic_corr_delay_1d,
)
from rotations import rotate_centroid_to_north_pole

# ========== Parameters ==========
BANDWIDTH_HZ = 50e3
SNR_DB = 50.0
EVENT_DIST_KM = 500.0
N_CHANNELS = 15               # only the farthest pair is used
F_MIN_HZ = 184e12
F_MAX_HZ = 196e12
CHANNEL_SPACING_HZ = 50e9
FS_FACTOR = 30
T_END_FACTOR = 160
A_PULSE = 1000*3.1e-4
FIBER_SEED = 59
NOISE_SEED = 59
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

# Use only the two farthest channels
ch_slow = 0
ch_fast = N_CHANNELS - 1
integrals_abs = abs(integrated_dispersion(wavelengths_nm[ch_slow], wavelengths_nm[ch_fast]))

# Compute true delay here, before it's needed for the window
true_delay = channel_delays[ch_slow] - channel_delays[ch_fast]
true_dist = EVENT_DIST_KM

# ========== Jones matrices (clean) ==========
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

# Extract clean Jones for the two channels
U_ref_clean = U_all[ch_fast]
U_test_clean = U_all[ch_slow]

# ========== Clean quaternions ==========
def jones_sequence_to_quaternions(U_seq):
    n = len(U_seq)
    Q = np.zeros((n, 4))
    for i in range(n):
        Q[i] = jones_to_quaternion(U_seq[i])
    return regularize_signs(Q)

q_ref_clean = jones_sequence_to_quaternions(U_ref_clean)
q_test_clean = jones_sequence_to_quaternions(U_test_clean)

# Clean rotation‑vector magnitudes
def jones_sequence_to_phi_mag(U_seq):
    n = len(U_seq)
    Q = np.zeros((n, 4))
    phi_mag = np.zeros(n)
    for i in range(n):
        Q[i] = jones_to_quaternion(U_seq[i])
    Q = regularize_signs(Q)
    for i in range(n):
        phi = quaternion_to_rotation_vector(Q[i])
        phi_mag[i] = np.linalg.norm(phi)
    return phi_mag

phi_ref_clean = jones_sequence_to_phi_mag(U_ref_clean)
phi_test_clean = jones_sequence_to_phi_mag(U_test_clean)

# ========== SNR definitions (all use AC peak power) ==========
rng = np.random.RandomState(NOISE_SEED)
snr_lin = 10**(SNR_DB / 10.0)

# Quaternion: AC peak power (sum of squares of 4 components)
q_ref_ac_clean = q_ref_clean - np.mean(q_ref_clean, axis=0, keepdims=True)
q_test_ac_clean = q_test_clean - np.mean(q_test_clean, axis=0, keepdims=True)
peak_power_quat = max(np.max(np.sum(q_ref_ac_clean**2, axis=1)),
                      np.max(np.sum(q_test_ac_clean**2, axis=1)))
noise_var_quat = peak_power_quat / (4.0 * snr_lin)   # 4 components share noise budget

# Rotation‑vector magnitude: AC peak power (scalar)
phi_ref_ac_clean = phi_ref_clean - np.mean(phi_ref_clean)
phi_test_ac_clean = phi_test_clean - np.mean(phi_test_clean)
peak_power_phi = max(np.max(phi_ref_ac_clean**2), np.max(phi_test_ac_clean**2))
noise_var_phi = peak_power_phi / snr_lin

# Stokes (unchanged, computed later)

# ========== Add noise ==========
# Quaternion noise
q_ref_noisy = q_ref_clean + np.sqrt(noise_var_quat) * rng.randn(n_samples, 4)
q_test_noisy = q_test_clean + np.sqrt(noise_var_quat) * rng.randn(n_samples, 4)

# RV noise
phi_ref_noisy = phi_ref_clean + np.sqrt(noise_var_phi) * rng.randn(n_samples)
phi_test_noisy = phi_test_clean + np.sqrt(noise_var_phi) * rng.randn(n_samples)

# ========== Filtering (optional) ==========
nyq = 0.5 / dt
b_lp, a_lp = butter(4, 3*BANDWIDTH_HZ/nyq, btype='low')

if USE_FILTER:
    # Quaternion filter per component
    for k in range(4):
        q_ref_noisy[:, k] = filtfilt(b_lp, a_lp, q_ref_noisy[:, k])
        q_test_noisy[:, k] = filtfilt(b_lp, a_lp, q_test_noisy[:, k])
    # RV filter
    phi_ref_noisy = filtfilt(b_lp, a_lp, phi_ref_noisy)
    phi_test_noisy = filtfilt(b_lp, a_lp, phi_test_noisy)

# ========== DC removal ==========
q_ref_ac = q_ref_noisy - np.mean(q_ref_noisy, axis=0, keepdims=True)
q_test_ac = q_test_noisy - np.mean(q_test_noisy, axis=0, keepdims=True)
phi_ref_ac = phi_ref_noisy - np.mean(phi_ref_noisy)
phi_test_ac = phi_test_noisy - np.mean(phi_test_noisy)

# ========== Quaternion 3D cross‑correlation (vector part) ==========
lags, _ = cross_correlation_1d_fft(q_ref_ac[:,1], q_test_ac[:,1], normalize=False)
corr_quat = np.zeros_like(lags, dtype=float)
for k in [1, 2, 3]:
    _, c = cross_correlation_1d_fft(q_ref_ac[:,k], q_test_ac[:,k], normalize=False)
    corr_quat += c
corr_quat /= np.sqrt(np.sum(q_ref_ac**2) * np.sum(q_test_ac**2))

# Parabolic peak helper
def parabolic_delay(lags, corr, dt):
    peak_idx = np.argmax(np.abs(corr))
    if 0 < peak_idx < len(corr)-1:
        yl, yc, yr = np.abs(corr[peak_idx-1]), np.abs(corr[peak_idx]), np.abs(corr[peak_idx+1])
        denom = yl - 2*yc + yr
        delta = (yl - yr) / (2 * denom) if abs(denom) > 1e-15 else 0.0
    else:
        delta = 0.0
    return (lags[peak_idx] + delta) * dt

delay_quat = parabolic_delay(lags, corr_quat, dt)
dist_quat = delay_quat * 1e12 / integrals_abs

# ========== Rotation‑vector 1D cross‑correlation ==========
lags1, corr_phi = cross_correlation_1d_fft(phi_ref_ac, phi_test_ac, normalize=True)
delay_phi = parabolic_delay(lags1, corr_phi, dt)
dist_phi = delay_phi * 1e12 / integrals_abs

# ========== 2D Stokes – BOUNDED WINDOW ESTIMATION ==========
s_in = np.array([1.0, 0.0, 0.0])
stokes_clean = np.zeros((N_CHANNELS, n_samples, 3))
for ch in range(N_CHANNELS):
    Q = np.zeros((n_samples, 4))
    for t in range(n_samples):
        Q[t] = jones_to_quaternion(U_all[ch, t])
    Q = regularize_signs(Q)
    stokes_clean[ch] = quaternion_rotate_stokes(Q, s_in)

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
    if USE_FILTER:
        for comp in range(3):
            stokes_noisy[ch, :, comp] = filtfilt(b_lp, a_lp, stokes_noisy[ch, :, comp])

stokes_rot = np.zeros_like(stokes_noisy)
for ch in range(N_CHANNELS):
    S_rot, _ = rotate_centroid_to_north_pole(stokes_noisy[ch])
    stokes_rot[ch] = S_rot

sig_ref_2d = stokes_rot[ch_fast, :, :2]
sig_test_2d = stokes_rot[ch_slow, :, :2]

# ----- 2D Stokes correlation (computed ourselves, used for estimation) -----
corr_2d = np.zeros_like(lags, dtype=float)
for comp in range(2):
    _, c = cross_correlation_1d_fft(sig_ref_2d[:,comp], sig_test_2d[:,comp], normalize=False)
    corr_2d += c
corr_2d /= np.sqrt(np.sum(sig_ref_2d**2) * np.sum(sig_test_2d**2))

# ----- Bounded window peak search -----
# True delay (physical maximum) plus a margin of 5 sigma_t
tau_max = true_delay + 5 * sigma_t   # you need true_delay already computed (it is)
mask = (lags * dt >= 0) & (lags * dt <= tau_max)
corr_windowed = np.where(mask, np.abs(corr_2d), 0.0)

peak_idx_2d = np.argmax(corr_windowed)

if 0 < peak_idx_2d < len(lags) - 1:
    y_left = np.abs(corr_2d[peak_idx_2d - 1])
    y_center = np.abs(corr_2d[peak_idx_2d])
    y_right = np.abs(corr_2d[peak_idx_2d + 1])
    denom = y_left - 2 * y_center + y_right
    if abs(denom) > 1e-15:
        delta = (y_left - y_right) / (2 * denom)
    else:
        delta = 0.0
else:
    delta = 0.0

lag_samples_2d = lags[peak_idx_2d] + delta
delay_2d = lag_samples_2d * dt
dist_2d = delay_2d * 1e12 / integrals_abs

# ========== Results ==========
true_delay = channel_delays[ch_slow] - channel_delays[ch_fast]
true_dist = EVENT_DIST_KM

print(f"True event distance: {true_dist:.1f} km")
print(f"True delay: {true_delay*1e12:.2f} ps")
print(f"Quaternion 3D estimate: {dist_quat:.1f} km  (err {dist_quat-true_dist:.1f} km)")
print(f"Rotation‑vector estimate: {dist_phi:.1f} km  (err {dist_phi-true_dist:.1f} km)")
print(f"2D Stokes estimate:    {dist_2d:.1f} km  (err {dist_2d-true_dist:.1f} km)")

# ========== Waveform plots ==========
t_span = 16 * sigma_t                     # half‑width of waveform plots
corr_span = 10 * sigma_t                 # half‑width of correlation zoom

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Quaternion vector part (ref)
ax = axes[0, 0]
for k, label in enumerate(['q1','q2','q3']):
    ax.plot(t_grid*1e6, q_ref_ac[:, k+1], label=label)
ax.set_title('Quat vec (ref, noisy, DC‑free)')
ax.set_xlabel('Time (µs)')
ax.set_xlim((t0 - t_span)*1e6, (t0 + t_span)*1e6)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Quaternion vector part (test)
ax = axes[0, 1]
for k, label in enumerate(['q1','q2','q3']):
    ax.plot(t_grid*1e6, q_test_ac[:, k+1], label=label)
ax.set_title('Quat vec (test, noisy, DC‑free)')
ax.set_xlabel('Time (µs)')
ax.set_xlim((t0 - t_span)*1e6, (t0 + t_span)*1e6)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# RV magnitude
ax = axes[0, 2]
ax.plot(t_grid*1e6, phi_ref_ac, label='|φ| ref')
ax.plot(t_grid*1e6, phi_test_ac, label='|φ| test')
ax.set_title('Rot‑vec magnitude (noisy, DC‑free)')
ax.set_xlabel('Time (µs)')
ax.set_xlim((t0 - t_span)*1e6, (t0 + t_span)*1e6)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Stokes ref
ax = axes[1, 0]
ax.plot(t_grid*1e6, stokes_rot[ch_fast, :, 0], label='S1 ref')
ax.plot(t_grid*1e6, stokes_rot[ch_fast, :, 1], label='S2 ref')
ax.set_title('Rotated Stokes (ref, noisy+filtered)')
ax.set_xlabel('Time (µs)')
ax.set_xlim((t0 - t_span)*1e6, (t0 + t_span)*1e6)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Stokes test
ax = axes[1, 1]
ax.plot(t_grid*1e6, stokes_rot[ch_slow, :, 0], label='S1 test')
ax.plot(t_grid*1e6, stokes_rot[ch_slow, :, 1], label='S2 test')
ax.set_title('Rotated Stokes (test, noisy+filtered)')
ax.set_xlabel('Time (µs)')
ax.set_xlim((t0 - t_span)*1e6, (t0 + t_span)*1e6)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Correlation comparison (zoomed around true delay)
ax = axes[1, 2]
lags_us = lags * dt * 1e6
ax.plot(lags_us, np.abs(corr_quat), label='Quaternion 3D')
ax.plot(lags_us, np.abs(corr_phi), label='Rot‑vec')
ax.plot(lags_us, np.abs(corr_2d), label='2D Stokes', linestyle='--')
ax.axvline(delay_quat*1e6, color='C0', linestyle='--')
ax.axvline(delay_phi*1e6, color='C1', linestyle='--')
ax.axvline(delay_2d*1e6, color='C2', linestyle='--')
ax.axvline(true_delay*1e6, color='g', linestyle=':', label='True')
ax.set_title('Correlation peaks')
ax.set_xlabel('Lag (µs)')
ax.set_xlim((true_delay - corr_span)*1e6, (true_delay + corr_span)*1e6)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('toy_waveforms_3methods.png', dpi=150)
plt.close()

print("Plots saved.")