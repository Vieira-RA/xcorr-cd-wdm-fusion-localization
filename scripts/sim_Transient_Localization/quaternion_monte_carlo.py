#!/usr/bin/env python3
"""
Monte‑Carlo comparison of 3D quaternion, rotation‑vector magnitude and 2D Stokes
(parabolic peak with bounded window) for CD localisation.
10 independent fibre seeds × 10 noise realisations = 100 estimates per method.
An IQR filter (factor=1.0) is applied to the distance estimates before computing
the standard deviation and bias.
"""
import numpy as np
from scipy.signal import butter, filtfilt
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
from signal_processing import cross_correlation_1d_fft
from rotations import rotate_centroid_to_north_pole
from estimation import iqr_filter      # added for outlier removal

# ========== Fixed parameters ==========
BANDWIDTH_HZ = 100e3
SNR_DB = 50.0
EVENT_DIST_KM = 500.0
N_CHANNELS = 15               # only the farthest pair is used
F_MIN_HZ = 184e12
F_MAX_HZ = 196e12
CHANNEL_SPACING_HZ = 50e9
FS_FACTOR = 30
T_END_FACTOR = 160
A_PULSE = 500*3.1e-4
USE_FILTER = False

N_FIBRES = 20
N_REAL = 20                  # noise realisations per fibre

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
    f_low.append(f_min); f_high.append(f_max)
    k = 1
    while len(f_low)+len(f_high) < n_channels:
        f_high.append(f_max - k*spacing)
        if len(f_low)+len(f_high) < n_channels:
            f_low.append(f_min + k*spacing)
        k += 1
    return np.sort(f_low + f_high)

freq_channels = generate_alternating_grid(F_MIN_HZ, F_MAX_HZ, N_CHANNELS, CHANNEL_SPACING_HZ)
omega_channels = 2*np.pi*freq_channels
wavelengths_nm = frequency_to_wavelength(freq_channels)
ch_slow, ch_fast = 0, N_CHANNELS-1
integrals_abs = abs(integrated_dispersion(wavelengths_nm[ch_slow], wavelengths_nm[ch_fast]))

# CD delays (independent of fibre seed)
L_km = 0.5; L = L_km*1e3; L_F = 20.0; D_pmd = 2.5298e-15
lambda0_nm = 1550; lambda0 = lambda0_nm*1e-9; c = 299792458.0
omega0 = 2*np.pi*c/lambda0

# channel delays (only depends on wavelengths)
channel_delays = relative_channel_delays(wavelengths_nm, EVENT_DIST_KM)
true_delay = channel_delays[ch_slow] - channel_delays[ch_fast]
true_dist = EVENT_DIST_KM

# envelope (common for all fibres)
g = np.exp(-((t_grid - t0)**2) / (2*sigma_t**2))
s_env = 1.0 + A_PULSE*g

# filter design (common for all fibres)
nyq = 0.5/dt
b_lp, a_lp = butter(4, 3*BANDWIDTH_HZ/nyq, btype='low')

# window for bounded 2D Stokes peak
tau_max = true_delay + 5*sigma_t

# ========== Helper: parabolic delay ==========
def parabolic_delay(lags, corr, dt):
    peak_idx = np.argmax(np.abs(corr))
    if 0 < peak_idx < len(corr)-1:
        yl, yc, yr = np.abs(corr[peak_idx-1]), np.abs(corr[peak_idx]), np.abs(corr[peak_idx+1])
        denom = yl - 2*yc + yr
        delta = (yl - yr)/(2*denom) if abs(denom)>1e-15 else 0.0
    else:
        delta = 0.0
    return (lags[peak_idx] + delta)*dt

# ========== Run Monte‑Carlo ==========
results_quat = []   # distance estimates (km)
results_phi  = []
results_2d   = []

for i_fibre in range(N_FIBRES):
    # fibre‑specific PMD profile
    fibre_seed = i_fibre   # seeds 0..9
    z, beta0, beta_prime, _ = generate_pmd_waveplates(L, L_F, D_pmd, lambda0, seed=fibre_seed)

    # Jones matrices for this fibre (same for all noise realisations)
    U_all = np.zeros((N_CHANNELS, n_samples, 2,2), dtype=complex)
    for ch_idx, omega_ch in enumerate(omega_channels):
        beta_base = beta0 + (omega_ch - omega0)*beta_prime
        for t_idx in range(n_samples):
            U_all[ch_idx,t_idx] = propagate_unitary(z, s_env[t_idx]*beta_base)
    for ch in range(N_CHANNELS):
        U_all[ch] = delay_jones_sequence(U_all[ch], dt, channel_delays[ch])

    U_ref_clean = U_all[ch_fast]
    U_test_clean = U_all[ch_slow]

    # Clean quaternions
    def jones_sequence_to_quaternions(U_seq):
        n = len(U_seq)
        Q = np.zeros((n,4))
        for i in range(n): Q[i] = jones_to_quaternion(U_seq[i])
        return regularize_signs(Q)

    q_ref_clean = jones_sequence_to_quaternions(U_ref_clean)
    q_test_clean = jones_sequence_to_quaternions(U_test_clean)

    # Clean RV magnitude
    def jones_sequence_to_phi_mag(U_seq):
        n = len(U_seq)
        Q = np.zeros((n,4)); phi_mag = np.zeros(n)
        for i in range(n): Q[i] = jones_to_quaternion(U_seq[i])
        Q = regularize_signs(Q)
        for i in range(n): phi_mag[i] = np.linalg.norm(quaternion_to_rotation_vector(Q[i]))
        return phi_mag

    phi_ref_clean = jones_sequence_to_phi_mag(U_ref_clean)
    phi_test_clean = jones_sequence_to_phi_mag(U_test_clean)

    # Clean Stokes (rotated) for SNR definition
    s_in = np.array([1.0,0.0,0.0])
    stokes_clean = np.zeros((N_CHANNELS, n_samples, 3))
    for ch in range(N_CHANNELS):
        Q = np.zeros((n_samples,4))
        for t in range(n_samples): Q[t] = jones_to_quaternion(U_all[ch,t])
        Q = regularize_signs(Q)
        stokes_clean[ch] = quaternion_rotate_stokes(Q, s_in)
    stokes_clean_rot = np.zeros_like(stokes_clean)
    for ch in range(N_CHANNELS):
        S_rot_clean, _ = rotate_centroid_to_north_pole(stokes_clean[ch])
        stokes_clean_rot[ch] = S_rot_clean
    peak_power_stokes = np.max(stokes_clean_rot[:,:,0]**2 + stokes_clean_rot[:,:,1]**2, axis=1)

    # SNR reference powers (AC)
    q_ref_ac_clean = q_ref_clean - np.mean(q_ref_clean, axis=0, keepdims=True)
    q_test_ac_clean = q_test_clean - np.mean(q_test_clean, axis=0, keepdims=True)
    peak_power_quat = max(np.max(np.sum(q_ref_ac_clean**2, axis=1)),
                          np.max(np.sum(q_test_ac_clean**2, axis=1)))
    phi_ref_ac_clean = phi_ref_clean - np.mean(phi_ref_clean)
    phi_test_ac_clean = phi_test_clean - np.mean(phi_test_clean)
    peak_power_phi = max(np.max(phi_ref_ac_clean**2), np.max(phi_test_ac_clean**2))

    for i_real in range(N_REAL):
        rng = np.random.RandomState(i_real)   # independent noise per realisation
        snr_lin = 10**(SNR_DB/10.0)

        # ---- Quaternion method ----
        noise_var_quat = peak_power_quat / (4.0*snr_lin)
        q_ref_noisy = q_ref_clean + np.sqrt(noise_var_quat)*rng.randn(n_samples,4)
        q_test_noisy = q_test_clean + np.sqrt(noise_var_quat)*rng.randn(n_samples,4)
        if USE_FILTER:
            for k in range(4):
                q_ref_noisy[:,k] = filtfilt(b_lp, a_lp, q_ref_noisy[:,k])
                q_test_noisy[:,k] = filtfilt(b_lp, a_lp, q_test_noisy[:,k])
        q_ref_ac = q_ref_noisy - np.mean(q_ref_noisy, axis=0, keepdims=True)
        q_test_ac = q_test_noisy - np.mean(q_test_noisy, axis=0, keepdims=True)
        lags, _ = cross_correlation_1d_fft(q_ref_ac[:,1], q_test_ac[:,1], normalize=False)
        corr_quat = np.zeros_like(lags, dtype=float)
        for k in [1,2,3]:
            _, c = cross_correlation_1d_fft(q_ref_ac[:,k], q_test_ac[:,k], normalize=False)
            corr_quat += c
        corr_quat /= np.sqrt(np.sum(q_ref_ac**2)*np.sum(q_test_ac**2))
        delay_quat = parabolic_delay(lags, corr_quat, dt)
        results_quat.append(delay_quat*1e12 / integrals_abs)

        # ---- Rotation‑vector method ----
        noise_var_phi = peak_power_phi / snr_lin
        phi_ref_noisy = phi_ref_clean + np.sqrt(noise_var_phi)*rng.randn(n_samples)
        phi_test_noisy = phi_test_clean + np.sqrt(noise_var_phi)*rng.randn(n_samples)
        if USE_FILTER:
            phi_ref_noisy = filtfilt(b_lp, a_lp, phi_ref_noisy)
            phi_test_noisy = filtfilt(b_lp, a_lp, phi_test_noisy)
        phi_ref_ac = phi_ref_noisy - np.mean(phi_ref_noisy)
        phi_test_ac = phi_test_noisy - np.mean(phi_test_noisy)
        lags1, corr_phi = cross_correlation_1d_fft(phi_ref_ac, phi_test_ac, normalize=True)
        delay_phi = parabolic_delay(lags1, corr_phi, dt)
        results_phi.append(delay_phi*1e12 / integrals_abs)

        # ---- 2D Stokes method (bounded window) ----
        noise_var_stokes = peak_power_stokes / (2.0*snr_lin)
        stokes_noisy = np.zeros_like(stokes_clean)
        for ch in range(N_CHANNELS):
            noise = np.sqrt(noise_var_stokes[ch])*rng.randn(n_samples,3)
            stokes_noisy[ch] = stokes_clean[ch] + noise
            if USE_FILTER:
                for comp in range(3):
                    stokes_noisy[ch,:,comp] = filtfilt(b_lp, a_lp, stokes_noisy[ch,:,comp])
        stokes_rot = np.zeros_like(stokes_noisy)
        for ch in range(N_CHANNELS):
            S_rot, _ = rotate_centroid_to_north_pole(stokes_noisy[ch])
            stokes_rot[ch] = S_rot
        sig_ref_2d = stokes_rot[ch_fast, :, :2]
        sig_test_2d = stokes_rot[ch_slow, :, :2]
        corr_2d = np.zeros_like(lags, dtype=float)
        for comp in range(2):
            _, c = cross_correlation_1d_fft(sig_ref_2d[:,comp], sig_test_2d[:,comp], normalize=False)
            corr_2d += c
        corr_2d /= np.sqrt(np.sum(sig_ref_2d**2)*np.sum(sig_test_2d**2))
        mask = (lags*dt >= 0) & (lags*dt <= tau_max)
        corr_windowed = np.where(mask, np.abs(corr_2d), 0.0)
        delay_2d = parabolic_delay(lags, corr_windowed, dt)
        results_2d.append(delay_2d*1e12 / integrals_abs)

# ========== Statistics with IQR filtering ==========
quat_arr = np.array(results_quat)
phi_arr  = np.array(results_phi)
stk_arr  = np.array(results_2d)

# IQR filter (factor=1.0)
mask_quat = iqr_filter(quat_arr, factor=1.0)
mask_phi  = iqr_filter(phi_arr,  factor=1.0)
mask_stk  = iqr_filter(stk_arr,  factor=1.0)

quat_filt = quat_arr[mask_quat]
phi_filt  = phi_arr[mask_phi]
stk_filt  = stk_arr[mask_stk]

print(f"Total samples per method: {len(quat_arr)}")
print(f"After IQR filtering (factor=1.0): Quat={np.sum(mask_quat)}, Rot‑vec={np.sum(mask_phi)}, 2D Stokes={np.sum(mask_stk)}")
print(f"True distance: {true_dist:.1f} km")
print()
print("Method            Std (km)   Bias (km)")
print("Quaternion 3D     {:.3f}      {:.3f}".format(np.std(quat_filt), np.mean(quat_filt)-true_dist))
print("Rot‑vec magnitude {:.3f}      {:.3f}".format(np.std(phi_filt),  np.mean(phi_filt)-true_dist))
print("2D Stokes (bounded) {:.3f}    {:.3f}".format(np.std(stk_filt),  np.mean(stk_filt)-true_dist))

# Optional: quick bar chart (filtered data)
import matplotlib.pyplot as plt
methods = ['Quat 3D', 'Rot‑vec', '2D Stokes']
stds = [np.std(quat_filt), np.std(phi_filt), np.std(stk_filt)]
biases = [np.mean(quat_filt)-true_dist, np.mean(phi_filt)-true_dist, np.mean(stk_filt)-true_dist]

fig, ax = plt.subplots(figsize=(6,4))
x = np.arange(len(methods))
width = 0.35
bars = ax.bar(x, stds, width, label='Std (km)')
ax.bar(x + width, biases, width, label='Bias (km)')
ax.set_ylabel('km')
ax.set_title('Monte‑Carlo comparison (IQR filtered)')
ax.set_xticks(x + width/2)
ax.set_xticklabels(methods)
ax.legend()
plt.tight_layout()
plt.savefig('mc_comparison.png', dpi=150)
plt.close()
print("Bar chart saved to mc_comparison.png")