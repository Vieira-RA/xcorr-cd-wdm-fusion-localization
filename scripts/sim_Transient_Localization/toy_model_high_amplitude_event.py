#!/usr/bin/env python3
"""
Toy Monte‑Carlo: 2D Stokes vs rotation‑vector magnitude, with three delay estimators.
Methods: parabolic xcorr, GCC‑PHAT, weighted phase‑slope.
10 fibre seeds × 10 noise realisations = 100 estimates per method.
IQR factor 1.0 applied before statistics.
One waveform example is saved for fibre 0, realisation 0.
"""
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

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
    phase_slope_delay_1d,
    gcc_phat_1d,
    phase_slope_delay_2d,
    gcc_phat_2d,
)
from rotations import rotate_centroid_to_north_pole
from estimation import iqr_filter

# ========== Parameters ==========
BANDWIDTH_HZ = 50e3
SNR_DB = 50.0
EVENT_DIST_KM = 500.0
N_CHANNELS = 15               # only the farthest pair is used
F_MIN_HZ = 184e12
F_MAX_HZ = 196e12
CHANNEL_SPACING_HZ = 50e9
FS_FACTOR = 45
T_END_FACTOR = 160
A_PULSE = 500 * 3.1e-4
FIBER_SEED = 5                # base seed for first fibre
NOISE_SEED = 2                # base noise seed (not used, we use i_real)
FMAX_FACTOR = 2.0
USE_FILTER = True
MIN_DELTA_LAMBDA_NM = 0

N_FIBRES = 10
N_REAL = 10

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

# Use only the two farthest channels
ch_slow, ch_fast = 0, N_CHANNELS - 1
list_ch = [ch_slow, ch_fast]

# ========== Fibre parameters ==========
L_km = 0.5
L = L_km * 1e3
L_F = 20.0
D_pmd = 2.5298e-15
lambda0_nm = 1550
lambda0 = lambda0_nm * 1e-9
c = 299792458.0
omega0 = 2*np.pi*c/lambda0

channel_delays = relative_channel_delays(wavelengths_nm, EVENT_DIST_KM)
true_delay = channel_delays[ch_slow] - channel_delays[ch_fast]
true_dist = EVENT_DIST_KM

integrals_abs = abs(integrated_dispersion(wavelengths_nm[ch_slow], wavelengths_nm[ch_fast]))

g = np.exp(-((t_grid - t0)**2) / (2*sigma_t**2))
s_env = 1.0 + A_PULSE * g

nyq = 0.5 / dt
b_lp, a_lp = butter(4, 10*BANDWIDTH_HZ/nyq, btype='low')
f_max = FMAX_FACTOR * BANDWIDTH_HZ

tau_max = true_delay + 5*sigma_t   # window for parabolic 2D

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

# ========== Results storage ==========
results = {
    'parabolic_2d': [],
    'phat_2d': [],
    'phase_2d': [],
    'parabolic_rv': [],
    'phat_rv': [],
    'phase_rv': [],
}

waveforms_saved = False

# ========== Monte Carlo loop ==========
for i_fibre in range(N_FIBRES):
    fibre_seed = FIBER_SEED + i_fibre
    z, beta0, beta_prime, _ = generate_pmd_waveplates(L, L_F, D_pmd, lambda0, seed=fibre_seed)

    # Jones matrices for the two extreme channels
    U_all = {}
    for ch in list_ch:
        omega_ch = omega_channels[ch]
        beta_base = beta0 + (omega_ch - omega0)*beta_prime
        U_seq = np.zeros((n_samples, 2, 2), dtype=complex)
        for t_idx in range(n_samples):
            beta_t = s_env[t_idx] * beta_base
            U_seq[t_idx] = propagate_unitary(z, beta_t)
        U_all[ch] = delay_jones_sequence(U_seq, dt, channel_delays[ch])

    U_ref_clean = U_all[ch_fast]   # earlier
    U_test_clean = U_all[ch_slow]  # later

    # Rotation‑vector magnitude (clean)
    def jones_sequence_to_phi_mag(U_seq):
        n = len(U_seq)
        Q = np.zeros((n,4))
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

    # Clean Stokes (pre‑rotation)
    s_in = np.array([1.0, 0.0, 0.0])
    stokes_clean = {}
    for ch in list_ch:
        Q = np.zeros((n_samples, 4))
        for t in range(n_samples):
            Q[t] = jones_to_quaternion(U_all[ch][t])
        Q = regularize_signs(Q)
        stokes_clean[ch] = quaternion_rotate_stokes(Q, s_in)

    # Clean rotated Stokes for SNR definition
    stokes_clean_rot = {}
    for ch in list_ch:
        S_rot_clean, _ = rotate_centroid_to_north_pole(stokes_clean[ch])
        stokes_clean_rot[ch] = S_rot_clean

    peak_power_stokes = max(np.max(stokes_clean_rot[ch][:,0]**2 + stokes_clean_rot[ch][:,1]**2)
                            for ch in list_ch)
    phi_ref_ac_clean = phi_ref_clean - np.mean(phi_ref_clean)
    phi_test_ac_clean = phi_test_clean - np.mean(phi_test_clean)
    peak_power_phi = max(np.max(phi_ref_ac_clean**2), np.max(phi_test_ac_clean**2))

    for i_real in range(N_REAL):
        rng = np.random.RandomState(i_real)
        snr_lin = 10**(SNR_DB / 10.0)

        # ---------- Rotation‑vector processing ----------
        noise_var_phi = peak_power_phi / snr_lin
        phi_ref_noisy = phi_ref_clean + np.sqrt(noise_var_phi)*rng.randn(n_samples)
        phi_test_noisy = phi_test_clean + np.sqrt(noise_var_phi)*rng.randn(n_samples)
        if USE_FILTER:
            phi_ref_noisy = filtfilt(b_lp, a_lp, phi_ref_noisy)
            phi_test_noisy = filtfilt(b_lp, a_lp, phi_test_noisy)
        phi_ref_ac = phi_ref_noisy - np.mean(phi_ref_noisy)
        phi_test_ac = phi_test_noisy - np.mean(phi_test_noisy)

        # RV – parabolic
        lags1, corr_phi = cross_correlation_1d_fft(phi_ref_ac, phi_test_ac, normalize=True)
        delay_rv_parabolic = parabolic_delay(lags1, corr_phi, dt)

        # RV – phase‑slope
        delay_rv_phase = phase_slope_delay_1d(phi_ref_ac, phi_test_ac, dt, weighted=True)

        # RV – GCC‑PHAT
        delay_rv_phat = gcc_phat_1d(phi_ref_ac, phi_test_ac, dt, f_max, 0.05)

        results['parabolic_rv'].append(delay_rv_parabolic * 1e12 / integrals_abs)
        results['phase_rv'].append(delay_rv_phase * 1e12 / integrals_abs)
        results['phat_rv'].append(delay_rv_phat * 1e12 / integrals_abs)

        # ---------- 2D Stokes processing ----------
        noise_var_stokes = peak_power_stokes / (2.0 * snr_lin)
        stokes_noisy = {}
        for ch in list_ch:
            noise = np.sqrt(noise_var_stokes)*rng.randn(n_samples, 3)
            sig = stokes_clean[ch] + noise
            if USE_FILTER:
                for comp in range(3):
                    sig[:, comp] = filtfilt(b_lp, a_lp, sig[:, comp])
            stokes_noisy[ch] = sig

        stokes_rot = {}
        for ch in list_ch:
            S_rot, _ = rotate_centroid_to_north_pole(stokes_noisy[ch])
            stokes_rot[ch] = S_rot

        sig_ref_2d = stokes_rot[ch_fast][:, :2]
        sig_test_2d = stokes_rot[ch_slow][:, :2]

        # 2D – parabolic (bounded window)
        corr_2d = np.zeros_like(lags1, dtype=float)
        for comp in range(2):
            _, c = cross_correlation_1d_fft(sig_ref_2d[:,comp], sig_test_2d[:,comp], normalize=False)
            corr_2d += c
        corr_2d /= np.sqrt(np.sum(sig_ref_2d**2)*np.sum(sig_test_2d**2))
        mask = (lags1*dt >= 0) & (lags1*dt <= tau_max)
        corr_windowed = np.where(mask, np.abs(corr_2d), 0.0)
        delay_2d_parabolic = parabolic_delay(lags1, corr_windowed, dt)

        # 2D – phase‑slope
        delay_2d_phase = phase_slope_delay_2d(sig_ref_2d, sig_test_2d, dt, weighted=True)

        # 2D – GCC‑PHAT
        delay_2d_phat = gcc_phat_2d(sig_ref_2d, sig_test_2d, dt, f_max, 0.05)

        results['parabolic_2d'].append(delay_2d_parabolic * 1e12 / integrals_abs)
        results['phase_2d'].append(delay_2d_phase * 1e12 / integrals_abs)
        results['phat_2d'].append(delay_2d_phat * 1e12 / integrals_abs)

        # Save waveform example
        if not waveforms_saved and i_fibre == 0 and i_real == 0:
            waveforms_saved = True
            fig_wave, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax1.plot(t_grid*1e6, sig_ref_2d[:,0], label='S1 ref (fast)')
            ax1.plot(t_grid*1e6, sig_ref_2d[:,1], label='S2 ref')
            ax1.plot(t_grid*1e6, sig_test_2d[:,0], '--', label='S1 test (slow)')
            ax1.plot(t_grid*1e6, sig_test_2d[:,1], '--', label='S2 test')
            ax1.set_ylabel('Rotated Stokes')
            ax1.set_title('2D Stokes waveforms')
            ax1.legend(fontsize=7)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([500, 700])

            ax2.plot(t_grid*1e6, phi_ref_ac, label='|φ| ref (fast)')
            ax2.plot(t_grid*1e6, phi_test_ac, '--', label='|φ| test (slow)')
            ax2.set_xlabel('Time (µs)')
            ax2.set_ylabel('|φ| (rad)')
            ax2.set_title('Rotation‑vector magnitude (DC‑free)')
            ax2.legend(fontsize=7)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([500, 700])

            plt.tight_layout()
            plt.savefig('waveform_example.png', dpi=150)
            plt.close(fig_wave)
            print("Waveform example saved to waveform_example.png")

# ========== Statistics with IQR ==========
print(f"Total samples per method: {N_FIBRES*N_REAL}")
print(f"True distance: {true_dist:.1f} km")
print()
print(f"{'Method':<25} {'Std (km)':>10} {'Bias (km)':>10} {'Inliers':>8}")
for key, label in [
    ('parabolic_2d', '2D Parabolic'),
    ('phat_2d', '2D GCC‑PHAT'),
    ('phase_2d', '2D Phase‑slope'),
    ('parabolic_rv', 'RV Parabolic'),
    ('phat_rv', 'RV GCC‑PHAT'),
    ('phase_rv', 'RV Phase‑slope'),
]:
    arr = np.array(results[key])
    mask = iqr_filter(arr, factor=1.0)
    filt = arr[mask]
    print(f"{label:<25} {np.std(filt):10.3f} {np.mean(filt)-true_dist:10.3f} {np.sum(mask):8d}")