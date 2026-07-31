"""
Tuning of robust adaptive weighting for CD localisation.
- Simulates a single (bandwidth, SNR) point with many noise realisations.
- Stores the raw pair distance estimates.
- Sweeps over scale method, scale multiplier, and bisquare tuning constant.
- Plots RMSE colour maps and prints the best parameter combinations.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import itertools
import time
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
from rotations import jones_to_rotation_matrix, rotate_centroid_to_north_pole
from signal_processing import phase_slope_delay_2d, gcc_phat_2d
from noise import add_noise_to_stokes
from estimation import iqr_filter

# ========================= Configuration =========================
# Fixed test point
BANDWIDTH_TEST = 1e3          # 20 kHz
SNR_TEST = 74.0                # dB
N_REAL = 200                   # number of noise realisations

# Fibre and channels (same as in your sweep)
L_km, L = 0.5, 0.5e3
L_F, D_pmd = 20.0, 2.5298e-15
lambda0, c = 1550e-9, 299792458.0
omega0 = 2 * np.pi * c / lambda0
A_pulse = 10 * 3.1e-4

N_CHANNELS = 20
f_min, f_max = 184e12, 196e12
freq_channels = np.linspace(f_min, f_max, N_CHANNELS)
omega_channels = 2 * np.pi * freq_channels
wavelengths_nm = frequency_to_wavelength(freq_channels)

event_distance_km = 500.0

FIBER_SEED = 0
BASE_NOISE_SEED = 0
FMAX_FACTOR = 2.0

# ========================= Pre‑compute static data ==============
z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
    L, L_F, D_pmd, lambda0, seed=FIBER_SEED)
channel_delays = relative_channel_delays(wavelengths_nm, event_distance_km)

pair_indices = list(itertools.combinations(range(N_CHANNELS), 2))
n_pairs = len(pair_indices)
integrals_abs = np.array([
    abs(integrated_dispersion(wavelengths_nm[i], wavelengths_nm[j]))
    for i, j in pair_indices
])

# ========================= Step 1: Generate raw pair distances =====
print("Generating raw pair distances...")
bw = BANDWIDTH_TEST
sigma_t = 0.3748 / bw
fs = 30 * bw
dt = 1.0 / fs
t_end = 160 * sigma_t
n_samples = int(t_end / dt) + 1
t_grid = np.linspace(0, t_end, n_samples)
t0 = t_end / 2

g = np.exp(-((t_grid - t0)**2) / (2 * sigma_t**2))
s_env = 1.0 + A_pulse * g

U_all = np.zeros((N_CHANNELS, n_samples, 2, 2), dtype=complex)
for ch_idx, omega_ch in enumerate(omega_channels):
    beta_base = beta0 + (omega_ch - omega0) * beta_prime
    for t_idx in range(n_samples):
        U_all[ch_idx, t_idx] = propagate_unitary(z, s_env[t_idx] * beta_base)

for ch in range(N_CHANNELS):
    U_all[ch] = delay_jones_sequence(U_all[ch], dt, channel_delays[ch])

s_in = np.array([1.0, 0.0, 0.0])
stokes_clean = np.zeros((N_CHANNELS, n_samples, 3))
for ch in range(N_CHANNELS):
    for t in range(n_samples):
        stokes_clean[ch, t] = jones_to_rotation_matrix(U_all[ch, t]) @ s_in

f_max = FMAX_FACTOR * bw
nyq = 0.5 * fs
b_lp, a_lp = butter(4, 3*bw/nyq, btype='low')

phase_d_all = np.zeros((N_REAL, n_pairs))   # storage for raw pair distances
phat_d_all  = np.zeros((N_REAL, n_pairs))

for i_real in tqdm(range(N_REAL), desc="Realisation"):
    rng = np.random.RandomState(BASE_NOISE_SEED + i_real)
    stokes_noisy = np.array([add_noise_to_stokes(stokes_clean[ch], SNR_TEST, rng)
                             for ch in range(N_CHANNELS)])
    # low‑pass filter
    stokes_filt = np.zeros_like(stokes_noisy)
    for ch in range(N_CHANNELS):
        for comp in range(3):
            stokes_filt[ch, :, comp] = filtfilt(b_lp, a_lp,
                                                stokes_noisy[ch, :, comp])
    stokes_rot = np.array([rotate_centroid_to_north_pole(s)[0]
                           for s in stokes_filt])

    phase_delays = np.zeros(n_pairs)
    phat_delays = np.zeros(n_pairs)
    for idx, (i, j) in enumerate(pair_indices):
        # i = slower, j = faster – we want positive delay
        sig_ref = stokes_rot[j, :, :2]
        sig_test = stokes_rot[i, :, :2]
        phase_delays[idx] = phase_slope_delay_2d(sig_ref, sig_test, dt, True)
        phat_delays[idx]  = gcc_phat_2d(sig_ref, sig_test, dt, f_max, 0.05)

    phase_d_all[i_real] = phase_delays * 1e12 / integrals_abs
    phat_d_all[i_real]  = phat_delays  * 1e12 / integrals_abs

# Save to disk
np.savez_compressed('tuning_data.npz',
                    phase_d_all=phase_d_all,
                    phat_d_all=phat_d_all,
                    integrals_abs=integrals_abs,
                    true_dist=event_distance_km)
print("Raw data saved to tuning_data.npz")

# ========================= Step 2: Tune parameters ==============
def tuned_weighted_mean(distances, D_abs,
                        iqr_factor=2.0,
                        scale_method='MAD',
                        scale_multiplier=1.0,
                        c_bisquare=2.5):
    """
    Adaptive weighted mean with configurable robustness parameters.
    """
    # 1. IQR pre‑filter
    mask = iqr_filter(distances, factor=iqr_factor)
    Z = distances[mask]
    D = D_abs[mask]
    if len(Z) == 0:
        return np.nan, np.zeros_like(distances)

    # 2. Wavelength‑separation weight (fixed D²)
    w_lambda = D ** 2
    w_lambda /= np.median(w_lambda)

    # 3. Robustness weight
    median_Z = np.median(Z)
    residuals = np.abs(Z - median_Z)

    if scale_method == 'MAD':
        mad = np.median(residuals)
        base_scale = 1.4826 * mad
    else:  # 'IQR'
        q1, q3 = np.percentile(Z, 25), np.percentile(Z, 75)
        base_scale = (q3 - q1)   # IQR itself

    scale = scale_multiplier * base_scale
    if scale < 1e-12:
        w_robust = np.ones_like(Z)
    else:
        t = residuals / (c_bisquare * scale)
        w_robust = np.where(t < 1.0, (1 - t**2)**2, 0.0)

    w_total = w_lambda * w_robust
    wmean = np.sum(w_total * Z) / np.sum(w_total)

    full_weights = np.zeros_like(distances)
    full_weights[mask] = w_total
    return wmean, full_weights

# Load the saved data
data = np.load('tuning_data.npz')
phase_d_all = data['phase_d_all']
phat_d_all  = data['phat_d_all']
integrals_abs = data['integrals_abs']
true_dist = data['true_dist'].item()

# Parameter grids
c_vals = np.linspace(1.0, 5.1, 60)      # bisquare tuning constant
sm_vals = np.linspace(0.5, 3.1, 60)     # scale multiplier
scale_methods = ['MAD', 'IQR']

# Result arrays
results = {}
for method in scale_methods:
    results[method] = {
        'phase': np.full((len(c_vals), len(sm_vals)), np.nan),
        'phat':  np.full((len(c_vals), len(sm_vals)), np.nan)
    }

print("Sweeping parameters...")
for method in scale_methods:
    for i_c, c_val in enumerate(tqdm(c_vals, desc=f"c_bisquare ({method})")):
        for j_sm, sm_val in enumerate(sm_vals):
            phase_est = []
            phat_est = []
            for k in range(N_REAL):
                wm_phase, _ = tuned_weighted_mean(
                    phase_d_all[k], integrals_abs,
                    iqr_factor=2.0,
                    scale_method=method,
                    scale_multiplier=sm_val,
                    c_bisquare=c_val)
                wm_phat, _ = tuned_weighted_mean(
                    phat_d_all[k], integrals_abs,
                    iqr_factor=2.0,
                    scale_method=method,
                    scale_multiplier=sm_val,
                    c_bisquare=c_val)
                if not np.isnan(wm_phase):
                    phase_est.append(wm_phase)
                if not np.isnan(wm_phat):
                    phat_est.append(wm_phat)
            if phase_est:
                rmse_phase = np.sqrt(np.mean((np.array(phase_est) - true_dist)**2))
                results[method]['phase'][i_c, j_sm] = rmse_phase
            if phat_est:
                rmse_phat = np.sqrt(np.mean((np.array(phat_est) - true_dist)**2))
                results[method]['phat'][i_c, j_sm] = rmse_phat

# ========================= Plotting ============================
def plot_rmse_heatmap(rmse, title, filename):
    fig, ax = plt.subplots(figsize=(8,6))
    X, Y = np.meshgrid(sm_vals, c_vals)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('white')
    c = ax.pcolormesh(X, Y, rmse, shading='auto', cmap=cmap)
    ax.set_xlabel('scale_multiplier')
    ax.set_ylabel('c_bisquare')
    ax.set_title(title)
    fig.colorbar(c, ax=ax, label='RMSE (km)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

for method in scale_methods:
    plot_rmse_heatmap(results[method]['phase'],
                      f'Phase‑slope RMSE – {method} scale',
                      f'tune_phase_{method}.png')
    plot_rmse_heatmap(results[method]['phat'],
                      f'GCC‑PHAT RMSE – {method} scale',
                      f'tune_phat_{method}.png')

# Print best parameters
for method in scale_methods:
    for est in ['phase', 'phat']:
        rmse = results[method][est]
        idx = np.nanargmin(rmse)
        i_c, j_sm = np.unravel_index(idx, rmse.shape)
        print(f"{est} {method}: best RMSE = {rmse[i_c,j_sm]:.2f} km "
              f"at c={c_vals[i_c]:.2f}, scale_multiplier={sm_vals[j_sm]:.2f}")

print("Tuning complete. Heatmaps saved.")