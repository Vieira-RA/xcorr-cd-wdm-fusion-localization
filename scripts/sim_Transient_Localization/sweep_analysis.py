"""
Multi‑channel CD localisation: sweep bandwidth & SNR.
Four delay‑estimation methods compared:
  - Weighted phase‑slope
  - Bandwidth‑masked GCC‑PHAT
  - Integer‑sample cross‑correlation peak
  - Parabolic‑refined cross‑correlation peak
Generates Figures 1,3,4,6,7,8 with all methods.
"""

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import itertools

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
from signal_processing import (
    phase_slope_delay_2d,
    gcc_phat_2d,
    integer_corr_delay_2d,
    parabolic_corr_delay_2d,
)
from noise import add_noise_to_stokes
from estimation import iqr_filter, adaptive_weighted_mean
from plotting import (
    plot_std_heatmap,
    plot_bias_heatmap,
    plot_success_rate_heatmap,
)
from scipy.signal import butter, filtfilt

# ========================= Configuration =========================
L_km, L = 0.5, 0.5e3
L_F, D_pmd = 20.0, 2.5298e-15
lambda0, c = 1550e-9, 299792458.0
omega0 = 2 * np.pi * c / lambda0
A_pulse = 10 * 3.1e-4

N_CHANNELS = 5
f_min, f_max = 184e12, 196e12
freq_channels = np.linspace(f_min, f_max, N_CHANNELS)
omega_channels = 2 * np.pi * freq_channels
wavelengths_nm = frequency_to_wavelength(freq_channels)

event_distance_km = 500.0

N_BANDWIDTHS, N_REAL = 5, 5
N_SNRS = N_BANDWIDTHS

BANDWIDTHS = np.logspace(np.log10(.1e3), np.log10(1e6), N_BANDWIDTHS)
SNRS = np.linspace(50, 100, N_SNRS)

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

# ========================= Storage =============================
phase_est = np.full((len(BANDWIDTHS), len(SNRS), N_REAL), np.nan)
phat_est  = np.full_like(phase_est, np.nan)
int_est   = np.full_like(phase_est, np.nan)   # integer peak
par_est   = np.full_like(phase_est, np.nan)   # parabolic peak

# For Fig 6: store raw pair distances for one representative setting
demo_bw = BANDWIDTHS[0]   # use the first bandwidth (adjust as desired)
demo_snr = SNRS[0]        # and first SNR
demo_phase_pairs = None
demo_phat_pairs = None
demo_int_pairs = None
demo_par_pairs = None

# For Fig 8: collect IQR‑filtered pair estimates (all methods)
phase_inlier_pairs = []
phat_inlier_pairs  = []
int_inlier_pairs   = []
par_inlier_pairs   = []

# ========================= Main sweep ===========================
total_bw = len(BANDWIDTHS)
bw_times = []
overall_start = time.time()

for i_bw, bw in enumerate(tqdm(BANDWIDTHS, desc="Bandwidth sweep")):
    bw_start = time.time()
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

    # --- Low‑pass filter design (zero‑phase, cutoff = 3×bandwidth) ---
    nyq = 0.5 * fs
    b_lp, a_lp = butter(4, 3*bw/nyq, btype='low')

    for i_snr, snr_db in enumerate(SNRS):
        for i_real in range(N_REAL):
            rng = np.random.RandomState(BASE_NOISE_SEED + i_real)
            stokes_noisy = np.array([add_noise_to_stokes(stokes_clean[ch], snr_db, rng)
                                     for ch in range(N_CHANNELS)])
            # Apply the zero‑phase low‑pass filter
            stokes_filt = np.zeros_like(stokes_noisy)
            for ch in range(N_CHANNELS):
                for comp in range(3):
                    stokes_filt[ch, :, comp] = filtfilt(b_lp, a_lp,
                                                        stokes_noisy[ch, :, comp])
            stokes_rot = np.array([rotate_centroid_to_north_pole(s)[0]
                                   for s in stokes_filt])

            # Allocate delay arrays for all four methods
            phase_delays = np.zeros(n_pairs)
            phat_delays  = np.zeros(n_pairs)
            int_delays   = np.zeros(n_pairs)
            par_delays   = np.zeros(n_pairs)

            for idx, (i, j) in enumerate(pair_indices):
                # i = slower (later), j = faster (earlier)
                sig_ref = stokes_rot[j, :, :2]   # earlier arrival
                sig_test = stokes_rot[i, :, :2]  # later arrival

                phase_delays[idx] = phase_slope_delay_2d(sig_ref, sig_test, dt, True)
                phat_delays[idx]  = gcc_phat_2d(sig_ref, sig_test, dt, f_max, 0.05)
                int_delays[idx]   = integer_corr_delay_2d(sig_ref, sig_test, dt)
                par_delays[idx]   = parabolic_corr_delay_2d(sig_ref, sig_test, dt)

            # Convert to distances
            phase_d = phase_delays * 1e12 / integrals_abs
            phat_d  = phat_delays  * 1e12 / integrals_abs
            int_d   = int_delays   * 1e12 / integrals_abs
            par_d   = par_delays   * 1e12 / integrals_abs

            # IQR filtering for each method
            phase_mask = iqr_filter(phase_d)
            phat_mask  = iqr_filter(phat_d)
            int_mask   = iqr_filter(int_d)
            par_mask   = iqr_filter(par_d)

            # Store inliers for global figure
            phase_inlier_pairs.append(phase_d[phase_mask])
            phat_inlier_pairs.append(phat_d[phat_mask])
            int_inlier_pairs.append(int_d[int_mask])
            par_inlier_pairs.append(par_d[par_mask])

            # if demo setting, keep raw pairs (all methods)
            if (abs(bw - demo_bw) < 1e-3 and abs(snr_db - demo_snr) < 1e-3
                and i_real == 0):
                demo_phase_pairs = phase_d.copy()
                demo_phat_pairs  = phat_d.copy()
                demo_int_pairs   = int_d.copy()
                demo_par_pairs   = par_d.copy()

            # Adaptive weighted mean for each method
            phase_est[i_bw, i_snr, i_real] = adaptive_weighted_mean(
                phase_d, phase_mask, integrals_abs)[0]
            phat_est[i_bw, i_snr, i_real]  = adaptive_weighted_mean(
                phat_d,  phat_mask,  integrals_abs)[0]
            int_est[i_bw, i_snr, i_real]   = adaptive_weighted_mean(
                int_d,   int_mask,   integrals_abs)[0]
            par_est[i_bw, i_snr, i_real]   = adaptive_weighted_mean(
                par_d,   par_mask,   integrals_abs)[0]

    bw_times.append(time.time() - bw_start)
    remaining = np.mean(bw_times) * (total_bw - i_bw - 1)
    print(f"B={bw:.0f} Hz done. Est. remaining: {remaining/60:.1f} min")

# ========================= Compute metrics =====================
true_dist = event_distance_km

def compute_metrics(estimates):
    """Return std, bias, success (relative 10%), success (absolute ±5 km)."""
    std = np.nanstd(estimates, axis=2)
    bias = np.nanmean(estimates, axis=2) - true_dist
    success_rel = np.nanmean(np.abs(estimates - true_dist) < 0.1*true_dist, axis=2)
    success_abs = np.nanmean(np.abs(estimates - true_dist) < 5.0, axis=2)
    return std, bias, success_rel, success_abs

phase_std, phase_bias, phase_success, phase_success_abs = compute_metrics(phase_est)
phat_std,  phat_bias,  phat_success,  phat_success_abs  = compute_metrics(phat_est)
int_std,   int_bias,   int_success,   int_success_abs   = compute_metrics(int_est)
par_std,   par_bias,   par_success,   par_success_abs   = compute_metrics(par_est)

# ========================= Figure 1: Schematic =================
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([0,1,2,3], [0,0.5,0,0], 'ko-', markersize=4)
ax.text(0,0.8,'Laser\ncomb', ha='center')
ax.text(1,0.8,'Sensing\nfibre', ha='center')
ax.text(2,0.8,'CD delay\nτ(λ)', ha='center')
ax.text(3,0.8,'Polarimeter\n→ S₁,S₂', ha='center')
ax.set_title('Simplified sensing principle')
ax.axis('off')
plt.tight_layout()
plt.savefig('fig01_schematic.png', dpi=150)
plt.close()

# ========================= Figure 3: Std heatmaps ==============
methods_std = {
    'Phase‑slope': phase_std,
    'GCC‑PHAT': phat_std,
    'Integer peak': int_std,
    'Parabolic peak': par_std,
}
for name, std in methods_std.items():
    plot_std_heatmap(std, SNRS, BANDWIDTHS,
                     f'{name} std (km)', f'fig03_{name.replace(" ","_").lower()}_std.png',
                     vmax=100)

# ========================= Figure 4: Bias & success ============
methods_bias = {
    'Phase‑slope': phase_bias,
    'GCC‑PHAT': phat_bias,
    'Integer peak': int_bias,
    'Parabolic peak': par_bias,
}
for name, bias in methods_bias.items():
    plot_bias_heatmap(bias, SNRS, BANDWIDTHS,
                      f'{name} bias (km)', f'fig04_{name.replace(" ","_").lower()}_bias.png',
                      vmin=-50, vmax=50)

methods_success_rel = {
    'Phase‑slope': phase_success,
    'GCC‑PHAT': phat_success,
    'Integer peak': int_success,
    'Parabolic peak': par_success,
}
for name, succ in methods_success_rel.items():
    plot_success_rate_heatmap(succ, SNRS, BANDWIDTHS,
                              f'{name} success rate (10%)',
                              f'fig04_{name.replace(" ","_").lower()}_success_rel.png')

methods_success_abs = {
    'Phase‑slope': phase_success_abs,
    'GCC‑PHAT': phat_success_abs,
    'Integer peak': int_success_abs,
    'Parabolic peak': par_success_abs,
}
for name, succ in methods_success_abs.items():
    plot_success_rate_heatmap(succ, SNRS, BANDWIDTHS,
                              f'{name} success rate (±5 km)',
                              f'fig04_{name.replace(" ","_").lower()}_success_abs.png')

# ========================= Figure 6: Pairwise scatter ==========
if demo_phase_pairs is not None:
    delta_lambda = np.array([
        np.abs(wavelengths_nm[i] - wavelengths_nm[j]) for i,j in pair_indices
    ])
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(delta_lambda, demo_phase_pairs, s=15, alpha=0.6, label='Phase‑slope')
    ax.scatter(delta_lambda, demo_phat_pairs, s=15, alpha=0.6, marker='s', label='GCC‑PHAT')
    ax.scatter(delta_lambda, demo_int_pairs, s=15, alpha=0.6, marker='D', label='Integer peak')
    ax.scatter(delta_lambda, demo_par_pairs, s=15, alpha=0.6, marker='^', label='Parabolic peak')
    ax.axhline(true_dist, color='k', ls='--')
    ax.set_xlabel('Wavelength separation (nm)')
    ax.set_ylabel('Estimated distance (km)')
    ax.set_title(f'Pairwise estimates, B={demo_bw*1e-3:.0f} kHz, SNR={demo_snr} dB')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig06_pairwise_scatter.png', dpi=150)
    plt.close()
else:
    print("Demo setting not found – pairwise scatter skipped.")

# ========================= Figure 7: Timing ====================
bw_samples = [int((160 * (0.3748/bw)) / (1/(30*bw))) + 1 for bw in BANDWIDTHS]
fig, ax = plt.subplots(figsize=(7,4))
ax.loglog(BANDWIDTHS, bw_times, 'o-')
ax.set_xlabel('Bandwidth (Hz)')
ax.set_ylabel('Computation time per bandwidth (s)')
ax.set_title('Execution time vs bandwidth (all SNRs, all realisations)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig07_timing.png', dpi=150)
plt.close()

# ========================= Figure 8: Global pair distribution =====
# Combine all inlier arrays
all_phase_in = np.concatenate(phase_inlier_pairs)
all_phat_in  = np.concatenate(phat_inlier_pairs)
all_int_in   = np.concatenate(int_inlier_pairs)
all_par_in   = np.concatenate(par_inlier_pairs)

fig, ax = plt.subplots(figsize=(12, 6))

# Determine sample size as min of lengths, capped at 5000
lens = [len(arr) for arr in (all_phase_in, all_phat_in, all_int_in, all_par_in)]
sample_size = min(5000, min(lens)) if all(l > 0 for l in lens) else 0

if sample_size > 0:
    idx_phase = np.random.choice(len(all_phase_in), sample_size, replace=False)
    idx_phat  = np.random.choice(len(all_phat_in),  sample_size, replace=False)
    idx_int   = np.random.choice(len(all_int_in),   sample_size, replace=False)
    idx_par   = np.random.choice(len(all_par_in),   sample_size, replace=False)

    ax.scatter(np.arange(sample_size), all_phase_in[idx_phase], s=2, alpha=0.4, label='Phase‑slope')
    ax.scatter(np.arange(sample_size), all_phat_in[idx_phat],   s=2, alpha=0.4, marker='s', label='GCC‑PHAT')
    ax.scatter(np.arange(sample_size), all_int_in[idx_int],     s=2, alpha=0.4, marker='D', label='Integer peak')
    ax.scatter(np.arange(sample_size), all_par_in[idx_par],     s=2, alpha=0.4, marker='^', label='Parabolic peak')

    mean_phase = np.mean(all_phase_in)
    mean_phat  = np.mean(all_phat_in)
    mean_int   = np.mean(all_int_in)
    mean_par   = np.mean(all_par_in)

    ax.axhline(mean_phase, color='C0', linestyle='-', linewidth=2,
               label=f'Phase‑slope mean: {mean_phase:.1f} km')
    ax.axhline(mean_phat, color='C1', linestyle='-', linewidth=2,
               label=f'GCC‑PHAT mean: {mean_phat:.1f} km')
    ax.axhline(mean_int, color='C2', linestyle='-', linewidth=2,
               label=f'Integer peak mean: {mean_int:.1f} km')
    ax.axhline(mean_par, color='C3', linestyle='-', linewidth=2,
               label=f'Parabolic peak mean: {mean_par:.1f} km')
else:
    ax.text(0.5, 0.5, 'No inliers to plot', transform=ax.transAxes, ha='center', va='center')

ax.axhline(true_dist, color='k', linestyle='--', linewidth=1.5,
           label=f'True distance: {true_dist:.1f} km')

ax.set_xlabel('Arbitrary sample index')
ax.set_ylabel('Estimated distance (km)')
ax.set_title('IQR‑filtered individual pair distance estimates across the entire sweep (all methods)')
ax.set_ylim([300,700])
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig('fig08_global_pair_distribution.png', dpi=150)
plt.close()

print("All figures saved.")