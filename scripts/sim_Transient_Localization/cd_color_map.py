"""
Sweep bandwidth & SNR – compute standard deviation of the final distance estimate.

- Fixed fibre seed (reproducible PMD profile).
- Bandwidths: log‑spaced from 200 Hz to 1 MHz.
- SNRs: linear from 30 dB to 100 dB.
- For each (B, SNR) combination, run N_REAL = 20 independent noise realisations.
- For each realisation, compute the adaptive weighted mean distance from all
  channel pairs (phase‑slope and GCC‑PHAT).
- Collect the estimates and compute the standard deviation across realisations.
- Plot 2‑D colour maps: y = bandwidth (log scale), x = SNR (dB), colour = std (km).
"""
import time  # add this import at the top of your script
from matplotlib.colors import PowerNorm
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
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
from signal_processing import phase_slope_delay_2d, gcc_phat_2d

# ============================================================
# Global configuration
# ============================================================
# Fibre
L_km = 0.500
L = L_km * 1e3
L_F = 20.0
D_pmd = 2.5298e-15
lambda0 = 1550e-9
c = 299792458.0
omega0 = 2 * np.pi * c / lambda0

# Transient (bandwidth will be swept, keep A_pulse constant)
A_pulse = 10 * 3.1e-4

# WDM channels
n_channels = 20
f_min = 184e12
f_max = 196e12
freq_channels = np.linspace(f_min, f_max, n_channels)
omega_channels = 2 * np.pi * freq_channels
wavelengths_nm = frequency_to_wavelength(freq_channels)

# Chromatic dispersion event
event_distance_km = 4000.0            # true distance (km)

# Sweep parameters
BANDWIDTHS = np.logspace(np.log10(.2e3), np.log10(10e6), 30)   # 10 points
SNRS = np.linspace(50, 100, 30)                               # 8 points
N_REAL = 20   # number of noise realisations per grid point

# Fixed seeds
FIBER_SEED = 0            # fibre PMD profile (constant across sweep)
BASE_NOISE_SEED = 0       # starting noise seed (incremented per realisation)

# PHAT frequency mask factor ( f_max = factor * bandwidth )
FMAX_FACTOR = 2.0

# ============================================================
# Helper: add noise to Stokes vectors given SNR, using a dedicated RNG
# ============================================================
def add_noise_to_stokes(stokes, snr_db, rng=None):
    if rng is None:
        rng = np.random
    signal_power = np.mean(np.sum(stokes**2, axis=1))
    snr_lin = 10**(snr_db / 10.0)
    noise_var = signal_power / snr_lin
    if noise_var <= 0:
        return stokes
    noise = np.sqrt(noise_var) * rng.randn(*stokes.shape)
    return stokes + noise

# ============================================================
# Helper: 1×IQR outlier filter
# ============================================================
def iqr_filter(data, factor=1.0):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (data >= lower) & (data <= upper)

# ============================================================
# Adaptive weighted mean (same as before)
# ============================================================
def adaptive_weighted_mean(distances, inlier_mask, D_abs, c_bisquare=2.5, iqr_factor=1.5):
    Z = distances[inlier_mask]
    D = D_abs[inlier_mask]
    if len(Z) == 0:
        return np.nan, np.zeros_like(distances)

    w_lambda = D ** 2
    w_lambda /= np.median(w_lambda)

    median_Z = np.median(Z)
    residuals = np.abs(Z - median_Z)
    q1 = np.percentile(Z, 25)
    q3 = np.percentile(Z, 75)
    iqr = q3 - q1
    scale = iqr_factor * iqr
    if scale < 1e-12:
        w_robust = np.ones_like(Z)
    else:
        t = residuals / (c_bisquare * scale)
        w_robust = np.where(t < 1.0, (1 - t**2)**2, 0.0)

    w_total = w_lambda * w_robust
    weighted_mean = np.sum(w_total * Z) / np.sum(w_total)

    full_weights = np.zeros_like(distances)
    full_weights[inlier_mask] = w_total
    return weighted_mean, full_weights

# ============================================================
# Pre‑compute static fibre profile (independent of bandwidth)
# ============================================================
z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
    L, L_F, D_pmd, lambda0, seed=FIBER_SEED
)

# Channel delays (depend only on wavelengths, not bandwidth)
channel_delays = relative_channel_delays(wavelengths_nm, event_distance_km)

# Pair indices and integrated dispersion (also bandwidth‑independent)
pair_indices = list(itertools.combinations(range(n_channels), 2))
n_pairs = len(pair_indices)
integrals_abs = np.zeros(n_pairs)
for idx, (i, j) in enumerate(pair_indices):
    lam_i = wavelengths_nm[i]
    lam_j = wavelengths_nm[j]
    integrals_abs[idx] = abs(integrated_dispersion(lam_i, lam_j))

# ============================================================
# Storage for final estimates
# shape: (n_bandwidth, n_snr, n_real)
phase_estimates = np.full((len(BANDWIDTHS), len(SNRS), N_REAL), np.nan)
phat_estimates = np.full_like(phase_estimates, np.nan)

# ============================================================
# Main sweep (with time‑remaining estimate)
# ============================================================
total_bw = len(BANDWIDTHS)
bw_times = []   # store elapsed time for each completed bandwidth

# We'll use a list to accumulate per‑bandwidth times and compute a moving average
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

    # Build the modulation envelope (noiseless)
    g = np.exp(-((t_grid - t0)**2) / (2 * sigma_t**2))
    s_env = 1.0 + A_pulse * g

    # Compute Jones matrices for all channels **once** for this bandwidth
    U_all = np.zeros((n_channels, n_samples, 2, 2), dtype=complex)
    for ch_idx, omega_ch in enumerate(omega_channels):
        delta_omega = omega_ch - omega0
        beta_base = beta0 + delta_omega * beta_prime
        for t_idx in range(n_samples):
            beta_t = s_env[t_idx] * beta_base
            U_all[ch_idx, t_idx] = propagate_unitary(z, beta_t)

    # Apply CD delays
    for ch in range(n_channels):
        U_all[ch] = delay_jones_sequence(U_all[ch], dt, channel_delays[ch])

    # Noiseless Stokes vectors
    s_in = np.array([1.0, 0.0, 0.0])
    stokes_clean = np.zeros((n_channels, n_samples, 3))
    for ch in range(n_channels):
        for t in range(n_samples):
            R = jones_to_rotation_matrix(U_all[ch, t])
            stokes_clean[ch, t] = R @ s_in

    f_max = FMAX_FACTOR * bw

    # Iterate over SNR and realisations
    for i_snr, snr_db in enumerate(SNRS):
        for i_real in range(N_REAL):
            # ... (rest of the inner loops exactly as before) ...
            noise_seed = BASE_NOISE_SEED + i_real
            rng = np.random.RandomState(noise_seed)

            stokes_noisy = np.zeros_like(stokes_clean)
            for ch in range(n_channels):
                stokes_noisy[ch] = add_noise_to_stokes(stokes_clean[ch], snr_db, rng=rng)

            stokes_rot = np.zeros_like(stokes_noisy)
            for ch in range(n_channels):
                S_rot, _ = rotate_centroid_to_north_pole(stokes_noisy[ch])
                stokes_rot[ch] = S_rot

            phase_delays = np.zeros(n_pairs)
            phat_delays = np.zeros(n_pairs)
            for idx, (i_ch, j_ch) in enumerate(pair_indices):
                sig_i = np.column_stack([stokes_rot[i_ch, :, 0], stokes_rot[i_ch, :, 1]])
                sig_j = np.column_stack([stokes_rot[j_ch, :, 0], stokes_rot[j_ch, :, 1]])
                phase_delays[idx] = phase_slope_delay_2d(sig_i, sig_j, dt, weighted=True)
                phat_delays[idx] = gcc_phat_2d(sig_i, sig_j, dt, f_max, mag_threshold=0.05)

            phase_distances = np.zeros(n_pairs)
            phat_distances = np.zeros(n_pairs)
            for idx in range(n_pairs):
                if integrals_abs[idx] < 1e-12:
                    continue
                delay_ps = phase_delays[idx] * 1e12
                phase_distances[idx] = delay_ps / integrals_abs[idx]
                delay_ps = phat_delays[idx] * 1e12
                phat_distances[idx] = delay_ps / integrals_abs[idx]

            phase_inliers = iqr_filter(phase_distances)
            phat_inliers = iqr_filter(phat_distances)

            wmean_phase, _ = adaptive_weighted_mean(phase_distances, phase_inliers, integrals_abs)
            wmean_phat, _ = adaptive_weighted_mean(phat_distances, phat_inliers, integrals_abs)

            phase_estimates[i_bw, i_snr, i_real] = wmean_phase
            phat_estimates[i_bw, i_snr, i_real] = wmean_phat

    # --- Time estimation ---
    bw_end = time.time()
    elapsed_this_bw = bw_end - bw_start
    bw_times.append(elapsed_this_bw)

    # Average time per bandwidth (use exponential smoothing or simple average)
    avg_per_bw = np.mean(bw_times)  # simple arithmetic mean of completed bandwidths
    remaining_bw = total_bw - (i_bw + 1)
    remaining_time = avg_per_bw * remaining_bw

    print(f"Bandwidth {bw:.0f} Hz completed. "
          f"Elapsed this step: {elapsed_this_bw:.1f} s, "
          f"estimated remaining: {remaining_time/60:.1f} min")

# ============================================================
# Compute standard deviation across realisations
# ============================================================
phase_std = np.nanstd(phase_estimates, axis=2)   # shape (n_bandwidth, n_snr)
phat_std = np.nanstd(phat_estimates, axis=2)

# Replace any zero or invalid entries with NaN for clean plotting
phase_std[phase_std == 0] = np.nan
phat_std[phat_std == 0] = np.nan

# ============================================================
# Colour maps
# ============================================================
def plot_heatmap(std_data, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(SNRS, BANDWIDTHS)

    # --- Choose colormap and normalisation ---
    cmap = plt.cm.plasma.copy()
    cmap.set_over('red')           # >100 km → red
    cmap.set_bad('white')          # NaN → white

    # Power‑law gamma to emphasise small values (gamma < 1)
    norm = PowerNorm(gamma=0.4, vmin=0, vmax=100)

    # Clip the data to the [0, 100] range for the base plot
    data_clipped = np.clip(std_data, 0, 100)

    c = ax.pcolormesh(X, Y, data_clipped, shading='auto',
                      cmap=cmap, norm=norm)

    # Overlay red for cells where std > 100
    red_mask = (std_data > 100) & (~np.isnan(std_data))
    if red_mask.any():
        red_data = np.ma.masked_where(~red_mask, np.ones_like(std_data))
        ax.pcolormesh(X, Y, red_data, shading='auto',
                      cmap=matplotlib.colors.ListedColormap(['red']),
                      vmin=0, vmax=1)

    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bandwidth (Hz)')
    ax.set_title(title)

    # Colour bar with extension arrow
    cbar = fig.colorbar(c, ax=ax, extend='max', label='Standard deviation (km)')
    # Keep linear ticks on the colour bar, even though the mapping is non‑linear
    cbar.set_ticks(np.linspace(0, 100, 6))
    cbar.set_ticklabels([f'{t:.0f}' for t in np.linspace(0, 100, 6)])

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

plot_heatmap(phase_std,
             'Phase‑slope method – std of final distance (km)',
             'sweep_phase_std.png')
plot_heatmap(phat_std,
             'GCC‑PHAT method – std of final distance (km)',
             'sweep_phat_std.png')

print("Sweep completed kkkk. Heatmaps saved.")