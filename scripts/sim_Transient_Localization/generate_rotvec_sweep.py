"""
Generator for rotation‑vector magnitude CD localisation sweep.
Saves results to sweep_rotvec.npz.
"""
import time
import numpy as np
from tqdm import tqdm
import itertools

from fiber_propagation import propagate_unitary
from pmd_model import generate_pmd_waveplates
from chromatic_dispersion import (
    frequency_to_wavelength,
    relative_channel_delays,
    delay_jones_sequence,
    integrated_dispersion,
)
from rotations import jones_to_rotation_matrix, rotate_centroid_to_north_pole
from quaternion import (
    jones_to_quaternion,
    regularize_signs,
    quaternion_to_rotation_vector,
)
from signal_processing import (
    phase_slope_delay_1d,
    gcc_phat_1d,
    integer_corr_delay_1d,
    parabolic_corr_delay_1d,
)
from noise import add_noise_to_stokes
from estimation import iqr_filter, adaptive_weighted_mean
from scipy.signal import butter, filtfilt

# ========================= Configuration =========================
L_km, L = 0.5, 0.5e3
L_F, D_pmd = 20.0, 2.5298e-15
lambda0, c = 1550e-9, 299792458.0
omega0 = 2 * np.pi * c / lambda0
A_pulse = 10 * 3.1e-4

N_CHANNELS = 30
f_min, f_max = 184e12, 196e12
freq_channels = np.linspace(f_min, f_max, N_CHANNELS)
omega_channels = 2 * np.pi * freq_channels
wavelengths_nm = frequency_to_wavelength(freq_channels)

event_distance_km = 500.0

N_BANDWIDTHS, N_REAL = 30, 20
N_SNRS = N_BANDWIDTHS

BANDWIDTHS = np.logspace(np.log10(.2e3), np.log10(2e6), N_BANDWIDTHS)
SNRS = np.linspace(50, 100, N_SNRS)

FIBER_SEED = 0
BASE_NOISE_SEED = 0
FMAX_FACTOR = 2.0

OUTPUT_FILE = "sweep_rotvec.npz"

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

# Control pair: (0, N_CHANNELS-1) – largest wavelength separation
ctrl_pair = (0, N_CHANNELS - 1)
# Find its index in the pair_indices list (to extract later)
ctrl_idx = pair_indices.index(ctrl_pair)

# ========================= Storage =============================
shape = (len(BANDWIDTHS), len(SNRS), N_REAL)
phase_est = np.full(shape, np.nan)
phat_est  = np.full(shape, np.nan)
int_est   = np.full(shape, np.nan)
par_est   = np.full(shape, np.nan)

# Control pair raw distance estimates (no IQR, no aggregation)
ctrl_phase = np.full(shape, np.nan)
ctrl_phat  = np.full(shape, np.nan)
ctrl_int   = np.full(shape, np.nan)
ctrl_par   = np.full(shape, np.nan)

# For demo (figure 6) and global inlier lists (figure 8)
demo_bw = BANDWIDTHS[0]
demo_snr = SNRS[-1]
demo_phase_pairs = None
demo_phat_pairs = None
demo_int_pairs = None
demo_par_pairs = None

phase_inlier_pairs = []
phat_inlier_pairs  = []
int_inlier_pairs   = []
par_inlier_pairs   = []

# ========================= Main sweep ===========================
total_bw = len(BANDWIDTHS)
bw_times = []

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

    # --- Jones matrices and CD delays (same as before) ---
    U_all = np.zeros((N_CHANNELS, n_samples, 2, 2), dtype=complex)
    for ch_idx, omega_ch in enumerate(omega_channels):
        beta_base = beta0 + (omega_ch - omega0) * beta_prime
        for t_idx in range(n_samples):
            U_all[ch_idx, t_idx] = propagate_unitary(z, s_env[t_idx] * beta_base)

    for ch in range(N_CHANNELS):
        U_all[ch] = delay_jones_sequence(U_all[ch], dt, channel_delays[ch])

    # --- Clean Stokes (for noise generation) ---
    s_in = np.array([1.0, 0.0, 0.0])
    stokes_clean = np.zeros((N_CHANNELS, n_samples, 3))
    for ch in range(N_CHANNELS):
        for t in range(n_samples):
            stokes_clean[ch, t] = jones_to_rotation_matrix(U_all[ch, t]) @ s_in

    # --- Convert Jones matrices to rotation‑vector magnitudes ---
    phi_mag = np.zeros((N_CHANNELS, n_samples))
    for ch in range(N_CHANNELS):
        quats = np.zeros((n_samples, 4))
        for t in range(n_samples):
            quats[t] = jones_to_quaternion(U_all[ch, t])
        quats = regularize_signs(quats)
        for t in range(n_samples):
            phi = quaternion_to_rotation_vector(quats[t])
            phi_mag[ch, t] = np.linalg.norm(phi)

    f_max = FMAX_FACTOR * bw
    nyq = 0.5 * fs
    b_lp, a_lp = butter(4, 3*bw/nyq, btype='low')

    for i_snr, snr_db in enumerate(SNRS):
        for i_real in range(N_REAL):
            rng = np.random.RandomState(BASE_NOISE_SEED + i_real)

            # Add noise to Stokes (as before), then recompute phi_mag from noisy Stokes?
            # Actually, the rotation vector is derived from the Jones matrices, which are noiseless.
            # The noise is added to the Stokes data only; the rotation vector remains deterministic.
            # To be consistent with earlier work, we add noise *after* computing phi_mag.
            # We'll add noise directly to the scalar phi_mag signals.
            # SNR definition: signal power = mean over time of phi_mag^2 for each channel,
            # then add Gaussian noise per channel.
            stokes_noisy = np.array([add_noise_to_stokes(stokes_clean[ch], snr_db, rng)
                                     for ch in range(N_CHANNELS)])
            # But the rotation vector was derived from U_all, not from stokes_noisy.
            # To keep physical consistency, we should derive a "noisy" phi_mag by using
            # the noisy Stokes to reconstruct an equivalent Jones matrix? That's complicated.
            # Simplification: treat phi_mag as the clean signal and add noise directly to it.
            # This matches the methodology where the noise is added to the SOP signal,
            # but here the signal is phi_mag. We'll add noise to phi_mag with the same SNR definition.
            # Compute noise variance per channel.
            signal_power = np.mean(phi_mag**2, axis=1)   # shape (N_CHANNELS,)
            snr_lin = 10**(snr_db / 10.0)
            noise_var = signal_power / snr_lin
            phi_noisy = phi_mag + np.sqrt(noise_var[:, None]) * rng.randn(*phi_mag.shape)

            # Low‑pass filter each channel
            phi_filt = np.zeros_like(phi_noisy)
            for ch in range(N_CHANNELS):
                phi_filt[ch] = filtfilt(b_lp, a_lp, phi_noisy[ch])
                            
            # Remove DC component (mean) from each channel
            for ch in range(N_CHANNELS):
                phi_filt[ch] = phi_filt[ch] - np.mean(phi_filt[ch])

            # Delay estimation for all pairs
            phase_delays = np.zeros(n_pairs)
            phat_delays  = np.zeros(n_pairs)
            int_delays   = np.zeros(n_pairs)
            par_delays   = np.zeros(n_pairs)

            for idx, (i, j) in enumerate(pair_indices):
                # i = slower (later), j = faster (earlier)
                sig_ref = phi_filt[j]    # earlier arrival
                sig_test = phi_filt[i]   # later arrival

                phase_delays[idx] = phase_slope_delay_1d(sig_ref, sig_test, dt, True)
                phat_delays[idx]  = gcc_phat_1d(sig_ref, sig_test, dt, f_max, 0.05)
                int_delays[idx]   = integer_corr_delay_1d(sig_ref, sig_test, dt)
                par_delays[idx]   = parabolic_corr_delay_1d(sig_ref, sig_test, dt)

            # Convert to distances
            phase_d = phase_delays * 1e12 / integrals_abs
            phat_d  = phat_delays  * 1e12 / integrals_abs
            int_d   = int_delays   * 1e12 / integrals_abs
            par_d   = par_delays   * 1e12 / integrals_abs

            # Control pair: raw distance (no filtering/aggregation)
            ctrl_phase[i_bw, i_snr, i_real] = phase_d[ctrl_idx]
            ctrl_phat[i_bw, i_snr, i_real]  = phat_d[ctrl_idx]
            ctrl_int[i_bw, i_snr, i_real]   = int_d[ctrl_idx]
            ctrl_par[i_bw, i_snr, i_real]   = par_d[ctrl_idx]

            # IQR filtering and adaptive weighted mean for all methods
            phase_mask = iqr_filter(phase_d)
            phat_mask  = iqr_filter(phat_d)
            int_mask   = iqr_filter(int_d)
            par_mask   = iqr_filter(par_d)

            phase_inlier_pairs.append(phase_d[phase_mask])
            phat_inlier_pairs.append(phat_d[phat_mask])
            int_inlier_pairs.append(int_d[int_mask])
            par_inlier_pairs.append(par_d[par_mask])

            if (abs(bw - demo_bw) < 1e-3 and abs(snr_db - demo_snr) < 1e-3
                and i_real == 0):
                demo_phase_pairs = phase_d.copy()
                demo_phat_pairs  = phat_d.copy()
                demo_int_pairs   = int_d.copy()
                demo_par_pairs   = par_d.copy()

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

# ========================= Save ==============================
np.savez_compressed(OUTPUT_FILE,
                    BANDWIDTHS=BANDWIDTHS,
                    SNRS=SNRS,
                    event_distance_km=event_distance_km,
                    phase_est=phase_est,
                    phat_est=phat_est,
                    int_est=int_est,
                    par_est=par_est,
                    ctrl_phase=ctrl_phase,
                    ctrl_phat=ctrl_phat,
                    ctrl_int=ctrl_int,
                    ctrl_par=ctrl_par,
                    demo_bw=demo_bw,
                    demo_snr=demo_snr,
                    demo_phase_pairs=demo_phase_pairs,
                    demo_phat_pairs=demo_phat_pairs,
                    demo_int_pairs=demo_int_pairs,
                    demo_par_pairs=demo_par_pairs,
                    phase_inlier_pairs=np.concatenate(phase_inlier_pairs) if phase_inlier_pairs else np.array([]),
                    phat_inlier_pairs=np.concatenate(phat_inlier_pairs) if phat_inlier_pairs else np.array([]),
                    int_inlier_pairs=np.concatenate(int_inlier_pairs) if int_inlier_pairs else np.array([]),
                    par_inlier_pairs=np.concatenate(par_inlier_pairs) if par_inlier_pairs else np.array([]),
                    bw_times=np.array(bw_times),
                    pair_indices=np.array(pair_indices),
                    wavelengths_nm=wavelengths_nm)
print(f"Data saved to {OUTPUT_FILE}")